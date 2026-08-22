"""Grounded answer generation with evidence sufficiency check and intelligent refusal.

This module implements the GroundedGenerator class that produces cited, evidence-backed
responses from retrieved documents. It strictly enforces grounding and refuses queries
when retrieved context is insufficient, missing, or lexically uncovered by the evidence.

Answer-language selection is decoupled from mere script detection: the caller can pass an
explicit ``language`` (``"en"`` | ``"hi"`` | ``"hi-en"``) coming from the Query Analyzer so
that an English or Hinglish (Romanized Hindi) query is *not* answered in Devanagari Hindi
just because Hindi happens to be supported. When no language is supplied the generator falls
back to Devanagari script detection, preserving the original behaviour.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Sequence

from modules.generation import GeneratedAnswer, Generator
from modules.retrieval import RetrievedDocument

logger = logging.getLogger(__name__)

# Canonical answer-language codes (aligned with QueryAnalyzer.QueryLanguage values).
_LANG_HI = "hi"
_LANG_HINGLISH = "hi-en"
_LANG_EN = "en"
_SUPPORTED_LANGS = frozenset({_LANG_HI, _LANG_HINGLISH, _LANG_EN})

# Low-content function/question words excluded from the content-coverage gate.
# Includes English, Romanized-Hindi, and Devanagari function words so the gate measures
# overlap on *content* terms only. This never lowers a score threshold — it can only add
# an additional refusal when the cited evidence shares zero content terms with the query.
_GATE_STOPWORDS: frozenset[str] = frozenset({
    # English function / question words
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "but", "if", "as", "it", "its", "this", "that", "these", "those",
    "what", "who", "when", "where", "which", "why", "how", "whom", "whose",
    "can", "could", "would", "should", "shall", "will", "may", "might", "must",
    "i", "you", "he", "she", "we", "they", "me", "my", "your", "our", "their",
    "about", "into", "over", "than", "then", "there", "here", "so", "such",
    # Romanized-Hindi function / question words
    "kya", "kyaa", "hai", "hain", "kaise", "kaisa", "kaisi", "kyun", "kyon",
    "kaun", "kab", "kahan", "kahaan", "mein", "me", "ka", "ki", "ke", "ko",
    "se", "par", "aur", "ya", "nahi", "nahin", "bhi", "to", "jo", "yeh", "ye",
    "woh", "wo", "hota", "hoti", "hote", "karta", "karti", "karte", "kar",
    "raha", "rahi", "rahe", "hume", "humein", "apna", "apni",
    # Devanagari function / question words
    "क्या", "है", "हैं", "था", "थी", "थे", "की", "का", "के", "को", "से", "में",
    "पर", "और", "या", "तो", "भी", "नहीं", "यह", "वह", "इस", "उस", "जो",
    "कैसे", "कैसा", "कैसी", "कौन", "कब", "कहाँ", "कहां", "करता", "करती",
    "करते", "किस", "तरह", "हुए", "हुई", "रहा", "रही", "रहे",
})

# Content-token pattern: word characters (Unicode) plus the Devanagari block, explicit for clarity.
_TOKEN_RE = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)


class GroundedGenerator(Generator):
    """Grounded answer generator with strict evidence thresholding and refusal fallback."""

    def __init__(
        self,
        llm_provider: str = "auto",
        api_key: str | None = None,
        confidence_threshold: float = 0.15,
    ) -> None:
        """Initialize GroundedGenerator.

        Args:
            llm_provider: Provider choice ('auto', 'gemini', 'openai', 'fallback')
            api_key: Optional API key for external LLM provider
            confidence_threshold: Minimum retrieval score required to attempt answer generation
        """
        self.llm_provider = llm_provider
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.confidence_threshold = confidence_threshold

    def is_hindi_query(self, query: str) -> bool:
        """Detect if query contains Devanagari script or Hindi phrasing."""
        return any('ऀ' <= char <= 'ॿ' for char in query)

    def _resolve_language(self, query: str, language: str | None) -> str:
        """Resolve the answer language.

        Prefers an explicit, caller-supplied language (from the Query Analyzer) so that
        English and Hinglish queries are answered in kind. Falls back to Devanagari script
        detection when nothing usable is supplied, preserving the original behaviour and
        keeping the unit-test contract (Devanagari → Hindi, otherwise English) intact.
        """
        if language:
            normalized = language.strip().lower()
            if normalized in _SUPPORTED_LANGS:
                return normalized
        return _LANG_HI if self.is_hindi_query(query) else _LANG_EN

    def _content_tokens(self, text: str) -> set[str]:
        """Extract content tokens (len ≥ 2, excluding function/question words)."""
        return {
            tok
            for tok in _TOKEN_RE.findall(text.lower())
            if len(tok) >= 2 and tok not in _GATE_STOPWORDS
        }

    async def generate(
        self,
        query: str,
        context: list[RetrievedDocument],
        language: str | None = None,
    ) -> GeneratedAnswer:
        """Generate an evidence-grounded answer or refuse if evidence is insufficient.

        Args:
            query: User text/voice query
            context: Retrieved candidate documents with scores and metadata
            language: Optional answer-language code ("en" | "hi" | "hi-en") from the
                Query Analyzer. When omitted, language is inferred from the query script.

        Returns:
            GeneratedAnswer object containing answer text, source list, refusal flag
        """
        resolved_lang = self._resolve_language(query, language)

        if not query or not query.strip():
            return GeneratedAnswer(
                text="Please provide a valid query.",
                sources=[],
                refused=True,
                refusal_reason="Empty or whitespace query provided.",
            )

        # 1. Evidence Sufficiency Check
        if not context:
            return self._build_refusal(
                query,
                "No relevant documents found in the available corpus.",
                resolved_lang,
            )

        # Find maximum retrieval score across candidate evidence
        max_score = max((doc.score for doc in context), default=0.0)
        logger.info("Retrieved %d candidate documents. Max score: %.4f", len(context), max_score)

        if max_score < self.confidence_threshold:
            return self._build_refusal(
                query,
                f"Retrieved evidence top score ({max_score:.3f}) is below confidence threshold ({self.confidence_threshold:.3f}).",
                resolved_lang,
            )

        # Filter top evidence documents above threshold
        relevant_sources = [doc for doc in context if doc.score >= self.confidence_threshold * 0.5][:5]
        if not relevant_sources:
            relevant_sources = context[:3]

        # 1b. Lexical content-coverage gate.
        # Guards against score inflation (e.g. min-max normalization pushing an unrelated
        # top hit toward 1.0) by requiring the cited evidence to share at least one content
        # term with the query. This strengthens grounding without weakening any threshold.
        query_content = self._content_tokens(query)
        if query_content:
            evidence_content: set[str] = set()
            for doc in relevant_sources:
                evidence_content |= self._content_tokens(doc.text)
            if not (query_content & evidence_content):
                return self._build_refusal(
                    query,
                    "Retrieved evidence does not lexically cover any content term in the query.",
                    resolved_lang,
                )

        # 2. Grounded Answer Synthesis
        try:
            # Check if an external LLM key is configured
            if self.api_key and self.llm_provider != "fallback":
                answer_text = await self._generate_with_llm(query, relevant_sources, resolved_lang)
            else:
                answer_text = self._generate_deterministic_grounded_answer(query, relevant_sources, resolved_lang)

            return GeneratedAnswer(
                text=answer_text,
                sources=relevant_sources,
                refused=False,
                refusal_reason=None,
            )
        except Exception as exc:
            logger.warning("External LLM generation failed, using deterministic grounded fallback: %s", exc)
            fallback_text = self._generate_deterministic_grounded_answer(query, relevant_sources, resolved_lang)
            return GeneratedAnswer(
                text=fallback_text,
                sources=relevant_sources,
                refused=False,
                refusal_reason=None,
            )

    def _build_refusal(
        self, query: str, reason: str, language: str | None = None
    ) -> GeneratedAnswer:
        """Construct a polite, language-aware refusal response."""
        lang = self._resolve_language(query, language)
        if lang == _LANG_HI:
            refusal_text = (
                "उपलब्ध साक्ष्यों के आधार पर इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली। "
                "कृपया अपना प्रश्न पुनः बदलें या उपलब्ध दस्तावेजों की जांच करें।"
            )
        elif lang == _LANG_HINGLISH:
            refusal_text = (
                "Available evidence ke aadhaar par is sawaal ka bharosemand jawab dene ke liye "
                "paryaapt jaankari nahi mili. Kripya apna sawaal dobara likhein ya available "
                "documents ka scope check karein."
            )
        else:
            refusal_text = (
                "I couldn't find enough evidence in the available sources to answer this question reliably. "
                "Please try rephrasing your query or checking available documentation scope."
            )

        return GeneratedAnswer(
            text=refusal_text,
            sources=[],
            refused=True,
            refusal_reason=reason,
        )

    def _generate_deterministic_grounded_answer(
        self, query: str, sources: Sequence[RetrievedDocument], language: str | None = None
    ) -> str:
        """Produce an evidence-backed answer strictly using retrieved text without external LLM call."""
        lang = self._resolve_language(query, language)
        primary_source = sources[0]
        excerpt = primary_source.text.strip()
        doc_title = (
            primary_source.metadata.get("title")
            or primary_source.metadata.get("source")
            or f"Document {primary_source.chunk_id}"
        )

        if lang == _LANG_HI:
            answer = f"प्राप्त साक्ष्यों (स्रोत: {doc_title}) के अनुसार:\n\n{excerpt}"
            if len(sources) > 1:
                answer += f"\n\nअतिरिक्त संदर्भ ({len(sources)-1} अन्य स्रोत भी मिले)।"
        elif lang == _LANG_HINGLISH:
            answer = f"Retrieved evidence ({doc_title}) ke anusaar:\n\n\"{excerpt}\""
            if len(sources) > 1:
                answer += f"\n\n(Total {len(sources)} evidence passages mile.)"
        else:
            answer = f"Based on retrieved evidence ({doc_title}):\n\n\"{excerpt}\""
            if len(sources) > 1:
                answer += f"\n\n(Supported by {len(sources)} total evidence passages)."

        return answer

    async def _generate_with_llm(
        self, query: str, sources: Sequence[RetrievedDocument], language: str | None = None
    ) -> str:
        """Call external LLM provider to synthesize a grounded answer."""
        lang = self._resolve_language(query, language)
        lang_instruction = {
            _LANG_HI: "Respond in clean, natural Hindi using Devanagari script.",
            _LANG_HINGLISH: (
                "Respond in natural Hinglish (Romanized Hindi mixed with English), "
                "matching the conversational style of the query. Do not switch to Devanagari."
            ),
            _LANG_EN: "Respond in clear, natural English.",
        }[lang]

        # Simple client fallback wrapper for Gemini or OpenAI API
        context_str = "\n\n".join(
            f"[Source {idx+1}: {doc.metadata.get('title', doc.chunk_id)}]\n{doc.text}"
            for idx, doc in enumerate(sources)
        )
        prompt = (
            f"You are ThinkZen, a grounded multilingual AI assistant. "
            f"Answer the query using ONLY the provided context snippets below. "
            f"Do not hallucinate or add facts not present in the context. "
            f"{lang_instruction}\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"QUERY: {query}\n\n"
            f"GROUNDED ANSWER:"
        )

        # Attempt google.genai if available
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            if response and hasattr(response, "text") and response.text:
                return response.text.strip()
        except Exception:
            pass

        # Fallback to deterministic synthesis if LLM library isn't present
        return self._generate_deterministic_grounded_answer(query, sources, lang)
