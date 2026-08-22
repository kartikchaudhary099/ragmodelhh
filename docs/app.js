/* ============================================================
   ThinkZen â€” voice-first frontend controller

   Honesty contract (do not break):
   â€¢ Voice uses the browser-native Web Speech API (real, client-side).
     No fake audio levels; the orb animation is STATE-driven only.
   â€¢ Every number/label shown comes from the real API:
       POST /api/v1/query   â†’ answer, sources, telemetry
       GET  /api/v1/judge   â†’ aggregate P50/P70/P100 (+ honest low-sample note)
   â€¢ Refusal â‰  Error. A refusal is a valid grounded decision (amber, calm).
     The red error state is reserved for real network/server/STT failures.
   â€¢ Nothing is hard-coded or mocked. Missing values render as "â€”".
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
    // ---- element refs -------------------------------------------------
    const $ = (id) => document.getElementById(id);

    const queryInput   = $('queryInput');
    const sendBtn      = $('sendBtn');
    const micBtn       = $('micBtn');
    const voiceOrb     = $('voiceOrb');
    const sttLang      = $('sttLang');

    const adaptiveToggle = $('adaptiveToggle');
    const alphaRange     = $('alphaRange');
    const alphaVal       = $('alphaVal');
    const alphaGroup     = $('alphaGroup');

    const statusPill = $('statusPill');
    const statusText = $('statusText');
    const orbLabel   = $('orbLabel');
    const orbHint    = $('orbHint');

    const judgeModeToggle = $('judgeModeToggle');
    const judgePanel      = $('judgePanel');

    const errorBanner  = $('errorBanner');
    const errorMessage = $('errorMessage');
    const errorDismiss = $('errorDismiss');
    const loadingCard  = $('loadingCard');

    const answerCard        = $('answerCard');
    const groundingBadge    = $('groundingBadge');
    const queryIntel        = $('queryIntel');
    const refusalCard       = $('refusalCard');
    const refusalSuggest    = $('refusalSuggest');
    const answerContent     = $('answerContent');
    const evidenceCount     = $('evidenceCount');
    const evidenceList      = $('evidenceList');

    // telemetry / pipeline
    const pipeline         = $('pipeline');
    const psteps           = pipeline ? Array.from(pipeline.querySelectorAll('.pstep')) : [];
    const telLang          = $('telLang');
    const telRetTime       = $('telRetTime');
    const telAlpha         = $('telAlpha');
    const telAlphaSource   = $('telAlphaSource');
    const telGrounding     = $('telGrounding');
    const telGenTime       = $('telGenTime');
    const telTotalMs       = $('telTotalMs');
    const telRetrievalMs   = $('telRetrievalMs');
    const telGenerationMs  = $('telGenerationMs');
    const telCandidates    = $('telCandidates');
    const telEvidenceCount = $('telEvidenceCount');
    const telDetectedLang  = $('telDetectedLang');
    const telQueryType     = $('telQueryType');
    const telComplexity    = $('telComplexity');
    const telStrategy      = $('telStrategy');
    const telAdaptiveAlpha = $('telAdaptiveAlpha');
    const telRationale     = $('telRationale');
    const telKeywords      = $('telKeywords');
    const telEbDecision    = $('telEbDecision');
    const telEbDiversity   = $('telEbDiversity');
    const telEbMaxScore    = $('telEbMaxScore');
    const telEbCoherence   = $('telEbCoherence');
    const telEbReason      = $('telEbReason');

    // aggregate
    const aggRuns = $('aggRuns');
    const aggP50  = $('aggP50');
    const aggP70  = $('aggP70');
    const aggP100 = $('aggP100');
    const aggNote = $('aggNote');

    // new UI chrome (all optional â€” guarded everywhere)
    const clearBtn        = $('clearBtn');
    const navAsk          = $('navAsk');
    const waveform        = $('waveform');
    const sysCorpus       = $('sysCorpus');
    const sysStatusDot    = $('sysStatusDot');
    const sysStatusText   = $('sysStatusText');
    const capDatasetVal   = $('capDatasetVal');
    const evidenceFoot    = $('evidenceFoot');
    const evidenceFootText= $('evidenceFootText');

    const prefersReducedMotion = window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const LANG_LABELS = { en: 'English', hi: 'à¤¹à¤¿à¤‚à¤¦à¥€ (Hindi)', 'hi-en': 'Hinglish', unknown: 'Auto-detect' };
    const LANG_SHORT  = { en: 'EN', hi: 'HI', 'hi-en': 'HI-EN', unknown: 'AUTO' };
    const langLabel = (c) => LANG_LABELS[c] || (c || 'â€”');

    // ---- helpers ------------------------------------------------------
    function escapeHtml(str) {
        return (str == null ? '' : String(str))
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }
    const fmtMs = (v) => (v == null || isNaN(v)) ? 'â€”' : `${Number(v).toFixed(2)} ms`;

    // ---- orb / status state machine -----------------------------------
    // A single source of truth for the visible app state. State is derived
    // from real events (mic listening, request in-flight, real API result).
    const ORB = {
        idle:       { label: 'Ask ThinkZen',            hint: 'Speak or type in Hindi, English, or Hinglish', pill: 'Ready',      cls: 'status-idle' },
        listening:  { label: 'Listeningâ€¦',           hint: 'Speak now â€” tap the mic again to stop',        pill: 'Listening',  cls: 'status-listening' },
        processing: { label: 'Searching evidenceâ€¦',  hint: 'Dense + BM25 retrieval and grounding check',   pill: 'Processing', cls: 'status-processing' },
        answer:     { label: 'Evidence found',       hint: 'Grounded answer ready below',                  pill: 'Ready',      cls: 'status-ready' },
        refused:    { label: 'Not enough evidence',  hint: 'Try a question about the indexed corpus',      pill: 'Ready',      cls: 'status-idle' },
        error:      { label: 'Something went wrong', hint: 'See the message below and try again',          pill: 'Error',      cls: 'status-error' },
    };
    function setState(state) {
        const s = ORB[state] || ORB.idle;
        if (voiceOrb) voiceOrb.setAttribute('data-state', state);
        document.body.setAttribute('data-appstate', state);
        if (orbLabel) orbLabel.textContent = s.label;
        if (orbHint)  orbHint.textContent  = s.hint;
        if (statusText) statusText.textContent = s.pill;
        if (statusPill) statusPill.className = `status-pill ${s.cls}`;
        // mirror the same real state into the sidebar status card
        if (sysStatusText) sysStatusText.textContent = s.pill;
        if (sysStatusDot)  sysStatusDot.className = `sys-dot ${s.cls}`;
        // mic button reflects listening/processing
        if (micBtn) {
            micBtn.classList.toggle('listening', state === 'listening');
            micBtn.classList.toggle('processing', state === 'processing');
        }
    }

    // ---- error banner (real errors only) ------------------------------
    function showError(msg) {
        if (errorMessage) errorMessage.textContent = msg;
        if (errorBanner) errorBanner.classList.remove('hidden');
    }
    function hideError() { if (errorBanner) errorBanner.classList.add('hidden'); }
    if (errorDismiss) errorDismiss.addEventListener('click', hideError);

    // ---- adaptive alpha ------------------------------------------------
    function syncAdaptive() {
        const adaptive = adaptiveToggle.checked;
        alphaRange.disabled = adaptive;
        if (alphaGroup) alphaGroup.classList.toggle('is-disabled', adaptive);
        alphaVal.textContent = adaptive ? 'auto' : parseFloat(alphaRange.value).toFixed(2);
    }
    alphaRange.addEventListener('input', () => {
        if (!adaptiveToggle.checked) alphaVal.textContent = parseFloat(alphaRange.value).toFixed(2);
    });
    adaptiveToggle.addEventListener('change', syncAdaptive);
    syncAdaptive();

    // ---- judge mode toggle --------------------------------------------
    judgeModeToggle.addEventListener('change', (e) => {
        judgePanel.classList.toggle('hidden', !e.target.checked);
    });

    // ---- textarea auto-grow + enter to submit -------------------------
    function autoGrow() {
        queryInput.style.height = 'auto';
        queryInput.style.height = Math.min(queryInput.scrollHeight, 168) + 'px';
    }
    queryInput.addEventListener('input', autoGrow);
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); executeSearch(); }
    });

    // ---- example chips (each runs a real query) -----------------------
    const exampleChips = Array.from(document.querySelectorAll('#exampleChips .chip'));
    exampleChips.forEach((chip) => {
        chip.addEventListener('click', () => {
            const q = chip.getAttribute('data-query');
            if (q) { queryInput.value = q; autoGrow(); executeSearch(); }
        });
    });
    // Build the refusal "try instead" suggestions from the real grounded examples.
    function buildRefusalSuggestions() {
        if (!refusalSuggest) return;
        refusalSuggest.innerHTML = '';
        exampleChips
            .filter((c) => !c.classList.contains('chip-refuse'))
            .forEach((c) => {
                const q = c.getAttribute('data-query');
                const b = document.createElement('button');
                b.type = 'button';
                b.className = 'chip';
                b.textContent = c.textContent;
                b.addEventListener('click', () => { queryInput.value = q; autoGrow(); executeSearch(); });
                refusalSuggest.appendChild(b);
            });
    }
    buildRefusalSuggestions();

    // ---- decorative waveform behind the orb (visual only, state-driven) ----
    function buildWaveform() {
        if (!waveform) return;
        const N = 48;
        const stops = [[34,211,238],[59,130,246],[124,58,237],[217,70,239]]; // cyanâ†’blueâ†’violetâ†’magenta
        const lerp = (a, b, t) => Math.round(a + (b - a) * t);
        for (let i = 0; i < N; i++) {
            const t = i / (N - 1);
            const seg = t * (stops.length - 1);
            const si = Math.min(stops.length - 2, Math.floor(seg));
            const f = seg - si;
            const c = [0, 1, 2].map((k) => lerp(stops[si][k], stops[si + 1][k], f));
            const h = 0.16 + 0.84 * Math.exp(-Math.pow((t - 0.5) / 0.3, 2)); // bell curve
            const bar = document.createElement('span');
            bar.className = 'wf-bar';
            bar.style.setProperty('--i', i);
            bar.style.setProperty('--h', h.toFixed(3));
            bar.style.background = `rgb(${c[0]},${c[1]},${c[2]})`;
            waveform.appendChild(bar);
        }
    }
    buildWaveform();

    // ---- language segmented control â†’ drives the (hidden) recognition <select> ----
    const langSegs = Array.from(document.querySelectorAll('.lang-seg-btn'));
    langSegs.forEach((btn) => {
        btn.addEventListener('click', () => {
            const code = btn.getAttribute('data-stt') || 'en-IN';
            if (sttLang) sttLang.value = code;
            langSegs.forEach((b) => {
                const on = b === btn;
                b.classList.toggle('is-active', on);
                b.setAttribute('aria-selected', on ? 'true' : 'false');
            });
        });
    });

    // ---- sidebar "Ask" focuses the composer ----
    if (navAsk) navAsk.addEventListener('click', () => { queryInput.focus(); });

    // ---- Clear: resets ONLY the current interaction (aggregate history is preserved) ----
    function clearInteraction() {
        queryInput.value = '';
        autoGrow();
        hideError();
        answerCard.classList.add('hidden');
        loadingCard.classList.add('hidden');
        queryIntel.innerHTML = '';
        evidenceList.innerHTML = '';
        evidenceCount.textContent = '0 sources';
        if (evidenceFoot) evidenceFoot.hidden = true;
        if (telKeywords) telKeywords.innerHTML = '';
        [telLang, telRetTime, telGrounding, telGenTime, telTotalMs, telRetrievalMs,
         telGenerationMs, telCandidates, telEvidenceCount, telDetectedLang, telQueryType,
         telComplexity, telStrategy, telAdaptiveAlpha, telRationale, telEbDecision,
         telEbDiversity, telEbMaxScore, telEbCoherence, telEbReason]
            .forEach((el) => { if (el) el.textContent = 'â€”'; });
        if (telAlpha) telAlpha.textContent = 'Î± = â€”';
        if (telAlphaSource) telAlphaSource.textContent = 'adaptive';
        psteps.forEach((s) => s.classList.remove('is-active', 'is-refused'));
        setState('idle');
        // NOTE: aggRuns / aggP50 / aggP70 / aggP100 / aggNote and sysCorpus are NOT reset.
    }
    if (clearBtn) clearBtn.addEventListener('click', clearInteraction);

    // ---- Web Speech API (primary, real voice path) --------------------
    let recognition = null;
    let isListening = false;
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SR) {
        recognition = new SR();
        recognition.continuous = false;
        recognition.interimResults = true;

        recognition.onstart = () => { isListening = true; setState('listening'); };
        recognition.onresult = (event) => {
            let transcript = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                transcript += event.results[i][0].transcript;
            }
            queryInput.value = transcript;
            autoGrow();
        };
        recognition.onerror = (event) => {
            isListening = false;
            // Only a genuine permission failure surfaces as a real error.
            if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
                setState('error');
                showError('Microphone access was blocked. Allow mic permission in your browser, or type your question instead.');
            } else if (event.error === 'no-speech') {
                setState('idle'); // benign: nothing captured
            } else {
                setState('idle');
            }
        };
        recognition.onend = () => {
            isListening = false;
            // Auto-submit when speech produced text; otherwise return to idle.
            if (queryInput.value.trim().length > 0) executeSearch();
            else if (voiceOrb && voiceOrb.getAttribute('data-state') === 'listening') setState('idle');
        };
    } else if (micBtn) {
        micBtn.classList.add('unsupported');
        micBtn.title = 'Speech recognition is not available in this browser (try Chrome or Edge).';
    }

    function toggleVoice() {
        if (!recognition) {
            showError('Speech recognition is not supported in this browser. Please type your question â€” everything else works the same.');
            return;
        }
        if (isListening) {
            recognition.stop();
            return;
        }
        hideError();
        queryInput.value = '';
        autoGrow();
        // Honour the selected voice language (falls back to English-India).
        recognition.lang = sttLang ? sttLang.value : 'en-IN';
        try { recognition.start(); }
        catch (err) { console.warn('Could not start recognition:', err); }
    }

    if (micBtn)   micBtn.addEventListener('click', toggleVoice);
    // voiceOrb is a native <button>, so Enter/Space activation already fires
    // 'click' â€” binding click alone avoids a double-trigger on keyboard use.
    if (voiceOrb) voiceOrb.addEventListener('click', toggleVoice);

    sendBtn.addEventListener('click', () => executeSearch());

    // ---- main query flow ----------------------------------------------
    function setLoading(loading) {
        loadingCard.classList.toggle('hidden', !loading);
        sendBtn.disabled = loading;
    }

    async function executeSearch() {
        const query = queryInput.value.trim();
        if (!query) {
            setState('error');
            showError('Please type or speak a question first.');
            return;
        }

        hideError();
        setState('processing');
        answerCard.classList.add('hidden');
        setLoading(true);

        // Build request. Adaptive mode omits alpha so the Query Analyzer selects it
        // (alpha_source = "adaptive"); otherwise send the slider value (alpha_source = "override").
        const body = {
            query,
            top_k: 5,
            confidence_threshold: 0.15,
            use_adaptive_retrieval: adaptiveToggle.checked,
        };
        if (!adaptiveToggle.checked) body.alpha = parseFloat(alphaRange.value);

        try {
            const response = await fetch('/api/v1/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });

            if (!response.ok) {
                let detail = `HTTP ${response.status}`;
                try { const j = await response.json(); if (j && j.detail) detail = j.detail; } catch (_) {}
                throw new Error(detail);
            }

            const data = await response.json();
            renderResponse(data);
            // A refusal is NOT an error â€” reflect the real decision in the orb.
            setState(data.refused ? 'refused' : 'answer');
            // Refresh aggregate latency from the real Judge endpoint.
            refreshAggregate();
        } catch (err) {
            console.error('Query execution error:', err);
            setState('error');
            showError(`Request failed: ${err.message}`);
        } finally {
            setLoading(false);
        }
    }

    // ---- render -------------------------------------------------------
    const ICON_CHECK = '<svg class="verdict-ic" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>';
    const ICON_INFO  = '<svg class="verdict-ic" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/></svg>';

    function renderResponse(data) {
        answerCard.classList.remove('hidden');

        // ---- verdict + refusal (refusal â‰  error) ----
        if (data.refused) {
            groundingBadge.className = 'verdict verdict-refused';
            groundingBadge.innerHTML = ICON_INFO + '<span>Not enough evidence</span>';
            refusalCard.classList.remove('hidden');
            // NORMAL view stays friendly & non-technical. The technical reason
            // (content-coverage / threshold) is shown only in Judge Mode below.
            answerContent.textContent = data.answer || '';
        } else {
            groundingBadge.className = 'verdict verdict-grounded';
            groundingBadge.innerHTML = ICON_CHECK + '<span>Grounded</span>';
            refusalCard.classList.add('hidden');
            answerContent.textContent = data.answer || '';
        }

        // ---- query intelligence chips (all from real telemetry) ----
        renderQueryIntel(data);

        // ---- evidence ----
        renderEvidence(data.sources || []);

        // ---- honest corpus / dataset + evidence footer (only from real metadata) ----
        bindCorpus(data);

        // ---- judge telemetry ----
        // refusal_reason is a TOP-LEVEL response field (not inside telemetry);
        // pass it through so the Judge panel can show the real technical reason.
        if (data.telemetry) renderTelemetry(data.telemetry, data.refused, data.refusal_reason);
    }

    function tag(text, tone) {
        return `<span class="intel-tag ${tone ? 'tone-' + tone : ''}">${escapeHtml(text)}</span>`;
    }
    function renderQueryIntel(data) {
        if (!queryIntel) return;
        const tel = data.telemetry || {};
        const qa = tel.query_analysis || {};
        const parts = [];
        const lang = tel.detected_language || qa.language;
        if (lang) parts.push(tag(LANG_SHORT[lang] || String(lang).toUpperCase(), 'cyan'));
        if (qa.query_type) parts.push(tag(String(qa.query_type).toUpperCase(), 'violet'));
        if (tel.alpha_source) {
            const a = (tel.alpha_used != null) ? Number(tel.alpha_used).toFixed(2) : '';
            parts.push(tag(`${tel.alpha_source === 'adaptive' ? 'ADAPTIVE' : 'OVERRIDE'} Î±${a ? ' ' + a : ''}`,
                tel.alpha_source === 'adaptive' ? 'cyan' : ''));
        }
        parts.push(tag('HYBRID', 'violet'));
        parts.push(tag(data.refused ? 'REFUSED' : 'GROUNDED', data.refused ? 'amber' : 'green'));
        queryIntel.innerHTML = parts.join('');
    }

    function renderEvidence(sources) {
        evidenceList.innerHTML = '';
        evidenceCount.textContent = `${sources.length} ${sources.length === 1 ? 'source' : 'sources'}`;
        if (sources.length === 0) {
            evidenceList.innerHTML = '<div class="ev-empty">No evidence passages met the grounding threshold for this query.</div>';
            return;
        }
        sources.forEach((src, i) => {
            const meta = src.metadata || {};
            const title = meta.title || meta.doc_id || src.chunk_id || `Source ${i + 1}`;
            const pct = (src.score != null && !isNaN(src.score)) ? Math.max(0, Math.min(100, src.score * 100)) : 0;
            const method = (src.method || '').toUpperCase();

            const item = document.createElement('details');
            item.className = 'ev-item';
            if (i === 0) item.open = true; // top passage expanded by default
            item.innerHTML = `
                <summary class="ev-summary">
                    <span class="ev-src">
                        <span class="ev-rank">${i + 1}</span>
                        <span class="ev-title">${escapeHtml(title)}</span>
                    </span>
                    <span class="ev-tags">
                        ${method ? `<span class="ev-method">${escapeHtml(method)}</span>` : ''}
                        <span class="ev-score"><span class="ev-bar"><span style="width:${pct.toFixed(0)}%"></span></span>${pct.toFixed(1)}%</span>
                        <svg class="ev-chevron" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>
                    </span>
                </summary>
                <div class="ev-detail">
                    <p class="ev-text">${escapeHtml(src.text)}</p>
                    <div class="ev-meta-row">chunk: ${escapeHtml(src.chunk_id || 'â€”')}${meta.doc_id ? ' Â· doc: ' + escapeHtml(meta.doc_id) : ''}${meta.language ? ' Â· ' + escapeHtml(meta.language) : ''}</div>
                </div>`;
            evidenceList.appendChild(item);
        });
    }

    // Bind corpus/dataset + the "grounded in ..." footer from REAL source metadata only.
    // If metadata carries no dataset/corpus, labels stay "â€”" and the footer text stays generic.
    function bindCorpus(data) {
        let label = null;
        for (const s of (data.sources || [])) {
            const m = s.metadata || {};
            const v = m.dataset || m.corpus;
            if (v) { label = String(v); break; }
        }
        if (label) {
            if (sysCorpus) sysCorpus.textContent = label;
            if (capDatasetVal) capDatasetVal.textContent = label;
        }
        if (evidenceFoot && evidenceFootText) {
            if (!data.refused && (data.sources || []).length > 0) {
                evidenceFootText.textContent = label
                    ? `Grounded in retrieved ${label} evidence`
                    : 'Grounded in retrieved evidence';
                evidenceFoot.hidden = false;
            } else {
                evidenceFoot.hidden = true;
            }
        }
    }

    function renderTelemetry(tel, refused) {
        const qa = tel.query_analysis || {};
        const eb = tel.evidence_bundle || {};

        // pipeline step metas
        telLang.textContent = langLabel(tel.detected_language);
        telRetTime.textContent = fmtMs(tel.retrieval_latency_ms);
        telAlpha.textContent = `Î± = ${(tel.alpha_used != null) ? Number(tel.alpha_used).toFixed(2) : 'â€”'}`;
        telAlphaSource.textContent = tel.alpha_source || 'â€”';
        telGrounding.textContent = (tel.grounding_status || 'â€”').toUpperCase();
        telGenTime.textContent = fmtMs(tel.generation_latency_ms);

        // latency hero + metric cards
        telTotalMs.textContent = fmtMs(tel.total_latency_ms);
        telRetrievalMs.textContent = fmtMs(tel.retrieval_latency_ms);
        telGenerationMs.textContent = fmtMs(tel.generation_latency_ms);
        telCandidates.textContent = (tel.candidate_count != null) ? tel.candidate_count : 'â€”';
        telEvidenceCount.textContent = (tel.evidence_count != null) ? tel.evidence_count : 'â€”';
        telDetectedLang.textContent = langLabel(tel.detected_language);

        // query analysis block
        telQueryType.textContent = qa.query_type || 'â€”';
        telComplexity.textContent = qa.complexity || 'â€”';
        telStrategy.textContent = qa.strategy_name || 'â€”';
        telAdaptiveAlpha.textContent = (qa.adaptive_alpha != null) ? Number(qa.adaptive_alpha).toFixed(2) : 'â€”';
        telRationale.textContent = qa.rationale || 'â€”';
        telKeywords.innerHTML = '';
        (qa.keywords || []).forEach((kw) => {
            const c = document.createElement('span');
            c.className = 'kw-chip';
            c.textContent = kw;
            telKeywords.appendChild(c);
        });

        // evidence intelligence block (this is where the technical reason lives)
        telEbDecision.textContent = (eb.grounding_decision || 'â€”').toUpperCase();
        telEbDiversity.textContent = (eb.source_diversity != null) ? eb.source_diversity : 'â€”';
        telEbMaxScore.textContent = (eb.max_retrieval_score != null) ? Number(eb.max_retrieval_score).toFixed(3) : 'â€”';
        telEbCoherence.textContent = (eb.mean_coherence_score != null) ? Number(eb.mean_coherence_score).toFixed(3) : 'â€”';
        telEbReason.textContent = eb.decision_reason || tel.refusal_reason || 'â€”';

        revealPipeline(refused);
    }

    // Reveal the pipeline trace AFTER real data arrives (presentation of real
    // returned values â€” never a fake "in-progress" indicator).
    function revealPipeline(refused) {
        psteps.forEach((s) => s.classList.remove('is-active', 'is-refused'));
        const apply = (idx) => {
            const step = psteps[idx];
            if (!step) return;
            // Step 04 (Evidence Intelligence) carries the refusal tone when refused.
            if (refused && step.getAttribute('data-step') === '4') step.classList.add('is-refused');
            else step.classList.add('is-active');
        };
        if (prefersReducedMotion) { psteps.forEach((_, i) => apply(i)); return; }
        psteps.forEach((_, i) => setTimeout(() => apply(i), i * 100));
    }

    // ---- aggregate latency (real GET /api/v1/judge) -------------------
    async function refreshAggregate() {
        try {
            const r = await fetch('/api/v1/judge');
            if (!r.ok) return;
            const j = await r.json();
            const stats = (j.latency_stats && j.latency_stats.REAL_RUN) || null;
            if (aggRuns) aggRuns.textContent = `${j.total_runs != null ? j.total_runs : 0} run${j.total_runs === 1 ? '' : 's'}`;
            if (stats) {
                if (aggP50)  aggP50.textContent  = fmtMs(stats.p50_ms);
                if (aggP70)  aggP70.textContent  = fmtMs(stats.p70_ms);
                if (aggP100) aggP100.textContent = fmtMs(stats.p100_ms);
                // Honest reporting: percentiles below 10 runs are not reliable.
                if (aggNote) aggNote.textContent = stats.low_sample_warning
                    || 'Live P50/P70/P100 across recorded RAG requests (excludes STT).';
            } else if (aggNote) {
                aggNote.textContent = 'Percentiles appear after live requests are recorded.';
            }
        } catch (_) { /* aggregate is best-effort; never fabricate values */ }
    }

    // ---- init ---------------------------------------------------------
    setState('idle');
    // Judge panel visibility follows the toggle's initial (checked) state.
    judgePanel.classList.toggle('hidden', !judgeModeToggle.checked);
});

