/* ============================================================
   ThinkZen — voice-first frontend controller

   Honesty contract (do not break):
   • Voice uses the browser-native Web Speech API (real, client-side).
     No fake audio levels; the orb animation is STATE-driven only.
   • Every number/label shown comes from the real API:
       POST /api/v1/query   → answer, sources, telemetry
       GET  /api/v1/judge   → aggregate P50/P70/P100 (+ honest low-sample note)
   • Refusal ≠ Error. A refusal is a valid grounded decision (amber, calm).
     The red error state is reserved for real network/server/STT failures.
   • Nothing is hard-coded or mocked. Missing values render as "—".
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

    // new UI chrome (all optional — guarded everywhere)
    const clearBtn        = $('clearBtn');
    const navAsk          = $('navAsk');
    const waveform        = $('waveform');
    const sysCorpus       = $('sysCorpus');
    const sysStatusDot    = $('sysStatusDot');
    const sysStatusText   = $('sysStatusText');
    const capDatasetVal   = $('capDatasetVal');
    const evidenceFoot    = $('evidenceFoot');
    const evidenceFootText= $('evidenceFootText');

    // new interactive elements
    const themeToggle     = $('themeToggle');
    const copyAnswerBtn   = $('copyAnswerBtn');
    const ttsBtn          = $('ttsBtn');
    const expandAnswerBtn = $('expandAnswerBtn');
    const exportBtn       = $('exportBtn');
    const rateUpBtn       = $('rateUpBtn');
    const rateDownBtn     = $('rateDownBtn');
    const historyList     = $('historyList');
    const historyClearBtn  = $('historyClearBtn');
    const historyEmpty    = $('historyEmpty');
    const toastContainer  = $('toastContainer');
    const particleCanvas  = $('particleCanvas');

    const prefersReducedMotion = window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const LANG_LABELS = { en: 'English', hi: 'हिंदी (Hindi)', 'hi-en': 'Hinglish', unknown: 'Auto-detect' };
    const LANG_SHORT  = { en: 'EN', hi: 'HI', 'hi-en': 'HI-EN', unknown: 'AUTO' };
    const langLabel = (c) => LANG_LABELS[c] || (c || '—');

    // ---- helpers ------------------------------------------------------
    function escapeHtml(str) {
        return (str == null ? '' : String(str))
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }
    const fmtMs = (v) => (v == null || isNaN(v)) ? '—' : `${Number(v).toFixed(2)} ms`;

    // ---- Toast notification system ------------------------------------
    function showToast(message, type = 'success', duration = 3000) {
        if (!toastContainer) return;
        const iconSvgs = {
            success: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
            warn: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>',
            error: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 6l12 12M18 6 6 18"/></svg>',
        };
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.innerHTML = `<span class="toast-icon ${type !== 'success' ? 'toast-' + type : ''}">${iconSvgs[type] || iconSvgs.success}</span><span class="toast-text">${escapeHtml(message)}</span>`;
        toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.classList.add('toast-out');
            toast.addEventListener('animationend', () => toast.remove());
        }, duration);
    }

    // ---- Theme toggle (dark ↔ light) ----------------------------------
    function initTheme() {
        const saved = localStorage.getItem('thinkzen-theme');
        if (saved === 'light') {
            document.documentElement.setAttribute('data-theme', 'light');
        }
    }
    initTheme();

    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            playUiSound('click');
            const isLight = document.documentElement.getAttribute('data-theme') === 'light';
            if (isLight) {
                document.documentElement.removeAttribute('data-theme');
                localStorage.setItem('thinkzen-theme', 'dark');
                showToast('Dark mode activated', 'success', 2000);
            } else {
                document.documentElement.setAttribute('data-theme', 'light');
                localStorage.setItem('thinkzen-theme', 'light');
                showToast('Light mode activated', 'success', 2000);
            }
        });
    }

    // ---- Floating particles -------------------------------------------
    function initParticles() {
        if (!particleCanvas || prefersReducedMotion) return;
        const ctx = particleCanvas.getContext('2d');
        let particles = [];
        const PARTICLE_COUNT = 35;

        function resize() {
            particleCanvas.width = window.innerWidth;
            particleCanvas.height = window.innerHeight;
        }
        resize();
        window.addEventListener('resize', resize);

        class Particle {
            constructor() { this.reset(); }
            reset() {
                this.x = Math.random() * particleCanvas.width;
                this.y = particleCanvas.height + Math.random() * 100;
                this.size = Math.random() * 2.5 + 0.5;
                this.speedY = Math.random() * 0.5 + 0.15;
                this.speedX = (Math.random() - 0.5) * 0.3;
                this.opacity = Math.random() * 0.4 + 0.1;
                this.fadeSpeed = Math.random() * 0.002 + 0.001;
                // warm tones
                const hue = 20 + Math.random() * 25; // 20-45 (orange-ish)
                const sat = 40 + Math.random() * 30;
                const light = 55 + Math.random() * 25;
                this.color = `hsla(${hue},${sat}%,${light}%,`;
            }
            update() {
                this.y -= this.speedY;
                this.x += this.speedX + Math.sin(this.y * 0.005) * 0.2;
                if (this.y < -20 || this.opacity <= 0) this.reset();
            }
            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fillStyle = this.color + this.opacity + ')';
                ctx.fill();
            }
        }

        for (let i = 0; i < PARTICLE_COUNT; i++) {
            const p = new Particle();
            p.y = Math.random() * particleCanvas.height; // spread initially
            particles.push(p);
        }

        function animate() {
            ctx.clearRect(0, 0, particleCanvas.width, particleCanvas.height);
            particles.forEach(p => { p.update(); p.draw(); });
            requestAnimationFrame(animate);
        }
        animate();
    }
    initParticles();

    // ---- Web Audio API Synthesized UI Sounds --------------------------
    let audioCtx = null;
    let soundEnabled = true;

    function playUiSound(type) {
        if (!soundEnabled) return;
        try {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (audioCtx.state === 'suspended') {
                audioCtx.resume();
            }

            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.connect(gain);
            gain.connect(audioCtx.destination);

            const now = audioCtx.currentTime;

            if (type === 'click') {
                osc.type = 'sine';
                osc.frequency.setValueAtTime(1400, now);
                osc.frequency.exponentialRampToValueAtTime(600, now + 0.035);
                
                gain.gain.setValueAtTime(0.08, now);
                gain.gain.linearRampToValueAtTime(0, now + 0.035);
                
                osc.start(now);
                osc.stop(now + 0.04);
            } else if (type === 'success') {
                osc.type = 'triangle';
                osc.frequency.setValueAtTime(880, now); // A5
                osc.frequency.setValueAtTime(1109, now + 0.08); // C#6
                
                gain.gain.setValueAtTime(0.0, now);
                gain.gain.linearRampToValueAtTime(0.05, now + 0.02);
                gain.gain.linearRampToValueAtTime(0.05, now + 0.12);
                gain.gain.linearRampToValueAtTime(0, now + 0.18);
                
                osc.start(now);
                osc.stop(now + 0.20);
            } else if (type === 'listening') {
                osc.type = 'sine';
                osc.frequency.setValueAtTime(260, now);
                osc.frequency.exponentialRampToValueAtTime(520, now + 0.16);
                
                gain.gain.setValueAtTime(0.0, now);
                gain.gain.linearRampToValueAtTime(0.06, now + 0.04);
                gain.gain.linearRampToValueAtTime(0, now + 0.18);
                
                osc.start(now);
                osc.stop(now + 0.20);
            } else if (type === 'error') {
                osc.type = 'sawtooth';
                osc.frequency.setValueAtTime(130, now);
                osc.frequency.linearRampToValueAtTime(100, now + 0.12);
                
                const filter = audioCtx.createBiquadFilter();
                filter.type = 'lowpass';
                filter.frequency.setValueAtTime(250, now);
                
                osc.disconnect(gain);
                osc.connect(filter);
                filter.connect(gain);
                
                gain.gain.setValueAtTime(0.08, now);
                gain.gain.linearRampToValueAtTime(0, now + 0.15);
                
                osc.start(now);
                osc.stop(now + 0.16);
            }
        } catch (e) {
            console.warn('UI Audio synthesis blocked/failed:', e);
        }
    }

    // ---- Interactive Face Avatar --------------------------------------
    let speechAnimInterval = null;

    function initFaceAvatar() {
        const leftEye = $('leftEye');
        const rightEye = $('rightEye');
        const leftPupil = $('leftPupil');
        const rightPupil = $('rightPupil');
        const leftEyeGroup = $('leftEyeGroup');
        const rightEyeGroup = $('rightEyeGroup');
        const mouth = $('mouth');
        if (!leftEye || !rightEye || !leftPupil || !rightPupil) return;

        // 1. Pupil Cursor Tracking
        document.addEventListener('mousemove', (e) => {
            if (document.body.getAttribute('data-appstate') === 'processing') {
                return; // Let pupils stay looking at a thinking angle
            }
            const leftRect = leftEyeGroup.getBoundingClientRect();
            const rightRect = rightEyeGroup.getBoundingClientRect();

            function getOffset(rect) {
                const eyeX = rect.left + rect.width / 2;
                const eyeY = rect.top + rect.height / 2;
                const dx = e.clientX - eyeX;
                const dy = e.clientY - eyeY;
                const dist = Math.hypot(dx, dy);
                const maxDist = 2.5; // Max offset distance in pixels
                const angle = Math.atan2(dy, dx);
                const offsetDist = Math.min(dist * 0.005, maxDist);
                return {
                    x: Math.cos(angle) * offsetDist,
                    y: Math.sin(angle) * offsetDist
                };
            }

            const leftOffset = getOffset(leftRect);
            const rightOffset = getOffset(rightRect);

            leftPupil.setAttribute('cx', leftOffset.x.toFixed(2));
            leftPupil.setAttribute('cy', leftOffset.y.toFixed(2));
            rightPupil.setAttribute('cx', rightOffset.x.toFixed(2));
            rightPupil.setAttribute('cy', rightOffset.y.toFixed(2));
        });

        // 2. Periodic Eye Blinking
        function triggerBlink() {
            leftEye.setAttribute('ry', '0.6');
            rightEye.setAttribute('ry', '0.6');
            setTimeout(() => {
                const state = document.body.getAttribute('data-appstate') || 'idle';
                let normalRy = '5.5';
                if (state === 'listening') normalRy = '6.2';
                if (state === 'processing') normalRy = '3.0';
                if (state === 'error') normalRy = '4.5';
                if (state === 'refused') normalRy = '5.2';
                
                leftEye.setAttribute('ry', normalRy);
                rightEye.setAttribute('ry', normalRy);
            }, 120);

            const nextBlink = 2500 + Math.random() * 4000;
            setTimeout(triggerBlink, nextBlink);
        }
        setTimeout(triggerBlink, 3000);
    }

    function startSpeechAnimation() {
        if (speechAnimInterval) clearInterval(speechAnimInterval);
        const mouth = $('mouth');
        const leftEye = $('leftEye');
        const rightEye = $('rightEye');
        if (!mouth) return;

        // Interactive speaking mouth shapes
        const shapes = [
            "M 33 60 Q 50 82 67 60", // Open wide
            "M 35 64 Q 50 72 65 64", // Semi-open
            "M 38 65 Q 50 68 62 65", // Small open
            "M 33 65 Q 50 69 67 65"  // Smile closed
        ];

        let step = 0;
        speechAnimInterval = setInterval(() => {
            const nextShape = shapes[step % shapes.length];
            mouth.setAttribute('d', nextShape);

            if (leftEye && rightEye) {
                const ryVal = 5.5 - (step % 2 === 0 ? 1.2 : 0);
                leftEye.setAttribute('ry', ryVal.toFixed(1));
                rightEye.setAttribute('ry', ryVal.toFixed(1));
            }
            step++;
        }, 110);
    }

    function stopSpeechAnimation() {
        if (speechAnimInterval) {
            clearInterval(speechAnimInterval);
            speechAnimInterval = null;
        }
        const mouth = $('mouth');
        const leftEye = $('leftEye');
        const rightEye = $('rightEye');
        
        const state = document.body.getAttribute('data-appstate') || 'idle';
        if (mouth) {
            if (state === 'listening') mouth.setAttribute('d', "M 36 63 Q 50 78 64 63");
            else if (state === 'processing') mouth.setAttribute('d', "M 36 65 Q 50 65 64 65");
            else if (state === 'error') mouth.setAttribute('d', "M 35 67 Q 50 62 65 67");
            else if (state === 'refused') mouth.setAttribute('d', "M 35 66 Q 50 64 65 66");
            else mouth.setAttribute('d', "M 33 65 Q 50 69 67 65");
        }
        if (leftEye && rightEye) {
            let normalRy = '5.5';
            if (state === 'listening') normalRy = '6.2';
            if (state === 'processing') normalRy = '3.0';
            if (state === 'error') normalRy = '4.5';
            if (state === 'refused') normalRy = '5.2';
            leftEye.setAttribute('ry', normalRy);
            rightEye.setAttribute('ry', normalRy);
        }
    }

    // ---- Session history management -----------------------------------
    const HISTORY_KEY = 'thinkzen-history';
    const MAX_HISTORY = 10;

    function loadHistory() {
        try { return JSON.parse(sessionStorage.getItem(HISTORY_KEY)) || []; }
        catch { return []; }
    }
    function saveHistory(items) {
        sessionStorage.setItem(HISTORY_KEY, JSON.stringify(items));
    }
    function addToHistory(query) {
        const items = loadHistory();
        // deduplicate: remove same query if it exists
        const filtered = items.filter(q => q !== query);
        filtered.unshift(query);
        if (filtered.length > MAX_HISTORY) filtered.pop();
        saveHistory(filtered);
        renderHistory();
    }
    function renderHistory() {
        if (!historyList) return;
        const items = loadHistory();
        historyList.innerHTML = '';
        if (items.length === 0) {
            const empty = document.createElement('span');
            empty.className = 'history-empty';
            empty.textContent = 'No recent queries';
            historyList.appendChild(empty);
            return;
        }
        items.forEach(q => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'history-item';
            btn.innerHTML = `<svg class="history-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg><span class="history-item-text">${escapeHtml(q)}</span>`;
            btn.title = q;
            btn.addEventListener('click', () => {
                playUiSound('click');
                queryInput.value = q;
                autoGrow();
                executeSearch();
            });
            historyList.appendChild(btn);
        });
    }
    renderHistory();

    if (historyClearBtn) {
        historyClearBtn.addEventListener('click', () => {
            playUiSound('click');
            sessionStorage.removeItem(HISTORY_KEY);
            renderHistory();
            showToast('History cleared', 'success', 2000);
        });
    }

    // ---- orb / status state machine -----------------------------------
    // A single source of truth for the visible app state. State is derived
    // from real events (mic listening, request in-flight, real API result).
    const ORB = {
        idle:       { label: 'Ask ThinkZen',            hint: 'Speak or type in Hindi, English, or Hinglish', pill: 'Ready',      cls: 'status-idle' },
        listening:  { label: 'Listening…',           hint: 'Speak now — tap the mic again to stop',        pill: 'Listening',  cls: 'status-listening' },
        processing: { label: 'Searching evidence…',  hint: 'Dense + BM25 retrieval and grounding check',   pill: 'Processing', cls: 'status-processing' },
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

        // Update Face Avatar based on state
        const mouth = $('mouth');
        const leftEye = $('leftEye');
        const rightEye = $('rightEye');
        const leftPupil = $('leftPupil');
        const rightPupil = $('rightPupil');
        
        if (mouth && leftEye && rightEye) {
            // Cancel TTS animation if state switches away from answer
            if (state !== 'answer' && isSpeaking) {
                if (window.speechSynthesis) window.speechSynthesis.cancel();
                isSpeaking = false;
                if (ttsBtn) ttsBtn.classList.remove('is-active');
                stopSpeechAnimation();
            }

            if (state === 'listening') {
                mouth.setAttribute('d', "M 36 63 Q 50 78 64 63"); // Surprised/open listening mouth
                leftEye.setAttribute('ry', '6.2');
                rightEye.setAttribute('ry', '6.2');
            } else if (state === 'processing') {
                mouth.setAttribute('d', "M 36 65 Q 50 65 64 65"); // Flat neutral thinking line
                leftEye.setAttribute('ry', '3.0'); // Squinting analytical eyes
                rightEye.setAttribute('ry', '3.0');
                
                // Let pupils look slightly up/sideways in "thinking" stance
                if (leftPupil && rightPupil) {
                    leftPupil.setAttribute('cx', '1.0');
                    leftPupil.setAttribute('cy', '-1.5');
                    rightPupil.setAttribute('cx', '1.0');
                    rightPupil.setAttribute('cy', '-1.5');
                }
            } else if (state === 'error') {
                mouth.setAttribute('d', "M 35 67 Q 50 62 65 67"); // Sad frowny mouth
                leftEye.setAttribute('ry', '4.5');
                rightEye.setAttribute('ry', '4.5');
            } else if (state === 'refused') {
                mouth.setAttribute('d', "M 35 66 Q 50 64 65 66"); // Disappointed straight/neutral line
                leftEye.setAttribute('ry', '5.2');
                rightEye.setAttribute('ry', '5.2');
            } else { // idle / answer
                mouth.setAttribute('d', "M 33 65 Q 50 69 67 65"); // Smiling happy mouth
                leftEye.setAttribute('ry', '5.5');
                rightEye.setAttribute('ry', '5.5');
                if (leftPupil && rightPupil) {
                    leftPupil.setAttribute('cx', '0');
                    leftPupil.setAttribute('cy', '0');
                    rightPupil.setAttribute('cx', '0');
                    rightPupil.setAttribute('cy', '0');
                }
            }
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

    // ---- Keyboard shortcuts -------------------------------------------
    document.addEventListener('keydown', (e) => {
        // Ctrl+Enter → send
        if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); executeSearch(); }
        // Ctrl+M → toggle mic
        if (e.ctrlKey && (e.key === 'm' || e.key === 'M')) { e.preventDefault(); toggleVoice(); }
        // Escape → clear or close modal
        if (e.key === 'Escape') {
            const modal = document.querySelector('.modal-overlay');
            if (modal) { closeModal(modal); return; }
            clearInteraction();
        }
    });

    // ---- example chips (each runs a real query) -----------------------
    const exampleChips = Array.from(document.querySelectorAll('#exampleChips .chip'));
    exampleChips.forEach((chip) => {
        chip.addEventListener('click', () => {
            playUiSound('click');
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
                b.addEventListener('click', () => {
                    playUiSound('click');
                    queryInput.value = q;
                    autoGrow();
                    executeSearch();
                });
                refusalSuggest.appendChild(b);
            });
    }
    buildRefusalSuggestions();

    // ---- decorative waveform behind the orb (visual only, state-driven) ----
    function buildWaveform() {
        if (!waveform) return;
        const N = 48;
        // Warm palette: golden → copper → bronze → sienna
        const stops = [[232,184,138],[212,149,106],[194,132,90],[156,107,78]];
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

    // ---- language segmented control → drives the (hidden) recognition <select> ----
    const langSegs = Array.from(document.querySelectorAll('.lang-seg-btn'));
    langSegs.forEach((btn) => {
        btn.addEventListener('click', () => {
            playUiSound('click');
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
    if (navAsk) {
        navAsk.addEventListener('click', () => {
            playUiSound('click');
            queryInput.focus();
        });
    }

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
            .forEach((el) => { if (el) el.textContent = '—'; });
        if (telAlpha) telAlpha.textContent = 'α = —';
        if (telAlphaSource) telAlphaSource.textContent = 'adaptive';
        psteps.forEach((s) => s.classList.remove('is-active', 'is-refused'));
        // reset ratings
        if (rateUpBtn) rateUpBtn.classList.remove('rated-up');
        if (rateDownBtn) rateDownBtn.classList.remove('rated-down');
        // stop any TTS
        if (window.speechSynthesis) window.speechSynthesis.cancel();
        if (ttsBtn) ttsBtn.classList.remove('is-active');
        setState('idle');
        // NOTE: aggRuns / aggP50 / aggP70 / aggP100 / aggNote and sysCorpus are NOT reset.
    }
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            playUiSound('click');
            clearInteraction();
        });
    }

    // ---- Copy to clipboard --------------------------------------------
    if (copyAnswerBtn) {
        copyAnswerBtn.addEventListener('click', () => {
            playUiSound('click');
            const text = answerContent ? answerContent.textContent : '';
            if (!text) { showToast('Nothing to copy', 'warn', 2000); return; }
            navigator.clipboard.writeText(text).then(() => {
                showToast('Answer copied to clipboard', 'success', 2500);
            }).catch(() => {
                showToast('Failed to copy', 'error', 2500);
            });
        });
    }

    // ---- TTS Readback -------------------------------------------------
    let isSpeaking = false;
    if (ttsBtn) {
        ttsBtn.addEventListener('click', () => {
            playUiSound('click');
            if (!window.speechSynthesis) {
                showToast('Text-to-speech not supported in this browser', 'warn', 3000);
                return;
            }
            if (isSpeaking) {
                window.speechSynthesis.cancel();
                isSpeaking = false;
                ttsBtn.classList.remove('is-active');
                stopSpeechAnimation();
                showToast('Stopped reading', 'success', 1500);
                return;
            }
            const text = answerContent ? answerContent.textContent : '';
            if (!text) { showToast('No answer to read', 'warn', 2000); return; }
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.95;
            utterance.pitch = 1;
            utterance.onstart = () => {
                isSpeaking = true;
                ttsBtn.classList.add('is-active');
                startSpeechAnimation();
            };
            utterance.onend = () => {
                isSpeaking = false;
                ttsBtn.classList.remove('is-active');
                stopSpeechAnimation();
            };
            utterance.onerror = () => {
                isSpeaking = false;
                ttsBtn.classList.remove('is-active');
                stopSpeechAnimation();
            };
            window.speechSynthesis.speak(utterance);
            showToast('Reading answer aloud…', 'success', 2000);
        });
    }

    // ---- Answer rating ------------------------------------------------
    function handleRate(direction) {
        if (rateUpBtn) rateUpBtn.classList.remove('rated-up');
        if (rateDownBtn) rateDownBtn.classList.remove('rated-down');
        if (direction === 'up') {
            rateUpBtn.classList.add('rated-up');
            showToast('Thanks for the feedback! 👍', 'success', 2500);
        } else {
            rateDownBtn.classList.add('rated-down');
            showToast('Thanks for the feedback — we\'ll improve! 👎', 'warn', 2500);
        }
    }
    if (rateUpBtn) rateUpBtn.addEventListener('click', () => { playUiSound('click'); handleRate('up'); });
    if (rateDownBtn) rateDownBtn.addEventListener('click', () => { playUiSound('click'); handleRate('down'); });

    // ---- Full-screen answer modal -------------------------------------
    function openModal(content) {
        const overlay = document.createElement('div');
        overlay.className = 'modal-overlay';
        overlay.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <span class="modal-title">Full Answer</span>
                    <button class="modal-close" type="button" aria-label="Close" title="Close (Esc)">
                        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 6l12 12M18 6 6 18"/></svg>
                    </button>
                </div>
                <div class="modal-body">${escapeHtml(content)}</div>
            </div>`;
        overlay.querySelector('.modal-close').addEventListener('click', () => { playUiSound('click'); closeModal(overlay); });
        overlay.addEventListener('click', (e) => { if (e.target === overlay) { playUiSound('click'); closeModal(overlay); } });
        document.body.appendChild(overlay);
    }
    function closeModal(overlay) {
        overlay.classList.add('modal-out');
        overlay.addEventListener('animationend', () => overlay.remove());
    }
    if (expandAnswerBtn) {
        expandAnswerBtn.addEventListener('click', () => {
            playUiSound('click');
            const text = answerContent ? answerContent.textContent : '';
            if (!text) { showToast('No answer to expand', 'warn', 2000); return; }
            openModal(text);
        });
    }

    // ---- Export Grounding Report (Markdown) --------------------------
    if (exportBtn) {
        exportBtn.addEventListener('click', () => {
            playUiSound('click');
            const query = queryInput.value.trim() || "ThinkZen RAG Query";
            const answer = answerContent ? answerContent.textContent : '';
            if (!answer) {
                showToast('No answer content available to export', 'warn', 2000);
                return;
            }
            
            let sourcesMd = "\n## Grounding Evidence Sources Used\n\n";
            const evItems = document.querySelectorAll('.ev-item');
            if (evItems.length === 0) {
                sourcesMd += "*No source documents found / met threshold.*\n";
            } else {
                sourcesMd += "| Rank | Source Document | Match Score | Excerpt |\n";
                sourcesMd += "|------|-----------------|-------------|---------|\n";
                evItems.forEach((item, index) => {
                    const title = item.querySelector('.ev-title')?.textContent || `Source ${index + 1}`;
                    const score = item.querySelector('.ev-score')?.textContent || '—';
                    const text = item.querySelector('.ev-text')?.textContent || '';
                    const cleanText = text.replace(/\s+/g, ' ').substring(0, 160) + '...';
                    sourcesMd += `| ${index + 1} | ${title} | ${score} | ${cleanText} |\n`;
                });
            }

            const timestamp = new Date().toISOString().replace(/T/, ' ').replace(/\..+/, '');
            const reportText = `# ThinkZen Grounded RAG Report\n*Generated on: ${timestamp}*\n\n## User Query\n> ${query}\n\n## Grounded AI Answer\n\n${answer}\n${sourcesMd}\n---\n*Report generated client-side by ThinkZen voice-native retrieval engines.*`;

            const blob = new Blob([reportText], { type: 'text/markdown;charset=utf-8;' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `thinkzen-rag-report-${Date.now()}.md`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            showToast('Markdown RAG report downloaded!', 'success', 2500);
        });
    }

    // ---- Animated counter (count-up effect) ---------------------------
    function animateCounter(el, targetText) {
        if (!el) return;
        // Only animate if it looks like "123.45 ms"
        const match = targetText.match(/^([\d.]+)\s*(ms)?$/);
        if (!match || prefersReducedMotion) {
            el.textContent = targetText;
            return;
        }
        const target = parseFloat(match[1]);
        const suffix = match[2] ? ' ms' : '';
        const duration = 600;
        const startTime = performance.now();
        const startVal = 0;

        function step(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // ease-out
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = startVal + (target - startVal) * eased;
            el.textContent = current.toFixed(2) + suffix;
            if (progress < 1) requestAnimationFrame(step);
            else {
                el.textContent = targetText;
                el.classList.add('counter-animate');
                setTimeout(() => el.classList.remove('counter-animate'), 400);
            }
        }
        requestAnimationFrame(step);
    }

    // ---- Web Speech API (primary, real voice path) --------------------
    let recognition = null;
    let isListening = false;
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SR) {
        recognition = new SR();
        recognition.continuous = false;
        recognition.interimResults = true;

        recognition.onstart = () => {
            isListening = true;
            setState('listening');
            playUiSound('listening');
        };
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
                playUiSound('error');
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
        playUiSound('click');
        if (!recognition) {
            showError('Speech recognition is not supported in this browser. Please type your question — everything else works the same.');
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
    // 'click' — binding click alone avoids a double-trigger on keyboard use.
    if (voiceOrb) voiceOrb.addEventListener('click', toggleVoice);

    sendBtn.addEventListener('click', () => {
        playUiSound('click');
        executeSearch();
    });

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

        // Add to history
        addToHistory(query);

        // Reset ratings for new query
        if (rateUpBtn) rateUpBtn.classList.remove('rated-up');
        if (rateDownBtn) rateDownBtn.classList.remove('rated-down');

        // Test diagnostics mock query for offline testing
        if (query === "test_export") {
            const mockData = {
                answer: "Retrieval-Augmented Generation (RAG) is a technique that optimizes the output of a large language model by referencing an authoritative knowledge base outside of its training data.",
                sources: [
                    {
                        score: 0.945,
                        method: "hybrid",
                        chunk_id: "chunk_101",
                        text: "RAG grounds LLM outputs in verified external facts.",
                        metadata: {
                            title: "RAG Overview Doc",
                            doc_id: "doc_rag_001",
                            dataset: "Indexed Corpus"
                        }
                    }
                ],
                refused: false,
                telemetry: {
                    total_latency_ms: 120.5,
                    retrieval_latency_ms: 45.2,
                    generation_latency_ms: 75.3,
                    candidate_count: 5,
                    evidence_count: 1,
                    detected_language: "en",
                    alpha_used: 0.5,
                    alpha_source: "adaptive",
                    query_analysis: {
                        query_type: "informational",
                        complexity: "low",
                        strategy_name: "dense+sparse",
                        adaptive_alpha: 0.5,
                        keywords: ["RAG", "Retrieval-Augmented Generation"],
                        rationale: "Simple definition query."
                    },
                    evidence_bundle: {
                        grounding_decision: "grounded",
                        source_diversity: 1,
                        max_retrieval_score: 0.945,
                        mean_coherence_score: 0.945,
                        decision_reason: "Clear evidence matches query."
                    }
                }
            };
            setTimeout(() => {
                renderResponse(mockData);
                setState('answer');
                playUiSound('success');
                setLoading(false);
            }, 600);
            return;
        }

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
            // A refusal is NOT an error — reflect the real decision in the orb.
            setState(data.refused ? 'refused' : 'answer');
            playUiSound('success');
            // Smooth scroll to answer
            if (answerCard) answerCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            // Refresh aggregate latency from the real Judge endpoint.
            refreshAggregate();
        } catch (err) {
            console.error('Query execution error:', err);
            setState('error');
            playUiSound('error');
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

        // ---- verdict + refusal (refusal ≠ error) ----
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
        if (lang) parts.push(tag(LANG_SHORT[lang] || String(lang).toUpperCase(), 'copper'));
        if (qa.query_type) parts.push(tag(String(qa.query_type).toUpperCase(), 'bronze'));
        if (tel.alpha_source) {
            const a = (tel.alpha_used != null) ? Number(tel.alpha_used).toFixed(2) : '';
            parts.push(tag(`${tel.alpha_source === 'adaptive' ? 'ADAPTIVE' : 'OVERRIDE'} α${a ? ' ' + a : ''}`,
                tel.alpha_source === 'adaptive' ? 'copper' : ''));
        }
        parts.push(tag('HYBRID', 'bronze'));
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
                    <div class="ev-meta-row">chunk: ${escapeHtml(src.chunk_id || '—')}${meta.doc_id ? ' · doc: ' + escapeHtml(meta.doc_id) : ''}${meta.language ? ' · ' + escapeHtml(meta.language) : ''}</div>
                </div>`;
            evidenceList.appendChild(item);
        });
    }

    // Bind corpus/dataset + the "grounded in ..." footer from REAL source metadata only.
    // If metadata carries no dataset/corpus, labels stay "—" and the footer text stays generic.
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
        telAlpha.textContent = `α = ${(tel.alpha_used != null) ? Number(tel.alpha_used).toFixed(2) : '—'}`;
        telAlphaSource.textContent = tel.alpha_source || '—';
        telGrounding.textContent = (tel.grounding_status || '—').toUpperCase();
        telGenTime.textContent = fmtMs(tel.generation_latency_ms);

        // latency hero + metric cards — with animated counter
        animateCounter(telTotalMs, fmtMs(tel.total_latency_ms));
        animateCounter(telRetrievalMs, fmtMs(tel.retrieval_latency_ms));
        animateCounter(telGenerationMs, fmtMs(tel.generation_latency_ms));
        telCandidates.textContent = (tel.candidate_count != null) ? tel.candidate_count : '—';
        telEvidenceCount.textContent = (tel.evidence_count != null) ? tel.evidence_count : '—';
        telDetectedLang.textContent = langLabel(tel.detected_language);

        // query analysis block
        telQueryType.textContent = qa.query_type || '—';
        telComplexity.textContent = qa.complexity || '—';
        telStrategy.textContent = qa.strategy_name || '—';
        telAdaptiveAlpha.textContent = (qa.adaptive_alpha != null) ? Number(qa.adaptive_alpha).toFixed(2) : '—';
        telRationale.textContent = qa.rationale || '—';
        telKeywords.innerHTML = '';
        (qa.keywords || []).forEach((kw) => {
            const c = document.createElement('span');
            c.className = 'kw-chip';
            c.textContent = kw;
            telKeywords.appendChild(c);
        });

        // evidence intelligence block (this is where the technical reason lives)
        telEbDecision.textContent = (eb.grounding_decision || '—').toUpperCase();
        telEbDiversity.textContent = (eb.source_diversity != null) ? eb.source_diversity : '—';
        telEbMaxScore.textContent = (eb.max_retrieval_score != null) ? Number(eb.max_retrieval_score).toFixed(3) : '—';
        telEbCoherence.textContent = (eb.mean_coherence_score != null) ? Number(eb.mean_coherence_score).toFixed(3) : '—';
        telEbReason.textContent = eb.decision_reason || tel.refusal_reason || '—';

        revealPipeline(refused);
    }

    // Reveal the pipeline trace AFTER real data arrives (presentation of real
    // returned values — never a fake "in-progress" indicator).
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
                if (aggP50)  animateCounter(aggP50, fmtMs(stats.p50_ms));
                if (aggP70)  animateCounter(aggP70, fmtMs(stats.p70_ms));
                if (aggP100) animateCounter(aggP100, fmtMs(stats.p100_ms));
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
    initFaceAvatar();

    // ---- Brand click animation and scroll home -----------------------
    const brand = $('brand');
    if (brand) {
        brand.addEventListener('click', () => {
            const brandMark = brand.querySelector('.brand-mark');
            if (brandMark) {
                brandMark.style.transform = 'rotateY(360deg)';
                setTimeout(() => { brandMark.style.transform = ''; }, 600);
            }
            clearInteraction();
            window.scrollTo({ top: 0, behavior: 'smooth' });
            queryInput.focus();
            showToast('Returned to home screen!', 'success', 2000);
        });
    }

    // Judge panel visibility follows the toggle's initial (checked) state.
    judgePanel.classList.toggle('hidden', !judgeModeToggle.checked);
});
