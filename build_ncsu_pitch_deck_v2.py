import os
import json

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IsotopePINN | NC State Technical Review & Progress (v2)</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0b0f19;
            --bg-slide: #111827;
            --surface: #1f2937;
            --surface-hover: #374151;
            --text-primary: #f9fafb;
            --text-secondary: #d1d5db;
            --text-dim: #9ca3af;
            --accent-amber: #f59e0b;
            --accent-blue: #38bdf8;
            --accent-emerald: #10b981;
            --accent-purple: #a855f7;
            --accent-rose: #f43f5e;
            --accent-teal: #14b8a6;
            --border: #374151;
            --border-bright: #4b5563;
            --gradient-hero: linear-gradient(135deg, #38bdf8 0%, #a855f7 50%, #f59e0b 100%);
            --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        html, body {
            font-family: var(--font-sans);
            background: var(--bg-primary);
            color: var(--text-primary);
            overflow: hidden;
            height: 100vh;
            width: 100vw;
        }

        #deck {
            display: flex;
            height: 100vh;
            transition: transform 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .slide {
            min-width: 100vw;
            height: 100vh;
            padding: 3.5rem 7vw 5.5rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            position: relative;
            overflow-y: auto;
            overflow-x: hidden;
            scrollbar-width: thin;
            scrollbar-color: var(--border-bright) transparent;
        }

        .slide::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(ellipse at 20% 40%, rgba(56, 189, 248, 0.04) 0%, transparent 70%),
                        radial-gradient(ellipse at 80% 80%, rgba(168, 85, 247, 0.03) 0%, transparent 60%);
            pointer-events: none;
            z-index: 0;
        }

        .slide > * { position: relative; z-index: 1; }

        /* Header typography */
        .kicker {
            font-family: var(--font-mono);
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--accent-amber);
            margin-bottom: 0.5rem;
        }

        .slide-title {
            font-size: clamp(2rem, 3.8vw, 3.2rem);
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1.15;
            margin-bottom: 1rem;
            color: var(--text-primary);
        }

        .slide-subtitle {
            font-size: clamp(1rem, 1.6vw, 1.25rem);
            color: var(--text-secondary);
            max-width: 900px;
            line-height: 1.5;
            margin-bottom: 2rem;
        }

        /* Grid Layouts */
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
        .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.25rem; }
        .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }

        @media (max-width: 900px) {
            .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; }
        }

        /* Cards */
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.4rem;
            transition: all 0.2s ease;
        }

        .card:hover {
            border-color: var(--border-bright);
            transform: translateY(-2px);
        }

        .card-amber { border-top: 4px solid var(--accent-amber); }
        .card-blue { border-top: 4px solid var(--accent-blue); }
        .card-emerald { border-top: 4px solid var(--accent-emerald); }
        .card-purple { border-top: 4px solid var(--accent-purple); }
        .card-teal { border-top: 4px solid var(--accent-teal); }

        .card-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.6rem;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .card-body {
            font-size: 0.92rem;
            color: var(--text-secondary);
            line-height: 1.55;
        }

        .card-body ul { margin-left: 1.2rem; margin-top: 0.4rem; }
        .card-body li { margin-bottom: 0.35rem; }

        /* Code snippets */
        code {
            font-family: var(--font-mono);
            font-size: 0.85rem;
            background: rgba(0, 0, 0, 0.4);
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            color: var(--accent-blue);
        }

        .code-block {
            font-family: var(--font-mono);
            font-size: 0.82rem;
            background: #070a12;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            color: #e2e8f0;
            overflow-x: auto;
            line-height: 1.5;
        }

        /* Status badges */
        .badge {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .badge-green { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4); }
        .badge-amber { background: rgba(245, 158, 11, 0.2); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.4); }
        .badge-blue { background: rgba(56, 189, 248, 0.2); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.4); }

        /* Navigation Bar */
        .nav-bar {
            position: fixed;
            bottom: 0; left: 0; right: 0;
            height: 60px;
            background: rgba(11, 15, 25, 0.92);
            backdrop-filter: blur(10px);
            border-top: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 2rem;
            z-index: 100;
        }

        .slide-counter {
            font-family: var(--font-mono);
            font-size: 0.9rem;
            color: var(--text-dim);
        }

        .slide-counter span { color: var(--text-primary); font-weight: 700; }

        .progress-bar {
            flex-grow: 1;
            max-width: 400px;
            height: 4px;
            background: var(--surface);
            border-radius: 2px;
            margin: 0 2rem;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: var(--gradient-hero);
            width: 10%;
            transition: width 0.3s ease;
        }

        .nav-btns { display: flex; gap: 0.5rem; }

        .nav-btn {
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 0.4rem 0.9rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .nav-btn:hover:not(:disabled) {
            background: var(--surface-hover);
            border-color: var(--border-bright);
        }

        .nav-btn:disabled { opacity: 0.4; cursor: not-allowed; }

        /* Speaker notes overlay */
        .notes-panel {
            position: fixed;
            bottom: 70px; right: 2rem;
            width: 420px;
            max-height: 400px;
            background: #0f172a;
            border: 1px solid var(--accent-amber);
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
            z-index: 99;
            overflow-y: auto;
            transition: all 0.3s ease;
        }

        .notes-panel.hidden { opacity: 0; pointer-events: none; transform: translateY(20px); }
        .notes-header { font-size: 0.85rem; font-weight: 700; color: var(--accent-amber); text-transform: uppercase; margin-bottom: 0.5rem; }
        .notes-content { font-size: 0.88rem; color: var(--text-secondary); line-height: 1.5; }

        .swipe-hint {
            position: fixed;
            top: 1rem; right: 1rem;
            font-size: 0.75rem;
            color: var(--text-dim);
            background: rgba(31, 41, 55, 0.8);
            padding: 0.3rem 0.6rem;
            border-radius: 4px;
            border: 1px solid var(--border);
            z-index: 100;
        }
    </style>
</head>
<body>

<div id="deck">

    <!-- SLIDE 1: Title & Overview -->
    <div class="slide">
        <div class="kicker">NC State Technical Review & Progress Update &middot; July 2026</div>
        <h1 class="slide-title" style="background: var(--gradient-hero); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            IsotopePINN: Physics-Informed Neural Surrogate for Ac-225 Production
        </h1>
        <p class="slide-subtitle">
            Sprint 4 & Sprint 5 Technical Progress: Evaluated Nuclear Data Spine, Exact Matrix Exponential Loss, 
            SOAP Optimizer, Locked-Test Protocol, and ISEF 2027 Strategic Roadmap.
        </p>
        <div class="grid-3" style="margin-top: 1rem;">
            <div class="card card-blue">
                <div class="card-title">Student Researcher</div>
                <div class="card-body"><b>Sam Ogunnubi</b><br>ISEF 2027 Project</div>
            </div>
            <div class="card card-amber">
                <div class="card-title">Mentor & Lab</div>
                <div class="card-body"><b>Jaden Palmer</b><br>NCSU ARTISANS Lab</div>
            </div>
            <div class="card card-emerald">
                <div class="card-title">Verification Standing</div>
                <div class="card-body"><b>22 / 22 Smoke Checks PASS</b><br><span class="badge badge-green">Bit-Identical Compat</span></div>
            </div>
        </div>
    </div>

    <!-- SLIDE 2: What I Improved Since Last Time -->
    <div class="slide">
        <div class="kicker">Core Technical Upgrades</div>
        <h2 class="slide-title">What I Improved Since Our Last Meeting</h2>
        <p class="slide-subtitle">Replaced empirical approximations with evaluated nuclear cross-sections, exact matrix exponential physics, and rigorous OOD testing.</p>
        <div class="grid-2">
            <div class="card card-emerald">
                <div class="card-title">1. Evaluated EXFOR/JENDL Spine</div>
                <div class="card-body">
                    <ul>
                        <li>Shifted from synthetic constants to <b>evaluated nuclear data (EXFOR/JENDL-4.0)</b> for Ra-226(n,2n) and (n,γ) cross-sections.</li>
                        <li>Implemented <code>f*</code> spectrum folding (effective factor 1.24×10⁻³) for thermal/fast neutron flux ratio.</li>
                    </ul>
                </div>
            </div>
            <div class="card card-blue">
                <div class="card-title">2. expmix Matrix Exponential Loss</div>
                <div class="card-body">
                    <ul>
                        <li>Replaced finite-difference ODE residuals with exact <code>expmix</code> physics loss using matrix exponential propagators.</li>
                        <li>Eliminates time-discretization errors on stiff species (Ac-225, Ra-225).</li>
                    </ul>
                </div>
            </div>
            <div class="card card-purple">
                <div class="card-title">3. Float64 Grid & EMA Weights</div>
                <div class="card-body">
                    <ul>
                        <li>Enforced Float64 precision on irradiation time grids (0.05 h to 500 h) to prevent float truncation.</li>
                        <li>Added Exponential Moving Average (EMA) weight smoothing during training for steady validation.</li>
                    </ul>
                </div>
            </div>
            <div class="card card-amber">
                <div class="card-title">4. 60-Scenario Locked Evaluation Protocol</div>
                <div class="card-body">
                    <ul>
                        <li>Created a fixed 60-scenario locked evaluation set (seed <code>20260725</code>) with shifted regime boundaries.</li>
                        <li>Zero data leakage: network never sees locked test scenarios during hyperparameter tuning.</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <!-- SLIDE 3: Architectural Innovations (Sprint 5 SOTA) -->
    <div class="slide">
        <div class="kicker">2025-2026 Literature Sweep</div>
        <h2 class="slide-title">Sprint 5 SOTA Architecture Additions</h2>
        <p class="slide-subtitle">Implemented 8 high-ROI techniques from recent PINN literature to achieve best-in-class stiff ODE learning.</p>
        <div class="grid-3">
            <div class="card card-amber">
                <div class="card-title">SOAP Optimizer <span class="badge badge-amber">arXiv:2409.11321</span></div>
                <div class="card-body">Shampoo preconditioner in Adam eigenbasis. Outperforms standard L-BFGS/Adam on ill-conditioned physics losses.</div>
            </div>
            <div class="card card-blue">
                <div class="card-title">JAWS + ACI Conformal UQ <span class="badge badge-blue">arXiv:2207.10716</span></div>
                <div class="card-body">Jackknife+ After Bootstrap + Adaptive Conformal Inference for honest coverage under distribution shift.</div>
            </div>
            <div class="card card-purple">
                <div class="card-title">Multi-Fidelity Head <span class="badge badge-purple">Meng & Karniadakis 2020</span></div>
                <div class="card-body">Log-space residual head bridging fast 0D scalar flux predictions to full 3D energy-folded spectra.</div>
            </div>
            <div class="card card-emerald">
                <div class="card-title">Causality-Weighted Loss <span class="badge badge-green">arXiv:2203.07404</span></div>
                <div class="card-body">Temporal causality weighting parameter <code>eps=0.05</code> enforcing sequential physics learning along time.</div>
            </div>
            <div class="card card-teal">
                <div class="card-title">minGRU Selective-Scan <span class="badge badge-blue">arXiv:2410.01201</span></div>
                <div class="card-body">Minimal recurrent baseline to evaluate whether recurrent gating outperforms feedforward models.</div>
            </div>
            <div class="card card-amber">
                <div class="card-title">Atom Budget Projection</div>
                <div class="card-body">Zero-parameter conservation projection enforcing mass conservation (sum ≤ initial target mass) to 1e-12.</div>
            </div>
        </div>
    </div>

    <!-- SLIDE 4: Budget-Conscious Colab Execution Plan -->
    <div class="slide">
        <div class="kicker">Execution Strategy</div>
        <h2 class="slide-title">What I Am Running Next (Colab Matrix)</h2>
        <p class="slide-subtitle">To respect GPU compute limits while maintaining scientific rigor, I designed a focused 2-run primary plan + analysis.</p>
        <div class="grid-3">
            <div class="card card-emerald">
                <div class="card-title">RUN_B0_control.ipynb <span class="badge badge-green">MUST RUN #1</span></div>
                <div class="card-body">
                    <ul>
                        <li><b>Baseline Control:</b> Committed best recipe (expmix + v3 coverage + adaptive weights).</li>
                        <li><b>Purpose:</b> Establishes our benchmark locked-set accuracy (TC-ACC-001 gate).</li>
                    </ul>
                </div>
            </div>
            <div class="card card-amber">
                <div class="card-title">RUN_S1_soap.ipynb <span class="badge badge-amber">MUST RUN #2</span></div>
                <div class="card-body">
                    <ul>
                        <li><b>SOAP Optimizer Swap:</b> Direct A/B test of SOAP vs Adam optimizer.</li>
                        <li><b>Purpose:</b> Tests the highest-ROI upgrade from 2025-26 literature.</li>
                    </ul>
                </div>
            </div>
            <div class="card card-purple">
                <div class="card-title">RUN_S5_full_stack.ipynb <span class="badge badge-purple">OPTIONAL</span></div>
                <div class="card-body">
                    <ul>
                        <li><b>Full Stack Combo:</b> SOAP + Causality + EMA + Rollout + Conserve combined.</li>
                        <li><b>Purpose:</b> Maximum accuracy run if compute budget permits.</li>
                    </ul>
                </div>
            </div>
        </div>
        <div class="code-block" style="margin-top: 1.5rem;">
# How Colab execution works:
# 1. Upload IsotopePINN_repo.zip to Google Drive root once (3.4 MB).
# 2. Open RUN_B0_control.ipynb & RUN_S1_soap.ipynb on GPU runtime.
# 3. Toggle FAST_SMOKE=False for full 6000-epoch training run (~3 h on T4 GPU).
# 4. Locked-set evaluation auto-runs post-training and downloads results zip.
        </div>
    </div>

    <!-- SLIDE 5: Locked-Test Protocol & Gate Criteria -->
    <div class="slide">
        <div class="kicker">Scientific Validation</div>
        <h2 class="slide-title">Locked-Test Protocol & Gate Criteria</h2>
        <p class="slide-subtitle">Strict evaluation protocol guaranteeing zero data leakage and honest error reporting for ISEF judges.</p>
        <div class="grid-2">
            <div class="card card-blue">
                <div class="card-title">Locked Test Harness (60 Scenarios)</div>
                <div class="card-body">
                    <ul>
                        <li>Fixed seed <code>20260725</code> evaluating shifted flux/energy boundaries.</li>
                        <li>Covers Thermal (0.025 eV), Epithermal (80-100 keV), and Fast (14 MeV) regimes.</li>
                        <li>Evaluates Ac-225 endpoint relative error against Bateman Radau solver.</li>
                    </ul>
                </div>
            </div>
            <div class="card card-amber">
                <div class="card-title">TC-ACC-001 Quality Gate</div>
                <div class="card-body">
                    <ul>
                        <li><b>Target Gate:</b> Ac-225 median relative error &lt; 3.00% on locked test set.</li>
                        <li>Currently flagged as <b>PENDING</b> on poster until full GPU runs finish.</li>
                        <li>Guarantees no unverified numbers go on presentation materials.</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <!-- SLIDE 6: Technical Questions for Jayden -->
    <div class="slide">
        <div class="kicker">Mentor Alignment</div>
        <h2 class="slide-title">10 Technical Questions for Jayden</h2>
        <p class="slide-subtitle">Key methodological questions to review during our technical discussion.</p>
        <div class="grid-2">
            <div class="card card-purple">
                <div class="card-title">UQ & Conformal Methods</div>
                <div class="card-body">
                    <ul>
                        <li>1. Is using Jacobian norm ratio as a weight-proxy in JAWS conformal prediction defensible?</li>
                        <li>2. How should ACI handle distribution shift when test streams are regime-stratified?</li>
                        <li>3. How should I report the &lt;3% gate on n=60 scenarios without overclaiming statistical power?</li>
                    </ul>
                </div>
            </div>
            <div class="card card-teal">
                <div class="card-title">Physics & Optimization</div>
                <div class="card-body">
                    <ul>
                        <li>4. Hyperparameter intuition for SOAP optimizer on stiff Bateman ODE losses?</li>
                        <li>5. Is framing <code>f*</code> as an "effective parameter with uncertainty" physical?</li>
                        <li>6. Is the 0D → spectrum multi-fidelity composite head framing scientifically honest?</li>
                        <li>7. Which ablations do ISEF judges actually care about most?</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <!-- SLIDE 7: Essential Favors & Resource Requests -->
    <div class="slide">
        <div class="kicker">Resource & Mentorship Requests</div>
        <h2 class="slide-title">How NCSU & Jayden Can Help</h2>
        <p class="slide-subtitle">High-leverage requests that will elevate the project to top-tier ISEF standards.</p>
        <div class="grid-3">
            <div class="card card-amber">
                <div class="card-title">1. Compute Sponsorship</div>
                <div class="card-body">
                    <ul>
                        <li>Can the lab run my 3 Colab notebooks on a spare GPU overnight?</li>
                        <li>Or sponsor access to <b>NCSU's Henry2 HPC cluster</b> / student GPU credits?</li>
                    </ul>
                </div>
            </div>
            <div class="card card-blue">
                <div class="card-title">2. PULSTAR Spectrum Intro</div>
                <div class="card-body">
                    <ul>
                        <li>Intro to an NCSU Nuclear Engineering faculty member to review my parametric spectrum (<code>f*</code> factor).</li>
                        <li>Validates cross-section folding against real reactor spectrum data.</li>
                    </ul>
                </div>
            </div>
            <div class="card card-emerald">
                <div class="card-title">3. Adversarial Review & Mock Judging</div>
                <div class="card-body">
                    <ul>
                        <li>30-minute adversarial protocol review: try to break my evaluation pipeline.</li>
                        <li>Mock judging session in December before regional fair competition.</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <!-- SLIDE 8: ISEF 2027 Roadmap & Poster Preview -->
    <div class="slide">
        <div class="kicker">Competition Strategy</div>
        <h2 class="slide-title">ISEF 2027 Roadmap & Poster Board v2</h2>
        <p class="slide-subtitle">Structured timeline leading up to regional and international science fairs.</p>
        <div class="grid-2">
            <div class="card card-blue">
                <div class="card-title">Poster Board v2 Design (board_v2_preview.html)</div>
                <div class="card-body">
                    <ul>
                        <li>Editorial palette with 48x36 tri-fold PDF export.</li>
                        <li>Interactive speaker notes overlay (press <code>N</code>).</li>
                        <li>Form 2A AI assistance disclosure & compliance footer.</li>
                    </ul>
                </div>
            </div>
            <div class="card card-emerald">
                <div class="card-title">Key Milestones (Aug 2026 - Jan 2027)</div>
                <div class="card-body">
                    <ul>
                        <li><b>Aug 2026:</b> Execute full GPU runs & update locked-set metrics.</li>
                        <li><b>Oct 2026:</b> Complete ISEF Research Plan & Paperwork (Forms 1, 1A, 1B, 2A).</li>
                        <li><b>Dec 2026:</b> Mock judging with NCSU mentors.</li>
                        <li><b>Jan 2027:</b> Regional Science Fair Competition.</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <!-- SLIDE 9: Summary & Project Links -->
    <div class="slide">
        <div class="kicker">Conclusion</div>
        <h2 class="slide-title">Summary & Project Artifacts</h2>
        <p class="slide-subtitle">All code, notebooks, and documentation are committed and ready in the project repository.</p>
        <div class="grid-3">
            <div class="card card-amber">
                <div class="card-title">GitHub Repository</div>
                <div class="card-body"><a href="https://github.com/samogunnubi0-del/PINN2.0" target="_blank" style="color: var(--accent-blue);">github.com/samogunnubi0-del/PINN2.0</a></div>
            </div>
            <div class="card card-blue">
                <div class="card-title">Live Streamlit App</div>
                <div class="card-body"><a href="https://lhyjrhmwzxqfpuuwsux7zh.streamlit.app" target="_blank" style="color: var(--accent-blue);">lhyjrhmwzxqfpuuwsux7zh.streamlit.app</a></div>
            </div>
            <div class="card card-emerald">
                <div class="card-title">NCSU Update Doc</div>
                <div class="card-body"><code>docs/NCSU_MENTOR_UPDATE.md</code></div>
            </div>
        </div>
    </div>

</div>

<!-- Navigation Bar -->
<div class="nav-bar">
    <div class="slide-counter"><span id="current">1</span> / <span id="total">9</span></div>
    <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width: 11%;"></div></div>
    <div class="nav-btns">
        <button class="nav-btn" id="prevBtn" disabled>&larr; Prev</button>
        <button class="nav-btn" id="nextBtn">Next &rarr;</button>
    </div>
</div>

<div class="swipe-hint" id="swipeHint">Use &larr; &rarr; or Space | 'N' for Notes</div>

<!-- Speaker Notes Panel -->
<div class="notes-panel hidden" id="notesPanel">
    <div class="notes-header">Speaker Notes & Talking Points</div>
    <div class="notes-content" id="notesContent"></div>
</div>

<script>
    let currentSlide = 0;
    const slides = document.querySelectorAll('.slide');
    const deck = document.getElementById('deck');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const currentSpan = document.getElementById('current');
    const totalSpan = document.getElementById('total');
    const progressFill = document.getElementById('progressFill');
    
    totalSpan.innerText = slides.length;

    const SPEAKER_NOTES = [
        "<strong>Slide 1 (Title):</strong> 'Hi Jayden, thanks again for your time! Today I want to walk you through the Sprint 4 and Sprint 5 technical progress on IsotopePINN, our new exact matrix-exponential physics loss, the SOAP optimizer integration, and how we can collaborate on compute and reactor spectrum validation.'",

        "<strong>Slide 2 (Improvements):</strong> 'Since our last talk, I upgraded the core physics: we replaced synthetic cross-sections with evaluated EXFOR/JENDL data, implemented expmix exact matrix-exponential loss to solve time-discretization errors on stiff species like Ac-225, enforced float64 time grids, and locked down a 60-scenario evaluation protocol with zero data leakage.'",

        "<strong>Slide 3 (Sprint 5 Architecture):</strong> 'I also ran a 2025-2026 literature sweep and built 8 high-ROI techniques: the SOAP optimizer (Shampoo in Adam basis), JAWS+ACI conformal prediction for distribution shift, a multi-fidelity log-space residual head, causality-weighted temporal loss, and minGRU baselines.'",

        "<strong>Slide 4 (Colab Plan):</strong> 'To manage GPU compute budget, I designed a focused Colab matrix: RUN_B0 (our control baseline) and RUN_S1 (the SOAP optimizer swap) are our top two must-runs. Each runs for 6000 epochs on a T4 GPU and evaluates on the locked test set.'",

        "<strong>Slide 5 (Locked Test):</strong> 'Our evaluation uses 60 locked scenarios with seed 20260725. We set a strict TC-ACC-001 quality gate (<3% median relative error). It's flagged as PENDING on our poster until the full GPU runs finish, ensuring we never overclaim numbers.'",

        "<strong>Slide 6 (10 Questions):</strong> 'I've prepared 10 technical questions for you today—covering JAWS weight-proxies, ACI under regime-stratified streams, SOAP hyperparameter tuning, and how to frame our spectrum-folding f* parameter.'",

        "<strong>Slide 7 (Favors):</strong> 'I have three key requests for how NCSU and your lab can help: first, compute sponsorship (running 3 Colab notebooks on lab GPUs or Henry2 cluster access); second, an intro to an NE faculty member for PULSTAR spectrum validation; and third, an adversarial review to try to break my evaluation protocol.'",

        "<strong>Slide 8 (ISEF Roadmap):</strong> 'Here is our timeline leading up to the January 2027 regional fair. Our poster board v2 is ready with full Form 2A AI assistance disclosures.'",

        "<strong>Slide 9 (Summary):</strong> 'All code, notebooks, and documentation are committed in the repository. Thank you so much for your time and guidance!'"
    ];

    function updateSlider() {
        deck.style.transform = `translateX(-${currentSlide * 100}vw)`;
        currentSpan.innerText = currentSlide + 1;
        prevBtn.disabled = currentSlide === 0;
        nextBtn.disabled = currentSlide === slides.length - 1;
        progressFill.style.width = `${((currentSlide + 1) / slides.length) * 100}%`;
        const nc = document.getElementById('notesContent');
        if (nc && SPEAKER_NOTES[currentSlide]) nc.innerHTML = SPEAKER_NOTES[currentSlide];
    }

    function nextSlide() {
        if (currentSlide < slides.length - 1) { currentSlide++; updateSlider(); }
    }

    function prevSlide() {
        if (currentSlide > 0) { currentSlide--; updateSlider(); }
    }

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); nextSlide(); }
        else if (e.key === 'ArrowLeft') { e.preventDefault(); prevSlide(); }
        else if (e.key === 'n' || e.key === 'N') {
            e.preventDefault();
            document.getElementById('notesPanel').classList.toggle('hidden');
        }
    });

    prevBtn.addEventListener('click', prevSlide);
    nextBtn.addEventListener('click', nextSlide);

    const ncInit = document.getElementById('notesContent');
    if (ncInit && SPEAKER_NOTES[0]) ncInit.innerHTML = SPEAKER_NOTES[0];
</script>
</body>
</html>
"""

def main():
    path = "ncsu_pitch_deck_v2.html"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(HTML_CONTENT)
    print(f"Successfully generated {path} ({len(HTML_CONTENT)} bytes)")

if __name__ == "__main__":
    main()
