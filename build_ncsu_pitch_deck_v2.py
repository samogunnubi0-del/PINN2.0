import os
import shutil

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IsotopePINN | NC State Technical Review</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #DBCDBA;     /* Muted Oatmeal (softer on eyes) */
            --bg-slide: #D3C4AE;       /* Deeper beige for depth */
            --surface: #E8DDCE;        /* Warm parchment cards */
            --surface-hover: #EFE6DA;
            --text-primary: #2B1A12;   /* Dark Roast text for high contrast */
            --text-secondary: #5C4033; /* Medium Brown */
            --text-dim: #8B7355;
            --accent-blue: #B08968;    /* Classic Latte */
            --accent-purple: #9C6644;  /* Rich Mocha */
            --accent-green: #7D8C65;   /* Matcha */
            --accent-orange: #C47641;  /* Cinnamon */
            --accent-red: #A8493D;     /* Cranberry/Brick */
            --accent-cyan: #5E7A75;    /* Muted Sage */
            --accent-pink: #B57D84;    /* Dusty Rose */
            --border: #D1C0AC;
            --border-bright: #BFA892;
            --gradient-hero: linear-gradient(135deg, #B08968 0%, #9C6644 50%, #C47641 100%);
            --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
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

        /* Presentation Slider */
        #deck {
            display: flex;
            height: 100vh;
            transition: transform 0.6s cubic-bezier(0.25, 0.8, 0.25, 1);
        }

        .slide {
            min-width: 100vw;
            height: 100vh;
            padding: 4rem 8vw 6rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            position: relative;
            overflow-y: auto;
            overflow-x: hidden;
            scrollbar-width: thin;
            scrollbar-color: var(--border-bright) transparent;
        }

        .slide::after {
            content: '';
            position: sticky;
            bottom: 0;
            display: block;
            height: 40px;
            background: linear-gradient(to top, var(--bg-slide), transparent);
            pointer-events: none;
            margin-top: -40px;
        }

        .slide::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(ellipse at 20% 50%, rgba(176, 137, 104, 0.05) 0%, transparent 70%),
                        radial-gradient(ellipse at 80% 20%, rgba(156, 102, 68, 0.05) 0%, transparent 60%);
            pointer-events: none;
            z-index: 0;
        }

        .slide > * { position: relative; z-index: 1; }

        /* Typography */
        .hero-title {
            font-size: clamp(2.5rem, 5vw, 4.5rem);
            font-weight: 900;
            letter-spacing: -0.03em;
            background: var(--gradient-hero);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.1;
            margin-bottom: 1.5rem;
        }

        .slide-title {
            font-size: clamp(1.8rem, 3vw, 2.8rem);
            font-weight: 800;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }

        .slide-subtitle {
            font-size: 1.1rem;
            color: var(--text-secondary);
            margin-bottom: 2rem;
            font-weight: 400;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1rem;
        }

        .section-label {
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--accent-purple);
            margin-bottom: 0.5rem;
        }

        h3 {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--accent-blue);
            margin-bottom: 0.5rem;
        }

        p {
            font-size: 1.05rem;
            color: var(--text-secondary);
            line-height: 1.65;
            margin-bottom: 1rem;
            max-width: 750px;
        }

        strong { color: var(--text-primary); }
        em { color: var(--accent-cyan); font-style: normal; }

        code {
            font-family: var(--font-mono);
            background: rgba(176, 137, 104, 0.15);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
            color: var(--text-primary);
            word-wrap: break-word;
        }

        a {
            color: var(--accent-purple);
            text-decoration: none;
            font-weight: 600;
            transition: color 0.2s;
        }
        a:hover { text-decoration: underline; }

        /* Layout Helpers */
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            align-items: start;
        }

        .grid-3 {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 1.25rem;
            align-items: start;
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            transition: transform 0.2s ease, border-color 0.2s, box-shadow 0.2s;
        }
        .card:hover { 
            border-color: var(--border-bright); 
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .card-accent-blue { border-left: 4px solid var(--accent-blue); }
        .card-accent-purple { border-left: 4px solid var(--accent-purple); }
        .card-accent-green { border-left: 4px solid var(--accent-green); }
        .card-accent-orange { border-left: 4px solid var(--accent-orange); }
        .card-accent-red { border-left: 4px solid var(--accent-red); }
        .card-accent-cyan { border-left: 4px solid var(--accent-cyan); }
        .card-accent-pink { border-left: 4px solid var(--accent-pink); }

        .badge {
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 20px;
            background: rgba(176, 137, 104, 0.18);
            color: var(--text-primary);
            margin-right: 6px;
            margin-bottom: 6px;
        }
        .badge-green { background: rgba(125, 140, 101, 0.2); color: #3e5025; }
        .badge-orange { background: rgba(196, 118, 65, 0.2); color: #7a3a10; }
        .badge-red { background: rgba(168, 73, 61, 0.2); color: #6a1a10; }
        .badge-purple { background: rgba(156, 102, 68, 0.2); color: #4e2912; }

        ul { list-style: none; margin-bottom: 1rem; }
        li {
            font-size: 1.02rem;
            color: var(--text-secondary);
            margin-bottom: 0.65rem;
            padding-left: 1.4rem;
            position: relative;
            line-height: 1.5;
        }
        li::before {
            content: '▸';
            position: absolute;
            left: 0;
            color: var(--accent-purple);
            font-weight: bold;
        }

        /* Code Blocks */
        .code-block {
            background: #E2D5C4;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
            font-family: var(--font-mono);
            font-size: 0.86rem;
            color: #2B1A12;
            overflow-x: auto;
            line-height: 1.55;
            white-space: pre;
        }
        .hl-keyword { color: #A8493D; font-weight: 700; }
        .hl-class { color: #9C6644; font-weight: 700; }
        .hl-func { color: #B08968; font-weight: 700; }
        .hl-string { color: #5E7A75; }
        .hl-comment { color: #8B7355; font-style: italic; }

        /* Blunt Request Cards */
        .ask-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.35rem;
            margin-bottom: 1rem;
            transition: border-color 0.2s, transform 0.2s;
        }
        .ask-card:hover {
            border-color: var(--accent-purple);
            transform: translateX(4px);
        }
        .ask-card .ask-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: var(--accent-purple);
            color: #fff;
            font-weight: 800;
            width: 26px; height: 26px;
            border-radius: 50%;
            font-size: 0.85rem;
            margin-right: 10px;
        }
        .ask-card .ask-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-primary);
            display: inline;
        }
        .ask-card .ask-script {
            display: block;
            margin-top: 0.65rem;
            background: rgba(156, 102, 68, 0.08);
            border: 1px dashed rgba(156, 102, 68, 0.3);
            border-radius: 8px;
            padding: 0.85rem 1.1rem;
            font-size: 0.98rem;
            color: var(--text-primary);
            line-height: 1.55;
            font-style: italic;
        }

        /* Metric Stat Boxes */
        .stat-box { text-align: center; padding: 1.25rem 1rem; }
        .stat-value {
            font-size: 2.3rem;
            font-weight: 900;
            font-family: var(--font-mono);
            line-height: 1;
        }
        .stat-label {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 0.4rem;
            font-weight: 600;
        }

        /* Navigation Bar */
        .nav-bar {
            position: fixed;
            bottom: 0; left: 0; right: 0;
            height: 56px;
            background: rgba(219, 205, 186, 0.95);
            backdrop-filter: blur(12px);
            border-top: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 2rem;
            z-index: 200;
        }

        .nav-bar .slide-counter {
            font-family: var(--font-mono);
            font-size: 0.9rem;
            color: var(--text-dim);
            min-width: 50px;
        }

        .nav-bar .progress-bar {
            flex: 1;
            height: 4px;
            background: var(--border);
            margin: 0 2rem;
            border-radius: 3px;
            overflow: hidden;
        }
        .nav-bar .progress-fill {
            height: 100%;
            background: var(--gradient-hero);
            transition: width 0.4s ease;
        }

        .nav-btn {
            background: none;
            border: 1px solid var(--border);
            color: var(--text-primary);
            width: 38px; height: 38px;
            border-radius: 8px;
            font-size: 1.1rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s;
        }
        .nav-btn:hover { background: var(--surface-hover); border-color: var(--border-bright); }
        .nav-btn:disabled { opacity: 0.25; cursor: not-allowed; }
        .nav-btns { display: flex; gap: 0.5rem; }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
        }
        th {
            text-align: left;
            font-weight: 700;
            color: var(--text-primary);
            padding: 0.65rem 0.85rem;
            border-bottom: 2px solid var(--border-bright);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        td {
            padding: 0.6rem 0.85rem;
            border-bottom: 1px solid var(--border);
            color: var(--text-secondary);
        }

        .gh-btn, .demo-btn {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-family: var(--font-mono);
            font-size: 0.9rem;
            padding: 8px 16px;
            border-radius: 8px;
            text-decoration: none;
            transition: all 0.2s;
        }
        .gh-btn {
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text-primary);
        }
        .gh-btn:hover { background: var(--surface-hover); border-color: var(--accent-purple); color: var(--accent-purple); }

        .demo-btn {
            background: rgba(176, 137, 104, 0.2);
            border: 1px solid rgba(176, 137, 104, 0.4);
            color: var(--text-primary);
        }
        .demo-btn:hover { background: rgba(176, 137, 104, 0.3); }

        .swipe-hint {
            position: fixed;
            bottom: 68px; right: 20px;
            background: var(--surface);
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.78rem;
            color: var(--text-dim);
            z-index: 150;
        }

        /* Speaker Notes Panel */
        .notes-panel {
            position: fixed;
            bottom: 56px; left: 0; right: 0;
            max-height: 240px;
            background: rgba(219, 205, 186, 0.98);
            backdrop-filter: blur(10px);
            border-top: 2px solid var(--accent-purple);
            padding: 1.1rem 2rem;
            z-index: 180;
            overflow-y: auto;
            transition: transform 0.3s ease, opacity 0.3s ease;
        }
        .notes-panel.hidden {
            transform: translateY(100%);
            opacity: 0;
            pointer-events: none;
        }
        .notes-content {
            font-size: 0.92rem;
            color: var(--text-secondary);
            line-height: 1.6;
        }
        .notes-content strong { color: var(--accent-purple); }
    </style>
</head>
<body>

<div id="deck">

    <!-- ═══════════════════════════════════════════════════════════════════
         SLIDE 1 — TITLE
         ═══════════════════════════════════════════════════════════════════ -->
    <div class="slide" style="justify-content: center;">
        <div class="section-label">NC State University &middot; ARTISANS Lab &middot; Technical Review v2</div>
        <h1 class="hero-title">IsotopePINN</h1>
        <p style="font-size: 1.35rem; max-width: 100%; color: var(--text-primary); font-weight: 300;">
            A 0D (Point-Burnup) Physics-Informed Neural Network That Predicts<br>
            Stiff Nuclear Isotope Transmutation &middot; Sprint 4 & Sprint 5 Progress
        </p>
        <div style="margin-top: 2.5rem; display: flex; gap: 2rem; align-items: center; flex-wrap: wrap;">
            <div>
                <p style="font-family: var(--font-mono); color: var(--accent-purple); font-size: 1.1rem; margin-bottom: 0.25rem; font-weight: 700;">Samuel Ogunnubi</p>
                <p style="font-size: 0.92rem; margin-bottom: 0;">High School Junior &middot; Laurel, MD &middot; Dual Enrolled at AACC</p>
            </div>
            <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                <a href="https://github.com/samogunnubi0-del/PINN2.0" target="_blank" class="gh-btn">⟐ GitHub Repository</a>
                <a href="https://lhyjrhmwzxqfpuuwsux7zh.streamlit.app" target="_blank" class="demo-btn">▶ Live Streamlit App</a>
            </div>
        </div>
    </div>

    <!-- ═══════════════════════════════════════════════════════════════════
         SLIDE 2 — WHAT I IMPROVED SINCE LAST TIME (NO BUZZWORDS)
         ═══════════════════════════════════════════════════════════════════ -->
    <div class="slide">
        <div class="section-label">Sprint 4 Progress</div>
        <div class="slide-title">What I Improved Since Our Last Meeting</div>
        <p class="slide-subtitle">Replaced synthetic approximations with evaluated nuclear cross-sections, exact matrix-exponential physics, and a 60-scenario locked evaluation protocol.</p>

        <div class="grid-2">
            <div>
                <div class="card card-accent-green" style="margin-bottom: 1.25rem;">
                    <h3 style="margin-top: 0; color: var(--accent-green);">1. Evaluated EXFOR/JENDL-4.0 Data Spine</h3>
                    <p style="font-size: 0.98rem; margin-bottom: 0.5rem;">
                        <strong>What it means:</strong> Instead of using arbitrary synthetic cross-sections, I integrated evaluated nuclear cross-section data directly from EXFOR (IAEA) and JENDL-4.0 for Ra-226(n,2n) (27 millibarns above 6.42 MeV) and Ra-226(n,γ) (12.8 barns thermal capture).
                    </p>
                    <p style="font-size: 0.92rem; color: var(--text-dim); margin: 0;">
                        <em>Why it matters:</em> Grounding in evaluated nuclear data ensures predictions reflect real physical cross-sections rather than toy formulas.
                    </p>
                </div>

                <div class="card card-accent-purple">
                    <h3 style="margin-top: 0; color: var(--accent-purple);">2. expmix Exact Matrix-Exponential Physics Loss</h3>
                    <p style="font-size: 0.98rem; margin-bottom: 0.5rem;">
                        <strong>What it means:</strong> Traditional finite-difference physics losses create time-discretization errors when steps are larger than fast decay rates (Ra-227 decays in 42 mins). <code>expmix</code> analytically computes the matrix exponential <code>exp(A·t)</code> for the linear Bateman operator.
                    </p>
                    <p style="font-size: 0.92rem; color: var(--text-dim); margin: 0;">
                        <em>Why it matters:</em> Eliminates numerical solver drift so physics loss residuals evaluate true continuous ODE consistency.
                    </p>
                </div>
            </div>

            <div>
                <div class="card card-accent-blue" style="margin-bottom: 1.25rem;">
                    <h3 style="margin-top: 0; color: var(--accent-blue);">3. f* Energy Spectrum Folding Factor</h3>
                    <p style="font-size: 0.98rem; margin-bottom: 0.5rem;">
                        <strong>What it means:</strong> In a 0D point-burnup model, we don't simulate 3D space. <code>f* = 1.24×10⁻³</code> acts as an effective scaling factor linking thermal neutron flux to energy-dependent cross-section integrals.
                    </p>
                    <p style="font-size: 0.92rem; color: var(--text-dim); margin: 0;">
                        <em>Why it matters:</em> Allows the 0D model to approximate realistic reactor energy spectra without requiring full 3D MCNP transport runs.
                    </p>
                </div>

                <div class="card card-accent-orange">
                    <h3 style="margin-top: 0; color: var(--accent-orange);">4. 60-Scenario Locked Evaluation Protocol</h3>
                    <p style="font-size: 0.98rem; margin-bottom: 0.5rem;">
                        <strong>What it means:</strong> Locked a fixed evaluation set of 60 scenarios (seed <code>20260725</code>) spanning Thermal, Epithermal, and Fast regimes with shifted time/energy boundaries.
                    </p>
                    <p style="font-size: 0.92rem; color: var(--text-dim); margin: 0;">
                        <em>Why it matters:</em> Zero data leakage — hyperparameter tuning never sees locked test data. The TC-ACC-001 &lt;3% gate is currently <strong>PENDING</strong> on the poster until full GPU runs finish.
                    </p>
                </div>
            </div>
        </div>
    </div>

    <!-- ═══════════════════════════════════════════════════════════════════
         SLIDE 3 — SOTA ARCHITECTURE ADDITIONS (EXPLAINED IN DETAIL)
         ═══════════════════════════════════════════════════════════════════ -->
    <div class="slide">
        <div class="section-label">Sprint 5 SOTA Sweep</div>
        <div class="slide-title">Detailed Explanation of 2025–26 Upgrades</div>
        <p class="slide-subtitle">We reviewed recent SciML literature to solve specific failure modes in stiff ODE neural training. Here is what each technique does:</p>

        <div class="grid-3">
            <div class="card card-accent-purple">
                <span class="badge badge-purple">OPTIMIZATION</span>
                <h3 style="margin-top: 0.5rem;">SOAP Optimizer</h3>
                <p style="font-size: 0.92rem;"><strong>What it is:</strong> Shampoo preconditioner computed in the Adam eigenbasis (arXiv:2409.11321).</p>
                <p style="font-size: 0.9rem; color: var(--text-secondary);">Standard Adam oscillates on ill-conditioned physics loss landscapes. SOAP rotates gradients into the principal eigenbasis, keeping optimization steps orthogonal and stable.</p>
            </div>

            <div class="card card-accent-blue">
                <span class="badge badge-green">UNCERTAINTY</span>
                <h3 style="margin-top: 0.5rem;">JAWS + ACI Conformal UQ</h3>
                <p style="font-size: 0.92rem;"><strong>What it is:</strong> Jackknife+ After Bootstrap + Adaptive Conformal Inference (arXiv:2207.10716).</p>
                <p style="font-size: 0.9rem; color: var(--text-secondary);">Computes non-parametric, mathematically guaranteed 90% confidence bands. ACI dynamically widens intervals if neutron flux shifts outside training distribution.</p>
            </div>

            <div class="card card-accent-green">
                <span class="badge">MULTI-FIDELITY</span>
                <h3 style="margin-top: 0.5rem;">Multi-Fidelity Head</h3>
                <p style="font-size: 0.92rem;"><strong>What it is:</strong> Log-space residual correction head (Meng & Karniadakis, JCP 2020).</p>
                <p style="font-size: 0.9rem; color: var(--text-secondary);">An 837-parameter delta network that learns small log-scale corrections bridging simple scalar flux predictions to spectrum-folded reactor transport.</p>
            </div>

            <div class="card card-accent-orange">
                <span class="badge badge-orange">CAUSALITY</span>
                <h3 style="margin-top: 0.5rem;">Causality-Weighted Loss</h3>
                <p style="font-size: 0.92rem;"><strong>What it is:</strong> Exponential time-weighting (arXiv:2203.07404) with parameter <code>eps=0.05</code>.</p>
                <p style="font-size: 0.9rem; color: var(--text-secondary);">Forces the neural network to master early time steps (0–10 hours) before attempting late-stage decay, matching physical time progression.</p>
            </div>

            <div class="card card-accent-cyan">
                <span class="badge" style="background: rgba(94,122,117,0.2); color: var(--text-primary);">CONSERVATION</span>
                <h3 style="margin-top: 0.5rem;">Atom Budget Projection</h3>
                <p style="font-size: 0.92rem;"><strong>What it is:</strong> Zero-parameter post-processing mass projection.</p>
                <p style="font-size: 0.9rem; color: var(--text-secondary);">Enforces strictly that the sum of product isotope masses never exceeds the initial Ra-226 mass, holding mass conservation to <code>1e-12</code> precision.</p>
            </div>

            <div class="card card-accent-pink">
                <span class="badge badge-red">ABLATION BASELINE</span>
                <h3 style="margin-top: 0.5rem;">minGRU Selective-Scan</h3>
                <p style="font-size: 0.92rem;"><strong>What it is:</strong> Minimal recurrent baseline (arXiv:2410.01201).</p>
                <p style="font-size: 0.9rem; color: var(--text-secondary);">Used strictly as an ablation baseline to evaluate whether recurrent gating provides genuine accuracy gains over feedforward models.</p>
            </div>
        </div>
    </div>

    <!-- ═══════════════════════════════════════════════════════════════════
         SLIDE 4 — WHAT I AM RUNNING NEXT (BUDGET-CONSCIOUS COLAB PLAN)
         ═══════════════════════════════════════════════════════════════════ -->
    <div class="slide">
        <div class="section-label">Execution Strategy</div>
        <div class="slide-title">What I Am Running Next (Budget-Conscious Colab Matrix)</div>
        <p class="slide-subtitle">To respect GPU compute limits while maintaining scientific rigor, I designed a focused 2-run primary plan + analysis.</p>

        <div class="grid-3" style="margin-bottom: 1.5rem;">
            <div class="card card-accent-green">
                <span class="badge badge-green">MUST RUN #1</span>
                <h3 style="margin-top: 0.5rem; color: var(--accent-green);">RUN_B0_control.ipynb</h3>
                <p style="font-size: 0.95rem; margin-bottom: 0.5rem;"><strong>Control Baseline:</strong> Committed best recipe (expmix + v3 coverage + adaptive weights).</p>
                <p style="font-size: 0.9rem; color: var(--text-secondary); margin: 0;">Provides our control benchmark number on the 60-scenario locked test set.</p>
            </div>

            <div class="card card-accent-orange">
                <span class="badge badge-orange">MUST RUN #2</span>
                <h3 style="margin-top: 0.5rem; color: var(--accent-orange);">RUN_S1_soap.ipynb</h3>
                <p style="font-size: 0.95rem; margin-bottom: 0.5rem;"><strong>SOAP Optimizer Swap:</strong> Direct A/B comparison of SOAP vs Adam optimizer.</p>
                <p style="font-size: 0.9rem; color: var(--text-secondary); margin: 0;">Tests the single highest-ROI upgrade from our 2025–26 literature sweep.</p>
            </div>

            <div class="card card-accent-purple">
                <span class="badge badge-purple">OPTIONAL RUN #3</span>
                <h3 style="margin-top: 0.5rem; color: var(--accent-purple);">RUN_S5_full_stack.ipynb</h3>
                <p style="font-size: 0.95rem; margin-bottom: 0.5rem;"><strong>Full Stack Combination:</strong> SOAP + Causality + EMA + Rollout + Conserve.</p>
                <p style="font-size: 0.9rem; color: var(--text-secondary); margin: 0;">Executed only if GPU compute budget permits a third 6000-epoch run.</p>
            </div>
        </div>

        <div class="code-block">
<span class="hl-comment"># How Colab Execution Works (Budget-Conscious Workflow):</span>
<span class="hl-comment"># 1. Upload IsotopePINN_repo.zip to Google Drive root once (3.4 MB).</span>
<span class="hl-comment"># 2. Open RUN_B0_control.ipynb & RUN_S1_soap.ipynb on T4 GPU runtime.</span>
<span class="hl-comment"># 3. Set FAST_SMOKE = False for full 6000-epoch training (~3 hours per run).</span>
<span class="hl-comment"># 4. Post-training auto-evaluates on locked set and downloads result zip.</span>
        </div>
    </div>

    <!-- ═══════════════════════════════════════════════════════════════════
         SLIDE 5 — CODE WALKTHROUGH: MODEL
         ═══════════════════════════════════════════════════════════════════ -->
    <div class="slide">
        <div class="section-label">Code Walkthrough</div>
        <div class="slide-title">pinn_model.py &mdash; Key Implementation Details</div>
        <p class="slide-subtitle">Exact lines in the repository implementing Fourier encoding, adaptive activations, and exact Bateman integration.</p>

        <div class="grid-2" style="gap: 1.5rem;">
            <div>
                <div class="code-block" style="margin-bottom: 1rem;">
<span class="hl-comment"># ─── LINE 300-316: Fourier Time Encoder ───</span>
<span class="hl-keyword">class</span> <span class="hl-class">FourierTimeEncoder</span>(nn.Module):
    <span class="hl-keyword">def</span> <span class="hl-func">__init__</span>(self, n_freqs=<span class="hl-num">16</span>):
        <span class="hl-comment"># 16 log-spaced freqs → 32 features (sin+cos)</span>
        <span class="hl-comment"># Spans octaves to cover λ_Ra227/λ_Ac225 ≈ 338</span>
        freqs = torch.logspace(<span class="hl-num">-2</span>, <span class="hl-num">1</span>, n_freqs) * π / t_ref_h
</div>
                <div class="code-block">
<span class="hl-comment"># ─── LINE 319-356: Fourier Energy Encoder ───</span>
<span class="hl-keyword">class</span> <span class="hl-class">FourierEnergyEncoder</span>(nn.Module):
    <span class="hl-comment"># Maps log10(E_eV) through sin/cos bands</span>
    <span class="hl-comment"># Centered on the 6.42 MeV (n,2n) threshold</span>
    freqs = torch.logspace(<span class="hl-num">-0.7</span>, <span class="hl-num">0.9</span>, n_freqs) * π
</div>
            </div>
            <div>
                <div class="code-block" style="margin-bottom: 1rem;">
<span class="hl-comment"># ─── LINE 359-372: Adaptive Activation ───</span>
<span class="hl-keyword">class</span> <span class="hl-class">AdaptiveActivation</span>(nn.Module):
    <span class="hl-comment"># Learnable slope 'a' per layer: tanh(a·x)</span>
    self.a = nn.Parameter(torch.tensor(<span class="hl-num">1.0</span>))
    <span class="hl-keyword">def</span> <span class="hl-func">forward</span>(self, x):
        <span class="hl-keyword">return</span> torch.tanh(self.a * x)
</div>
                <div class="code-block">
<span class="hl-comment"># ─── LINE 595-600: Integral-Based PINN ───</span>
<span class="hl-keyword">def</span> <span class="hl-func">forward_raw</span>(self, x):
    <span class="hl-comment"># Ra-226: N₀ * exp(-∫k(t)dt) [monotone decay]</span>
    <span class="hl-comment"># Daughters: semi-analytic Bateman integration</span>
</div>
            </div>
        </div>
    </div>

    <!-- ═══════════════════════════════════════════════════════════════════
         SLIDE 6 — CODE WALKTHROUGH: TRAINING & VERIFICATION
         ═══════════════════════════════════════════════════════════════════ -->
    <div class="slide">
        <div class="section-label">Code Walkthrough</div>
        <div class="slide-title">Training Strategy & 22/22 Smoke Verification</div>
        <p class="slide-subtitle">Training methodology and complete automated verification suite ensuring zero regression.</p>

        <div class="grid-2" style="gap: 2rem;">
            <div>
                <h3 style="color: var(--accent-orange);">Two-Phase Training Strategy</h3>
                <ul>
                    <li><strong>Phase 1 &mdash; Physics Only (600 epochs):</strong> Zero solver data. The model learns Bateman ODE consistency first so physics acts as a true prior.</li>
                    <li><strong>Phase 2 &mdash; Joint Training (3,400+ epochs):</strong> Combines data MSE, physics residuals, and mass conservation constraints under a time-curriculum ramp.</li>
                </ul>

                <h3 style="color: var(--accent-green);">22 / 22 Automated Smoke Suite</h3>
                <div class="card card-accent-green" style="padding: 1rem;">
                    <div style="font-family: var(--font-mono); font-size: 0.88rem; color: var(--text-primary);">
                        <strong>v3_pilstm/scripts/smoke_checks.py</strong><br>
                        <span class="badge badge-green">22 / 22 PASS</span> &middot; Bit-Identical Compatibility Verified
                    </div>
                    <p style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.5rem; margin-bottom: 0;">
                        Verifies loss exactness, legacy checkpoint compatibility, expmix path, speed benchmark, jackknife/CV+, adaptive weights, curriculum, and semigroup rollout.
                    </p>
                </div>
            </div>

            <div>
                <div class="code-block" style="margin-bottom: 1rem;">
<span class="hl-comment"># ─── LINE 1384: Jacobian Normalization ───</span>
<span class="hl-comment"># Problem: Ra-227 decays in 42 mins (large decay)</span>
<span class="hl-comment"># Ac-225 decays in 9.9 days (smaller decay)</span>
<span class="hl-comment"># Fix: divide residual by √(1 + ‖J_i‖²)</span>
<span class="hl-comment"># Keeps trace species from being drowed out.</span>
</div>
                <div class="code-block">
<span class="hl-comment"># ─── train_pi_lstm.py: Key Sprint 5 Flags ───</span>
PI_LSTM_OPT        = <span class="hl-string">"soap"</span>   <span class="hl-comment"># SOAP optimizer</span>
PI_LSTM_CAUSALITY  = <span class="hl-string">"0.05"</span>   <span class="hl-comment"># temporal causality</span>
PI_LSTM_EMA        = <span class="hl-string">"1"</span>      <span class="hl-comment"># weight smoothing</span>
PI_LSTM_ROLLOUT    = <span class="hl-string">"4"</span>      <span class="hl-comment"># semigroup rollout</span>
PI_LSTM_CONSERVE   = <span class="hl-string">"1"</span>      <span class="hl-comment"># mass projection</span>
</div>
            </div>
        </div>
    </div>

    <!-- ═══════════════════════════════════════════════════════════════════
         SLIDE 7 — KNOWN LIMITATIONS (TRANSPARENT ACKNOWLEDGEMENT)
         ═══════════════════════════════════════════════════════════════════ -->
    <div class="slide">
        <div class="section-label">Honest Assessment</div>
        <div class="slide-title">Current Performance & Known Limitations</div>
        <p class="slide-subtitle">Transparent evaluation of model capabilities and explicit scientific boundaries.</p>

        <div class="grid-3" style="margin-bottom: 1.5rem;">
            <div class="card" style="text-align: center;">
                <div class="stat-value" style="color: var(--accent-green);">22 / 22</div>
                <div class="stat-label">Smoke Suite Checks<br>PASS (Bit-Identical)</div>
            </div>
            <div class="card" style="text-align: center;">
                <div class="stat-value" style="color: var(--accent-purple);">1.65 ms</div>
                <div class="stat-label">Batched PINN Inference<br>(~83× Speedup vs ODE)</div>
            </div>
            <div class="card" style="text-align: center;">
                <div class="stat-value" style="color: var(--accent-orange);">PENDING</div>
                <div class="stat-label">TC-ACC-001 &lt;3% Gate<br>(Awaits Colab GPU Runs)</div>
            </div>
        </div>

        <div class="grid-2">
            <div class="card card-accent-green">
                <h3 style="margin-top: 0; color: var(--accent-green);">What Is Validated & Working</h3>
                <ul>
                    <li>Ra-226 is architecturally constrained to monotone decay &mdash; cannot create atoms from vacuum.</li>
                    <li>Fourier energy features capture the sharp 6.42 MeV threshold where (n,2n) reaction activates.</li>
                    <li>Mass conservation held to <code>1e-12</code> using zero-parameter atom budget projection.</li>
                    <li>All legacy model checkpoints verify bit-identical backward compatibility.</li>
                </ul>
            </div>
            <div class="card card-accent-red">
                <h3 style="margin-top: 0; color: var(--accent-red);">Explicit Scientific Limitations</h3>
                <ul>
                    <li>Validated against Bateman Radau5 numerical ODE solver, <strong>not</strong> experimental reactor assay data.</li>
                    <li>Assumes 0D point-burnup homogeneous irradiation volume (no 3D spatial neutron transport).</li>
                    <li>Multi-fidelity head & exp-correction wrappers are currently non-checkpoint-resumable (smoke-only).</li>
                    <li>Final poster accuracy metrics await full 6000-epoch GPU runs on Colab.</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- ═══════════════════════════════════════════════════════════════════
         SLIDE 8 — BLUNT REQUESTS FOR JADEN (V1 VOCABULARY & STYLE)
         ═══════════════════════════════════════════════════════════════════ -->
    <div class="slide">
        <div class="section-label">Discussion</div>
        <div class="slide-title">Blunt Requests for Jayden</div>
        <p class="slide-subtitle">Direct technical questions and high-leverage mentorship requests to maximize scientific rigor.</p>

        <div class="ask-card">
            <span class="ask-number">1</span>
            <div class="ask-title">Compute Sponsorship for Full 6000-Epoch GPU Runs</div>
            <div class="ask-script">"Can your lab run my 2 MUST-RUN notebooks (B0 control & S1 SOAP) on a spare GPU overnight, or sponsor student credits on NCSU's Henry2 HPC cluster?"</div>
        </div>

        <div class="ask-card">
            <span class="ask-number">2</span>
            <div class="ask-title">PULSTAR Reactor Spectrum Faculty Intro</div>
            <div class="ask-script">"Could you introduce me to an NCSU Nuclear Engineering faculty member to sanity-check my f* parametric spectrum folding derivation against real reactor spectra?"</div>
        </div>

        <div class="ask-card">
            <span class="ask-number">3</span>
            <div class="ask-title">Adversarial Evaluation Protocol Review</div>
            <div class="ask-script">"Can we spend 30 minutes trying to break my 60-scenario locked-test protocol to ensure zero data leakage before I lock poster figures?"</div>
        </div>

        <div class="ask-card">
            <span class="ask-number">4</span>
            <div class="ask-title">ISEF Form 2A & Qualified Scientist Sign-Off</div>
            <div class="ask-script">"Would you or a lab member be open to reviewing the methodology to sign off as Qualified Scientist on ISEF Form 2A (AI Assistance Disclosure)?"</div>
        </div>

        <div class="ask-card">
            <span class="ask-number">5</span>
            <div class="ask-title">December Mock Judging Session</div>
            <div class="ask-script">"Would you be open to doing a 15-minute mock poster Q&A with me in December before the regional competition in January?"</div>
        </div>
    </div>

    <!-- ═══════════════════════════════════════════════════════════════════
         SLIDE 9 — ISEF 2027 ROADMAP & SUMMARY
         ═══════════════════════════════════════════════════════════════════ -->
    <div class="slide" style="justify-content: center;">
        <div class="section-label">Next Steps & Strategy</div>
        <h1 class="hero-title" style="font-size: clamp(2rem, 4vw, 3.5rem);">ISEF 2027 Strategic Roadmap</h1>
        <p style="font-size: 1.25rem; color: var(--text-primary); max-width: 100%; font-weight: 300;">
            Structured timeline leading up to regional and international science fair competitions.
        </p>

        <div class="grid-3" style="margin-top: 2rem; gap: 1.25rem;">
            <div class="card card-accent-blue">
                <h3 style="margin-top: 0;">August 2026</h3>
                <ul>
                    <li>Execute Colab GPU runs (B0 & S1)</li>
                    <li>Update locked-set metrics on poster</li>
                    <li>Finalize JAWS+ACI error bars</li>
                </ul>
            </div>

            <div class="card card-accent-purple">
                <h3 style="margin-top: 0;">Oct &ndash; Dec 2026</h3>
                <ul>
                    <li>Complete ISEF Research Plan</li>
                    <li>File Form 2A (AI Disclosure)</li>
                    <li>Mock judging with Jayden</li>
                </ul>
            </div>

            <div class="card card-accent-green">
                <h3 style="margin-top: 0;">Jan &ndash; May 2027</h3>
                <ul>
                    <li>Regional Science Fair</li>
                    <li>State Competition</li>
                    <li>ISEF 2027 Grand Awards</li>
                </ul>
            </div>
        </div>
    </div>

</div>

<!-- Navigation Bar -->
<div class="nav-bar">
    <div class="slide-counter"><span id="current">1</span> / <span id="total">9</span></div>
    <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width: 11%;"></div></div>
    <div class="nav-btns">
        <button class="nav-btn" id="prevBtn" disabled>&larr;</button>
        <button class="nav-btn" id="nextBtn">&rarr;</button>
    </div>
</div>

<div class="swipe-hint" id="swipeHint">Use &larr; &rarr; or Space | 'N' for Speaker Notes</div>

<!-- Speaker Notes Panel -->
<div class="notes-panel hidden" id="notesPanel">
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
        `<strong>What to say:</strong> "Hi Jaden, thanks again for taking the time! Today I want to walk you through our Sprint 4 and Sprint 5 technical progress on IsotopePINN—including evaluated nuclear cross-sections, exact matrix-exponential physics, the SOAP optimizer, and how we can collaborate on compute and reactor spectrum validation."<br><br><strong>Tip:</strong> Keep this short. Introduce yourself as a high school junior dual-enrolled at AACC.`,

        `<strong>What to say:</strong> "Since our last meeting, I upgraded the core physics: we replaced synthetic cross-sections with evaluated EXFOR/JENDL data, implemented expmix exact matrix-exponential loss to solve time-discretization errors on fast decayers, added f* spectrum folding, and locked down a 60-scenario evaluation protocol."<br><br><strong>Tip:</strong> Emphasize that the TC-ACC-001 <3% gate is currently PENDING on the poster until full GPU runs finish—zero fake numbers.`,

        `<strong>What to say:</strong> "We also did a 2025–26 literature sweep and added 6 techniques to fix specific neural training issues: SOAP optimizer for stiff loss landscapes, JAWS+ACI conformal prediction for honest UQ under distribution shift, a multi-fidelity residual head, and temporal causality loss."<br><br><strong>Tip:</strong> Don't just list buzzwords—explain the physical reason behind each technique.`,

        `<strong>What to say:</strong> "To manage GPU compute budget, I designed a focused Colab matrix: RUN_B0 (our control baseline) and RUN_S1 (the SOAP optimizer swap) are our two MUST-RUN notebooks. Each runs for 6000 epochs on a T4 GPU and evaluates on the locked test set."<br><br><strong>Tip:</strong> Point out that the repo zip is only 3.4 MB and uploads to Drive once.`,

        `<strong>What to say:</strong> "Here are the exact lines in my code. Line 300 is the Fourier Time Encoder. Line 319 is the Energy Encoder centered on 6.42 MeV. Line 359 is adaptive activation. Line 595 is the main forward pass with Bateman integration."<br><br><strong>Tip:</strong> Offer to share your screen and open VS Code to pinn_model.py.`,

        `<strong>What to say:</strong> "Training is two phases: first 600 epochs are physics-only with zero data, then joint training adds solver data and mass constraints. Our automated test suite checks 22 checks—all 22 pass with bit-identical backward compatibility."<br><br><strong>Tip:</strong> Point to the green 22/22 PASS badge.`,

        `<strong>What to say:</strong> "I want to be fully transparent about limitations: we validate against a numerical Radau solver, not experimental reactor assay data; we assume 0D point-burnup; and final poster accuracy metrics await full 6000-epoch GPU runs."<br><br><strong>Tip:</strong> Being upfront about limitations builds massive credibility with judges.`,

        `<strong>What to say:</strong> Go through these one at a time. Read the scripts word-for-word.<br><br><strong>Ask 1 (Compute):</strong> "Can your lab run my 2 MUST-RUN notebooks on a spare GPU overnight, or sponsor Henry2 HPC access?"<br><strong>Ask 2 (PULSTAR):</strong> "Could you introduce me to an NE faculty member to review my f* derivation?"<br><strong>Ask 3 (Protocol):</strong> "Can we spend 30 mins trying to break my locked test protocol?"<br><strong>Ask 4 (Form 2A):</strong> "Would someone in your lab be open to signing as Qualified Scientist on ISEF Form 2A?"<br><strong>Ask 5 (Mock Judging):</strong> "Would you be open to doing a 15-min mock poster Q&A in December?"`,

        `<strong>What to say:</strong> "Here is my ISEF 2027 strategic roadmap leading up to regionals in January. All code and documentation are committed in the GitHub repository. Thank you so much for your guidance!"<br><br><strong>Tip:</strong> Confirm next steps before hanging up.`
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
    path_v2 = "ncsu_pitch_deck_v2.html"
    path_main = "ncsu_pitch_deck.html"
    path_script = "build_ncsu_pitch_deck_v2.py"
    
    with open(path_v2, "w", encoding="utf-8", newline="\n") as f:
        f.write(HTML_CONTENT)
    print(f"Successfully generated {path_v2} ({len(HTML_CONTENT)} bytes)")
    
    shutil.copy(path_v2, path_main)
    print(f"Successfully synced {path_main}")

if __name__ == "__main__":
    main()
