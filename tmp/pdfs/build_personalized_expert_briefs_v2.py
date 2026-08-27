from __future__ import annotations

from html import escape
from itertools import count
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(r"C:\Users\ogunn\Downloads\New folder")
OUT = ROOT / "Ac225_Personalized_Expert_Briefs_2026"
OUT.mkdir(parents=True, exist_ok=True)

INK = colors.HexColor("#17232d")
TEXT = colors.HexColor("#354653")
MUTED = colors.HexColor("#687783")
LINE = colors.HexColor("#cbd6dc")
PAPER = colors.white
SOFT = colors.HexColor("#f4f7f8")
RED = colors.HexColor("#a92532")
TEAL = colors.HexColor("#006c70")
BLUE = colors.HexColor("#285f8e")
GOLD = colors.HexColor("#9b660d")
GREEN = colors.HexColor("#26734d")
PURPLE = colors.HexColor("#6b4f86")
CYAN = colors.HexColor("#287d8b")

PAGE_W, PAGE_H = letter
MARGIN_X = 34
TOP = 33
BOTTOM = 30
CONTENT_W = PAGE_W - 2 * MARGIN_X
_style_ids = count(1)


def shade(color: colors.Color, amount: float = 0.9) -> colors.Color:
    return colors.Color(
        color.red + (1 - color.red) * amount,
        color.green + (1 - color.green) * amount,
        color.blue + (1 - color.blue) * amount,
    )


def style(**kwargs) -> ParagraphStyle:
    base = {
        "fontName": "Helvetica",
        "fontSize": 8.5,
        "leading": 10.6,
        "textColor": TEXT,
        "spaceBefore": 0,
        "spaceAfter": 0,
        "alignment": TA_LEFT,
    }
    base.update(kwargs)
    return ParagraphStyle(f"s{next(_style_ids)}", **base)


def p(text: str, **kwargs) -> Paragraph:
    return Paragraph(text, style(**kwargs))


def bullet_list(items: list[str], *, size: float = 8.2, color=TEXT) -> ListFlowable:
    return ListFlowable(
        [
            ListItem(
                p(item, fontSize=size, leading=size + 2.1, textColor=color),
                leftIndent=10,
            )
            for item in items
        ],
        bulletType="bullet",
        start="square",
        leftIndent=13,
        bulletFontName="Helvetica-Bold",
        bulletFontSize=5,
        bulletColor=color,
        spaceBefore=2,
        spaceAfter=1,
    )


def box(
    content,
    *,
    background=SOFT,
    border=LINE,
    padding=8,
    left_bar=None,
    widths=None,
) -> Table:
    if not isinstance(content, list):
        content = [[content]]
    table = Table(content, colWidths=widths, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("BOX", (0, 0), (-1, -1), 0.7, border),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), padding),
        ("RIGHTPADDING", (0, 0), (-1, -1), padding),
        ("TOPPADDING", (0, 0), (-1, -1), padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
    ]
    if left_bar:
        commands.append(("LINEBEFORE", (0, 0), (0, -1), 4, left_bar))
    table.setStyle(TableStyle(commands))
    return table


def line_label(text: str, accent) -> Table:
    label = p(
        text.upper(),
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=8.6,
        textColor=accent,
    )
    table = Table([[label, ""]], colWidths=[160, CONTENT_W - 160])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LINEBELOW", (1, 0), (1, 0), 0.7, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return table


def document_header(
    kicker: str,
    title: str,
    recipient: str,
    role: str,
    accent,
    *,
    subtitle: str | None = None,
) -> list:
    title_block = [
        p(
            kicker.upper(),
            fontName="Helvetica-Bold",
            fontSize=7.4,
            leading=8.5,
            textColor=accent,
        ),
        Spacer(1, 4),
        p(
            title,
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=21.5,
            textColor=INK,
        ),
    ]
    if subtitle:
        title_block.extend(
            [
                Spacer(1, 4),
                p(subtitle, fontSize=8.5, leading=10.2, textColor=MUTED),
            ]
        )
    identity = box(
        [
            [
                p("PERSONAL NOTE FOR", fontName="Helvetica-Bold", fontSize=6.8, textColor=accent),
            ],
            [p(recipient, fontName="Helvetica-Bold", fontSize=10.5, leading=11.2, textColor=INK)],
            [p(role, fontSize=7.3, leading=8.7, textColor=MUTED)],
        ],
        background=shade(accent, 0.94),
        border=shade(accent, 0.72),
        padding=7,
    )
    header = Table([[title_block, identity]], colWidths=[CONTENT_W - 170, 170])
    header.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 12),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return [header, Spacer(1, 10)]


def build_pdf(path: Path, accent, story: list, short_title: str) -> None:
    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(accent)
        canvas.rect(0, PAGE_H - 7, PAGE_W, 7, fill=1, stroke=0)
        canvas.setStrokeColor(LINE)
        canvas.line(MARGIN_X, 23, PAGE_W - MARGIN_X, 23)
        canvas.setFont("Helvetica", 6.8)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN_X, 12, "Samuel Ogunnubi | Grade 12 | Glen Burnie High School | Maryland")
        canvas.drawRightString(PAGE_W - MARGIN_X, 12, short_title)
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=TOP,
        bottomMargin=BOTTOM,
        title=short_title,
        author="Samuel Ogunnubi",
        subject="Personalized technical research question",
    )
    frame = Frame(
        MARGIN_X,
        BOTTOM,
        CONTENT_W,
        PAGE_H - TOP - BOTTOM,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="main",
    )
    doc.addPageTemplates([PageTemplate(id="note", frames=[frame], onPage=on_page)])
    doc.build(story)


def lee_story() -> list:
    accent = RED
    story = document_header(
        "Berkeley experiment follow-up",
        "I tested V3 on the Berkeley cases. It failed, and I want to understand why.",
        "Professor Lee A. Bernstein",
        "UC Berkeley and Lawrence Berkeley National Laboratory",
        accent,
        subtitle="This is the first real outside test of my frozen Ra-226 to Ac-225 model.",
    )
    story += [
        p("Professor Bernstein,", fontName="Helvetica-Bold", fontSize=9.2, textColor=INK),
        Spacer(1, 4),
        p(
            "The neutron-spectrum code and papers you sent gave me something I had been missing for months: a real experiment close to the pathway I modeled. I kept my seed-42 checkpoint frozen, rebuilt the 33 and 40 MeV cases as carefully as I could, and ran the comparison without changing the model afterward.",
            fontSize=9,
            leading=11.5,
        ),
        Spacer(1, 8),
        box(
            [[
                p("33 MeV case", fontName="Helvetica-Bold", fontSize=8.2, textColor=accent),
                p("Raw V3 activity was about <b>623 times too high</b>.", fontSize=8.4, leading=10.2),
                p("40 MeV case", fontName="Helvetica-Bold", fontSize=8.2, textColor=accent),
                p("Raw V3 activity was about <b>600 times too high</b>.", fontSize=8.4, leading=10.2),
            ]],
            background=shade(accent, 0.94),
            border=shade(accent, 0.72),
            padding=7,
            widths=[70, 190, 70, 190],
        ),
        Spacer(1, 9),
        line_label("Where I got stuck", accent),
        Spacer(1, 5),
        p(
            "I do not think it would be honest to tune V3 until those two points look good. Before I call this a model failure, though, I need to know whether I reconstructed the experiment correctly. The dissertation reports the recovered Ac-225 activities, while my equations predict atoms produced in the target. Those are not automatically the same quantity.",
            fontSize=8.8,
            leading=11.1,
        ),
        Spacer(1, 8),
        Table(
            [[
                box(
                    [[p("Details I can already document", fontName="Helvetica-Bold", fontSize=9, textColor=TEAL)], [bullet_list([
                        "The published 33 and 40 MeV neutron-spectrum reconstruction.",
                        "A 1 mg Ra-226 target and the reported Ac-225 activities.",
                        "The frozen model, source hashes, inputs, and failed predictions.",
                    ])]],
                    background=shade(TEAL, 0.94),
                    border=shade(TEAL, 0.72),
                    padding=7,
                ),
                box(
                    [[p("Details I still cannot pin down", fontName="Helvetica-Bold", fontSize=9, textColor=GOLD)], [bullet_list([
                        "Spectrum normalization or fluence at the radium target position.",
                        "Target geometry, position, irradiation, separation, assay, and reference times.",
                        "The exact meaning of produced versus recovered activity and any numerical Ac-227 detection limit.",
                    ])]],
                    background=shade(GOLD, 0.93),
                    border=shade(GOLD, 0.72),
                    padding=7,
                ),
            ]],
            colWidths=[CONTENT_W / 2 - 4, CONTENT_W / 2 - 4],
            style=[
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 4),
                ("LEFTPADDING", (1, 0), (1, 0), 4),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
            ],
        ),
        Spacer(1, 9),
        box(
            p(
                "<b>What I am asking:</b> Could you point me to where those details are defined, correct anything I misunderstood, or tell me who would know? Even a short reference would let me rebuild the comparison correctly and report the failure for the right scientific reason.",
                fontSize=9,
                leading=11.3,
                textColor=INK,
            ),
            background=shade(BLUE, 0.93),
            border=shade(BLUE, 0.68),
            padding=9,
            left_bar=BLUE,
        ),
        Spacer(1, 7),
        p(
            "If it helps, I can send the exact inputs, reconstruction table, source code, frozen checkpoint, and the two failed outputs. I am not asking for private facility data or for Berkeley to validate my model.",
            fontSize=7.7,
            leading=9.4,
            textColor=MUTED,
        ),
    ]
    return story


def jaden_story() -> list:
    accent = TEAL
    story = document_header(
        "Mentor decision note",
        "Jaden, I need your honest read on what V3 actually proves.",
        "Jaden Palmer",
        "ARTISANS Lab, North Carolina State University",
        accent,
        subtitle="The model is trained and frozen. My problem now is deciding what claim is fair.",
    )
    story += [
        p(
            "Your advice during our last call honestly changed the project. I finished the LSTM run, froze it before outside testing, and then tested it much harder than I originally planned. Some results are strong. Some are uncomfortable. I want the final project to show both.",
            fontSize=9.1,
            leading=11.5,
        ),
        Spacer(1, 8),
        line_label("The scorecard I would show a judge", accent),
        Spacer(1, 5),
    ]
    score_rows = [
        ["0.97%", "Matched synthetic test", "Worst-mesh median Ac-225 endpoint error."],
        ["3.94%", "Wider 10,000-case grid", "Productive-case median; the p95 rose to 183.84%."],
        ["Rank 2", "Schedule screening", "V3 selected Radau's second-ranked schedule with 0.0470% objective regret."],
        ["Mixed", "LSTM versus matched MLP", "The evidence does not support saying the LSTM is always better."],
        ["Failed", "Berkeley outside test", "The two cases were outside training support and about 600 times too high."],
        ["Slower", "Exact analytic baseline", "The exact solver beats V3 on this reduced constant-rate system."],
    ]
    score = Table(
        [[
            p("RESULT", fontName="Helvetica-Bold", fontSize=7, textColor=PAPER),
            p("TEST", fontName="Helvetica-Bold", fontSize=7, textColor=PAPER),
            p("WHAT IT MEANS", fontName="Helvetica-Bold", fontSize=7, textColor=PAPER),
        ]]
        + [
            [
                p(a, fontName="Helvetica-Bold", fontSize=8.6, textColor=accent),
                p(b, fontName="Helvetica-Bold", fontSize=7.7, textColor=INK),
                p(c, fontSize=7.5, leading=9.1),
            ]
            for a, b, c in score_rows
        ],
        colWidths=[58, 137, CONTENT_W - 195],
    )
    score.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), accent),
                ("GRID", (0, 0), (-1, -1), 0.5, LINE),
                ("BACKGROUND", (0, 1), (-1, -1), PAPER),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PAPER, SOFT]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story += [
        score,
        Spacer(1, 9),
        Table(
            [[
                box(
                    [[p("What I think I can defend", fontName="Helvetica-Bold", fontSize=9, textColor=GREEN)], [bullet_list([
                        "V3 can be studied as a frozen surrogate for solver-verified screening inside a declared reduced domain.",
                        "Its median ranking behavior can still be useful even when every finalist is checked by a trusted solver.",
                        "The failure map and warning system are part of the result, not something to hide.",
                    ], size=7.8)]],
                    background=shade(GREEN, 0.93),
                    border=shade(GREEN, 0.7),
                    padding=7,
                ),
                box(
                    [[p("What I am still unsure about", fontName="Helvetica-Bold", fontSize=9, textColor=RED)], [bullet_list([
                        "Which error statistic should be the headline without making the model look better than it is.",
                        "How to define uncertainty and an out-of-range warning that a judge will accept.",
                        "Whether the current model still has a useful research role when the exact reduced solver is faster.",
                    ], size=7.8)]],
                    background=shade(RED, 0.94),
                    border=shade(RED, 0.72),
                    padding=7,
                ),
            ]],
            colWidths=[CONTENT_W / 2 - 4, CONTENT_W / 2 - 4],
            style=[
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 4),
                ("LEFTPADDING", (1, 0), (1, 0), 4),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
            ],
        ),
        Spacer(1, 9),
        box(
            [[p("The decision I need help making", fontName="Helvetica-Bold", fontSize=9.3, textColor=accent)], [p(
                "If you were judging this project, what exact research question and claim would you accept from these results? I also need your opinion on the minimum additional evidence that matters most before December, especially if I cannot afford another full training run.",
                fontSize=9,
                leading=11.2,
                textColor=INK,
            )]],
            background=shade(accent, 0.93),
            border=shade(accent, 0.65),
            padding=9,
            left_bar=accent,
        ),
    ]
    return story


def brockman_story() -> list:
    accent = BLUE
    story = document_header(
        "Production workflow question",
        "My model ends at cooling. Real isotope production does not.",
        "Dr. John Brockman",
        "University of Missouri Research Reactor",
        accent,
        subtitle="I need a public, realistic production timeline so I stop optimizing only the easy half.",
    )
    story += [
        p(
            "My name is Samuel Ogunnubi, and I am a 12th grader in Glen Burnie High School's Biomedical and Allied Health program in Maryland. I started this project after learning what Ac-225 can do in targeted cancer treatment. My original goal was simple: see whether a computer model could help compare production plans faster and eventually help make the process more practical.",
            fontSize=8.9,
            leading=11.2,
        ),
        Spacer(1, 8),
        line_label("The problem I finally noticed", accent),
        Spacer(1, 5),
        p(
            "I built a physics-guided LSTM around irradiation, isotope buildup and decay, and cooling. That is useful for testing the code, but it is not the whole production process. An exact solver is already faster for my small constant-rate equation set. If this project is going to matter, it has to study the real end-to-end timeline and the constraints that actually slow a facility down.",
            fontSize=8.8,
            leading=11,
        ),
        Spacer(1, 9),
    ]
    steps = [
        ("1", "Irradiate", True),
        ("2", "Build and decay", True),
        ("3", "Cool", True),
        ("4", "Separate", False),
        ("5", "Purify", False),
        ("6", "Quality check", False),
        ("7", "Recycle or deliver", False),
    ]
    process_cells = []
    for number, label, modeled in steps:
        color = GREEN if modeled else GOLD
        process_cells.append(
            box(
                [[p(number, fontName="Helvetica-Bold", fontSize=12, textColor=color, alignment=TA_CENTER)], [p(label, fontName="Helvetica-Bold", fontSize=7.2, leading=8.4, textColor=INK, alignment=TA_CENTER)], [p("MODELED" if modeled else "MISSING", fontName="Helvetica-Bold", fontSize=5.8, textColor=color, alignment=TA_CENTER)]],
                background=shade(color, 0.94),
                border=shade(color, 0.68),
                padding=5,
            )
        )
    process = Table([process_cells], colWidths=[CONTENT_W / 7] * 7)
    process.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story += [
        process,
        Spacer(1, 9),
        Table(
            [[
                box(
                    [[p("What I already have", fontName="Helvetica-Bold", fontSize=9, textColor=GREEN)], [bullet_list([
                        "A frozen reduced model for irradiation, decay, and cooling.",
                        "A 10,000-case schedule study with every finalist checked by Radau.",
                        "Produced and recovered activity stored separately in my Berkeley reconstruction.",
                    ], size=7.8)]],
                    background=shade(GREEN, 0.94),
                    border=shade(GREEN, 0.72),
                    padding=7,
                ),
                box(
                    [[p("What I need from real operations", fontName="Helvetica-Bold", fontSize=9, textColor=GOLD)], [bullet_list([
                        "Public ranges for separation, purification, quality-control, and turnaround time.",
                        "Recovery losses, target reuse, staffing or equipment constraints, and the step that usually controls the schedule.",
                        "One public process example I can cite and reproduce without using private facility data.",
                    ], size=7.8)]],
                    background=shade(GOLD, 0.93),
                    border=shade(GOLD, 0.7),
                    padding=7,
                ),
            ]],
            colWidths=[CONTENT_W / 2 - 4, CONTENT_W / 2 - 4],
            style=[
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 4),
                ("LEFTPADDING", (1, 0), (1, 0), 4),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
            ],
        ),
        Spacer(1, 9),
        box(
            p(
                "<b>My question:</b> Could you recommend a public example that follows an isotope from irradiation through a usable, quality-checked product, or point me to the MURR person who understands that workflow best? I would use it to change my goal from 'predict atoms quickly' to 'compare total time and usable activity under realistic constraints.'",
                fontSize=9,
                leading=11.3,
                textColor=INK,
            ),
            background=shade(accent, 0.93),
            border=shade(accent, 0.65),
            padding=9,
            left_bar=accent,
        ),
        Spacer(1, 7),
        p(
            "I am not asking for proprietary MURR numbers or a radioactive experiment. A published workflow, reasonable public ranges, or the right person to ask would be enough to move the project forward.",
            fontSize=7.7,
            leading=9.4,
            textColor=MUTED,
        ),
    ]
    return story


def hobbs_story() -> list:
    accent = PURPLE
    story = document_header(
        "Medical meaning and reporting",
        "I can calculate Ac-225 activity. I need help deciding what makes that output scientifically useful.",
        "Dr. Robert F. Hobbs",
        "Johns Hopkins Radiation Oncology and Molecular Radiation Sciences",
        accent,
        subtitle="I want the production output to be useful without turning it into a clinical claim.",
    )
    story += [
        p(
            "I am a 12th grader at Glen Burnie High School in Maryland, and I built a reduced model of one Ra-226 to Ac-225 production pathway after learning about targeted alpha therapy. Your work in alpha-particle therapy, radiopharmaceutical dosimetry, modeling, and treatment planning is a close match to the question I am trying to answer. My concern is that a mathematically correct activity value can still be misleading if I report it at the wrong time, use the wrong impurity definition, or connect it to treatment in a way the project has not earned.",
            fontSize=8.9,
            leading=11.2,
        ),
        Spacer(1, 9),
    ]
    left = box(
        [[p("What the model currently gives me", fontName="Helvetica-Bold", fontSize=9.5, textColor=BLUE)], [bullet_list([
            "Predicted Ac-225 atoms and activity throughout irradiation and cooling.",
            "Predicted trace Ac-227 from the reduced side pathway.",
            "A candidate schedule and a stated calculation time.",
            "Synthetic error statistics and a warning when inputs leave the studied range.",
        ], size=7.9)]],
        background=shade(BLUE, 0.94),
        border=shade(BLUE, 0.72),
        padding=8,
    )
    right = box(
        [[p("What those numbers still do not tell me", fontName="Helvetica-Bold", fontSize=9.5, textColor=accent)], [bullet_list([
            "Which reference time makes a production comparison fair.",
            "Whether impurity should be reported by activity, atoms, mass, or another convention.",
            "How daughter products and measurement uncertainty should be stated.",
            "What information is useful to a medical reader without implying dose, safety, or clinical validation.",
        ], size=7.9)]],
        background=shade(accent, 0.94),
        border=shade(accent, 0.72),
        padding=8,
    )
    story += [
        Table(
            [[left, right]],
            colWidths=[CONTENT_W / 2 - 4, CONTENT_W / 2 - 4],
            style=[
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 4),
                ("LEFTPADDING", (1, 0), (1, 0), 4),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
            ],
        ),
        Spacer(1, 10),
        line_label("The reporting choice I am trying not to guess", accent),
        Spacer(1, 5),
        p(
            "I found an IAEA example that uses a 2% Ac-227/Ac-225 activity specification, but I am not treating it as a universal medical or regulatory limit. I would rather cite the right convention and clearly label what is only an example. The project has no patient data, dose model, or clinical validation.",
            fontSize=8.8,
            leading=11,
        ),
        Spacer(1, 9),
        box(
            [[p("What I would value from you", fontName="Helvetica-Bold", fontSize=9.5, textColor=accent)], [p(
                "Could you recommend one or two public references that show how an Ac-225 production or product-quality study should report activity, reference time, radionuclidic impurity, daughter equilibrium, and uncertainty? If there is a better person at Hopkins for this exact question, a referral would also help.",
                fontSize=9.1,
                leading=11.5,
                textColor=INK,
            )]],
            background=shade(accent, 0.93),
            border=shade(accent, 0.65),
            padding=9,
            left_bar=accent,
        ),
        Spacer(1, 9),
        box(
            [[p("How this changes my project", fontName="Helvetica-Bold", fontSize=9.2, textColor=GREEN)], [p(
                "I would use the reference to define the output before ranking schedules, remove any unsupported purity claim, and explain the medical relevance in plain language while keeping the final claim strictly about production-side screening.",
                fontSize=8.8,
                leading=11,
            )]],
            background=shade(GREEN, 0.94),
            border=shade(GREEN, 0.72),
            padding=8,
        ),
    ]
    return story


def engle_story() -> list:
    accent = GOLD
    story = document_header(
        "Targetry and feasibility",
        "What would make one of my Ac-225 production plans physically possible?",
        "Professor Jonathan W. Engle",
        "University of Wisconsin-Madison Cyclotron Laboratory",
        accent,
        subtitle="Right now my equations can rank a schedule that a real target could never survive.",
    )
    story += [
        p(
            "My name is Samuel Ogunnubi, and I am a 12th grader in Maryland. I spent my summer building a physics-guided LSTM for a reduced Ra-226(n,2n)Ra-225 to Ac-225 chain. The neutron pathway I am studying comes from high-energy deuterons on beryllium, which is why your work in radionuclide production, accelerator targetry, radiochemistry, and nuclear data stood out to me.",
            fontSize=8.9,
            leading=11.2,
        ),
        Spacer(1, 8),
        line_label("The model only sees one layer of the real problem", accent),
        Spacer(1, 6),
    ]
    layers = [
        ("Neutron source and transport", "Partly represented", "A two-group spectrum cannot preserve the full accelerator field."),
        ("Target geometry and heat", "Not represented", "No beam spot, power density, cooling, target damage, or shielding geometry."),
        ("Isotope buildup and decay", "Represented", "A reduced five-nuclide constant-rate chain checked against Radau."),
        ("Separation and recovery", "Not represented", "No chemistry time, recovery fraction, purification loss, or turnaround."),
    ]
    layer_table = Table(
        [
            [
                p("PHYSICAL LAYER", fontName="Helvetica-Bold", fontSize=7, textColor=PAPER),
                p("CURRENT STATUS", fontName="Helvetica-Bold", fontSize=7, textColor=PAPER),
                p("WHY IT MATTERS", fontName="Helvetica-Bold", fontSize=7, textColor=PAPER),
            ]
        ]
        + [
            [
                p(a, fontName="Helvetica-Bold", fontSize=8, textColor=INK),
                p(b, fontName="Helvetica-Bold", fontSize=7.5, textColor=GREEN if b == "Represented" else accent),
                p(c, fontSize=7.6, leading=9.2),
            ]
            for a, b, c in layers
        ],
        colWidths=[145, 110, CONTENT_W - 255],
    )
    layer_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), accent),
                ("GRID", (0, 0), (-1, -1), 0.5, LINE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PAPER, SOFT]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story += [
        layer_table,
        Spacer(1, 9),
        box(
            p(
                "The Berkeley stress test exposed this problem. I pushed the detailed 33 and 40 MeV experimental conditions into my reduced two-group input, even though they were outside the training range, and V3 overpredicted the activity by about 600 times. I kept that failed result instead of retraining on it.",
                fontSize=8.8,
                leading=11.1,
            ),
            background=shade(RED, 0.94),
            border=shade(RED, 0.72),
            padding=8,
            left_bar=RED,
        ),
        Spacer(1, 9),
        Table(
            [[
                box(
                    [[p("What I can send", fontName="Helvetica-Bold", fontSize=9, textColor=BLUE)], [bullet_list([
                        "Frozen source, checkpoint, and declared input ranges.",
                        "The reduced reaction chain and solver checks.",
                        "The failed Berkeley reconstruction and schedule-ranking results.",
                    ], size=7.8)]],
                    background=shade(BLUE, 0.94),
                    border=shade(BLUE, 0.72),
                    padding=7,
                ),
                box(
                    [[p("What I need to learn", fontName="Helvetica-Bold", fontSize=9, textColor=accent)], [bullet_list([
                        "The beam, geometry, thermal, cooling, damage, and recovery variables that bound a feasible case.",
                        "A public targetry or radiochemistry benchmark with enough inputs and outputs to reproduce.",
                        "Which constraint would reject an attractive schedule first in real work.",
                    ], size=7.8)]],
                    background=shade(accent, 0.93),
                    border=shade(accent, 0.7),
                    padding=7,
                ),
            ]],
            colWidths=[CONTENT_W / 2 - 4, CONTENT_W / 2 - 4],
            style=[
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 4),
                ("LEFTPADDING", (1, 0), (1, 0), 4),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
            ],
        ),
        Spacer(1, 9),
        box(
            p(
                "<b>My question:</b> Could you recommend a public benchmark and the smallest set of physical feasibility limits I should add before calling this a production-planning model? Your answer would let me define a realistic domain, reject impossible schedules, and give the surrogate a real job beyond solving a small equation set.",
                fontSize=9,
                leading=11.3,
                textColor=INK,
            ),
            background=shade(accent, 0.93),
            border=shade(accent, 0.65),
            padding=9,
            left_bar=accent,
        ),
    ]
    return story


def severin_story() -> list:
    accent = GREEN
    story = document_header(
        "Separation and recovery",
        "I can predict what is produced. I cannot yet predict what becomes usable.",
        "Professor Gregory W. Severin",
        "Michigan State University and FRIB Isotope Harvesting Group",
        accent,
        subtitle="The gap between those two numbers may matter more than another small accuracy improvement.",
    )
    story += [
        p(
            "I am a 12th grader at Glen Burnie High School, and I built a reduced model of the Ra-226 to Ac-225 production pathway. My equations follow irradiation, isotope buildup, decay, and cooling. They stop at the exact point where isotope harvesting and chemistry begin, which is why I am reaching out to your group.",
            fontSize=8.9,
            leading=11.2,
        ),
        Spacer(1, 8),
        line_label("The result that changed how I see the project", accent),
        Spacer(1, 5),
        p(
            "When I reconstructed the published Berkeley experiments, I had to keep produced activity and recovered activity as different quantities. The recovery was not identical between the two experiments. That made something obvious: a schedule that produces the most atoms is not automatically the schedule that delivers the most usable Ac-225, or does it fastest.",
            fontSize=8.9,
            leading=11.2,
        ),
        Spacer(1, 10),
    ]
    flow = Table(
        [[
            box([[p("Produced in target", fontName="Helvetica-Bold", fontSize=9, textColor=BLUE, alignment=TA_CENTER)], [p("What my model predicts", fontSize=7.5, alignment=TA_CENTER)]], background=shade(BLUE, 0.94), border=shade(BLUE, 0.7), padding=7),
            p("->", fontName="Helvetica-Bold", fontSize=15, textColor=MUTED, alignment=TA_CENTER),
            box([[p("Separate and purify", fontName="Helvetica-Bold", fontSize=9, textColor=GOLD, alignment=TA_CENTER)], [p("Time and losses unknown", fontSize=7.5, alignment=TA_CENTER)]], background=shade(GOLD, 0.93), border=shade(GOLD, 0.7), padding=7),
            p("->", fontName="Helvetica-Bold", fontSize=15, textColor=MUTED, alignment=TA_CENTER),
            box([[p("Quality checked", fontName="Helvetica-Bold", fontSize=9, textColor=accent, alignment=TA_CENTER)], [p("The usable result I need", fontSize=7.5, alignment=TA_CENTER)]], background=shade(accent, 0.93), border=shade(accent, 0.65), padding=7),
        ]],
        colWidths=[150, 26, 150, 26, 150],
    )
    flow.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story += [
        flow,
        Spacer(1, 10),
        Table(
            [[
                box(
                    [[p("The practical numbers I am missing", fontName="Helvetica-Bold", fontSize=9.2, textColor=GOLD)], [bullet_list([
                        "Time from end of irradiation to completed separation.",
                        "Recovery fraction and how it should be defined.",
                        "Losses during purification, transfers, and quality checks.",
                        "The point when material is considered usable rather than merely produced.",
                    ], size=7.8)]],
                    background=shade(GOLD, 0.93),
                    border=shade(GOLD, 0.7),
                    padding=8,
                ),
                box(
                    [[p("What I can do without retraining V3", fontName="Helvetica-Bold", fontSize=9.2, textColor=accent)], [bullet_list([
                        "Add a separate chemistry and recovery layer after the frozen physics model.",
                        "Run sensitivity ranges instead of pretending one facility value is universal.",
                        "Rank usable activity and total turnaround, then verify the nuclear part with the trusted solver.",
                        "Clearly label every value as published, estimated, or still unknown.",
                    ], size=7.8)]],
                    background=shade(accent, 0.93),
                    border=shade(accent, 0.68),
                    padding=8,
                ),
            ]],
            colWidths=[CONTENT_W / 2 - 4, CONTENT_W / 2 - 4],
            style=[
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 4),
                ("LEFTPADDING", (1, 0), (1, 0), 4),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
            ],
        ),
        Spacer(1, 10),
        box(
            p(
                "<b>What I am asking:</b> Could you recommend one public isotope-harvesting or radiochemistry process example that reports separation time, recovery, purification losses, and quality-control steps? If a person in your group works more directly with those measurements, I would be grateful for the right name.",
                fontSize=9,
                leading=11.3,
                textColor=INK,
            ),
            background=shade(accent, 0.93),
            border=shade(accent, 0.62),
            padding=9,
            left_bar=accent,
        ),
    ]
    return story


def zimmerman_story() -> list:
    accent = CYAN
    story = document_header(
        "Activity and measurement timing",
        "I do not want a timestamp mistake to look like model error.",
        "Dr. Brian E. Zimmerman",
        "NIST Radioactivity Group",
        accent,
        subtitle="My model predicts atoms. The experiments report activity measured after several real steps.",
    )
    story += [
        p(
            "My name is Samuel Ogunnubi, and I am a 12th grader in Maryland working on a reduced Ac-225 production model. I found NIST's Ac-225 standardization work while trying to compare my calculations with published experiments. That comparison looks simple at first because activity and atoms are connected by A = lambda N. The hard part is knowing which N and which time the reported activity actually represents.",
            fontSize=8.9,
            leading=11.2,
        ),
        Spacer(1, 9),
        line_label("The four clocks I need to keep separate", accent),
        Spacer(1, 6),
    ]
    clock_data = [
        ("1", "End of bombardment", "Production stops and decay continues."),
        ("2", "Chemical separation", "Ra, Ac, and daughters may no longer share one history."),
        ("3", "Assay time", "The instrument measures activity under stated conditions."),
        ("4", "Reported reference time", "The published value may be decay-corrected to another time."),
    ]
    clock_cells = []
    for number, name, meaning in clock_data:
        clock_cells.append(
            box(
                [[p(number, fontName="Helvetica-Bold", fontSize=11, textColor=accent)], [p(name, fontName="Helvetica-Bold", fontSize=7.8, leading=9, textColor=INK)], [p(meaning, fontSize=6.8, leading=8.2, textColor=MUTED)]],
                background=shade(accent, 0.95),
                border=shade(accent, 0.7),
                padding=6,
            )
        )
    clocks = Table([clock_cells], colWidths=[CONTENT_W / 4] * 4)
    clocks.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story += [
        clocks,
        Spacer(1, 9),
        box(
            [[
                p("A = lambda N", fontName="Helvetica-Bold", fontSize=16, textColor=accent, alignment=TA_CENTER),
                p(
                    "The equation is not my main uncertainty. I need to know which decay constant and nuclear-data source to cite, whether progeny are in equilibrium, whether activity is for Ac-225 alone or a measurement window containing daughters, and how timing and uncertainty are carried through the conversion.",
                    fontSize=8.7,
                    leading=10.9,
                ),
            ]],
            background=shade(accent, 0.94),
            border=shade(accent, 0.68),
            padding=9,
            widths=[120, CONTENT_W - 120],
        ),
        Spacer(1, 10),
        Table(
            [[
                box(
                    [[p("What I have already prepared", fontName="Helvetica-Bold", fontSize=9.2, textColor=BLUE)], [bullet_list([
                        "Predicted atom inventories from the frozen reduced decay chain.",
                        "Published produced and recovered activities with reported uncertainties.",
                        "A timeline table for the Berkeley irradiation, separation, assay, and reference times.",
                    ], size=7.9)]],
                    background=shade(BLUE, 0.94),
                    border=shade(BLUE, 0.72),
                    padding=8,
                ),
                box(
                    [[p("What I need to get right", fontName="Helvetica-Bold", fontSize=9.2, textColor=accent)], [bullet_list([
                        "The recommended decay data and activity-to-atoms procedure.",
                        "How to state progeny equilibrium and the exact activity reference time.",
                        "How measurement and nuclear-data uncertainty should appear beside model error.",
                    ], size=7.9)]],
                    background=shade(accent, 0.94),
                    border=shade(accent, 0.68),
                    padding=8,
                ),
            ]],
            colWidths=[CONTENT_W / 2 - 4, CONTENT_W / 2 - 4],
            style=[
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 4),
                ("LEFTPADDING", (1, 0), (1, 0), 4),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
            ],
        ),
        Spacer(1, 9),
        box(
            p(
                "<b>My question:</b> Could you recommend the best public method or reference for making this comparison correctly? I would use it to rebuild the external error calculation, separate measurement uncertainty from model error, and avoid claiming that an apparent mismatch came from V3 when it may have come from my timing or activity definition.",
                fontSize=9,
                leading=11.3,
                textColor=INK,
            ),
            background=shade(accent, 0.93),
            border=shade(accent, 0.62),
            padding=9,
            left_bar=accent,
        ),
        Spacer(1, 6),
        p(
            "I would cite any guidance as a measurement reference only. I would not describe it as NIST validation of the production model.",
            fontSize=7.6,
            leading=9.2,
            textColor=MUTED,
        ),
    ]
    return story


BRIEFS = [
    {
        "number": 1,
        "slug": "Lee_Bernstein",
        "name": "Professor Lee A. Bernstein",
        "short": "Berkeley reconstruction",
        "role": "UC Berkeley and Lawrence Berkeley National Laboratory",
        "email": "labernstein@berkeley.edu",
        "accent": "#a92532",
        "title": "I tested V3 on the Berkeley cases. It failed, and I want to understand why.",
        "summary": "An honest follow-up about the two outside experiments, the 600x miss, and the exact experimental definitions still missing.",
        "story": lee_story,
        "subject": "One follow-up after I tested the Berkeley cases",
        "salutation": "Dear Professor Bernstein,",
        "email_paragraphs": [
            "Thank you again for sending the spectrum code and papers. I finally used them to run the first real outside test of my frozen model.",
            "I want to be honest about the result: V3 was about 600 times too high on both Berkeley cases. I am not changing the model to make those points look better. I am trying to find out whether the failure comes from the model's limited range, my reconstruction, or both.",
            "The details I still cannot pin down are the spectrum normalization at the radium target, the target geometry and position, the full timing, the meaning of produced versus recovered activity, and any numerical Ac-227 detection limit. Could you point me to where those are defined, or to the person who would know?",
            "I made the attached page just for this follow-up so you can see exactly what I tried. Thank you again for getting me this far.",
        ],
    },
    {
        "number": 2,
        "slug": "Jaden_Palmer",
        "name": "Jaden Palmer",
        "short": "V3 claim decision",
        "role": "ARTISANS Lab, North Carolina State University",
        "email": "jspalme2@ncsu.edu",
        "accent": "#006c70",
        "title": "Jaden, I need your honest read on what V3 actually proves.",
        "summary": "A mentor note showing the good results, ugly results, exact-solver problem, and the claim decisions still open.",
        "story": jaden_story,
        "subject": "Can I get your honest read on my final V3 claim?",
        "salutation": "Hi Jaden,",
        "email_paragraphs": [
            "Your advice from our last call honestly changed the project, so thank you again. I finished V3, froze it, and tested it much harder.",
            "The matched result got below 1%, but the wider grid had a huge error tail, the Berkeley test failed outside the training range, and I found that an exact solver is faster for my small equation set. I kept all of that instead of only showing the best number.",
            "Could we do a short Zoom so I can show you the full scorecard? I mainly need your honest opinion on what the final research question should be, which result should be the headline, and what I can still finish before December without another expensive training run.",
            "The attached page is the exact decision I need help making.",
        ],
    },
    {
        "number": 3,
        "slug": "John_Brockman",
        "name": "Dr. John Brockman",
        "short": "Production workflow",
        "role": "University of Missouri Research Reactor",
        "email": "brockmanjd@missouri.edu",
        "accent": "#285f8e",
        "title": "My model ends at cooling. Real isotope production does not.",
        "summary": "A production-timeline question about the chemistry, quality, recovery, and turnaround steps the model currently misses.",
        "story": brockman_story,
        "subject": "What am I missing from a real isotope-production timeline?",
        "salutation": "Dear Dr. Brockman,",
        "email_paragraphs": [
            "My name is Sam Ogunnubi, and I am a 12th grader in Glen Burnie High School's Biomedical and Allied Health program in Maryland. I spent my summer building a model around one Ac-225 production pathway because I wanted to see whether production plans could be compared faster.",
            "I realized my model stops too early. It covers irradiation, isotope buildup, and cooling, but not separation, purification, quality checks, recovery losses, target reuse, or delivery. Those steps may control the real timeline more than the part I modeled.",
            "Could you recommend one public example that follows an isotope from irradiation to a usable product, or point me to the MURR person who knows that workflow best?",
            "I put the exact gap and how I would use the answer on the attached page. Thank you for your time.",
        ],
    },
    {
        "number": 4,
        "slug": "Robert_Hobbs",
        "name": "Dr. Robert F. Hobbs",
        "short": "Medical reporting",
        "role": "Johns Hopkins Radiation Oncology and Molecular Radiation Sciences",
        "email": "rhobbs3@jhmi.edu",
        "accent": "#6b4f86",
        "title": "I can calculate Ac-225 activity. I need help deciding what makes that output scientifically useful.",
        "summary": "A focused request about reference time, impurity, uncertainty, and keeping production claims separate from clinical claims.",
        "story": hobbs_story,
        "subject": "Question about making my Ac-225 model output scientifically useful",
        "salutation": "Dear Dr. Hobbs,",
        "email_paragraphs": [
            "My name is Sam Ogunnubi, and I am a 12th grader in Glen Burnie High School's Biomedical and Allied Health program. I found your work in alpha-particle therapy and radiopharmaceutical dosimetry while trying to decide what my Ac-225 production model should report.",
            "The code can report Ac-225 and trace Ac-227 activity, but I do not want to guess what makes those numbers medically meaningful or accidentally turn a production result into a clinical claim.",
            "Could you recommend one or two public references for reporting Ac-225 activity, reference time, radionuclidic impurity, daughter equilibrium, and uncertainty?",
            "I made a one-page note showing exactly what the model gives me, what I am unsure about, and how I would use your answer. Thank you for any direction you can give me.",
        ],
    },
    {
        "number": 5,
        "slug": "Jonathan_Engle",
        "name": "Professor Jonathan W. Engle",
        "short": "Targetry limits",
        "role": "University of Wisconsin-Madison Cyclotron Laboratory",
        "email": "jwengle@wisc.edu",
        "accent": "#9b660d",
        "title": "What would make one of my Ac-225 production plans physically possible?",
        "summary": "A targetry question about geometry, heat, transport, recovery, and the benchmark needed to reject impossible schedules.",
        "story": engle_story,
        "subject": "What physical limits should my Ac-225 model include?",
        "salutation": "Dear Professor Engle,",
        "email_paragraphs": [
            "My name is Sam Ogunnubi, and I am a 12th grader in Maryland. I built a physics-guided model for a reduced Ra-226 to Ac-225 pathway using high-energy deuteron-on-beryllium neutron conditions.",
            "The model can rank irradiation and cooling schedules, but it does not know whether a target could actually handle the beam, heat, geometry, cooling, or chemistry. My Berkeley outside test made that weakness very clear.",
            "Could you recommend a public targetry or radiochemistry benchmark and the smallest set of physical limits I should add before calling this a realistic screening problem?",
            "The attached page shows what is modeled, what is missing, and how your answer would change the project. Thank you for reading it.",
        ],
    },
    {
        "number": 6,
        "slug": "Gregory_Severin",
        "name": "Professor Gregory W. Severin",
        "short": "Recovery layer",
        "role": "Michigan State University and FRIB Isotope Harvesting Group",
        "email": "severin9@msu.edu",
        "accent": "#26734d",
        "title": "I can predict what is produced. I cannot yet predict what becomes usable.",
        "summary": "A recovery question about separation time, purification losses, quality checks, and adding a post-processing layer without retraining.",
        "story": severin_story,
        "subject": "How should I model the gap between produced and usable Ac-225?",
        "salutation": "Dear Professor Severin,",
        "email_paragraphs": [
            "My name is Sam Ogunnubi, and I am a 12th grader at Glen Burnie High School. I built a reduced model of the Ra-226 to Ac-225 pathway, but it currently stops before separation and recovery.",
            "The Berkeley experiment showed me why that matters: produced activity and recovered activity were not the same, and the recovery changed between experiments. A schedule can look great in my model and still be a bad real process.",
            "Could you recommend one public isotope-harvesting or radiochemistry example that reports separation time, recovery, purification losses, and quality checks? If someone in your group is a better fit for the question, I would really appreciate the right name.",
            "I made the attached page so the problem is clear without turning this email into an essay. Thank you for your time.",
        ],
    },
    {
        "number": 7,
        "slug": "Brian_Zimmerman",
        "name": "Dr. Brian E. Zimmerman",
        "short": "Activity timing",
        "role": "NIST Radioactivity Group",
        "email": "brian.zimmerman@nist.gov",
        "accent": "#287d8b",
        "title": "I do not want a timestamp mistake to look like model error.",
        "summary": "A measurement question about atoms-to-activity conversion, decay data, progeny, reference times, and uncertainty.",
        "story": zimmerman_story,
        "subject": "Question about comparing predicted atoms with measured Ac-225 activity",
        "salutation": "Dear Dr. Zimmerman,",
        "email_paragraphs": [
            "My name is Sam Ogunnubi, and I am a 12th grader in Maryland comparing a reduced Ac-225 production model with published measurements.",
            "My model predicts atoms, but the experiments report activity after irradiation, separation, assay, and sometimes decay correction to another reference time. I do not want a timing or progeny mistake to look like model error.",
            "Could you recommend the best public method for converting Ac-225 activity to atoms and reporting the decay data, daughter equilibrium, reference time, and uncertainty correctly?",
            "The attached page shows the four times I am trying to keep separate. Thank you for any reference or correction you can share.",
        ],
    },
]


HTML_BASE_CSS = """
:root{--ink:#17232d;--text:#354653;--muted:#687783;--line:#cbd6dc;--paper:#fff;--canvas:#e8eef1;--accent:#285f8e;--soft:#f4f7f8}*{box-sizing:border-box}html,body{margin:0;max-width:100%;color:var(--ink);background:var(--canvas);font-family:Arial,Helvetica,sans-serif;font-size:14px;line-height:1.48;letter-spacing:0}a{color:#285f8e;text-underline-offset:2px}.actions{width:min(8.5in,calc(100% - 20px));margin:14px auto 8px;display:flex;flex-wrap:wrap;justify-content:flex-end;gap:7px}.action{min-height:34px;display:inline-flex;align-items:center;justify-content:center;padding:7px 11px;border:1px solid #aebdc6;border-radius:4px;color:var(--ink);background:var(--paper);font-weight:800;text-decoration:none}.sheet{position:relative;width:min(8.5in,calc(100% - 20px));min-height:11in;margin:0 auto 24px;padding:.43in .47in .38in;overflow:hidden;background:var(--paper);box-shadow:0 8px 28px rgba(24,42,55,.14)}.sheet:before{content:"";position:absolute;inset:0 0 auto;height:7px;background:var(--accent)}header{display:grid;grid-template-columns:minmax(0,1fr)2.25in;gap:.25in;align-items:start;margin-bottom:16px;padding-bottom:13px;border-bottom:1px solid var(--line)}.kicker{margin:0 0 5px;color:var(--accent);font-size:10px;font-weight:900;text-transform:uppercase}.sheet h1{margin:0 0 6px;font-size:26px;line-height:1.08}.subtitle{margin:0;color:var(--muted);font-size:12px}.recipient{padding:10px;border:1px solid color-mix(in srgb,var(--accent) 28%,white);border-left:4px solid var(--accent);background:color-mix(in srgb,var(--accent) 7%,white);font-size:10px;line-height:1.35}.recipient b{display:block;margin:2px 0;color:var(--ink);font-size:14px}.recipient small{display:block;color:var(--accent);font-size:9px;font-weight:900;text-transform:uppercase}.letter p{margin:0 0 10px;color:var(--text)}.letter strong{color:var(--ink)}.label{display:flex;align-items:center;gap:8px;margin:15px 0 7px;color:var(--accent);font-size:10px;font-weight:900;text-transform:uppercase}.label:after{content:"";height:1px;flex:1;background:var(--line)}.note{padding:12px 14px;border:1px solid var(--line);border-left:5px solid var(--accent);background:color-mix(in srgb,var(--accent) 7%,white)}.note p:last-child{margin-bottom:0}.split{display:grid;grid-template-columns:1fr 1fr;gap:10px}.panel{padding:11px 12px;border:1px solid var(--line);background:var(--soft)}.panel h2{margin:0 0 6px;color:var(--accent);font-size:13px}.panel p:last-child{margin-bottom:0}.panel ul{margin:0;padding-left:18px;color:var(--text);font-size:12px}.panel li{margin:4px 0}.result-strip{display:grid;grid-template-columns:repeat(2,1fr);gap:1px;margin:12px 0;border:1px solid var(--line);background:var(--line)}.result-strip div{padding:10px 12px;background:color-mix(in srgb,var(--accent) 7%,white)}.result-strip b{display:block;color:var(--accent);font-size:18px}.score{width:100%;border-collapse:collapse;font-size:11px}.score th{padding:7px;color:#fff;background:var(--accent);text-align:left}.score td{padding:7px;border:1px solid var(--line);vertical-align:top}.score tr:nth-child(odd) td{background:var(--soft)}.process{display:grid;grid-template-columns:repeat(7,1fr);gap:5px}.process div{min-height:72px;padding:8px 5px;border:1px solid var(--line);background:var(--soft);text-align:center}.process b{display:block;margin-bottom:3px;color:var(--accent);font-size:16px}.process span{display:block;font-size:9px;font-weight:800}.process small{display:block;margin-top:4px;color:var(--muted);font-size:8px}.layers{display:grid;grid-template-columns:1.1fr .8fr 1.8fr;border:1px solid var(--line)}.layers div{padding:8px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);font-size:11px}.layers div:nth-child(3n){border-right:0}.layers b{color:var(--accent)}.flow{display:grid;grid-template-columns:1fr 30px 1fr 30px 1fr;align-items:stretch;gap:5px}.flow .stage{display:flex;min-height:84px;flex-direction:column;justify-content:center;padding:9px;border:1px solid var(--line);background:var(--soft);text-align:center}.flow .arrow{display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:22px}.flow b{color:var(--accent)}.clocks{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.clock{padding:9px;border:1px solid var(--line);background:color-mix(in srgb,var(--accent) 7%,white)}.clock b{display:block;color:var(--accent);font-size:17px}.clock strong{display:block;font-size:11px}.clock small{display:block;margin-top:4px;color:var(--muted);font-size:9px}.equation{display:grid;grid-template-columns:130px 1fr;gap:12px;align-items:center;padding:13px;border:1px solid var(--line);background:color-mix(in srgb,var(--accent) 7%,white)}.equation b{color:var(--accent);font-size:22px;text-align:center}footer{margin-top:17px;padding-top:8px;border-top:1px solid var(--line);color:var(--muted);font-size:9px}@media(max-width:700px){.actions{flex-direction:column}.action{width:100%}.sheet{min-height:auto;padding:30px 20px}.sheet h1{font-size:23px}header,.split,.result-strip,.process,.layers,.flow,.clocks,.equation{grid-template-columns:1fr}.flow .arrow{min-height:28px;transform:rotate(90deg)}.layers div{border-right:0}.recipient{margin-top:3px}}@page{size:Letter;margin:0}@media print{html,body{width:8.5in;min-height:11in;background:#fff}.actions{display:none}.sheet{width:8.5in;height:11in;min-height:11in;margin:0;padding:.43in .47in .38in;box-shadow:none;print-color-adjust:exact;-webkit-print-color-adjust:exact}header{grid-template-columns:minmax(0,1fr)2.25in}.split{grid-template-columns:1fr 1fr}.result-strip{grid-template-columns:repeat(2,1fr)}.process{grid-template-columns:repeat(7,1fr)}.layers{grid-template-columns:1.1fr .8fr 1.8fr}.flow{grid-template-columns:1fr 30px 1fr 30px 1fr}.clocks{grid-template-columns:repeat(4,1fr)}.equation{grid-template-columns:130px 1fr}}
"""


def html_shell(brief: dict, body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="A personal technical research note from Samuel Ogunnubi to {escape(brief['name'])}."><title>{escape(brief['short'])} | Samuel Ogunnubi</title><style>{HTML_BASE_CSS}</style></head>
<body style="--accent:{brief['accent']}"><nav class="actions"><a class="action" href="index.html">All personal notes</a><a class="action" href="../Ac225_Expert_Email_Templates_2026.html">Matching email</a><button class="action" type="button" onclick="window.print()">Print / PDF</button></nav><main class="sheet letter">{body}<footer>Samuel Ogunnubi | Grade 12 | Glen Burnie High School | Maryland</footer></main></body></html>"""


def html_header(brief: dict, kicker: str, subtitle: str) -> str:
    return f"""<header><div><p class="kicker">{escape(kicker)}</p><h1>{escape(brief['title'])}</h1><p class="subtitle">{escape(subtitle)}</p></div><aside class="recipient"><small>Personal note for</small><b>{escape(brief['name'])}</b>{escape(brief['role'])}</aside></header>"""


def html_list(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"


def lee_html(b: dict) -> str:
    return html_shell(b, html_header(b, "Berkeley experiment follow-up", "This is the first real outside test of my frozen Ra-226 to Ac-225 model.") + """
<p><strong>Professor Bernstein,</strong></p><p>The neutron-spectrum code and papers you sent gave me something I had been missing for months: a real experiment close to the pathway I modeled. I kept my seed-42 checkpoint frozen, rebuilt the 33 and 40 MeV cases as carefully as I could, and ran the comparison without changing the model afterward.</p>
<div class="result-strip"><div><b>about 623x high</b>33 MeV raw V3 activity</div><div><b>about 600x high</b>40 MeV raw V3 activity</div></div>
<p class="label">Where I got stuck</p><p>I do not think it would be honest to tune V3 until those points look good. Before I call this only a model failure, I need to know whether I reconstructed the experiment correctly. The dissertation reports recovered Ac-225 activity, while my equations predict atoms produced in the target. Those are not automatically the same quantity.</p>
<div class="split"><section class="panel"><h2>What I can already document</h2>""" + html_list([
        "The published 33 and 40 MeV spectrum reconstruction.",
        "A 1 mg Ra-226 target and the reported Ac-225 activities.",
        "My frozen source, inputs, checkpoint, and failed predictions.",
    ]) + """</section><section class="panel"><h2>What I still cannot pin down</h2>""" + html_list([
        "Spectrum normalization or fluence at the radium target.",
        "Target geometry, position, irradiation, separation, assay, and reference times.",
        "Produced versus recovered activity definitions and any Ac-227 detection limit.",
    ]) + """</section></div>
<div class="note"><p><strong>What I am asking:</strong> Could you point me to where those details are defined, correct anything I misunderstood, or tell me who would know? Even a short reference would let me rebuild the comparison and report the failure for the right scientific reason.</p></div>
<p><small>If it helps, I can send the exact inputs, reconstruction table, code, checkpoint, and failed outputs. I am not asking for private facility data or for Berkeley to validate my model.</small></p>""")


def jaden_html(b: dict) -> str:
    rows = [
        ("0.97%", "Matched synthetic test", "Strong median result."),
        ("3.94% / 183.84%", "Wider grid median / p95", "The tail is still a major problem."),
        ("Rank 2", "Schedule screening", "0.0470% objective regret after solver verification."),
        ("Mixed", "LSTM versus matched MLP", "I cannot claim the LSTM always wins."),
        ("Failed", "Berkeley outside test", "About 600x high outside training support."),
        ("Slower", "Exact analytic baseline", "V3 is not the fastest solver for this small system."),
    ]
    table = "<table class='score'><thead><tr><th>Result</th><th>Test</th><th>What it means</th></tr></thead><tbody>" + "".join(f"<tr><td><strong>{escape(a)}</strong></td><td>{escape(c)}</td><td>{escape(d)}</td></tr>" for a, c, d in rows) + "</tbody></table>"
    return html_shell(b, html_header(b, "Mentor decision note", "The model is trained and frozen. My problem now is deciding what claim is fair.") + """
<p>Your advice during our last call honestly changed the project. I finished the LSTM run, froze it before outside testing, and then tested it much harder than I originally planned. Some results are strong. Some are uncomfortable. I want the final project to show both.</p>
<p class="label">The scorecard I would show a judge</p>""" + table + """
<div class="split"><section class="panel"><h2>What I think I can defend</h2>""" + html_list([
        "A frozen surrogate can be studied for solver-verified screening inside a declared reduced domain.",
        "Median ranking behavior may still be useful when every finalist is checked.",
        "The warning system and failure map are part of the result.",
    ]) + """</section><section class="panel"><h2>What I am still unsure about</h2>""" + html_list([
        "Which error statistic should be the headline.",
        "How to define uncertainty and the out-of-range warning.",
        "Whether the model has a useful job when the exact reduced solver is faster.",
    ]) + """</section></div>
<div class="note"><p><strong>The decision I need help making:</strong> If you were judging this project, what research question and claim would you accept from these results? I also need your opinion on the most valuable work I can finish before December without another full training run.</p></div>""")


def brockman_html(b: dict) -> str:
    stages = [("1", "Irradiate", "Modeled"), ("2", "Build and decay", "Modeled"), ("3", "Cool", "Modeled"), ("4", "Separate", "Missing"), ("5", "Purify", "Missing"), ("6", "Quality check", "Missing"), ("7", "Recycle or deliver", "Missing")]
    process = "<div class='process'>" + "".join(f"<div><b>{n}</b><span>{escape(name)}</span><small>{state}</small></div>" for n, name, state in stages) + "</div>"
    return html_shell(b, html_header(b, "Production workflow question", "I need a public timeline so I stop optimizing only the easy half.") + """
<p>My name is Samuel Ogunnubi, and I am a 12th grader in Glen Burnie High School's Biomedical and Allied Health program in Maryland. I started this project after learning what Ac-225 can do in targeted cancer treatment. My goal was to see whether a computer model could help compare production plans faster.</p>
<p>I built a physics-guided LSTM around irradiation, isotope buildup and decay, and cooling. Then I found that an exact solver is already faster for my small constant-rate equation set. If the project is going to matter, it has to study the real end-to-end timeline and the constraints that actually slow a facility down.</p>
<p class="label">Where my model stops</p>""" + process + """
<div class="split"><section class="panel"><h2>What I already have</h2>""" + html_list([
        "A frozen model for irradiation, decay, and cooling.",
        "A 10,000-case study with solver-verified finalists.",
        "Produced and recovered activity separated in the Berkeley analysis.",
    ]) + """</section><section class="panel"><h2>What I need from real operations</h2>""" + html_list([
        "Public timing ranges after cooling.",
        "Recovery losses, target reuse, and quality constraints.",
        "One process example I can cite and reproduce.",
    ]) + """</section></div>
<div class="note"><p><strong>My question:</strong> Could you recommend a public example that follows an isotope from irradiation through a usable, quality-checked product, or point me to the MURR person who knows that workflow best? I would use it to compare total time and usable activity instead of only predicted atoms.</p></div>""")


def hobbs_html(b: dict) -> str:
    return html_shell(b, html_header(b, "Medical meaning and reporting", "I want the output to be useful without turning it into a clinical claim.") + """
<p>I am a 12th grader at Glen Burnie High School in Maryland, and I built a reduced model of one Ra-226 to Ac-225 pathway after learning about targeted alpha therapy. Your work in alpha-particle therapy, radiopharmaceutical dosimetry, modeling, and treatment planning is a close match to the question I am trying to answer. My concern is that a mathematically correct value can still be misleading if I report it at the wrong time, use the wrong impurity definition, or connect it to treatment in a way the project has not earned.</p>
<div class="split"><section class="panel"><h2>What the model gives me</h2>""" + html_list([
        "Ac-225 atoms and activity during irradiation and cooling.",
        "Trace Ac-227 from the reduced side pathway.",
        "A candidate schedule and synthetic error statistics.",
    ]) + """</section><section class="panel"><h2>What the numbers do not tell me</h2>""" + html_list([
        "The right reference time for a fair product comparison.",
        "The correct basis for radionuclidic impurity.",
        "How daughter products and uncertainty should be stated.",
        "What is medically useful without implying clinical validation.",
    ]) + """</section></div>
<p class="label">The line I am trying not to cross</p><p>I found an IAEA example using a 2% Ac-227/Ac-225 activity specification, but I am not treating it as a universal medical or regulatory limit. The project has no patient data, dose model, or clinical validation.</p>
<div class="note"><p><strong>What I would value from you:</strong> Could you recommend one or two public references showing how an Ac-225 production or product-quality study should report activity, reference time, radionuclidic impurity, daughter equilibrium, and uncertainty? I would use them to define the output before ranking schedules and keep the final claim strictly on the production side.</p></div>""")


def engle_html(b: dict) -> str:
    rows = [
        ("Neutron source and transport", "Partly represented", "A two-group spectrum loses detail."),
        ("Target geometry and heat", "Not represented", "No beam spot, cooling, damage, or shielding geometry."),
        ("Isotope buildup and decay", "Represented", "Reduced chain checked against Radau."),
        ("Separation and recovery", "Not represented", "No chemistry time, recovery, or turnaround."),
    ]
    layers = "<div class='layers'>" + "".join(f"<div><strong>{escape(a)}</strong></div><div><b>{escape(c)}</b></div><div>{escape(d)}</div>" for a, c, d in rows) + "</div>"
    return html_shell(b, html_header(b, "Targetry and feasibility", "Right now my equations can rank a schedule that a real target could never survive.") + """
<p>My name is Samuel Ogunnubi, and I am a 12th grader in Maryland. I built a physics-guided LSTM for a reduced Ra-226(n,2n)Ra-225 to Ac-225 chain. The neutron pathway I am studying comes from high-energy deuterons on beryllium, which is why your work in radionuclide production, accelerator targetry, radiochemistry, and nuclear data stood out to me.</p>
<p class="label">The model only sees one layer of the real problem</p>""" + layers + """
<div class="note"><p>The Berkeley outside test exposed this weakness. I pushed the 33 and 40 MeV experimental conditions into my reduced input, even though they were outside the training range, and V3 overpredicted activity by about 600 times. I kept that failed result instead of retraining on it.</p></div>
<div class="split"><section class="panel"><h2>What I can send</h2>""" + html_list([
        "Frozen source, checkpoint, and input ranges.",
        "The reduced chain and solver checks.",
        "The failed Berkeley reconstruction.",
    ]) + """</section><section class="panel"><h2>What I need to learn</h2>""" + html_list([
        "The smallest set of beam, geometry, thermal, damage, and recovery limits.",
        "A public targetry or radiochemistry benchmark.",
        "Which real constraint would reject a schedule first.",
    ]) + """</section></div>
<p><strong>My question:</strong> Could you recommend a public benchmark and the feasibility limits I should add before calling this a production-planning model? Your answer would let me reject impossible schedules and give the surrogate a real job beyond solving a small equation set.</p>""")


def severin_html(b: dict) -> str:
    flow = """<div class="flow"><div class="stage"><b>Produced in target</b><small>What my model predicts</small></div><div class="arrow">-&gt;</div><div class="stage"><b>Separate and purify</b><small>Time and losses unknown</small></div><div class="arrow">-&gt;</div><div class="stage"><b>Quality checked</b><small>The usable result I need</small></div></div>"""
    return html_shell(b, html_header(b, "Separation and recovery", "The gap between produced and usable may matter more than another small accuracy gain.") + """
<p>I am a 12th grader at Glen Burnie High School, and I built a reduced model of the Ra-226 to Ac-225 pathway. My equations follow irradiation, isotope buildup, decay, and cooling. They stop at the exact point where isotope harvesting and chemistry begin, which is why I am reaching out to your group.</p>
<p>When I reconstructed the Berkeley experiments, I had to keep produced activity and recovered activity as different numbers. The recovery was not identical between the experiments. A schedule that produces the most atoms is not automatically the one that delivers the most usable Ac-225.</p>""" + flow + """
<div class="split"><section class="panel"><h2>The practical numbers I am missing</h2>""" + html_list([
        "Separation and purification time.",
        "Recovery fraction and transfer losses.",
        "Quality-control steps and the definition of usable material.",
    ]) + """</section><section class="panel"><h2>What I can do without retraining</h2>""" + html_list([
        "Add a chemistry and recovery layer after frozen V3.",
        "Use public sensitivity ranges, not fake facility precision.",
        "Rank usable activity and total turnaround.",
    ]) + """</section></div>
<div class="note"><p><strong>What I am asking:</strong> Could you recommend one public isotope-harvesting or radiochemistry example that reports separation time, recovery, purification losses, and quality checks? If someone in your group handles those measurements more directly, I would be grateful for the right name.</p></div>""")


def zimmerman_html(b: dict) -> str:
    clocks = [("1", "End of bombardment", "Production stops."), ("2", "Chemical separation", "The isotope history changes."), ("3", "Assay time", "Activity is measured."), ("4", "Reference time", "The value may be decay-corrected.")]
    clock_html = "<div class='clocks'>" + "".join(f"<div class='clock'><b>{n}</b><strong>{escape(name)}</strong><small>{escape(note)}</small></div>" for n, name, note in clocks) + "</div>"
    return html_shell(b, html_header(b, "Activity and measurement timing", "My model predicts atoms. The experiments report activity after several real steps.") + """
<p>My name is Samuel Ogunnubi, and I am a 12th grader in Maryland comparing a reduced Ac-225 production model with published experiments. I found NIST's Ac-225 standardization work while trying to make that comparison correctly.</p>
<p>At first it looked simple because activity and atoms are connected by A = lambda N. The hard part is knowing which N and which time the reported activity represents.</p>
<p class="label">The four clocks I need to keep separate</p>""" + clock_html + """
<div class="equation"><b>A = lambda N</b><p>The equation is not my main uncertainty. I need the right decay data, progeny assumption, activity definition, measurement time, and uncertainty treatment.</p></div>
<div class="split"><section class="panel"><h2>What I have prepared</h2>""" + html_list([
        "Predicted atom inventories from the frozen chain.",
        "Published activities and reported uncertainties.",
        "A timeline for irradiation, separation, assay, and reference time.",
    ]) + """</section><section class="panel"><h2>What I need to get right</h2>""" + html_list([
        "The recommended decay data and conversion method.",
        "How to state progeny equilibrium and reference time.",
        "How uncertainty should appear beside model error.",
    ]) + """</section></div>
<div class="note"><p><strong>My question:</strong> Could you recommend the best public method for making this comparison correctly? I would use it to rebuild the external error calculation and avoid blaming the model for a mistake in my timing or activity definition.</p></div>
<p><small>I would cite the guidance as a measurement reference only, not as NIST validation of the production model.</small></p>""")


HTML_BUILDERS = {
    1: lee_html,
    2: jaden_html,
    3: brockman_html,
    4: hobbs_html,
    5: engle_html,
    6: severin_html,
    7: zimmerman_html,
}


def base_name(brief: dict) -> str:
    return f"{brief['number']:02d}_{brief['slug']}_Technical_Brief"


def index_html() -> str:
    rows = []
    for brief in BRIEFS:
        base = base_name(brief)
        rows.append(
            f"""<article style="--accent:{brief['accent']}"><p class="for">Personal note {brief['number']}</p><h2>{escape(brief['name'])}</h2><p class="title">{escape(brief['title'])}</p><p>{escape(brief['summary'])}</p><div><a href="{base}.html">Open HTML</a><a href="{base}.pdf">Open PDF</a></div></article>"""
        )
    return """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Samuel's Personal Expert Notes</title><style>
:root{--ink:#17232d;--text:#354653;--muted:#687783;--line:#cbd6dc;--paper:#fff;--canvas:#e8eef1}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--canvas);font-family:Arial,Helvetica,sans-serif;line-height:1.5;letter-spacing:0}.page{width:min(980px,calc(100% - 24px));margin:20px auto;padding:38px;background:var(--paper);box-shadow:0 8px 28px rgba(24,42,55,.13)}header{max-width:760px;margin-bottom:26px}h1{margin:0 0 8px;font-size:31px}.intro{color:var(--text)}.back{display:inline-block;margin-bottom:18px;color:#285f8e;font-weight:800}.list{border-top:1px solid var(--line)}article{display:grid;grid-template-columns:155px 1.2fr 1.5fr auto;gap:16px;align-items:center;padding:18px 4px;border-bottom:1px solid var(--line);border-left:5px solid var(--accent)}article>*:first-child{margin-left:12px}.for{color:var(--accent);font-size:10px;font-weight:900;text-transform:uppercase}article h2{margin:0;font-size:17px}.title{margin:0;color:var(--ink);font-weight:800}article p{margin:0;color:var(--text);font-size:12px}article div{display:flex;gap:6px}article a{white-space:nowrap;padding:6px 8px;border:1px solid var(--line);border-radius:3px;color:#285f8e;font-size:10px;font-weight:900;text-decoration:none}@media(max-width:760px){.page{padding:26px 18px}article{grid-template-columns:1fr;gap:7px}article>*:first-child{margin-left:10px}article div{margin-left:10px}}
</style></head><body><main class="page"><a class="back" href="../Ac225_Expert_Email_Templates_2026.html">Matching emails</a><header><h1>Seven different people. Seven different notes.</h1><p class="intro">These are not the same brief with different names. Each page starts from the exact part of my project that matches that person's work, explains what I have already done, asks one focused question, and says how the answer would change my project.</p></header><section class="list">""" + "".join(rows) + """</section></main></body></html>"""


EMAIL_CSS = """
:root{--ink:#17232d;--text:#354653;--muted:#687783;--line:#cbd6dc;--paper:#fff;--canvas:#e8eef1}*{box-sizing:border-box}html,body{margin:0;max-width:100%;color:var(--ink);background:var(--canvas);font-family:Arial,Helvetica,sans-serif;line-height:1.48;letter-spacing:0}.actions{width:min(980px,calc(100% - 20px));margin:16px auto 8px;display:flex;flex-wrap:wrap;justify-content:flex-end;gap:7px}.actions a{padding:8px 11px;border:1px solid #aebdc6;border-radius:4px;color:var(--ink);background:var(--paper);font-weight:800;text-decoration:none}.page{width:min(980px,calc(100% - 20px));margin:0 auto 28px;padding:38px;background:var(--paper);box-shadow:0 8px 28px rgba(24,42,55,.13)}header{max-width:780px;margin-bottom:24px}h1{margin:0 0 8px;font-size:31px}.intro{color:var(--text)}.email-entry{margin:0 0 22px;padding:0 0 22px;border-bottom:1px solid var(--line)}.email-head{display:grid;grid-template-columns:1fr auto;gap:12px;align-items:end;margin-bottom:9px}.email-head h2{margin:0;color:var(--accent);font-size:20px}.email-head p{margin:2px 0 0;color:var(--muted);font-size:11px}.email-head a{color:#285f8e;font-size:11px;font-weight:800}.letter{padding:15px 17px;border-left:5px solid var(--accent);background:#f4f7f8;color:var(--text);font-size:12px}.letter p{margin:0 0 9px}.letter p:last-child{margin-bottom:0}.tools{display:flex;flex-wrap:wrap;gap:7px;margin-top:8px}.tools a,.tools button{padding:6px 9px;border:1px solid var(--line);border-radius:3px;color:#285f8e;background:#fff;font:inherit;font-size:10px;font-weight:900;text-decoration:none;cursor:pointer}@media(max-width:650px){.actions{flex-direction:column}.actions a{width:100%;text-align:center}.page{padding:27px 19px}.email-head{grid-template-columns:1fr}.email-head a{overflow-wrap:anywhere}}
"""


def email_kit_html() -> str:
    entries = []
    for brief in BRIEFS:
        base = base_name(brief)
        paragraphs = "".join(f"<p>{escape(text)}</p>" for text in brief["email_paragraphs"])
        entries.append(
            f"""<section class="email-entry" style="--accent:{brief['accent']}"><div class="email-head"><div><h2>{escape(brief['name'])}</h2><p>{escape(brief['role'])}</p></div><a href="mailto:{escape(brief['email'])}">{escape(brief['email'])}</a></div><div class="letter" id="email-{brief['number']}"><p><strong>Subject: {escape(brief['subject'])}</strong></p><p>{escape(brief['salutation'])}</p>{paragraphs}<p>Best,<br>Sam Ogunnubi</p></div><div class="tools"><a href="Ac225_Personalized_Expert_Briefs_2026/{base}.html">Read their personal note</a><a href="Ac225_Personalized_Expert_Briefs_2026/{base}.pdf">Attach their PDF</a><button type="button" data-copy="email-{brief['number']}">Copy email</button></div></section>"""
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Samuel's Personal Expert Emails</title><style>{EMAIL_CSS}</style></head><body><nav class="actions"><a href="Ac225_Personalized_Expert_Briefs_2026/index.html">Open all personal notes</a><a href="Ac225_Project_Goals_2026.html">Project roadmap</a></nav><main class="page"><header><h1>Emails written for the actual person</h1><p class="intro">Each email has a different opening, level of detail, and request because each person knows a different part of the problem. Read the matching note before sending and attach only that person's PDF.</p></header>{''.join(entries)}</main><script>document.querySelectorAll('[data-copy]').forEach((button)=>{{button.addEventListener('click',async()=>{{const target=document.getElementById(button.dataset.copy);const original=button.textContent;try{{await navigator.clipboard.writeText(target.innerText.trim());button.textContent='Copied'}}catch(_){{button.textContent='Select and copy'}}setTimeout(()=>{{button.textContent=original}},1800)}})}});</script></body></html>"""


def main() -> None:
    for brief in BRIEFS:
        base = base_name(brief)
        html_path = OUT / f"{base}.html"
        pdf_path = OUT / f"{base}.pdf"
        html_path.write_text(HTML_BUILDERS[brief["number"]](brief), encoding="utf-8")
        build_pdf(pdf_path, colors.HexColor(brief["accent"]), brief["story"](), brief["short"])
        print(html_path)
        print(pdf_path)
    (OUT / "index.html").write_text(index_html(), encoding="utf-8")
    (ROOT / "Ac225_Expert_Email_Templates_2026.html").write_text(email_kit_html(), encoding="utf-8")
    print(OUT / "index.html")
    print(ROOT / "Ac225_Expert_Email_Templates_2026.html")


if __name__ == "__main__":
    main()
