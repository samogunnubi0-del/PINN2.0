from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
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


OUTPUT = Path(r"C:\Users\ogunn\Downloads\New folder\Ac225_Expert_Request_OnePager.pdf")

INK = colors.HexColor("#17232d")
TEXT = colors.HexColor("#354653")
MUTED = colors.HexColor("#687783")
LINE = colors.HexColor("#ccd7dd")
RED = colors.HexColor("#a92532")
RED_SOFT = colors.HexColor("#f8e9eb")
TEAL = colors.HexColor("#006c70")
TEAL_SOFT = colors.HexColor("#e2f1f0")
BLUE = colors.HexColor("#285f8e")
BLUE_SOFT = colors.HexColor("#e7eff7")
GOLD = colors.HexColor("#9b660d")
GOLD_SOFT = colors.HexColor("#fbf1dc")
GREEN = colors.HexColor("#26734d")
GREEN_SOFT = colors.HexColor("#e5f2eb")
GRAY_SOFT = colors.HexColor("#f4f7f8")
WHITE = colors.white

PAGE_W, PAGE_H = letter
LEFT = 29
RIGHT = 29
TOP = 24
BOTTOM = 22
CONTENT_W = PAGE_W - LEFT - RIGHT

styles = getSampleStyleSheet()


def ps(name, **kwargs):
    base = {
        "fontName": "Helvetica",
        "textColor": TEXT,
        "fontSize": 7.4,
        "leading": 9.0,
        "spaceAfter": 0,
        "spaceBefore": 0,
        "alignment": TA_LEFT,
    }
    base.update(kwargs)
    return ParagraphStyle(name, **base)


EYEBROW = ps("eyebrow", fontName="Helvetica-Bold", textColor=RED, fontSize=8.0, leading=9.2)
TITLE = ps("title", fontName="Helvetica-Bold", textColor=INK, fontSize=20.2, leading=21.0)
SUBTITLE = ps("subtitle", textColor=TEXT, fontSize=8.5, leading=10.1)
IDENTITY = ps("identity", textColor=TEXT, fontSize=7.9, leading=9.8)
IDENTITY_NAME = ps("identity_name", fontName="Helvetica-Bold", textColor=INK, fontSize=10.2, leading=11.2)
SECTION = ps("section", fontName="Helvetica-Bold", textColor=INK, fontSize=10.1, leading=11.5)
LABEL_RED = ps("label_red", fontName="Helvetica-Bold", textColor=RED, fontSize=7.8, leading=9.0)
BODY = ps("body", textColor=TEXT, fontSize=7.8, leading=9.5)
SMALL = ps("small", textColor=TEXT, fontSize=7.05, leading=8.65)
TINY = ps("tiny", textColor=TEXT, fontSize=6.55, leading=7.8)
FOOT = ps("foot", textColor=MUTED, fontSize=6.25, leading=7.4)
METRIC_VALUE = ps("metric_value", fontName="Helvetica-Bold", textColor=INK, fontSize=11.2, leading=12.0)
METRIC_NOTE = ps("metric_note", textColor=TEXT, fontSize=6.45, leading=7.5)
FACT_HEAD = ps("fact_head", fontName="Helvetica-Bold", textColor=BLUE, fontSize=8.35, leading=9.4)
TABLE_HEAD = ps("table_head", fontName="Helvetica-Bold", textColor=WHITE, fontSize=6.65, leading=7.7)
TABLE_BODY = ps("table_body", textColor=TEXT, fontSize=6.25, leading=7.45)
TABLE_ROUTE = ps("table_route", fontName="Helvetica-Bold", textColor=INK, fontSize=6.4, leading=7.5)


def page_decoration(canvas, doc):
    canvas.saveState()
    widths = [PAGE_W * 0.5, PAGE_W * 0.25, PAGE_W * 0.125, PAGE_W * 0.125]
    x = 0
    for width, color in zip(widths, [RED, TEAL, GOLD, BLUE]):
        canvas.setFillColor(color)
        canvas.rect(x, PAGE_H - 6, width, 6, stroke=0, fill=1)
        x += width
    canvas.restoreState()


doc = BaseDocTemplate(
    str(OUTPUT),
    pagesize=letter,
    leftMargin=LEFT,
    rightMargin=RIGHT,
    topMargin=TOP,
    bottomMargin=BOTTOM,
    title="Ac-225 Expert Evidence Request - Samuel Ogunnubi",
    author="Samuel Ogunnubi",
)
frame = Frame(LEFT, BOTTOM, CONTENT_W, PAGE_H - TOP - BOTTOM, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
doc.addPageTemplates([PageTemplate(id="one-page", frames=[frame], onPage=page_decoration)])

story = [Spacer(1, 2)]

header_left = [
    Paragraph("TECHNICAL EVIDENCE REQUEST | AUGUST 23, 2026", EYEBROW),
    Spacer(1, 2),
    Paragraph("Ra-226 to Ac-225 Surrogate Study", TITLE),
    Spacer(1, 3),
    Paragraph("What is already complete, what the frozen model can and cannot support, and the specific outside evidence now needed.", SUBTITLE),
]
header_right = [
    Paragraph("Samuel Ogunnubi", IDENTITY_NAME),
    Paragraph("Grade 12, Glen Burnie High School<br/>Biomedical and Allied Health Program<br/>Maryland, United States", IDENTITY),
]
header = Table([[header_left, header_right]], colWidths=[CONTENT_W - 165, 165])
header.setStyle(TableStyle([
    ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
    ("LEFTPADDING", (0, 0), (0, 0), 0),
    ("RIGHTPADDING", (0, 0), (0, 0), 12),
    ("LEFTPADDING", (1, 0), (1, 0), 10),
    ("RIGHTPADDING", (1, 0), (1, 0), 0),
    ("LINEBEFORE", (1, 0), (1, 0), 3, TEAL),
    ("LINEBELOW", (0, 0), (-1, -1), 0.6, LINE),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
]))
story.extend([header, Spacer(1, 7)])

purpose = Table([
    [Paragraph("PURPOSE", LABEL_RED), Paragraph(
        "I am seeking public data, references, or a brief technical review to determine whether a frozen surrogate can support fast, solver-verified screening of Ac-225 production scenarios. I am not requesting private facility data, a radioactive experiment, clinical advice, or an endorsement.",
        BODY,
    )]
], colWidths=[61, CONTENT_W - 61])
purpose.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), RED_SOFT),
    ("LINEBEFORE", (0, 0), (0, 0), 3.5, RED),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (0, 0), 7),
    ("RIGHTPADDING", (0, 0), (0, 0), 5),
    ("LEFTPADDING", (1, 0), (1, 0), 5),
    ("RIGHTPADDING", (1, 0), (1, 0), 7),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.extend([purpose, Spacer(1, 7)])

metric_data = [[
    [Paragraph("0.97%", METRIC_VALUE), Paragraph("Matched synthetic worst-mesh median Ac-225 endpoint error, seed 42", METRIC_NOTE)],
    [Paragraph("3.94% / 183.84%", METRIC_VALUE), Paragraph("Median / p95 error across 9,000 productive cases in a 10,000-case grid", METRIC_NOTE)],
    [Paragraph("Rank 2", METRIC_VALUE), Paragraph("V3-selected schedule after Radau verification; 0.047% objective regret", METRIC_NOTE)],
    [Paragraph("84.9x / 56.7x", METRIC_VALUE), Paragraph("Faster than Radau, but slower than the exact analytic batch baseline at 10,000 cases", METRIC_NOTE)],
]]
metrics = Table(metric_data, colWidths=[CONTENT_W / 4.0] * 4)
metrics.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), GRAY_SOFT),
    ("BOX", (0, 0), (-1, -1), 0.5, LINE),
    ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ("LINEABOVE", (0, 0), (0, 0), 3, BLUE),
    ("LINEABOVE", (1, 0), (1, 0), 3, GOLD),
    ("LINEABOVE", (2, 0), (2, 0), 3, GREEN),
    ("LINEABOVE", (3, 0), (3, 0), 3, TEAL),
]))
story.extend([metrics, Spacer(1, 6), Paragraph("Current Technical Contract", SECTION), Spacer(1, 3)])


def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(item, SMALL), leftIndent=7) for item in items],
        bulletType="bullet",
        start="square",
        leftIndent=10,
        bulletFontName="Helvetica",
        bulletFontSize=5.0,
        bulletColor=BLUE,
        spaceBefore=0,
        spaceAfter=0,
    )


facts = Table([[ 
    [Paragraph("Frozen model and pathway", FACT_HEAD), Spacer(1, 2), bullets([
        "Ra-226(n,2n)Ra-225 to Ac-225, with Ra-226(n,gamma)Ra-227 to Ac-227 as a competing branch.",
        "Physics-informed LSTM, reduced five-nuclide system, 1,400 Radau-generated training scenarios, 6,000 epochs, seed 42.",
        "Constant irradiation followed by exact decay-only cooling; activity reported per 1 g initial Ra-226.",
        "Checkpoint and source hashes are preserved. No tuning is allowed after outside results are seen.",
    ])],
    [Paragraph("Post-lock reality checks", FACT_HEAD), Spacer(1, 2), bullets([
        "The LSTM beat a matched MLP on pointwise error, but the MLP ranked the full grid more accurately and ran faster.",
        "Berkeley 33 and 40 MeV cases were outside training support; raw V3 activity predictions were about 600 times too high.",
        "The reduced constant-rate equations have an exact analytic solution that is faster and more accurate than V3.",
        "Current value is limited to screening candidates that are then checked with a trusted solver.",
    ])],
]], colWidths=[CONTENT_W / 2.0] * 2)
facts.setStyle(TableStyle([
    ("BOX", (0, 0), (-1, -1), 0.5, LINE),
    ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
]))
story.extend([facts, Spacer(1, 5)])

warning = Table([[Paragraph(
    "<b>Claim boundary:</b> The project does not yet demonstrate reactor accuracy, experimental validation, a universal error below 3%, clinical readiness, production cost savings, or a decisive LSTM advantage.",
    SMALL,
)]], colWidths=[CONTENT_W])
warning.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), GOLD_SOFT),
    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#ddc489")),
    ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story.extend([warning, Spacer(1, 6), Paragraph("One Deliverable Requested From Each Expert Route", SECTION), Spacer(1, 3)])

requests = [
    [Paragraph("Expert route", TABLE_HEAD), Paragraph("Requested technical deliverable", TABLE_HEAD), Paragraph("How it strengthens the study", TABLE_HEAD)],
    [Paragraph("Berkeley / LBNL", TABLE_ROUTE), Paragraph("A reproducible 33/40 MeV specification: target-plane spectrum and normalization; target inventory, position, and geometry; beam/fluence history; irradiation, cooling, and separation times; produced versus recovered Ac-225 with uncertainty; any numerical Ac-227 detection limit.", TABLE_BODY), Paragraph("Correct the published-case reconstruction and define a future model domain without tuning V3 to the answer.", TABLE_BODY)],
    [Paragraph("NC State ARTISANS", TABLE_ROUTE), Paragraph("A short critique of the frozen validation protocol, metrics, uncertainty, applicability-domain warning, and minimum evidence for a useful solver-verified surrogate.", TABLE_BODY), Paragraph("Make the ML claim statistically and scientifically defensible.", TABLE_BODY)],
    [Paragraph("MURR", TABLE_ROUTE), Paragraph("A public example or realistic ranges for irradiation, cooling, separation, recovery, quality control, target recycling, and the constraints that dominate turnaround.", TABLE_BODY), Paragraph("Replace an irradiation-only objective with an end-to-end workflow.", TABLE_BODY)],
    [Paragraph("Johns Hopkins", TABLE_ROUTE), Paragraph("A published reference for activity reference time, radionuclidic impurity quantity, uncertainty, and other outputs a production-screening tool should report.", TABLE_BODY), Paragraph("Define a medically relevant objective without making a clinical claim.", TABLE_BODY)],
    [Paragraph("UW-Madison", TABLE_ROUTE), Paragraph("A public targetry/radiochemistry benchmark and guidance on beam, thermal, geometry, recovery, and turnaround variables for a realistic accelerator search.", TABLE_BODY), Paragraph("Add feasibility limits and a stronger comparison problem.", TABLE_BODY)],
    [Paragraph("MSU / FRIB", TABLE_ROUTE), Paragraph("A public example for separation, recovery fraction, purification time, quality checks, and material losses between produced and usable isotope.", TABLE_BODY), Paragraph("Add the missing chemistry and recovery layer to total-process time.", TABLE_BODY)],
    [Paragraph("NIST", TABLE_ROUTE), Paragraph("The recommended activity-to-atoms and reporting treatment for progeny equilibrium, assay/separation/reference times, decay data, and uncertainty.", TABLE_BODY), Paragraph("Make experimental activity comparisons dimensionally and metrologically correct.", TABLE_BODY)],
]
request_table = Table(requests, colWidths=[91, 291, CONTENT_W - 382], repeatRows=1)
request_style = [
    ("BACKGROUND", (0, 0), (-1, 0), INK),
    ("BOX", (0, 0), (-1, -1), 0.5, LINE),
    ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 4.4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4.4),
]
for row in range(2, len(requests), 2):
    request_style.append(("BACKGROUND", (0, row), (-1, row), GRAY_SOFT))
request_table.setStyle(TableStyle(request_style))
story.extend([request_table, Spacer(1, 6)])

boundaries = Table([[ 
    [Paragraph("What I can provide", ps("bound_h1", fontName="Helvetica-Bold", textColor=TEAL, fontSize=7.8, leading=8.8)), Paragraph("Frozen checkpoint hash, model ranges, reduced reaction chain, full result tables, failure map, Berkeley reconstruction, source code, and assumption log.", TINY)],
    [Paragraph("What I will do with help", ps("bound_h2", fontName="Helvetica-Bold", textColor=TEAL, fontSize=7.8, leading=8.8)), Paragraph("Cite public sources, preserve negative results, document assumptions, verify finalists with a trusted solver, and acknowledge guidance only with permission.", TINY)],
]], colWidths=[CONTENT_W / 2.0] * 2)
boundaries.setStyle(TableStyle([
    ("BOX", (0, 0), (-1, -1), 0.5, LINE),
    ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.extend([boundaries, Spacer(1, 5)])

footer = Paragraph(
    "<b>Supporting public sources:</b> Morrell, <i>Next-Generation Isotope Production via Deuteron Breakup</i> (UC Berkeley, 2021); Berkeley Nuclear Data Program; NIST Ac-225 activity-standard work. All percentages above describe preserved computational artifacts, not facility or clinical performance.",
    FOOT,
)
story.append(footer)

doc.build(story)
print(OUTPUT)
