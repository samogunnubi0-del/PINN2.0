from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(r"C:\Users\ogunn\Downloads\New folder")
OUTPUT = ROOT / "output" / "pdf" / "Samuel_Ogunnubi_Ac225_Berkeley_OnePager_20260827.pdf"

REGULAR = r"C:\Windows\Fonts\segoeui.ttf"
SEMIBOLD = r"C:\Windows\Fonts\seguisb.ttf"
pdfmetrics.registerFont(TTFont("BriefRegular", REGULAR))
pdfmetrics.registerFont(TTFont("BriefBold", SEMIBOLD))

INK = colors.HexColor("#14232B")
TEXT = colors.HexColor("#42545D")
MUTED = colors.HexColor("#66777F")
LINE = colors.HexColor("#D6E0E3")
TEAL = colors.HexColor("#086B72")
TEAL_DARK = colors.HexColor("#084D54")
TEAL_PALE = colors.HexColor("#E8F3F3")
BLUE = colors.HexColor("#347EAA")
BLUE_PALE = colors.HexColor("#EBF3F8")
CORAL = colors.HexColor("#D65F47")
CORAL_PALE = colors.HexColor("#FCEDE9")
GOLD = colors.HexColor("#C88B1B")
GOLD_PALE = colors.HexColor("#FCF4E2")
PURPLE = colors.HexColor("#7254A5")
PALE = colors.HexColor("#F7F9FA")


def pstyle(name, size, leading, color=TEXT, bold=False):
    return ParagraphStyle(
        name,
        fontName="BriefBold" if bold else "BriefRegular",
        fontSize=size,
        leading=leading,
        textColor=color,
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=0,
    )


def para(pdf, text, x, top, width, sty):
    item = Paragraph(text, sty)
    _, height = item.wrap(width, 500)
    item.drawOn(pdf, x, top - height)
    return height


def panel(pdf, x, y, w, h, fill=PALE, stroke=LINE, radius=5):
    pdf.setFillColor(fill)
    pdf.setStrokeColor(stroke)
    pdf.setLineWidth(0.8)
    pdf.roundRect(x, y, w, h, radius, fill=1, stroke=1)


def heading(pdf, text, x, y, color=TEAL):
    pdf.setFillColor(color)
    pdf.setFont("BriefBold", 8.2)
    pdf.drawString(x, y, text.upper())


def step(pdf, x, y, w, number, title, body, accent, fill):
    panel(pdf, x, y, w, 61, fill=fill)
    pdf.setFillColor(accent)
    pdf.circle(x + 14, y + 44, 8, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("BriefBold", 7)
    pdf.drawCentredString(x + 14, y + 41.5, str(number))
    pdf.setFillColor(INK)
    pdf.setFont("BriefBold", 7.5)
    pdf.drawString(x + 27, y + 43, title.upper())
    para(pdf, body, x + 10, y + 31, w - 20, pstyle(f"step-{number}", 6.1, 7.2, TEXT))


def metric(pdf, x, y, value, title, note, accent):
    pdf.setFillColor(accent)
    pdf.rect(x, y, 3, 42, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont("BriefBold", 15)
    pdf.drawString(x + 10, y + 24, value)
    pdf.setFillColor(TEXT)
    pdf.setFont("BriefBold", 6.2)
    pdf.drawString(x + 10, y + 13, title.upper())
    pdf.setFillColor(MUTED)
    pdf.setFont("BriefRegular", 5.9)
    pdf.drawString(x + 10, y + 4, note)


def bullet(pdf, x, y, text, width, accent=TEAL, size=6.6, leading=8):
    pdf.setFillColor(accent)
    pdf.circle(x + 3, y - 3, 2.2, fill=1, stroke=0)
    return para(pdf, text, x + 10, y + 2, width - 10, pstyle(f"bullet-{x}-{y}", size, leading, TEXT))


def build_pdf():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(OUTPUT), pagesize=letter, pageCompression=1)
    pdf.setTitle("Ac-225 Surrogate Research Brief | Samuel Ogunnubi")
    pdf.setAuthor("Samuel Ogunnubi")
    pdf.setSubject("Simple one-page project explanation and Berkeley evidence request")

    # Header.
    pdf.setFillColor(TEAL_DARK)
    pdf.rect(0, 784, 612, 8, fill=1, stroke=0)
    heading(pdf, "One-page research brief | August 2026", 28, 760)
    para(
        pdf,
        "Can a Physics-Guided Model Screen Ac-225 Production Plans Faster?",
        28,
        744,
        410,
        pstyle("title", 18.2, 20, INK, True),
    )
    para(
        pdf,
        "A student-built surrogate study for comparing irradiation and cooling conditions before trusted-solver verification.",
        28,
        700,
        420,
        pstyle("subtitle", 7.6, 9, MUTED),
    )
    pdf.setStrokeColor(LINE)
    pdf.line(462, 690, 462, 756)
    pdf.setFillColor(INK)
    pdf.setFont("BriefBold", 10.5)
    pdf.drawRightString(584, 742, "Samuel Ogunnubi")
    pdf.setFillColor(TEXT)
    pdf.setFont("BriefBold", 6.6)
    pdf.drawRightString(584, 728, "12th-grade student researcher")
    pdf.setFillColor(MUTED)
    pdf.setFont("BriefRegular", 6.2)
    pdf.drawRightString(584, 715, "Glen Burnie High School, Maryland")
    pdf.drawRightString(584, 704, "Biomedical and Allied Health")

    # Project goal.
    panel(pdf, 28, 626, 556, 54, fill=TEAL_PALE, stroke=colors.HexColor("#B9D7D9"))
    heading(pdf, "What I am trying to do", 40, 663)
    para(
        pdf,
        "Ac-225 is promising for targeted alpha therapy, but production is difficult. I am testing whether a fast model can narrow many production plans, reject conditions it does not understand, and send only the strongest candidates to trusted nuclear calculations. It screens possibilities; it does not replace the physics.",
        40,
        651,
        528,
        pstyle("goal", 6.8, 8.1, TEXT),
    )

    # How it works.
    heading(pdf, "How the project works", 28, 610)
    step(pdf, 28, 536, 132, 1, "Inputs", "Beam energy, target geometry, irradiation time, and cooling time.", BLUE, BLUE_PALE)
    step(pdf, 169, 536, 132, 2, "Reaction rates", "Predict the Ra-226(n,2n) and competing (n,gamma) rates.", TEAL, TEAL_PALE)
    step(pdf, 310, 536, 132, 3, "Isotope chain", "Track Ra-225 to Ac-225 and the Ra-227 to Ac-227 impurity branch.", CORAL, CORAL_PALE)
    step(pdf, 451, 536, 133, 4, "Safety check", "Warn outside the training domain; verify accepted finalists with physics.", GOLD, GOLD_PALE)

    # Work completed and results.
    panel(pdf, 28, 324, 344, 196)
    heading(pdf, "What I built and tested", 42, 501)
    pdf.setFillColor(INK)
    pdf.setFont("BriefBold", 9.2)
    pdf.drawString(42, 481, "V3: physics-informed isotope-timeline model")
    bullet(pdf, 42, 465, "Reduced five-nuclide system with physics penalties and exact decay-only cooling checks.", 316)
    bullet(pdf, 42, 441, "1,400 Radau trajectories; 6,000 epochs; frozen seed-42 checkpoint and source hashes.", 316)
    bullet(pdf, 42, 417, "Compared with a matched MLP, shifted tests, and a 10,000-schedule audit.", 316)

    pdf.setStrokeColor(LINE)
    pdf.line(42, 397, 358, 397)
    pdf.setFillColor(INK)
    pdf.setFont("BriefBold", 9.2)
    pdf.drawString(42, 379, "V4: geometry, uncertainty, and domain test")
    bullet(pdf, 42, 361, "GP versus five-seed LSTM on the same finite-geometry/TENDL proxy data.", 316)
    bullet(pdf, 42, 337, "Split: 160 train, 24 select, 24 calibrate, and 48 locked test; paired-bootstrap ranking.", 316)

    panel(pdf, 384, 324, 200, 196, fill=colors.white)
    heading(pdf, "Current results", 398, 501, CORAL)
    metric(pdf, 398, 447, "1.83%", "LSTM median", "p95: 3.93%", CORAL)
    metric(pdf, 490, 447, "1.97%", "GP median", "p95: 3.79%", BLUE)
    metric(pdf, 398, 391, "95.8%", "joint coverage", "calibrated 95%", TEAL)
    metric(pdf, 490, 391, "100%", "OOD rejection", "184 of 184 warned", GOLD)
    pdf.setStrokeColor(LINE)
    pdf.line(398, 376, 570, 376)
    para(pdf, "<b>V3 endpoint test:</b> 0.97% matched median; 1.57% shifted; matched MLP 2.91%.", 398, 365, 172, pstyle("v3metric", 6.2, 7.4, TEXT))
    para(pdf, "<b>Schedule audit:</b> rank 2, 0.047% below Radau's best; 84.9x faster.", 398, 338, 172, pstyle("schedule", 6.2, 7.4, TEXT))

    # Interpretation.
    panel(pdf, 28, 264, 556, 45, fill=colors.HexColor("#F3F0FA"), stroke=colors.HexColor("#D9CEEA"))
    heading(pdf, "What these results actually mean", 40, 292, PURPLE)
    para(
        pdf,
        "Both V4 models passed every predeclared gate, but the paired bootstrap did not resolve a winner. GP is only the provisional leader because its observed p95 was slightly lower. V4 is a static one-step geometry test, so it cannot prove that LSTM memory helps. The exact analytic baseline also remains faster for this small equation set.",
        40,
        280,
        528,
        pstyle("meaning", 6.45, 7.7, TEXT),
    )

    # Failure and evidence request.
    panel(pdf, 28, 102, 205, 147, fill=CORAL_PALE, stroke=colors.HexColor("#F2C7BE"))
    heading(pdf, "What failed", 42, 230, CORAL)
    pdf.setFillColor(INK)
    pdf.setFont("BriefBold", 9.4)
    pdf.drawString(42, 210, "The Berkeley reality check")
    para(
        pdf,
        "The first reconstruction of the published 33 and 40 MeV Ra-226 experiments fell outside V3's training support. Raw activity predictions were about 600 times too high.",
        42,
        196,
        177,
        pstyle("failure", 6.7, 8, TEXT),
    )
    para(
        pdf,
        "I kept this failure instead of tuning the model to make two outside points look correct. The next test must distinguish a true model failure from missing experiment inputs or a reconstruction mismatch.",
        42,
        153,
        177,
        pstyle("failure2", 6.7, 8, TEXT),
    )

    panel(pdf, 245, 102, 339, 147, fill=GOLD_PALE, stroke=colors.HexColor("#EACD91"))
    heading(pdf, "What I need now", 259, 230, GOLD)
    para(pdf, "To run a frozen Berkeley external test without tuning, I need:", 259, 214, 311, pstyle("needintro", 7, 8.2, INK, True))
    bullet(pdf, 259, 195, "Target-integrated neutron spectra or fluence for 33 and 40 MeV, including bins, units, and normalization.", 311, BLUE, 6.5, 7.7)
    bullet(pdf, 259, 164, "Beamspot and target-geometry correction inputs, plus beam-current or fluence history.", 311, TEAL, 6.5, 7.7)
    bullet(pdf, 259, 141, "Spectrum uncertainty, covariance, or parameter samples.", 311, GOLD, 6.5, 7.7)
    bullet(pdf, 259, 122, "A numerical Ac-227 detection limit, confidence level, and reference time, if available.", 311, CORAL, 6.5, 7.7)

    # Footer and claim boundary.
    pdf.setStrokeColor(LINE)
    pdf.line(28, 84, 584, 84)
    para(
        pdf,
        "<b>Claim boundary:</b> Current evidence is modeled proxy data. It does not yet establish experimental, reactor, clinical, cost-savings, or universal under-3% performance.",
        28,
        74,
        556,
        pstyle("claim", 6.2, 7.3, MUTED),
    )
    pdf.setFillColor(MUTED)
    pdf.setFont("BriefRegular", 5.8)
    pdf.drawString(28, 39, "Frozen artifacts, source hashes, full tables, failure analysis, and code are available.")
    pdf.setFillColor(INK)
    pdf.setFont("BriefBold", 6.2)
    pdf.drawRightString(584, 39, "sam.ogunnubi0@gmail.com")

    pdf.save()
    print(OUTPUT)


if __name__ == "__main__":
    build_pdf()
