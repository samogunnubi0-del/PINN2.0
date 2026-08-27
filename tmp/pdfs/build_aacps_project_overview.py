from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(r"C:\Users\ogunn\Downloads\New folder")
OUTPUT = ROOT / "Ac225_AACPS_Project_Overview_2026.pdf"


def register_fonts() -> tuple[str, str]:
    regular = Path(r"C:\Windows\Fonts\arial.ttf")
    bold = Path(r"C:\Windows\Fonts\arialbd.ttf")
    if regular.is_file() and bold.is_file():
        pdfmetrics.registerFont(TTFont("AACPSArial", str(regular)))
        pdfmetrics.registerFont(TTFont("AACPSArialBold", str(bold)))
        return "AACPSArial", "AACPSArialBold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = register_fonts()


INK = colors.HexColor("#0f172a")
TEXT = colors.HexColor("#334155")
MUTED = colors.HexColor("#64748b")
LINE = colors.HexColor("#d7e0e8")
CRIMSON = colors.HexColor("#e11d48")
AMBER = colors.HexColor("#d97706")
BLUE = colors.HexColor("#2563eb")
CYAN = colors.HexColor("#0284c7")
EMERALD = colors.HexColor("#059669")
PURPLE = colors.HexColor("#7c3aed")
TEAL = colors.HexColor("#0d9488")


def style(name: str, size: float, leading: float, color=TEXT, align=TA_LEFT, bold=False):
    return ParagraphStyle(
        name,
        fontName=FONT_BOLD if bold else FONT,
        fontSize=size,
        leading=leading,
        textColor=color,
        alignment=align,
        spaceAfter=0,
        spaceBefore=0,
        allowWidows=0,
        allowOrphans=0,
    )


def draw_paragraph(pdf, text, x, top, width, paragraph_style):
    paragraph = Paragraph(text, paragraph_style)
    _, height = paragraph.wrap(width, 500)
    paragraph.drawOn(pdf, x, top - height)
    return height


def draw_card(pdf, x, y, width, height, number, accent, background, tag, title, body):
    pdf.setFillColor(background)
    pdf.setStrokeColor(colors.Color(accent.red, accent.green, accent.blue, alpha=0.24))
    pdf.setLineWidth(0.8)
    pdf.roundRect(x, y, width, height, 5, fill=1, stroke=1)

    pdf.setFillColor(accent)
    pdf.circle(x + 13, y + height - 13, 8.5, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont(FONT_BOLD, 7.2)
    pdf.drawCentredString(x + 13, y + height - 15.5, str(number))

    text_x = x + 25
    text_width = width - 33
    top = y + height - 7
    draw_paragraph(pdf, tag.upper(), text_x, top, text_width, style(f"tag-{number}", 5.2, 6.1, accent, bold=True))
    title_height = draw_paragraph(pdf, title, text_x, top - 8, text_width, style(f"title-{number}", 7.6, 8.4, INK, bold=True))
    draw_paragraph(pdf, body, text_x, top - 10 - title_height, text_width, style(f"body-{number}", 5.8, 6.7, TEXT))


def build_pdf():
    pdf = canvas.Canvas(str(OUTPUT), pagesize=letter, pageCompression=1)
    pdf.setTitle("Ac-225 Project Overview for AACPS | Samuel Ogunnubi")
    page_width, page_height = letter

    accents = [CRIMSON, AMBER, BLUE, EMERALD, PURPLE]
    segment = page_width / len(accents)
    for index, accent in enumerate(accents):
        pdf.setFillColor(accent)
        pdf.rect(index * segment, page_height - 4, segment + 1, 4, fill=1, stroke=0)

    pdf.setFillColor(colors.HexColor("#fff1f2"))
    pdf.setStrokeColor(colors.HexColor("#fecdd3"))
    pdf.roundRect(28, 756, 216, 15, 4, fill=1, stroke=1)
    pdf.setFillColor(CRIMSON)
    pdf.setFont(FONT_BOLD, 6.7)
    pdf.drawString(35, 761, "PREPARED FOR DR. MAUREEN MCMAHON | AACPS")

    title_style = style("main-title", 18.2, 19.2, INK, bold=True)
    draw_paragraph(pdf, "Can AI Help Screen Ac-225 Production Plans Faster?", 28, 748, 405, title_style)
    subtitle = (
        "I am testing whether a physics-guided computer model can narrow thousands of irradiation and cooling "
        "schedules before trusted nuclear calculations check every finalist."
    )
    draw_paragraph(pdf, subtitle, 28, 707, 410, style("subtitle", 7.5, 9.0, MUTED))

    pdf.setStrokeColor(LINE)
    pdf.setLineWidth(1.2)
    pdf.line(450, 704, 450, 764)
    pdf.setFillColor(INK)
    pdf.setFont(FONT_BOLD, 10.5)
    pdf.drawRightString(584, 748, "Samuel Ogunnubi")
    pdf.setFillColor(TEXT)
    pdf.setFont(FONT_BOLD, 7.0)
    pdf.drawRightString(584, 735, "12th-grade student researcher")
    pdf.setFillColor(MUTED)
    pdf.setFont(FONT, 6.4)
    pdf.drawRightString(584, 723, "Biomedical and Allied Health")
    pdf.drawRightString(584, 713, "Glen Burnie High School, Maryland")
    pdf.setStrokeColor(LINE)
    pdf.setLineWidth(0.8)
    pdf.line(28, 681, 584, 681)

    pdf.saveState()
    pdf.setStrokeColor(colors.HexColor("#cbd5e1"))
    pdf.setLineWidth(1.1)
    pdf.setDash(4, 4)
    pdf.ellipse(63, 316, 549, 668, fill=0, stroke=1)
    pdf.setStrokeColor(colors.HexColor("#d8cde5"))
    pdf.setDash(2, 5)
    pdf.ellipse(181, 375, 431, 609, fill=0, stroke=1)
    pdf.restoreState()

    draw_card(pdf, 221, 611, 170, 62, 1, CRIMSON, colors.HexColor("#fff1f2"), "Medical need", "Why I chose Ac-225", "Ac-225 can deliver short-range alpha radiation to targeted cancer cells, but it is scarce and difficult to produce.")
    draw_card(pdf, 424, 559, 160, 75, 2, AMBER, colors.HexColor("#fffbeb"), "Chosen pathway", "How this route starts", "Ra-226 is exposed to fast neutrons made by deuterons striking beryllium. Some becomes Ra-225.")
    draw_card(pdf, 447, 445, 145, 88, 3, BLUE, colors.HexColor("#eff6ff"), "Simplified science", "The system I modeled", "Five connected isotopes are tracked through irradiation and cooling, including a small Ac-227 side pathway.")
    draw_card(pdf, 369, 309, 180, 89, 4, CYAN, colors.HexColor("#f0f9ff"), "Reference calculations", "How I created evidence", "Radau generated 1,400 synthetic examples from evaluated data. An exact Bateman calculation checked the reduced equations.")
    draw_card(pdf, 63, 309, 180, 89, 5, EMERALD, colors.HexColor("#ecfdf5"), "Model design", "The model I built", "I trained an LSTM for 6,000 epochs. It follows the isotope timeline while physics rules discourage impossible behavior.")
    draw_card(pdf, 20, 445, 145, 88, 6, PURPLE, colors.HexColor("#f5f3ff"), "Frozen results", "What worked", "Median Ac-225 error was 0.97% matched and 1.57% shifted. A matched MLP scored 2.91%.")
    draw_card(pdf, 28, 559, 160, 75, 7, TEAL, colors.HexColor("#f0fdfa"), "Outside reality check", "What failed", "Two Berkeley cases were overpredicted by about 600 times outside the training range. I kept and analyzed the failure.")

    pdf.setFillColor(colors.white)
    pdf.setStrokeColor(colors.HexColor("#d7e0e8"))
    pdf.setLineWidth(1.2)
    pdf.circle(306, 493, 83, fill=1, stroke=1)
    pdf.setFillColor(colors.HexColor("#fff1f2"))
    pdf.setStrokeColor(colors.HexColor("#fecdd3"))
    pdf.roundRect(275, 543, 62, 13, 5, fill=1, stroke=1)
    pdf.setFillColor(CRIMSON)
    pdf.setFont(FONT_BOLD, 6.1)
    pdf.drawCentredString(306, 547, "MY GOAL")
    draw_paragraph(pdf, "Make early Ac-225 production planning faster to explore", 244, 535, 124, style("center-title", 10.2, 11.0, INK, TA_CENTER, True))
    draw_paragraph(pdf, "Use a fast screening model to narrow many possibilities, then verify every finalist with trusted science.", 249, 487, 114, style("center-body", 6.3, 7.4, MUTED, TA_CENTER))
    pdf.setFillColor(colors.HexColor("#f8fafc"))
    pdf.setStrokeColor(LINE)
    pdf.roundRect(257, 427, 98, 24, 8, fill=1, stroke=1)
    draw_paragraph(pdf, "Ra-226 + fast n -> Ra-225 -> <b>Ac-225</b>", 262, 445, 88, style("reaction", 6.2, 7.0, CRIMSON, TA_CENTER))

    panel_y = 178
    panel_h = 114
    pdf.setFillColor(colors.HexColor("#f8fafc"))
    pdf.setStrokeColor(LINE)
    pdf.roundRect(28, panel_y, 330, panel_h, 5, fill=1, stroke=1)
    pdf.setFillColor(BLUE)
    pdf.setFont(FONT_BOLD, 8.1)
    pdf.drawString(39, panel_y + panel_h - 16, "WHAT I HAVE COMPLETED SO FAR")

    completed = [
        "137-source scoping comparison",
        "Reduced five-isotope physics model",
        "1,400 Radau training examples",
        "V3 LSTM trained and frozen",
        "Parameter-matched MLP comparison",
        "Untouched matched and shifted tests",
        "10,000-schedule decision audit",
        "Berkeley outside stress test",
    ]
    for index, item in enumerate(completed):
        column = index // 4
        row = index % 4
        x = 40 + column * 160
        y = panel_y + panel_h - 34 - row * 19
        pdf.setFillColor(EMERALD)
        pdf.circle(x + 3, y + 2, 2.6, fill=1, stroke=0)
        draw_paragraph(pdf, item, x + 10, y + 8, 145, style(f"done-{index}", 6.2, 7.2, TEXT))

    pdf.setFillColor(colors.HexColor("#fffaf0"))
    pdf.setStrokeColor(colors.HexColor("#f2d39b"))
    pdf.roundRect(369, panel_y, 215, panel_h, 5, fill=1, stroke=1)
    pdf.setFillColor(AMBER)
    pdf.setFont(FONT_BOLD, 8.1)
    pdf.drawString(380, panel_y + panel_h - 16, "WHAT I AM TRYING TO FINISH")
    next_steps = (
        "<b>1.</b> Define the exact planning decision.<br/>"
        "<b>2.</b> Obtain realistic spectrum, geometry, timing, recovery, and uncertainty data.<br/>"
        "<b>3.</b> Get expert review of the chain and reporting rules.<br/>"
        "<b>4.</b> Build a richer benchmark with an unseen final test.<br/>"
        "<b>5.</b> Present the successes and failures honestly."
    )
    draw_paragraph(pdf, next_steps, 380, panel_y + panel_h - 28, 192, style("next", 6.0, 8.2, TEXT))

    boundary_y = 70
    boundary_h = 96
    pdf.setFillColor(colors.HexColor("#f5f3ff"))
    pdf.setStrokeColor(colors.HexColor("#d9cbea"))
    pdf.roundRect(28, boundary_y, 556, boundary_h, 5, fill=1, stroke=1)
    evidence = (
        "<b>Current evidence and limit:</b> On 10,000 simplified scenarios, V3 recovered 18 of Radau's top 20 candidates and selected Radau's second-ranked plan, only 0.047% below the best score. It was 84.9 times faster than Radau but slower than the exact calculation for this small equation set. This is a screening prototype, not proof of real production savings or reactor accuracy.<br/><br/>"
        "<b>Where AACPS support would help:</b> Guidance on the science-fair pathway and documentation, help finding a qualified mentor to review the nuclear assumptions, and support connecting the project with appropriate university or laboratory resources. AACPS is not being asked to certify the model or provide radioactive materials."
    )
    draw_paragraph(pdf, evidence, 40, boundary_y + boundary_h - 12, 532, style("boundary", 6.25, 7.6, TEXT))

    pdf.setStrokeColor(LINE)
    pdf.line(28, 55, 584, 55)
    pdf.setFillColor(MUTED)
    pdf.setFont(FONT, 5.9)
    pdf.drawString(28, 43, "AACPS project overview | Physics-informed machine learning | Updated August 23, 2026")
    pdf.drawRightString(584, 43, "sam.ogunnubi0@gmail.com")
    pdf.save()
    print(OUTPUT)


if __name__ == "__main__":
    build_pdf()
