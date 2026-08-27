from pathlib import Path
import json
import re

from pypdf import PdfReader


ROOT = Path(r"C:\Users\ogunn\Downloads\New folder")
EMAIL_KIT = ROOT / "Ac225_Expert_Email_Templates_2026.html"

text = EMAIL_KIT.read_text(encoding="utf-8")
goals_text = (ROOT / "Ac225_Project_Goals_2026.html").read_text(encoding="utf-8")
brief_dir = ROOT / "Ac225_Personalized_Expert_Briefs_2026"
email_entries = re.findall(r'<section class="email-entry"', text)
template_ids = re.findall(r'<div class="letter" id="([^"]+)"', text)
copy_targets = re.findall(r'data-copy="([^"]+)"', text)
mailtos = re.findall(r'href="mailto:([^"]+)"', text)
pdf_links = re.findall(r'href="([^"]+\.pdf)"', text)
emoji_like = sorted(
    {
        character
        for character in text
        if ord(character) > 0xFFFF or 0x1F300 <= ord(character) <= 0x1FAFF
    }
)

expected_briefs = {
    "01_Lee_Bernstein_Technical_Brief.pdf": "Professor Lee A. Bernstein",
    "02_Jaden_Palmer_Technical_Brief.pdf": "Jaden Palmer",
    "03_John_Brockman_Technical_Brief.pdf": "Dr. John Brockman",
    "04_Robert_Hobbs_Technical_Brief.pdf": "Dr. Robert F. Hobbs",
    "05_Jonathan_Engle_Technical_Brief.pdf": "Professor Jonathan W. Engle",
    "06_Gregory_Severin_Technical_Brief.pdf": "Professor Gregory W. Severin",
    "07_Brian_Zimmerman_Technical_Brief.pdf": "Dr. Brian E. Zimmerman",
}
cross_person_checks = {}
all_target_names = set(expected_briefs.values())
brief_html_texts = []
for filename, target_name in expected_briefs.items():
    pdf_text = "\n".join(
        page.extract_text() or "" for page in PdfReader(brief_dir / filename).pages
    )
    cross_person_checks[filename] = {
        "contains_target": target_name in pdf_text,
        "contains_other_targets": sorted(
            name for name in all_target_names - {target_name} if name in pdf_text
        ),
    }
    brief_html_texts.append(
        (brief_dir / filename.replace(".pdf", ".html")).read_text(encoding="utf-8")
    )

generic_headings = (
    "Why I Am Reaching Out to You",
    "What I Built and Where I Need Help",
    "My specific question",
)

print(
    json.dumps(
        {
            "email_entries": len(email_entries),
            "template_ids": len(template_ids),
            "unique_template_ids": len(set(template_ids)),
            "copy_targets_match": set(template_ids) == set(copy_targets),
            "mailtos": len(mailtos),
            "unique_mailtos": len(set(mailtos)),
            "pdf_links": len(pdf_links),
            "all_pdf_links_exist": all((ROOT / link).is_file() for link in pdf_links),
            "old_generic_link": "Ac225_Expert_Request_OnePager" in text,
            "referral_only_names": any(
                name in text for name in ("Bowman", "backup contact")
            ),
            "emoji_like": emoji_like,
            "roadmap_uses_personalized_index": (
                "Ac225_Personalized_Expert_Briefs_2026/index.html" in goals_text
            ),
            "roadmap_uses_old_generic": "Ac225_Expert_Request_OnePager" in goals_text,
            "generic_heading_hits": {
                heading: sum(heading in page for page in brief_html_texts)
                for heading in generic_headings
            },
            "cross_person_checks": cross_person_checks,
        },
        indent=2,
    )
)
