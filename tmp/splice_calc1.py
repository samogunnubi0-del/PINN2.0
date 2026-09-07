from pathlib import Path

root = Path(r"c:\Users\ogunn\Downloads\New folder")
html = (root / "Calc1_A_Plan.html").read_text(encoding="utf-8")
head = html.split("<body>", 1)[0]
head = head.replace(
    "</style>\n<link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css\">\n<script defer src=\"https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js\"></script>\n",
    "</style>\n",
)
body = (root / "tmp" / "calc1_body.html").read_text(encoding="utf-8")
js = (root / "tmp" / "calc1_app.js").read_text(encoding="utf-8")
if not body.startswith("<body>"):
    raise SystemExit("body fragment missing <body>")
out = head + body.replace("/*APP_JS*/", js)
dest = root / "Calc1_The_A_Plan.html"
(root / "Calc1_A_Plan.html").write_text(out, encoding="utf-8") if False else None
dest.write_text(out, encoding="utf-8")
print("wrote", dest, "bytes", len(out), "has APP_JS", "/*APP_JS*/" in out)
print("has PLAYLIST", "PLO1y6V1SXjjNSSOZvV3PcFu4B1S8nfXBM" in out)
print("has The A Plan", "The A Plan" in out)
print("has formula gym", "Formula gym" in out)
