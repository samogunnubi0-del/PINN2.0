from pathlib import Path
t = Path(r"c:\Users\ogunn\Downloads\New folder\Calc1_The_A_Plan.html").read_text(encoding="utf-8")
start = t.rfind("<script>\n")
end = t.rfind("</script>")
s = t[start+9:end]
out = Path(r"c:\Users\ogunn\Downloads\New folder\tmp\calc1_check.js")
out.write_text(s, encoding="utf-8")
print("js bytes", len(s))
