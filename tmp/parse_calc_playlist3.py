import json
import re
import urllib.request

html_path = r"c:\Users\ogunn\Downloads\New folder\tmp\yt_playlist_raw.html"
with open(html_path, encoding="utf-8") as f:
    html = f.read()

m = re.search(r"ytInitialData\s*=\s*(\{.*?\});</script>", html)
data = json.loads(m.group(1))

def extract_panel(obj, out):
    if isinstance(obj, dict):
        if "playlistPanelVideoRenderer" in obj:
            r = obj["playlistPanelVideoRenderer"]
            t = r.get("title", {})
            title = t.get("simpleText") or "".join(x.get("text", "") for x in t.get("runs", []))
            out.append({
                "index": r.get("indexText", {}).get("simpleText"),
                "id": r.get("videoId"),
                "title": title,
                "length": r.get("lengthText", {}).get("simpleText", ""),
            })
        for v in obj.values():
            extract_panel(v, out)
    elif isinstance(obj, list):
        for v in obj:
            extract_panel(v, out)

items = []
extract_panel(data, items)
# unique preserve order
seen = set()
uniq = []
for it in items:
    if it["id"] in seen:
        continue
    seen.add(it["id"])
    uniq.append(it)
print("from watch page", len(uniq))
out_txt = r"c:\Users\ogunn\Downloads\New folder\tmp\calc_playlist.txt"
with open(out_txt, "w", encoding="utf-8") as tf:
    for it in uniq:
        tf.write(f"{it['index']}\t{it['id']}\t{it['length']}\t{it['title']}\n")

# Fetch the dedicated playlist page which usually has more items
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
url = "https://www.youtube.com/playlist?list=PLO1y6V1SXjjNSSOZvV3PcFu4B1S8nfXBM"
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=40) as resp:
    plist = resp.read().decode("utf-8", "replace")

with open(r"c:\Users\ogunn\Downloads\New folder\tmp\yt_playlist_page.html", "w", encoding="utf-8") as f:
    f.write(plist)

print("playlist page len", len(plist), "playlistVideoRenderer", plist.count("playlistVideoRenderer"), "lockupViewModel", plist.count("lockupViewModel"))

pm = re.search(r"ytInitialData\s*=\s*(\{.*?\});</script>", plist)
print("ytInitialData", bool(pm))
pitems = []
if pm:
    pdata = json.loads(pm.group(1))

    def walk(obj):
        if isinstance(obj, dict):
            if "playlistVideoRenderer" in obj:
                r = obj["playlistVideoRenderer"]
                t = r.get("title", {})
                title = t.get("simpleText") or "".join(x.get("text", "") for x in t.get("runs", []))
                pitems.append({
                    "index": r.get("index", {}).get("simpleText"),
                    "id": r.get("videoId"),
                    "title": title,
                    "length": r.get("lengthText", {}).get("simpleText", ""),
                })
            if "lockupViewModel" in obj:
                r = obj["lockupViewModel"]
                if r.get("contentType") == "LOCKUP_CONTENT_TYPE_VIDEO":
                    title = ""
                    meta = r.get("metadata", {})
                    # try nested
                    def find_title(o):
                        if isinstance(o, dict):
                            if "content" in o and isinstance(o["content"], str) and len(o["content"]) > 8:
                                return o.get("content")
                            for v in o.values():
                                found = find_title(v)
                                if found:
                                    return found
                        elif isinstance(o, list):
                            for v in o:
                                found = find_title(v)
                                if found:
                                    return found
                        return None
                    title = find_title(meta) or ""
                    pitems.append({"id": r.get("contentId"), "title": title, "lockup": True})
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
    walk(pdata)

print("pitems", len(pitems))
with open(out_txt, "a", encoding="utf-8") as tf:
    tf.write("\n--- playlist page ---\n")
    for it in pitems:
        tf.write(json.dumps(it, ensure_ascii=False) + "\n")

out = r"c:\Users\ogunn\Downloads\New folder\tmp\calc_playlist.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump({"watch": uniq, "playlist_page": pitems}, f, indent=2)
