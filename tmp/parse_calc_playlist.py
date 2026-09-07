import json
import re

path = r"c:\Users\ogunn\Downloads\New folder\tmp\yt_playlist_raw.html"
with open(path, encoding="utf-8") as f:
    html = f.read()

m = re.search(r"ytInitialData\s*=\s*(\{.*?\});</script>", html)
print("ytInitialData found", bool(m), "len", len(m.group(1)) if m else 0)

items = []
if m:
    data = json.loads(m.group(1))
    # walk for playlistVideoRenderer
    def walk(obj, path=""):
        if isinstance(obj, dict):
            if "playlistVideoRenderer" in obj:
                r = obj["playlistVideoRenderer"]
                vid = r.get("videoId")
                title = ""
                t = r.get("title", {})
                if "runs" in t:
                    title = "".join(x.get("text", "") for x in t["runs"])
                elif "simpleText" in t:
                    title = t["simpleText"]
                length = ""
                lt = r.get("lengthText", {})
                if "simpleText" in lt:
                    length = lt["simpleText"]
                index = r.get("index", {})
                idx = index.get("simpleText") if isinstance(index, dict) else None
                items.append({"index": idx, "id": vid, "title": title, "length": length})
            for k, v in obj.items():
                walk(v, path + "/" + str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, path + f"[{i}]")
    walk(data)

print("items", len(items))
for it in items:
    print(f"{it['index']}\t{it['id']}\t{it['length']}\t{it['title']}")

# also dump unique playlistVideoRenderer count via regex
print("playlistVideoRenderer count", html.count("playlistVideoRenderer"))

out = r"c:\Users\ogunn\Downloads\New folder\tmp\calc_playlist.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(items, f, indent=2)
print("wrote", out)
