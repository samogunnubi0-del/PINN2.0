import json
import re

path = r"c:\Users\ogunn\Downloads\New folder\tmp\yt_playlist_raw.html"
with open(path, encoding="utf-8") as f:
    html = f.read()

# find renderer keys
keys = sorted(set(re.findall(r'"([A-Za-z]+Renderer)"', html)))
print("renderer keys containing playlist/video:")
for k in keys:
    if any(x in k.lower() for x in ["playlist", "video", "lockup", "compact"]):
        print(" ", k, html.count(k))

# playlistPanelVideoRenderer is common on watch pages
for name in ["playlistPanelVideoRenderer", "playlistVideoRenderer", "compactVideoRenderer", "lockupViewModel"]:
    print(name, html.count(name))

m = re.search(r"ytInitialData\s*=\s*(\{.*?\});</script>", html)
data = json.loads(m.group(1))

items = []

def walk(obj):
    if isinstance(obj, dict):
        if "playlistPanelVideoRenderer" in obj:
            r = obj["playlistPanelVideoRenderer"]
            vid = r.get("videoId")
            title = ""
            t = r.get("title", {})
            if "simpleText" in t:
                title = t["simpleText"]
            elif "runs" in t:
                title = "".join(x.get("text", "") for x in t["runs"])
            length = r.get("lengthText", {}).get("simpleText", "")
            idx = r.get("indexText", {}).get("simpleText", "")
            items.append({"index": idx, "id": vid, "title": title, "length": length})
        if "lockupViewModel" in obj:
            r = obj["lockupViewModel"]
            items.append({"lockup": {k: (str(v)[:80] if not isinstance(v, (dict, list)) else type(v).__name__) for k, v in r.items()}})
        for v in obj.values():
            walk(v)
    elif isinstance(obj, list):
        for v in obj:
            walk(v)

walk(data)
print("items", len(items))
print(json.dumps(items[:8], indent=2)[:3000])

# Try contents.twoColumnWatchNextResults.playlist
try:
    playlist = data["contents"]["twoColumnWatchNextResults"]["playlist"]
    print("playlist keys", playlist.keys() if isinstance(playlist, dict) else type(playlist))
    print(json.dumps(playlist, indent=2)[:2000])
except Exception as e:
    print("no twoColumn playlist", e)
    # print top keys
    print("top", list(data.keys()))
    if "contents" in data:
        print("contents", list(data["contents"].keys()) if isinstance(data["contents"], dict) else type(data["contents"]))
