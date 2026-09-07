import json
import re
import urllib.request

urls = [
    "https://www.understandthemath.com/calculus-1",
    "https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=3dVaOT3pQqc&format=json",
    "https://www.youtube.com/watch?v=3dVaOT3pQqc&list=PLO1y6V1SXjjNSSOZvV3PcFu4B1S8nfXBM",
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

for url in urls:
    print("=" * 80)
    print("URL", url)
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode("utf-8", "replace")
    except Exception as e:
        print("ERR", e)
        continue
    print("len", len(data))
    ids = re.findall(r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/)|youtu\.be/)([A-Za-z0-9_-]{11})", data)
    print("ids", len(ids), list(dict.fromkeys(ids))[:40])
    if "oembed" in url:
        print(data[:500])
    if "youtube.com/watch" in url:
        # playlist initial data
        m = re.search(r"ytInitialData\s*=\s*(\{.*?\});</script>", data)
        if m:
            print("found ytInitialData", len(m.group(1)))
        vids = re.findall(r'"videoId":"([A-Za-z0-9_-]{11})"', data)
        titles = re.findall(r'"title":\{"runs":\[\{"text":"(.*?)"\}\]', data)
        print("videoIds unique", len(set(vids)))
        print("first 20 videoIds", list(dict.fromkeys(vids))[:20])
        print("titles sample", titles[:15])
        with open(r"c:\Users\ogunn\Downloads\New folder\tmp\yt_playlist_raw.html", "w", encoding="utf-8") as f:
            f.write(data)
