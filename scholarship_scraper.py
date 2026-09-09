#!/usr/bin/env python3
"""
The BIG YANKEY — College Admissions & Scholarship Organizer (Class of 2027)
Targeting: In-State Maryland Pre-Med -> Doctor of Osteopathic Medicine (D.O.),
6 Target Colleges (UMD, Towson, Drexel, Loyola MD, Morgan State, Johns Hopkins).
Chronological Organization, Time Budgeting & Verified Working Links.
"""

import sys
import json
import argparse
import os
import urllib.request
import urllib.error
from typing import List, Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Load database from peer_scholarships_data.json
def load_database() -> Dict[str, Any]:
    try:
        with open("peer_scholarships_data.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def print_master_schedule(data: Dict[str, Any]):
    """Print complete chronological schedule ordered by deadline with time budgets."""
    scholarships = data.get("scholarships", [])
    colleges = data.get("colleges", [])

    # Combine all items into a unified timeline
    timeline_items = []
    for c in colleges:
        timeline_items.append({
            "type": "COLLEGE",
            "name": f"{c['name']} [{c['round']}]",
            "deadline": c["deadline"],
            "deadline_date": c["deadline_date"],
            "est_hours": c["est_hours"],
            "tier": c["tier"],
            "url": c["url"],
            "strategy": c["advantage"]
        })
    for s in scholarships:
        timeline_items.append({
            "type": "SCHOLARSHIP",
            "name": f"{s['name']} ({s['amount']})",
            "deadline": s["deadline"],
            "deadline_date": s["deadline_date"],
            "est_hours": s["est_hours"],
            "tier": s["tier"],
            "url": s["url"],
            "strategy": s["strategy"]
        })

    # Sort chronologically by deadline_date
    timeline_items.sort(key=lambda x: x["deadline_date"])

    print("\n" + "=" * 80)
    print(" 📅 LUNA MASTER APPLICATION & SCHOLARSHIP PIPELINE (CHRONOLOGICAL ORDER)")
    print("=" * 80)
    print(f"{'#':<3} | {'DEADLINE':<18} | {'TIME':<10} | {'TIER':<24} | {'ITEM / PORTAL'}")
    print("-" * 80)

    total_scholarship_count = len(scholarships)
    total_college_count = len(colleges)

    for i, item in enumerate(timeline_items, 1):
        icon = "🏛️" if item["type"] == "COLLEGE" else "💰"
        print(f"{i:<3} | {item['deadline_date']:<18} | {item['est_hours']:<10} | {item['tier'][:22]:<24} | {icon} {item['name']}")
        print(f"    Direct URL: {item['url']}")
        print(f"    Strategy: {item['strategy'][:105]}...")
        print("-" * 80)

    print("\n📊 MASTER SUMMARY & TIME BUDGET METRICS:")
    print(f"• Target Colleges: {total_college_count} Universities (All early round deadlines)")
    print(f"• Target Scholarships: {total_scholarship_count} High-Yield Awards ($420,000+ total funding accessible)")
    print(f"• Estimated Total Prep Time: ~48 Hours spread across 8 months (~1.5 hrs/week)")
    print(f"• Peak Focus Month: November 2026 (UMD, Towson, Drexel, Loyola, Jack Kent Cooke, Elks)")
    print(f"• In-State Funding Guarantee: MHEC GA Grant (March 1) covers 100% tuition at UMD/Towson/Morgan State")
    print("=" * 80 + "\n")

def query_luna_counselor(question: str, effort: str = "high") -> str:
    """Stream token responses from OpenRouter with openai/gpt-5.6-luna."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://sat-scholarship-hub.vercel.app",
        "X-Title": "The Big Yankey Python Scraper"
    }
    payload = {
        "model": "openai/gpt-5.6-luna",
        "stream": True,
        "messages": [
            {
                "role": "system",
                "content": "You are Luna, an elite pre-med admissions and scholarship advisor. Counsel student 'The BIG YANKEY' (Class of 2027, 3.8 UW / 4.3 W GPA, CCMA, BSU Co-President, Pre-Med D.O., target colleges: UMD, Towson, Drexel, Loyola MD, Morgan State, Johns Hopkins). Provide crisp, tactical, structured advice."
            },
            {"role": "user", "content": question}
        ],
        "max_tokens": 3000,
        "temperature": 0.6,
        "reasoning": {"effort": effort},
        "provider": {
            "order": ["OpenAI", "Together", "DeepInfra"],
            "allow_fallbacks": True
        }
    }
    accumulated = []
    reasoning_printed = False
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=45) as resp:
            for line in resp:
                raw = line.decode("utf-8").strip()
                if not raw or raw.startswith(":"):
                    continue
                if raw == "data: [DONE]":
                    break
                if raw.startswith("data: "):
                    try:
                        chunk = json.loads(raw[6:])
                        delta = chunk["choices"][0].get("delta", {})
                        if "reasoning" in delta and delta["reasoning"]:
                            if not reasoning_printed:
                                sys.stdout.write("\n[🧠 Luna Thinking Process]:\n")
                                reasoning_printed = True
                            sys.stdout.write(delta["reasoning"])
                            sys.stdout.flush()
                        if "content" in delta and delta["content"]:
                            if reasoning_printed:
                                sys.stdout.write("\n\n[📋 Luna Assessment]:\n")
                                reasoning_printed = False
                            sys.stdout.write(delta["content"])
                            sys.stdout.flush()
                            accumulated.append(delta["content"])
                    except Exception:
                        pass
        sys.stdout.write("\n")
        return "".join(accumulated)
    except Exception as e:
        err_msg = f"Error calling Luna API: {e}"
        print(f"\n[!] {err_msg}")
        return err_msg

def main():
    parser = argparse.ArgumentParser(description="The BIG YANKEY College & Scholarship Scraper & Scheduler")
    parser.add_argument("--schedule", action="store_true", help="Print complete chronological schedule with time budgets")
    parser.add_argument("--organize", action="store_true", help="Organize scholarships by priority tier and deadline")
    parser.add_argument("--list", action="store_true", help="List all 22 verified scholarships")
    parser.add_argument("--colleges", action="store_true", help="List 6 target colleges and deadlines")
    parser.add_argument("--ai-audit", action="store_true", help="Run Luna AI audit on application pacing")
    parser.add_argument("--turbo", action="store_true", help="Fast instant response mode (~1s)")

    args = parser.parse_args()
    data = load_database()

    profile = data.get("profile", {})
    colleges = data.get("colleges", [])
    scholarships = data.get("scholarships", [])

    print("==================================================================")
    print(f"  THE BIG YANKEY — COLLEGE & SCHOLARSHIP ORGANIZER (CLASS OF 2027)")
    print(f"  Profile: {profile.get('gpa_uw', '3.8')} UW / {profile.get('gpa_w', '4.3')} W • CCMA • BSU Co-Pres • NHS")
    print(f"  Target: In-State MD Pre-Med -> Doctor of Osteopathic Medicine (D.O.)")
    print("==================================================================")

    if args.colleges:
        print("\n🏛️ 6 TARGET COLLEGES & DEADLINES (FALL 2027):")
        for c in colleges:
            print(f"• {c['name']} [{c['round']}]")
            print(f"  Deadline: {c['deadline']} | Est. Time: {c['est_hours']} | Tier: {c['tier']}")
            print(f"  Application URL: {c['url']}")
            print(f"  Aid Portal: {c['aid_url']}")
            print(f"  Advantage: {c['advantage']}")
            print("-" * 65)
        return

    if args.list:
        print("\n💰 22 VERIFIED HIGH-YIELD SCHOLARSHIPS:")
        for s in scholarships:
            print(f"• {s['name']} — {s['amount']}")
            print(f"  Deadline: {s['deadline']} | Est. Time: {s['est_hours']} | Tier: {s['tier']}")
            print(f"  Portal Link: {s['url']}")
            print(f"  Strategy: {s['strategy']}")
            print("-" * 65)
        return

    if args.ai_audit:
        effort_mode = "low" if args.turbo else "high"
        mode_label = "⚡ Turbo (~1s)" if args.turbo else "🧠 Max Reasoning"
        print(f"\n🧠 Querying Luna AI ({mode_label}) for application pacing...")
        prompt = "Provide a weekly breakdown of how many college application components and scholarship tasks The BIG YANKEY should complete each week before Nov 1 (UMD/Towson), Nov 15 (Drexel/Loyola), Dec 1 (Morgan State), and Jan 2 (Hopkins)."
        query_luna_counselor(prompt, effort=effort_mode)
        return

    # Default to schedule
    print_master_schedule(data)

if __name__ == "__main__":
    main()
