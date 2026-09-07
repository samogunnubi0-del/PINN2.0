// api/daily-evidence.js
import { defaultEvidencePack } from './_data/defaultDrillPacks.js';

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,POST');
  res.setHeader(
    'Access-Control-Allow-Headers',
    'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version, Authorization'
  );

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const payload = req.method === 'POST' ? (req.body || {}) : (req.query || {});
  const today = new Date().toISOString().slice(0, 10);
  const date = payload.date || today;
  const force = Boolean(payload.force);

  const apiKey = process.env.OPENROUTER_API_KEY;

  if (apiKey && force) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 20000);

      const prompt = `Generate a daily 5-check set for the Evidence Anchor Engine for ${date}.
Rules:
1. Provide exactly 5 evidenceDrills (including 2 quantitative chart drills and 3 textual evidence drills).
2. For quantitative drills, include valid chartData, chartAssertionPrompt, and expectedTrend ('INCREASES' or 'DECREASES').
3. Zero hallucinations, zero ambiguity, exactly one mathematically/textually correct choice per drill.
Return valid JSON only with 'evidenceDrills' array.`;

      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
          'HTTP-Referer': 'https://sat-scholarship-hub.vercel.app',
          'X-Title': 'Reading 700 Daily Evidence Anchor'
        },
        signal: controller.signal,
        body: JSON.stringify({
          model: 'openai/gpt-5.6-luna',
          messages: [
            {
              role: 'system',
              content: 'You are an elite Digital SAT Reading Coach specializing in Information & Ideas evidence analysis.'
            },
            { role: 'user', content: prompt }
          ],
          reasoning: { effort: 'medium' },
          max_tokens: 8000,
          response_format: { type: 'json_object' }
        })
      });

      clearTimeout(timeout);

      if (response.ok) {
        const result = await response.json();
        const content = result.choices?.[0]?.message?.content;
        if (content) {
          const parsed = JSON.parse(content);
          if (Array.isArray(parsed.evidenceDrills) && parsed.evidenceDrills.length === 5) {
            return res.status(200).json({
              ok: true,
              source: 'luna-live',
              pack: parsed
            });
          }
        }
      }
    } catch (e) {
      console.warn('Live Luna evidence generation fell back:', e.message);
    }
  }

  return res.status(200).json({
    ok: true,
    source: 'curated-5pack',
    pack: defaultEvidencePack
  });
}
