// api/daily-questions.js
import { defaultDailyPack } from './_data/defaultDailyPack.js';

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
      const timeout = setTimeout(() => controller.abort(), 25000);

      const prompt = `Create one complete 27-question Digital SAT Reading and Writing practice module for ${date}. Set ID: sat-rw-${date}.
Use standard Digital SAT sections: Q1-Q7 Craft and Structure, Q8-Q15 Information and Ideas, Q16-Q22 Standard English Conventions, Q23-Q27 Expression of Ideas.
Aim at a student scoring 540 advancing to 700+. Zero grammar ambiguities, exactly 4 plausible choices per question, 0-3 zero-indexed correct answer. Return valid JSON only with 'generated_date', 'set_id', and 'questions' array.`;

      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
          'HTTP-Referer': 'https://sat-scholarship-hub.vercel.app',
          'X-Title': 'Reading 700 Daily Coach'
        },
        signal: controller.signal,
        body: JSON.stringify({
          model: 'openai/gpt-5.6-luna',
          messages: [
            {
              role: 'system',
              content: 'You are the SAT Reading and Writing 700+ Coach. Produce rigorous, precise Digital SAT questions in valid JSON format.'
            },
            { role: 'user', content: prompt }
          ],
          reasoning: { effort: 'medium' },
          max_tokens: 18000,
          response_format: { type: 'json_object' }
        })
      });

      clearTimeout(timeout);

      if (response.ok) {
        const result = await response.json();
        const content = result.choices?.[0]?.message?.content;
        if (content) {
          const parsed = JSON.parse(content);
          if (Array.isArray(parsed.questions) && parsed.questions.length === 27) {
            return res.status(200).json({
              ok: true,
              source: 'luna-live',
              pack: parsed
            });
          }
        }
      }
    } catch (e) {
      console.warn('Live Luna generation fell back to curated daily pack:', e.message);
    }
  }

  // Fallback / standard delivery: Curated, verified 27-question pack
  const pack = {
    ...defaultDailyPack,
    generated_date: date,
    set_id: `sat-rw-${date}`
  };

  return res.status(200).json({
    ok: true,
    source: 'curated-daily',
    pack
  });
}
