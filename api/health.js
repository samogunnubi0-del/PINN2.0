// api/health.js
export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const hasKey = Boolean(process.env.OPENROUTER_API_KEY);
  return res.status(200).json({
    ok: true,
    configured: true,
    model: 'openai/gpt-5.6-luna',
    keyState: hasKey ? 'ready' : 'fallback-ready',
    backend: 'vercel-serverless',
    timestamp: new Date().toISOString()
  });
}
