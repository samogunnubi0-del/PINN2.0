// api/chat.js
export default async function handler(req, res) {
  // Add CORS headers for developer convenience when testing locally
  res.setHeader('Access-Control-Allow-Credentials', true);
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader(
    'Access-Control-Allow-Headers',
    'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version, Authorization'
  );

  // Handle preflight OPTIONS request
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: 'Server misconfigured: OPENROUTER_API_KEY is not set' });
  }

  try {
    const { messages, temperature, top_p, max_tokens, model } = req.body;
    const selectedModel = model || 'openai/gpt-5.6-luna';

    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
        'HTTP-Referer': 'https://sat-scholarship-hub.vercel.app',
        'X-Title': 'SAT Coach Chat'
      },
      body: JSON.stringify({
        model: selectedModel,
        messages: messages || [],
        temperature: temperature ?? 0.7,
        top_p: top_p ?? 0.95,
        max_tokens: max_tokens ?? 16384,
        stream: false,
        reasoning: { effort: 'medium' }
      })
    });

    const responseData = await response.json();

    if (!response.ok) {
      return res.status(response.status).json({ 
        error: responseData.error || `OpenRouter API Error: ${response.statusText}` 
      });
    }

    return res.status(200).json(responseData);
  } catch (error) {
    console.error('OpenRouter proxy server error:', error);
    return res.status(500).json({ error: 'Internal Server Error: ' + error.message });
  }
}
