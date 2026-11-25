// Pont HTTP vers l'agent d'Ismail, pour le benchmark (à copier dans src/scripts/ de son repo).
//   POST /tour     {thread, message} -> un tour de l'agent offer-advisor (mêmes réglages que son interface : maxSteps 5)
//   POST /extraire {text}            -> l'agent profile-extractor sur une conversation complète
// Lancement :  npx tsx --env-file=.env src/scripts/bench-bridge.ts
import http from 'node:http';
import { mastra } from '../mastra';
import { profileSchema } from '../mastra/agents/profile-extractor';
import { getProfile } from '../mastra/profile';

const advisor = mastra.getAgent('offerAdvisorAgent');
const extractor = mastra.getAgent('profileExtractorAgent');

type ToolResult = { name: string; result: any };

function toolResults(res: any): ToolResult[] {
  const raw: any[] = [...(res.steps ?? []).flatMap((s: any) => s.toolResults ?? []), ...(res.toolResults ?? [])];
  const seen = new Set<any>();
  const out: ToolResult[] = [];
  for (const tr of raw) {
    const id = tr?.payload?.toolCallId ?? tr?.toolCallId ?? tr;
    if (seen.has(id)) continue;
    seen.add(id);
    out.push({ name: tr?.payload?.toolName ?? tr?.toolName ?? '', result: tr?.payload?.result ?? tr?.result ?? tr?.output });
  }
  return out;
}

async function tour(thread: string, message: string) {
  const t = performance.now();
  const res: any = await advisor.generate(message, { memory: { thread, resource: 'benchmark' }, maxSteps: 5 } as any);
  const results = toolResults(res);
  const update = results.filter((r) => /update/i.test(r.name)).pop()?.result;
  const search = results.filter((r) => /search/i.test(r.name)).pop()?.result;
  return {
    text: res.text ?? '',
    tools: results.map((r) => r.name),
    missing: update?.missing ?? null,
    rejected: update?.rejected ?? [],
    search: search
      ? {
          status: search.status,
          refusal: search.refusal,
          offers: (search.offers ?? []).map((o: any) => ({ insurer: o.insurer, name: o.name, monthlyPremium: o.monthlyPremium })),
        }
      : null,
    profile: getProfile(thread),
    ms: Math.round(performance.now() - t),
  };
}

async function extraire(text: string) {
  const t = performance.now();
  const res: any = await extractor.generate(text, { structuredOutput: { schema: profileSchema } } as any);
  return { profile: res.object ?? {}, ms: Math.round(performance.now() - t) };
}

http
  .createServer(async (req, rep) => {
    let body = '';
    for await (const chunk of req) body += chunk;
    try {
      const data = body ? JSON.parse(body) : {};
      const out = req.url === '/tour' ? await tour(data.thread, data.message) : req.url === '/extraire' ? await extraire(data.text) : null;
      rep.writeHead(out ? 200 : 404, { 'Content-Type': 'application/json' });
      rep.end(JSON.stringify(out ?? { error: 'route inconnue' }));
    } catch (err) {
      console.error(err);
      rep.writeHead(500, { 'Content-Type': 'application/json' });
      rep.end(JSON.stringify({ error: String((err as Error)?.message ?? err) }));
    }
  })
  .listen(4200, () => console.log('Pont benchmark prêt sur http://localhost:4200'));
