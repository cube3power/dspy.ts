/**
 * GEPA example: reflective Pareto-frontier prompt evolution, with the frontier
 * persisted to AgentDB so a re-run resumes warm.
 *
 *   npx ts-node examples/gepa/index.ts
 */
import { GEPA } from '../../src/optimize/gepa';
import { TrainingExample } from '../../src/optimize/base';
import { PredictModule } from '../../src/modules/predict';
import { AgentDBClient } from '../../src/memory/agentdb/client';
import { Signature } from '../../src/core/signature';
import { configureLM, DummyLM, getLM } from '../../src/index';

type In = { question: string };
type Out = { answer: string };

const signature: Signature = {
  inputs: [{ name: 'question', type: 'string', required: true }],
  outputs: [{ name: 'answer', type: 'string', required: true }],
};

export async function evolvePrompt(): Promise<{ instruction: string; meanScore: number; frontier: number }> {
  // Swap DummyLM for a real provider (OpenAI / Anthropic / etc.) to see real evolution.
  if (!safeGetLM()) {
    const lm = new DummyLM();
    await lm.init();
    configureLM(lm);
  }

  const program = new PredictModule<In, Out>({
    name: 'QA',
    signature,
    promptTemplate: (i) => `Answer the question.\nQuestion: ${i.question}\nAnswer:`,
  });
  const trainset: TrainingExample<In, Out>[] = [
    { input: { question: 'capital of France?' }, output: { answer: 'Paris' } },
    { input: { question: '2 + 2?' }, output: { answer: '4' } },
    { input: { question: 'largest planet?' }, output: { answer: 'Jupiter' } },
  ];
  const metric = (_i: In, o: Out, e?: Out) => (o && e && typeof o.answer === 'string' && o.answer.trim() === e.answer ? 1 : o && o.answer ? 0.3 : 0);

  const store = new AgentDBClient({ vectorDimension: 64, storage: { path: './data/agentdb', inMemory: true } });
  await store.init();

  const gepa = new GEPA(metric, { numIterations: 10, mutationsPerStep: 2, frontierSize: 8, frontierStore: store });
  await gepa.compile(program, trainset);
  const r = gepa.result!;
  return { instruction: r.best.instruction, meanScore: r.best.meanScore, frontier: r.frontier.length };
}

function safeGetLM() {
  try { return getLM(); } catch { return null; }
}

if (require.main === module) {
  evolvePrompt()
    .then((r) => {
      console.log(`GEPA best meanScore: ${r.meanScore.toFixed(3)}  (frontier size ${r.frontier})`);
      console.log(`Evolved instruction:\n  ${r.instruction}`);
    })
    .catch((err) => console.error('GEPA example failed:', err.message));
}
