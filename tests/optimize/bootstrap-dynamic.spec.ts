/**
 * BootstrapFewShot dynamic demo selection (P2c) — at run time the compiled
 * module picks the demos nearest to the current input (AgentDB vector search),
 * rather than baking a single fixed demo set into the prompt.
 */
import { BootstrapFewShot, BootstrapOptimizedModule } from '../../src/optimize/bootstrap';
import { TrainingExample } from '../../src/optimize/base';
import { PredictModule } from '../../src/modules/predict';
import { AgentDBClient } from '../../src/memory/agentdb/client';
import { Signature } from '../../src/core/signature';
import { configureLM, DummyLM } from '../../src/index';

type In = { topic: string };
type Out = { fact: string };
const signature: Signature = {
  inputs: [{ name: 'topic', type: 'string', required: true }],
  outputs: [{ name: 'fact', type: 'string', required: true }],
};
const program = (): PredictModule<In, Out> =>
  new PredictModule<In, Out>({ name: 'Facts', signature, promptTemplate: (i) => `Topic: ${i.topic}` });
const trainset: TrainingExample<In, Out>[] = [
  { input: { topic: 'France' }, output: { fact: 'Paris is the capital of France' } },
  { input: { topic: 'Italy' }, output: { fact: 'Rome is the capital of Italy' } },
  { input: { topic: 'Japan' }, output: { fact: 'Tokyo is the capital of Japan' } },
];
const metric = (_i: In, o: Out) => (o && typeof o.fact === 'string' && o.fact.length > 0 ? 1 : 0);

function store(): AgentDBClient {
  return new AgentDBClient({
    vectorDimension: 64,
    storage: { path: '.', inMemory: true },
    performance: { maxConcurrency: 1, cacheSize: 8, batchEnabled: true },
  });
}

describe('BootstrapFewShot · dynamic demo selection', () => {
  let lm: DummyLM;
  beforeEach(async () => {
    lm = new DummyLM();
    await lm.init();
    configureLM(lm);
  });

  it('indexes demos and selects the one nearest the current input', async () => {
    const s = store();
    await s.init();
    const opt = new BootstrapFewShot(metric, { maxLabeledDemos: 3, dynamicDemos: { store: s, k: 1 } });
    const compiled = (await opt.compile(program(), trainset)) as BootstrapOptimizedModule<In, Out>;
    expect(compiled).toBeInstanceOf(BootstrapOptimizedModule);
    expect(s.getStats().totalVectors).toBe(3);

    for (const topic of ['France', 'Italy', 'Japan']) {
      const picked = await compiled.selectDemos({ topic });
      expect(picked).toHaveLength(1);
      expect(picked[0].input.topic).toBe(topic);
    }
    await s.cleanup();
  });

  it("run() builds the prompt from the input-conditioned demo only", async () => {
    const s = store();
    await s.init();
    const opt = new BootstrapFewShot(metric, { maxLabeledDemos: 3, dynamicDemos: { store: s, k: 1 } });
    const compiled = (await opt.compile(program(), trainset)) as BootstrapOptimizedModule<In, Out>;
    const seen: string[] = [];
    const orig = lm.generate.bind(lm);
    lm.generate = async (p: string) => {
      seen.push(p);
      return orig(p);
    };
    await compiled.run({ topic: 'Italy' });
    expect(seen[0]).toContain('Rome');
    expect(seen[0]).not.toContain('Paris'); // the France demo is not in the prompt
    await s.cleanup();
  });

  it('without dynamicDemos, selectDemos returns the full fixed set', async () => {
    const opt = new BootstrapFewShot(metric, { maxLabeledDemos: 3 });
    const compiled = (await opt.compile(program(), trainset)) as BootstrapOptimizedModule<In, Out>;
    const all = await compiled.selectDemos({ topic: 'France' });
    expect(all).toHaveLength(3);
    await compiled.run({ topic: 'France' }); // still runs
  });

  it('rejects a non-Module program', async () => {
    await expect(new BootstrapFewShot(metric).compile({} as any, [])).rejects.toThrow(/Module/i);
  });

  it('save() / load() round-trips the fixed demo set', async () => {
    const fs = await import('node:fs');
    const os = await import('node:os');
    const path = await import('node:path');
    const file = path.join(os.tmpdir(), `bootstrap-${Date.now()}.json`);
    try {
      const opt = new BootstrapFewShot(metric, { maxLabeledDemos: 3 });
      const compiled = (await opt.compile(program(), trainset)) as BootstrapOptimizedModule<In, Out>;
      opt.save(file);
      const data = JSON.parse(fs.readFileSync(file, 'utf8'));
      expect(data.program.demos).toHaveLength(compiled.demos.length);
      const opt2 = new BootstrapFewShot(metric);
      expect(() => opt2.load(file)).not.toThrow();
    } finally {
      try { fs.unlinkSync(file); } catch { /* ignore */ }
    }
  });
});
