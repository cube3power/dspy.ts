/**
 * GEPA optimizer (P3) — reflective Pareto-frontier prompt evolution.
 */
import { GEPA } from '../../src/optimize/gepa';
import { OptimizedModule } from '../../src/optimize/miprov2';
import { TrainingExample } from '../../src/optimize/base';
import { PredictModule } from '../../src/modules/predict';
import { AgentDBClient } from '../../src/memory/agentdb/client';
import { Signature } from '../../src/core/signature';
import { configureLM, DummyLM } from '../../src/index';

type In = { q: string };
type Out = { a: string };
const signature: Signature = {
  inputs: [{ name: 'q', type: 'string', required: true }],
  outputs: [{ name: 'a', type: 'string', required: true }],
};
const program = (): PredictModule<In, Out> =>
  new PredictModule<In, Out>({ name: 'QA', signature, promptTemplate: (i) => `Q: ${i.q}\nA:` });
const trainset: TrainingExample<In, Out>[] = [
  { input: { q: 'x' }, output: { a: '1' } },
  { input: { q: 'y' }, output: { a: '2' } },
  { input: { q: 'z' }, output: { a: '3' } },
];
const metric = (_i: In, o: Out) => (o && typeof o.a === 'string' && o.a.length > 0 ? 1 : 0);

function store(): AgentDBClient {
  return new AgentDBClient({
    vectorDimension: 64,
    storage: { path: '.', inMemory: true },
    performance: { maxConcurrency: 1, cacheSize: 8, batchEnabled: true },
  });
}

describe('GEPA', () => {
  let lm: DummyLM;
  beforeEach(async () => {
    lm = new DummyLM();
    await lm.init();
    configureLM(lm);
  });

  it('compiles a Module into an OptimizedModule and produces a non-empty, deduplicated Pareto frontier', async () => {
    const opt = new GEPA(metric, { numIterations: 5, mutationsPerStep: 2, frontierSize: 6, seed: 7 });
    const compiled = await opt.compile(program(), trainset);
    expect(compiled).toBeInstanceOf(OptimizedModule);
    const r = opt.result!;
    expect(r.iterations).toBe(5);
    expect(r.reflections).toHaveLength(5);
    for (const rf of r.reflections) {
      expect(typeof rf.from).toBe('string');
      expect(Array.isArray(rf.weakExamples)).toBe(true);
      expect(Array.isArray(rf.mutated)).toBe(true);
    }
    expect(r.frontier.length).toBeGreaterThan(0);
    expect(r.frontier.length).toBeLessThanOrEqual(6);
    const instr = r.frontier.map((c) => c.instruction);
    expect(new Set(instr).size).toBe(instr.length); // deduplicated
    // `best` is on the frontier and has the maximal meanScore
    expect(r.frontier).toContain(r.best);
    for (const c of r.frontier) expect(r.best.meanScore).toBeGreaterThanOrEqual(c.meanScore);
    // every candidate carries per-example scores aligned to the eval set
    for (const c of r.frontier) expect(c.perExampleScores).toHaveLength(trainset.length);
    // the compiled module runs end-to-end
    expect(typeof (await (compiled as OptimizedModule<In, Out>).run({ q: 'anything' })).a).toBe('string');
  });

  it('is deterministic for a fixed seed', async () => {
    const a = new GEPA(metric, { numIterations: 4, seed: 11 });
    const b = new GEPA(metric, { numIterations: 4, seed: 11 });
    await a.compile(program(), trainset);
    await b.compile(program(), trainset);
    expect(a.result!.reflections.map((x) => x.from)).toEqual(b.result!.reflections.map((x) => x.from));
    expect(a.result!.best.instruction).toBe(b.result!.best.instruction);
  });

  it('persists the frontier and warm-starts a later compile of the same task', async () => {
    const s = store();
    await s.init();
    const o1 = new GEPA(metric, { numIterations: 4, frontierStore: s, seed: 1 });
    await o1.compile(program(), trainset);
    expect(o1.result!.warmStarted).toBe(false);
    expect(s.getStats().totalVectors).toBeGreaterThan(0);
    const o2 = new GEPA(metric, { numIterations: 3, frontierStore: s, seed: 2 });
    await o2.compile(program(), trainset);
    expect(o2.result!.warmStarted).toBe(true);
    await s.cleanup();
  });

  it('works without a frontier store (no warm start)', async () => {
    const opt = new GEPA(metric, { numIterations: 3 });
    await opt.compile(program(), trainset);
    expect(opt.result!.warmStarted).toBe(false);
  });

  it('rejects a non-Module program', async () => {
    await expect(new GEPA(metric).compile({} as any, [])).rejects.toThrow(/Module/i);
  });

  it('save() / load() round-trips the best instruction and result', async () => {
    const fs = await import('node:fs');
    const os = await import('node:os');
    const path = await import('node:path');
    const file = path.join(os.tmpdir(), `gepa-${Date.now()}.json`);
    try {
      const opt = new GEPA(metric, { numIterations: 3, seed: 5 });
      const compiled = (await opt.compile(program(), trainset)) as OptimizedModule<In, Out>;
      opt.save(file);
      const data = JSON.parse(fs.readFileSync(file, 'utf8'));
      expect(data.program.instruction).toBe(compiled.instruction);
      expect(data.result.iterations).toBe(3);
      const opt2 = new GEPA(metric);
      expect(() => opt2.load(file)).not.toThrow();
      expect(opt2.result!.best.instruction).toBe(compiled.instruction);
    } finally {
      try { fs.unlinkSync(file); } catch { /* ignore */ }
    }
  });
});
