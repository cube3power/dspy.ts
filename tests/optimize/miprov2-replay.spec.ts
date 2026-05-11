/**
 * MIPROv2 experience replay (P2b) — trials persisted to AgentDB; later compiles
 * of the same task fingerprint warm-start from the prior best instruction.
 */
import { MIPROv2 } from '../../src/optimize/miprov2';
import { TrainingExample } from '../../src/optimize/base';
import { PredictModule } from '../../src/modules/predict';
import { AgentDBClient } from '../../src/memory/agentdb/client';
import { Signature } from '../../src/core/signature';
import { configureLM, DummyLM } from '../../src/index';

type QA = { q: string };
type Ans = { a: string };

const signature: Signature = {
  inputs: [{ name: 'q', type: 'string', required: true }],
  outputs: [{ name: 'a', type: 'string', required: true }],
};
const program = (name = 'QA'): PredictModule<QA, Ans> =>
  new PredictModule<QA, Ans>({ name, signature, promptTemplate: (i) => `Q: ${i.q}\nA:` });
const trainset: TrainingExample<QA, Ans>[] = [
  { input: { q: 'x' }, output: { a: '1' } },
  { input: { q: 'y' }, output: { a: '2' } },
  { input: { q: 'z' } },
];
const metric = (_i: QA, o: Ans) => (o && typeof o.a === 'string' && o.a.length > 0 ? 0.7 : 0);

function store(): AgentDBClient {
  return new AgentDBClient({
    vectorDimension: 64,
    storage: { path: '.', inMemory: true },
    performance: { maxConcurrency: 1, cacheSize: 8, batchEnabled: true },
  });
}

describe('MIPROv2 · experience replay', () => {
  let lm: DummyLM;
  beforeEach(async () => {
    lm = new DummyLM();
    await lm.init();
    configureLM(lm);
  });

  it('first compile is a cold start and records one best per compile', async () => {
    const s = store();
    await s.init();
    const o1 = new MIPROv2(metric, { numTrials: 4, replayStore: s, seed: 1 });
    await o1.compile(program(), trainset);
    expect(o1.result!.warmStarted).toBe(false);
    expect(o1.result!.recalledInstructions).toBe(0);
    expect(s.getStats().totalVectors).toBe(1); // one `mipro-best` record
    await s.cleanup();
  });

  it('a later compile of the same task warm-starts from the prior best instruction', async () => {
    const s = store();
    await s.init();
    const o1 = new MIPROv2(metric, { numTrials: 4, replayStore: s, seed: 1 });
    await o1.compile(program(), trainset);
    const o2 = new MIPROv2(metric, { numTrials: 4, replayStore: s, seed: 2 });
    await o2.compile(program(), trainset);
    expect(o2.result!.warmStarted).toBe(true);
    expect(o2.result!.recalledInstructions).toBeGreaterThanOrEqual(1);
    expect(s.getStats().totalVectors).toBe(2);
    await s.cleanup();
  });

  it('a different program (different fingerprint) does not warm-start from another task', async () => {
    const s = store();
    await s.init();
    await new MIPROv2(metric, { numTrials: 3, replayStore: s, seed: 1 }).compile(program('QA'), trainset);
    const other = new MIPROv2(metric, { numTrials: 3, replayStore: s, seed: 2 });
    await other.compile(program('SUMMARIZE'), trainset);
    expect(other.result!.warmStarted).toBe(false);
    expect(other.result!.recalledInstructions).toBe(0);
    await s.cleanup();
  });

  it('without a replay store the search is always a cold start', async () => {
    const o = new MIPROv2(metric, { numTrials: 2 });
    await o.compile(program(), trainset);
    expect(o.result!.warmStarted).toBe(false);
    expect(o.result!.recalledInstructions).toBe(0);
  });

  it('AgentDBClient.hashEmbed is deterministic and dimension-fitted', async () => {
    const s = store();
    expect(s.vectorDimension).toBe(64);
    const a = s.hashEmbed('hello world');
    const b = s.hashEmbed('hello world');
    expect(a).toEqual(b);
    expect(a).toHaveLength(64);
    expect(s.hashEmbed('different')).not.toEqual(a);
  });
});
