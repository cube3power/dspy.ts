/**
 * CompilationTracer (P5b) — optimizer-run observability: causal chains, AgentDB
 * persistence, optional MLflow (a no-op when @mlflow/tracking is absent).
 */
import { CompilationTracer } from '../../src/observability/tracer';
import { MIPROv2 } from '../../src/optimize/miprov2';
import { TrainingExample } from '../../src/optimize/base';
import { PredictModule } from '../../src/modules/predict';
import { AgentDBClient } from '../../src/memory/agentdb/client';
import { Signature } from '../../src/core/signature';
import { configureLM, DummyLM } from '../../src/index';

function store(): AgentDBClient {
  return new AgentDBClient({
    vectorDimension: 64,
    storage: { path: '.', inMemory: true },
    performance: { maxConcurrency: 1, cacheSize: 8, batchEnabled: true },
  });
}

describe('CompilationTracer', () => {
  it('records a run + a causal chain of trials and a summary', async () => {
    const t = new CompilationTracer();
    const runId = await t.startRun('demo', { a: 1, b: 'x' });
    expect(typeof runId).toBe('string');
    const t1 = await t.logTrial(runId, { label: 'first', score: 0.5 });
    const t2 = await t.logTrial(runId, { label: 'second', params: { k: 2 }, score: 0.8 });
    await t.endRun(runId, { bestScore: 0.8, trials: 2 });

    const trace = t.getTrace(runId)!;
    expect(trace.optimizer).toBe('demo');
    expect(trace.params).toEqual({ a: 1, b: 'x' });
    expect(trace.trials).toHaveLength(2);
    expect(trace.bestScore).toBe(0.8);
    expect(typeof trace.endedAt).toBe('number');

    const chain = t.causalChain(runId);
    expect(chain.map((c) => c.label)).toEqual(['first', 'second']);
    expect(chain[0].causedBy).toBeUndefined();
    expect(chain[1].causedBy).toBe(t1); // the second trial links back to the first
    expect(chain[1].id).toBe(t2);
    expect(t.runIds).toEqual([runId]);
    expect(t.runCount).toBe(1);
    expect(t.mlflowAvailable).toBe(false); // mlflow not requested
  });

  it('persists run / trial / end records when given an AgentDB store', async () => {
    const s = store();
    await s.init();
    const t = new CompilationTracer({ store: s });
    const runId = await t.startRun('opt', {});
    await t.logTrial(runId, { label: 'a', score: 1 });
    await t.logTrial(runId, { label: 'b', score: 0 });
    await t.endRun(runId, { bestScore: 1, trials: 2 });
    expect(s.getStats().totalVectors).toBe(4); // run + 2 trials + end
    await s.cleanup();
  });

  it('mlflowAvailable is false when @mlflow/tracking is not installed (no throw)', async () => {
    const t = new CompilationTracer({ mlflow: true });
    const runId = await t.startRun('m', { p: 1 });
    await t.logTrial(runId, { label: 'x', score: 0.3 });
    await t.endRun(runId, { bestScore: 0.3 });
    expect(t.mlflowAvailable).toBe(false);
  });

  it('multiple runs are tracked independently in order', async () => {
    const t = new CompilationTracer();
    const r1 = await t.startRun('A', {});
    const r2 = await t.startRun('B', {});
    await t.logTrial(r1, { label: 'a1', score: 1 });
    expect(t.runIds).toEqual([r1, r2]);
    expect(t.getTrace(r1)!.trials).toHaveLength(1);
    expect(t.getTrace(r2)!.trials).toHaveLength(0);
  });
});

describe('CompilationTracer · MIPROv2 integration', () => {
  type In = { q: string };
  type Out = { a: string };
  const signature: Signature = {
    inputs: [{ name: 'q', type: 'string', required: true }],
    outputs: [{ name: 'a', type: 'string', required: true }],
  };
  const program = (): PredictModule<In, Out> => new PredictModule<In, Out>({ name: 'QA', signature, promptTemplate: (i) => `Q: ${i.q}` });
  const trainset: TrainingExample<In, Out>[] = [
    { input: { q: '1' }, output: { a: '1' } },
    { input: { q: '2' }, output: { a: '2' } },
    { input: { q: '3' } },
  ];
  const metric = (_i: In, o: Out) => (o && typeof o.a === 'string' && o.a.length > 0 ? 0.7 : 0);

  beforeEach(async () => {
    const lm = new DummyLM();
    await lm.init();
    configureLM(lm);
  });

  it('traces a MIPROv2 compile (run + one trial per search trial + summary)', async () => {
    const tracer = new CompilationTracer();
    const opt = new MIPROv2(metric, { numTrials: 4, seed: 1, tracer });
    await opt.compile(program(), trainset);
    expect(tracer.runCount).toBe(1);
    const trace = tracer.getTrace(tracer.runIds[0])!;
    expect(trace.optimizer).toBe('MIPROv2');
    expect(trace.params.numTrials).toBe(4);
    expect(trace.trials).toHaveLength(4);
    expect(typeof trace.bestScore).toBe('number');
    // each trial carries a score; trials past the first link causally to the previous
    expect(trace.trials.every((t) => typeof t.score === 'number')).toBe(true);
    expect(trace.trials[1].causedBy).toBe(trace.trials[0].id);
  });
});
