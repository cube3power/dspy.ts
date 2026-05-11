/**
 * MIPROv2 optimizer (P2a) — instruction proposal + demo bootstrapping + search.
 */
import { MIPROv2, OptimizedModule } from '../../src/optimize/miprov2';
import { TrainingExample } from '../../src/optimize/base';
import { PredictModule } from '../../src/modules/predict';
import { Signature } from '../../src/core/signature';
import { configureLM, DummyLM } from '../../src/index';

const signature: Signature = {
  inputs: [{ name: 'question', type: 'string', required: true }],
  outputs: [{ name: 'answer', type: 'string', required: true }],
};

function program(): PredictModule<{ question: string }, { answer: string }> {
  return new PredictModule({
    name: 'QA',
    signature,
    promptTemplate: (i) => `Answer the question.\nQuestion: ${i.question}\nAnswer:`,
  });
}

const trainset: TrainingExample<{ question: string }, { answer: string }>[] = [
  { input: { question: 'capital of France?' }, output: { answer: 'Paris' } },
  { input: { question: '2 + 2?' }, output: { answer: '4' } },
  { input: { question: 'color of the sky?' } }, // unlabeled → bootstrap candidate
  { input: { question: 'a fruit?' } },
];

// Every parseable, non-empty answer scores 0.7 (DummyLM always returns a non-empty string).
const metric = (_i: { question: string }, o: { answer: string }) =>
  o && typeof o.answer === 'string' && o.answer.length > 0 ? 0.7 : 0;

describe('MIPROv2', () => {
  let lm: DummyLM;
  beforeEach(async () => {
    lm = new DummyLM();
    await lm.init();
    configureLM(lm);
  });

  it('compiles a Module into an OptimizedModule with an instruction and demos', async () => {
    const opt = new MIPROv2(metric, { numTrials: 4, numCandidateInstructions: 4, debug: false });
    const compiled = await opt.compile(program(), trainset);
    expect(compiled).toBeInstanceOf(OptimizedModule);
    const om = compiled as OptimizedModule<{ question: string }, { answer: string }>;
    expect(typeof om.instruction).toBe('string');
    expect(om.instruction.length).toBeGreaterThan(0);
    expect(Array.isArray(om.demos)).toBe(true);
    // the compiled module runs end-to-end
    const out = await om.run({ question: 'anything?' });
    expect(typeof out.answer).toBe('string');
  });

  it('records the search trace; the best score is ≥ every trial score', async () => {
    const opt = new MIPROv2(metric, { numTrials: 6, numCandidateInstructions: 5 });
    await opt.compile(program(), trainset);
    const r = opt.result!;
    expect(r.trials).toHaveLength(6);
    for (const t of r.trials) {
      expect(typeof t.instruction).toBe('string');
      expect(typeof t.numDemos).toBe('number');
      expect(typeof t.score).toBe('number');
      expect(r.score).toBeGreaterThanOrEqual(t.score);
    }
    // with this metric every candidate scores 0.7
    expect(r.score).toBeCloseTo(0.7, 5);
  });

  it('is deterministic for a fixed seed', async () => {
    const a = new MIPROv2(metric, { numTrials: 5, seed: 7 });
    const b = new MIPROv2(metric, { numTrials: 5, seed: 7 });
    await a.compile(program(), trainset);
    await b.compile(program(), trainset);
    expect(a.result!.trials.map((t) => t.instruction)).toEqual(b.result!.trials.map((t) => t.instruction));
    expect(a.result!.trials.map((t) => t.numDemos)).toEqual(b.result!.trials.map((t) => t.numDemos));
  });

  it('rejects a non-Module program', async () => {
    await expect(new MIPROv2(metric).compile({} as any, [])).rejects.toThrow(/Module/i);
  });

  it('save() / load() round-trips the optimized program', async () => {
    const fs = await import('node:fs');
    const os = await import('node:os');
    const path = await import('node:path');
    const file = path.join(os.tmpdir(), `miprov2-${Date.now()}.json`);
    try {
      const opt = new MIPROv2(metric, { numTrials: 3 });
      const compiled = (await opt.compile(program(), trainset)) as OptimizedModule<{ question: string }, { answer: string }>;
      opt.save(file);

      const data = JSON.parse(fs.readFileSync(file, 'utf8'));
      expect(data.program.instruction).toBe(compiled.instruction);
      expect(data.program.demos.length).toBe(compiled.demos.length);
      expect(data.result.trials).toHaveLength(3);

      const opt2 = new MIPROv2(metric);
      expect(() => opt2.load(file)).not.toThrow();
      expect(opt2.result!.trials).toHaveLength(3);
      expect(opt2.result!.instruction).toBe(compiled.instruction);
    } finally {
      try { fs.unlinkSync(file); } catch { /* ignore */ }
    }
  });

  it('OptimizedModule.buildPrompt includes the instruction and labeled demos', () => {
    const p = OptimizedModule.buildPrompt('Be precise.', [{ input: { q: 1 }, output: { a: 2 } }, { input: { q: 3 } }], { q: 9 });
    expect(p).toContain('Be precise.');
    expect(p).toContain('"q":1');
    expect(p).toContain('"a":2');
    expect(p).not.toContain('"q":3'); // unlabeled demo (no output) is dropped from the prompt
    expect(p).toContain('"q":9'); // the actual input
  });
});
