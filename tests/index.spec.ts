import * as dspy from '../src/index';
import { configureLM, getLM, LMError } from '../src/index';
import { DummyLM } from '../src/lm/dummy';

describe('package barrel exports', () => {
  it('re-exports the core types and the LM registry', () => {
    expect(typeof dspy.configureLM).toBe('function');
    expect(typeof dspy.getLM).toBe('function');
    expect(typeof dspy.Module).toBe('function');
    expect(typeof dspy.Pipeline).toBe('function');
    expect(typeof dspy.DummyLM).toBe('function');
  });

  it('re-exports the modules', () => {
    expect(typeof dspy.PredictModule).toBe('function');
    expect(typeof dspy.ChainOfThought).toBe('function');
    expect(typeof dspy.ReAct).toBe('function');
    expect(typeof dspy.ReActReflexion).toBe('function');
    expect(typeof dspy.RetrieveModule).toBe('function');
  });

  it('re-exports the optimizers', () => {
    expect(typeof dspy.Optimizer).toBe('function');
    expect(typeof dspy.BootstrapFewShot).toBe('function');
    expect(typeof dspy.BootstrapOptimizedModule).toBe('function');
    expect(typeof dspy.MIPROv2).toBe('function');
    expect(typeof dspy.GEPA).toBe('function');
    expect(typeof dspy.OptimizedModule).toBe('function');
  });

  it('re-exports the AgentDB memory layer, the caching LM, and the tracer', () => {
    expect(typeof dspy.AgentDBClient).toBe('function');
    expect(typeof dspy.ReasoningBank).toBe('function');
    expect(typeof dspy.CachingLM).toBe('function');
    expect(typeof dspy.CompilationTracer).toBe('function');
  });
});

describe('Global LM Configuration', () => {
  afterEach(() => {
    // Reset global LM after each test
    configureLM(null as any);
  });

  it('should throw error when LM is not configured', () => {
    expect(() => getLM()).toThrow(LMError);
  });

  it('should allow configuring and retrieving LM', async () => {
    const dummyLM = new DummyLM();
    await dummyLM.init();
    
    configureLM(dummyLM);
    expect(getLM()).toBe(dummyLM);
  });

  it('should work with custom responses', async () => {
    const dummyLM = new DummyLM(new Map([
      ['test', 'response']
    ]));
    await dummyLM.init();
    
    configureLM(dummyLM);
    const lm = getLM();
    const response = await lm.generate('test');
    expect(response).toBe('response');
  });
});
