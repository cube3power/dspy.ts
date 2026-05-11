/**
 * DSPy.ts — public API.
 *
 * Re-exports the core types, modules, optimizers, AgentDB-backed memory layer,
 * LM drivers, and observability so consumers can `import { … } from 'dspy.ts'`
 * without reaching into `dist/` subpaths.
 */
import { LMDriver, LMError } from './lm/base';

// Global LM registry — the one this package's modules and optimizers read from.
let globalLM: LMDriver | null = null;

/** Configure the global language model driver. */
export function configureLM(lm: LMDriver): void {
  globalLM = lm;
}

/**
 * Get the currently configured LM driver.
 * @throws {LMError} if no LM is configured.
 */
export function getLM(): LMDriver {
  if (!globalLM) {
    throw new LMError('No language model configured. Call configureLM() first.');
  }
  return globalLM;
}

// --- barrel re-exports -----------------------------------------------------
export * from './core'; // Module, Signature, Pipeline, factory helpers
export * from './lm'; // DummyLM, CachingLM, OnnxLM, TorchLM, providers, LM types
export * from './memory'; // AgentDBClient, ReasoningBank, SAFLA, MemoryTier, …
export * from './modules'; // PredictModule, ChainOfThoughtModule, ReAct, ReActReflexion, RetrieveModule, …
export * from './optimize'; // Optimizer, BootstrapFewShot, MIPROv2, GEPA, OptimizedModule, …
export * from './observability'; // CompilationTracer
