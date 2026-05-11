/**
 * CompilationTracer — observability for optimizer runs.
 *
 * Records each compile run and its trials as a causal chain (every trial links
 * back to the one it was derived from). When an `AgentDBClient` store is given,
 * the run / trial / end records are persisted (`type: 'compile-run' | 'compile-trial'
 * | 'compile-run-end'`, with a `causedBy` edge per trial) so the chain can be
 * replayed later — the `agentdb.CausalMemoryGraph`-style view of a compilation.
 * When `mlflow: true` and `@mlflow/tracking` (an optional dependency) is present,
 * params and per-trial metrics are also logged to MLflow; absent, that's a no-op.
 *
 * (Delegating to `agentdb.CausalMemoryGraph` directly needs a sql.js `db` handle
 * that `AgentDBClient` doesn't yet expose — noted on #8.)
 */
import { AgentDBClient } from '../memory/agentdb/client';

export interface CompileRun {
  id: string;
  optimizer: string;
  params: Record<string, unknown>;
  startedAt: number;
  endedAt?: number;
  bestScore?: number;
  summary?: Record<string, unknown>;
  trials: CompileTrial[];
}

export interface CompileTrial {
  id: string;
  runId: string;
  /** The trial this one was derived from — the reflective/causal link. */
  causedBy?: string;
  label: string;
  params: Record<string, unknown>;
  score: number;
  at: number;
}

export class CompilationTracer {
  private store?: AgentDBClient;
  private useMlflow: boolean;
  private runs = new Map<string, CompileRun>();
  private order: string[] = [];
  private lastTrial = new Map<string, string>();
  private counter = 0;
  private mlflow: unknown = null;
  private mlflowResolved = false;

  constructor(opts: { store?: AgentDBClient; mlflow?: boolean } = {}) {
    this.store = opts.store;
    this.useMlflow = opts.mlflow ?? false;
  }

  private mkId(prefix: string): string {
    return `${prefix}-${Date.now().toString(36)}-${(++this.counter).toString(36)}`;
  }

  async startRun(optimizer: string, params: Record<string, unknown> = {}): Promise<string> {
    const run: CompileRun = { id: this.mkId('run'), optimizer, params, startedAt: Date.now(), trials: [] };
    this.runs.set(run.id, run);
    this.order.push(run.id);
    await this.persist(run.id, { type: 'compile-run', runId: run.id, optimizer, params, startedAt: run.startedAt });
    await this.tryMlflow((m) => {
      m.startRun?.({ runName: `${optimizer}-${run.id}` });
      for (const [k, v] of Object.entries(params)) m.logParam?.(String(k), String(v));
    });
    return run.id;
  }

  async logTrial(runId: string, trial: { label: string; params?: Record<string, unknown>; score: number }): Promise<string> {
    const run = this.runs.get(runId);
    const t: CompileTrial = {
      id: this.mkId('trial'),
      runId,
      causedBy: this.lastTrial.get(runId),
      label: trial.label,
      params: trial.params ?? {},
      score: trial.score,
      at: Date.now(),
    };
    if (run) run.trials.push(t);
    this.lastTrial.set(runId, t.id);
    await this.persist(runId, { type: 'compile-trial', trialId: t.id, runId, causedBy: t.causedBy ?? null, label: t.label, params: t.params, score: t.score, at: t.at });
    await this.tryMlflow((m) => m.logMetric?.(`${trial.label}.score`, trial.score, run ? run.trials.length : undefined));
    return t.id;
  }

  async endRun(runId: string, summary: { bestScore: number } & Record<string, unknown>): Promise<void> {
    const run = this.runs.get(runId);
    if (run) {
      run.endedAt = Date.now();
      run.bestScore = summary.bestScore;
      run.summary = summary;
    }
    await this.persist(runId, { type: 'compile-run-end', runId, ...summary, endedAt: Date.now() });
    await this.tryMlflow((m) => {
      m.logMetric?.('bestScore', summary.bestScore);
      m.endRun?.();
    });
  }

  /** The full trace (run + ordered trials) for a run id. */
  getTrace(runId: string): CompileRun | undefined {
    return this.runs.get(runId);
  }

  /** The causal chain of trials for a run, oldest first (each carries `causedBy`). */
  causalChain(runId: string): CompileTrial[] {
    return this.runs.get(runId)?.trials.slice() ?? [];
  }

  /** Run ids in start order. */
  get runIds(): string[] {
    return this.order.slice();
  }

  get runCount(): number {
    return this.runs.size;
  }

  /** Whether `@mlflow/tracking` was found (only meaningful after a run when `mlflow: true`). */
  get mlflowAvailable(): boolean {
    return this.mlflowResolved && !!this.mlflow;
  }

  private async persist(runId: string, record: Record<string, unknown>): Promise<void> {
    if (!this.store) return;
    try {
      await this.store.store(this.store.hashEmbed(`${runId}|${record.trialId ?? record.type}`), record);
    } catch {
      /* best-effort */
    }
  }

  private async tryMlflow(fn: (m: Record<string, (...args: unknown[]) => unknown>) => void): Promise<void> {
    if (!this.useMlflow) return;
    if (!this.mlflowResolved) {
      this.mlflowResolved = true;
      try {
        // Optional dependency — the specifier is cast so `tsc` doesn't try to resolve it when absent.
        this.mlflow = await import('@mlflow/tracking' as string);
      } catch {
        this.mlflow = null;
      }
    }
    if (this.mlflow) {
      try {
        const m = (this.mlflow as { default?: unknown }).default ?? this.mlflow;
        fn(m as Record<string, (...args: unknown[]) => unknown>);
      } catch {
        /* MLflow logging is best-effort */
      }
    }
  }
}
