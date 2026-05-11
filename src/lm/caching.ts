/**
 * CachingLM — an `LMDriver` wrapper that serves `generate()` from an AgentDB
 * vector cache: prompts are keyed by an embedding, and a query is a *fuzzy* hit
 * when a cached prompt is within `similarityThreshold` cosine of it (and the
 * generation options + TTL are compatible). On a miss it calls the wrapped LM
 * and stores `(prompt embedding, response)`.
 *
 * Embedding mode `'hash'` (default) uses `AgentDBClient.hashEmbed` — deterministic
 * and dependency-free; near-identical prompts (whitespace / a changed number)
 * still hit. Mode `'model'` uses the AgentDB EmbeddingService for semantic-fuzzy
 * caching when its native deps are available (falls back to hash otherwise).
 */
import { LMDriver, GenerationOptions } from './base';
import { AgentDBClient } from '../memory/agentdb/client';

interface NormOpts {
  model?: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
}

export interface CachingLMOptions {
  inner: LMDriver;
  cache: AgentDBClient;
  /** Minimum cosine similarity for a cache hit (default 0.97). */
  similarityThreshold?: number;
  /** Time-to-live for cache entries, in ms (default: no expiry). */
  ttlMs?: number;
  /** How to embed prompts for the cache key (default 'hash'). */
  embed?: 'hash' | 'model';
}

export class CachingLM implements LMDriver {
  private inner: LMDriver;
  private cache: AgentDBClient;
  private threshold: number;
  private ttlMs?: number;
  private embedMode: 'hash' | 'model';
  private hits = 0;
  private misses = 0;

  constructor(opts: CachingLMOptions) {
    this.inner = opts.inner;
    this.cache = opts.cache;
    this.threshold = opts.similarityThreshold ?? 0.97;
    this.ttlMs = opts.ttlMs;
    this.embedMode = opts.embed ?? 'hash';
  }

  async init(): Promise<void> {
    if (this.inner.init) await this.inner.init();
    await this.cache.init();
  }

  async cleanup(): Promise<void> {
    if (this.inner.cleanup) await this.inner.cleanup();
  }

  /** Cache statistics for this wrapper instance. */
  get stats(): { hits: number; misses: number; hitRate: number } {
    const total = this.hits + this.misses;
    return { hits: this.hits, misses: this.misses, hitRate: total > 0 ? this.hits / total : 0 };
  }

  async generate(prompt: string, options?: GenerationOptions): Promise<string> {
    const emb = await this.embedPrompt(prompt);
    try {
      const cached = await this.lookup(emb, options);
      if (cached !== null) {
        this.hits++;
        return cached;
      }
    } catch {
      /* cache lookup failed — fall through to the real LM */
    }
    this.misses++;
    const response = await this.inner.generate(prompt, options);
    try {
      await this.cache.store(emb, {
        type: 'lm-cache',
        prompt: prompt.length > 2000 ? prompt.slice(0, 2000) : prompt,
        response,
        opts: this.normalize(options),
        ts: Date.now(),
      });
    } catch {
      /* best-effort caching */
    }
    return response;
  }

  private async embedPrompt(prompt: string): Promise<number[]> {
    if (this.embedMode === 'model') {
      try {
        return await this.cache.embed(prompt);
      } catch {
        /* fall back to hash */
      }
    }
    return this.cache.hashEmbed(prompt);
  }

  private async lookup(emb: number[], options?: GenerationOptions): Promise<string | null> {
    const hits = await this.cache.search(emb, { k: 5, minScore: this.threshold });
    for (const h of hits) {
      const m = h.data.metadata;
      if (!m || m.type !== 'lm-cache') continue;
      if (this.ttlMs && m.ts && Date.now() - Number(m.ts) > this.ttlMs) continue;
      if (!this.optionsCompatible(m.opts, options)) continue;
      return String(m.response ?? '');
    }
    return null;
  }

  private normalize(o?: GenerationOptions): NormOpts {
    if (!o) return {};
    return {
      model: (o as { model?: string }).model,
      temperature: o.temperature,
      maxTokens: o.maxTokens,
      topP: o.topP,
    };
  }

  private optionsCompatible(stored: unknown, query?: GenerationOptions): boolean {
    const a = (stored as NormOpts) ?? {};
    const b = this.normalize(query);
    return a.model === b.model && a.temperature === b.temperature && a.maxTokens === b.maxTokens && a.topP === b.topP;
  }
}
