/**
 * ReasoningBank — real embeddings (agentdb EmbeddingService, hash fallback)
 * and semantic retrieval over the AgentDB vector index (P0c).
 */
import { AgentDBClient } from '../../src/memory/agentdb/client';
import { ReasoningBank } from '../../src/memory/reasoning-bank/bank';
import { KnowledgeUnit } from '../../src/memory/reasoning-bank/types';

function unit(id: string, pattern: string, reasoning: string[], confidence = 0.8, transferable = true, success = true): KnowledgeUnit {
  return {
    id,
    pattern,
    context: { domain: 'qa', inputFeatures: {}, conditions: {} },
    success,
    reasoning,
    transferable,
    confidence,
    usageCount: 1,
    successRate: success ? 1 : 0,
    relatedUnits: [],
    createdAt: new Date(),
    updatedAt: new Date(),
    metadata: {},
  };
}

function fresh(): { db: AgentDBClient; rb: ReasoningBank } {
  const db = new AgentDBClient({
    vectorDimension: 768,
    storage: { path: '.', inMemory: true },
    performance: { maxConcurrency: 1, cacheSize: 8, batchEnabled: true },
  });
  return { db, rb: new ReasoningBank(db, { autoEvolve: false }) };
}

describe('ReasoningBank · embeddings', () => {
  it('AgentDBClient.embed returns a vector of the configured dimension, or fails with a clear error', async () => {
    const db = new AgentDBClient({ vectorDimension: 64, storage: { path: '.', inMemory: true } });
    await db.init();
    try {
      const v = await db.embed('hello world');
      expect(Array.isArray(v)).toBe(true);
      expect(v.length).toBe(64); // fitted (padded/truncated) to vectorDimension
    } catch (err) {
      expect((err as Error).message).toMatch(/EmbeddingService not available/i);
    }
    await db.cleanup();
  });
});

describe('ReasoningBank · semantic retrieval', () => {
  let db: AgentDBClient;
  let rb: ReasoningBank;
  beforeEach(async () => {
    ({ db, rb } = fresh());
    await rb.init();
    await rb.store(unit('u1', 'retry transient http errors with exponential backoff', ['detect 5xx', 'wait 2^n s', 'retry 3x']));
    await rb.store(unit('u2', 'parse JSON safely with a schema validator', ['try parse', 'on fail return error', 'validate with zod']));
    await rb.store(unit('u3', 'cache embeddings keyed by text hash', ['hash text', 'lookup cache', 'embed on miss'], 0.4));
    await rb.store(unit('u4', 'experimental sketch', ['x'], 0.9, /*transferable*/ false, /*success*/ false));
  });
  afterEach(async () => {
    await rb.cleanup();
  });

  it('returns PatternMatches with similarity scores, capped at limit', async () => {
    const matches = await rb.retrieveSemantic('how do I handle flaky network calls', { limit: 2 });
    expect(matches.length).toBeGreaterThan(0);
    expect(matches.length).toBeLessThanOrEqual(2);
    for (const m of matches) {
      expect(m.unit).toBeDefined();
      expect(typeof m.similarity).toBe('number');
      expect(typeof m.explanation).toBe('string');
    }
  });

  it('honours minConfidence (excludes low-confidence units)', async () => {
    const matches = await rb.retrieveSemantic('caching strategy', { limit: 10, minConfidence: 0.5 });
    expect(matches.map((m) => m.unit.id)).not.toContain('u3'); // confidence 0.4
  });

  it('honours successfulOnly and transferableOnly', async () => {
    const ok = await rb.retrieveSemantic('anything', { limit: 10, successfulOnly: true });
    expect(ok.map((m) => m.unit.id)).not.toContain('u4'); // success: false
    const t = await rb.retrieveSemantic('anything', { limit: 10, transferableOnly: true });
    expect(t.map((m) => m.unit.id)).not.toContain('u4'); // transferable: false
  });

  it('the in-memory retrieve() path still works (regression)', async () => {
    const r = await rb.retrieve({ context: { domain: 'qa' }, minConfidence: 0.5, limit: 10 });
    const ids = r.map((u) => u.id).sort();
    expect(ids).toContain('u1');
    expect(ids).toContain('u2');
    expect(ids).not.toContain('u3'); // confidence 0.4 < 0.5
    expect(ids).toContain('u4'); // confidence 0.9 — retrieve() doesn't filter on success/transferable here
  });

  it('getStats reflects stored units', async () => {
    const s = rb.getStats();
    expect(s.totalUnits).toBe(4);
    expect(s.successfulUnits).toBe(3);
    expect(s.transferableUnits).toBe(3);
  });
});
