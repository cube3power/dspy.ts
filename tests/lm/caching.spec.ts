/**
 * CachingLM (P5a) — AgentDB-backed LM cache with fuzzy embedding hits.
 */
import { CachingLM } from '../../src/lm/caching';
import { DummyLM } from '../../src/lm/dummy';
import { AgentDBClient } from '../../src/memory/agentdb/client';

function cacheStore(): AgentDBClient {
  return new AgentDBClient({
    vectorDimension: 64,
    storage: { path: '.', inMemory: true },
    performance: { maxConcurrency: 1, cacheSize: 8, batchEnabled: true },
  });
}

describe('CachingLM', () => {
  it('returns the cached response for a repeated prompt (does not re-call the LM)', async () => {
    const inner = new DummyLM();
    const lm = new CachingLM({ inner, cache: cacheStore() });
    await lm.init();
    inner.setResponse('hello world', 'A');
    expect(await lm.generate('hello world')).toBe('A'); // miss
    inner.setResponse('hello world', 'B'); // underlying changes — cache should still serve A
    expect(await lm.generate('hello world')).toBe('A'); // hit
    expect(lm.stats.hits).toBe(1);
    expect(lm.stats.misses).toBe(1);
    expect(lm.stats.hitRate).toBeCloseTo(0.5, 5);
    await lm.cleanup();
  });

  it('serves a fuzzy hit for a near-identical prompt', async () => {
    const inner = new DummyLM();
    const lm = new CachingLM({ inner, cache: cacheStore(), similarityThreshold: 0.9 });
    await lm.init();
    inner.setResponse('the quick brown fox', 'R');
    await lm.generate('the quick brown fox'); // miss, cached
    inner.setResponse('the quick brown fox ', 'OTHER'); // a different prompt to the LM…
    const out = await lm.generate('the quick brown fox '); // …but a fuzzy match to the cache
    expect(out).toBe('R');
    expect(lm.stats.hits).toBe(1);
    await lm.cleanup();
  });

  it('misses on a clearly different prompt', async () => {
    const inner = new DummyLM();
    const lm = new CachingLM({ inner, cache: cacheStore() });
    await lm.init();
    inner.setResponse('alpha', 'X');
    inner.setResponse('a totally unrelated and much longer prompt', 'Y');
    await lm.generate('alpha');
    expect(await lm.generate('a totally unrelated and much longer prompt')).toBe('Y');
    expect(lm.stats.misses).toBe(2);
    expect(lm.stats.hits).toBe(0);
    await lm.cleanup();
  });

  it('does not reuse a cache entry generated with incompatible options', async () => {
    const inner = new DummyLM();
    const lm = new CachingLM({ inner, cache: cacheStore() });
    await lm.init();
    inner.setResponse('p', 'temp-low');
    await lm.generate('p', { temperature: 0.2 }); // cached with temp 0.2
    inner.setResponse('p', 'temp-high');
    expect(await lm.generate('p', { temperature: 0.9 })).toBe('temp-high'); // miss — options differ
    inner.setResponse('p', 'temp-low-changed');
    expect(await lm.generate('p', { temperature: 0.2 })).toBe('temp-low'); // hit — same options as the first call
    await lm.cleanup();
  });

  it('respects ttlMs (expired entries are not served)', async () => {
    const inner = new DummyLM();
    const lm = new CachingLM({ inner, cache: cacheStore(), ttlMs: 5 });
    await lm.init();
    inner.setResponse('q', 'first');
    await lm.generate('q'); // cached
    await new Promise((r) => setTimeout(r, 12));
    inner.setResponse('q', 'second');
    expect(await lm.generate('q')).toBe('second'); // entry expired → miss → fresh response
    await lm.cleanup();
  });

  it('init() / cleanup() are safe even when the inner LM lacks them', async () => {
    const inner = { generate: async () => 'ok' } as { generate: () => Promise<string> };
    const lm = new CachingLM({ inner: inner as any, cache: cacheStore() });
    await expect(lm.init()).resolves.toBeUndefined();
    expect(await lm.generate('anything')).toBe('ok');
    await expect(lm.cleanup()).resolves.toBeUndefined();
  });
});
