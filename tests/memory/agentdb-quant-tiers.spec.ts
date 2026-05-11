/**
 * AgentDBClient — RaBitQ 1-bit quantization + hierarchical memory tiers (P0b).
 */
import { AgentDBClient } from '../../src/memory/agentdb/client';

const A = [1, 1, 1, 1, 0, 0, 0, 0];
const B = [0, 0, 0, 0, 1, 1, 1, 1];
const C = [1, 1, 1, 0.1, 0, 0, 0, 0];

describe('AgentDBClient · RaBitQ quantization', () => {
  it('still returns the nearest neighbour, and reports a 32× compression ratio', async () => {
    const c = new AgentDBClient({
      vectorDimension: 8,
      storage: { path: '.', inMemory: true },
      performance: { maxConcurrency: 1, cacheSize: 8, batchEnabled: true, quantization: 'rabitq', rerankFactor: 3 },
    });
    await c.init();
    const a = await c.store(A, { l: 'a' });
    await c.store(B, { l: 'b' });
    const cc = await c.store(C, { l: 'c' });

    const r = await c.search(A, { k: 3 });
    expect(r[0].id).toBe(a);
    expect(r[0].score).toBeGreaterThan(0.99);
    // 'c' (close to A) ranks above 'b' (orthogonal)
    const order = r.map((x) => x.id);
    const bId = order.find((id) => id !== a && id !== cc)!;
    expect(order.indexOf(cc)).toBeLessThan(order.indexOf(bId));

    const info = c.quantizationInfo();
    expect(info.mode).toBe('rabitq');
    expect(info.codes).toBe(3);
    expect(info.compressionRatio).toBe((8 * 4) / 1); // float32 bytes / 1-bit-code bytes

    await c.delete(a);
    expect((await c.search(A, { k: 5 })).length).toBe(2); // bit code cleaned up too
    await c.cleanup();
  });

  it('default mode is "none" (full-precision search, no codes)', async () => {
    const c = new AgentDBClient({ vectorDimension: 8, storage: { path: '.', inMemory: true } });
    await c.init();
    await c.store(A);
    expect(c.quantizationInfo().mode).toBe('none');
    expect(c.quantizationInfo().codes).toBe(0);
    expect((await c.search(A, { k: 1 }))[0].score).toBeGreaterThan(0.99);
    await c.cleanup();
  });
});

describe('AgentDBClient · hierarchical tiers', () => {
  let c: AgentDBClient;
  beforeEach(async () => {
    c = new AgentDBClient({ vectorDimension: 8, storage: { path: '.', inMemory: true }, performance: { maxConcurrency: 1, cacheSize: 8, batchEnabled: true } });
    await c.init();
  });
  afterEach(async () => {
    await c.cleanup();
  });

  it('defaults new items to the "long" tier', async () => {
    await c.store(A);
    expect(c.tierCounts()).toEqual({ working: 0, short: 0, long: 1 });
  });

  it('stores into a chosen tier and searches tier subsets', async () => {
    const a = await c.store(A, { l: 'a' }, { tier: 'long' });
    const b = await c.store(B, { l: 'b' }, { tier: 'short' });
    const cc = await c.store(C, { l: 'c' }, { tier: 'working' });
    expect(c.tierCounts()).toEqual({ working: 1, short: 1, long: 1 });

    const r = await c.searchTiered(A, { tiers: ['long', 'working'], k: 5 });
    const ids = r.map((x) => x.id);
    expect(ids).toContain(a);
    expect(ids).toContain(cc);
    expect(ids).not.toContain(b); // 'short' excluded
  });

  it('promotes an item between tiers', async () => {
    const a = await c.store(A, {}, { tier: 'working' });
    expect(c.tierCounts().working).toBe(1);
    c.promote(a, 'long');
    expect(c.tierCounts()).toEqual({ working: 0, short: 0, long: 1 });
    c.promote('does-not-exist', 'long'); // no-op
    expect(c.tierCounts().long).toBe(1);
  });

  it('evicts a tier by count cap (oldest first) and by age', async () => {
    const first = await c.store(A, {}, { tier: 'working' });
    await new Promise((r) => setTimeout(r, 5));
    await c.store(B, {}, { tier: 'working' });
    await c.store(C, {}, { tier: 'working' });
    expect(c.tierCounts().working).toBe(3);

    const evicted = await c.evictTier('working', { max: 1 });
    expect(evicted).toBe(2);
    expect(c.tierCounts().working).toBe(1);
    // the oldest ('first') was evicted
    expect((await c.search(A, { k: 5 })).map((x) => x.id)).not.toContain(first);

    // age-based eviction: wait, then evict anything older than 5ms
    await new Promise((r) => setTimeout(r, 12));
    const gone = await c.evictTier('working', { maxAgeMs: 5 });
    expect(gone).toBe(1);
    expect(c.tierCounts().working).toBe(0);
  });
});
