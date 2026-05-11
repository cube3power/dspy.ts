/**
 * RetrieveModule — AgentDB-backed retrieval with optional MMR diversity (P1).
 */
import { AgentDBClient } from '../../src/memory/agentdb/client';
import { RetrieveModule } from '../../src/modules/retrieve';

async function seed(db: AgentDBClient) {
  // Two near-identical "A" passages, one "A-ish", plus a distinct "B".
  await db.store([1, 0, 0, 0], { text: 'A1' });
  await db.store([0.98, 0.05, 0, 0], { text: 'A2' });
  await db.store([0.9, 0.2, 0, 0], { text: 'A3' });
  await db.store([0.6, 0.6, 0.2, 0], { text: 'B' }); // still somewhat relevant, but more diverse
  await db.store([0, 0, 1, 0], { text: 'C' }); // ~irrelevant
}

function db(): AgentDBClient {
  return new AgentDBClient({
    vectorDimension: 4,
    storage: { path: '.', inMemory: true },
    performance: { maxConcurrency: 1, cacheSize: 8, batchEnabled: true },
  });
}

describe('RetrieveModule', () => {
  it('returns k structured passages ranked by relevance, plus a joined context', async () => {
    const client = db();
    await client.init();
    await seed(client);
    const r = new RetrieveModule({ client, k: 3, useMMR: false, overFetchFactor: 5 });
    const out = await r.run({ queryVector: [1, 0, 0, 0] });
    expect(out.passages).toHaveLength(3);
    expect(out.passages[0].text).toBe('A1'); // most relevant
    for (const p of out.passages) {
      expect(typeof p.id).toBe('string');
      expect(typeof p.text).toBe('string');
      expect(typeof p.score).toBe('number');
      expect(p.metadata).toBeDefined();
    }
    expect(out.context).toBe(out.passages.map((p) => p.text).join('\n\n'));
    expect(r.name).toBe('Retrieve');
    expect(r.strategy).toBe('Retrieve');
    await client.cleanup();
  });

  it('per-call k overrides the default', async () => {
    const client = db();
    await client.init();
    await seed(client);
    const r = new RetrieveModule({ client, k: 4, useMMR: false, overFetchFactor: 5 });
    expect((await r.run({ queryVector: [1, 0, 0, 0], k: 2 })).passages).toHaveLength(2);
    await client.cleanup();
  });

  it('the MMR path runs and still returns the top relevant passage', async () => {
    const client = db();
    await client.init();
    await seed(client);
    const r = new RetrieveModule({ client, k: 3, useMMR: true, mmrLambda: 0.5, overFetchFactor: 5 });
    const out = await r.run({ queryVector: [1, 0, 0, 0] });
    expect(out.passages.length).toBeLessThanOrEqual(3);
    expect(out.passages.map((p) => p.text)).toContain('A1');
  });

  it('MMR with a strong diversity weight pulls in a less-redundant passage', async () => {
    const client = db();
    await client.init();
    await seed(client);
    const relevance = await new RetrieveModule({ client, k: 3, useMMR: false, overFetchFactor: 5 }).run({ queryVector: [1, 0, 0, 0] });
    const diverse = await new RetrieveModule({ client, k: 3, useMMR: true, mmrLambda: 0.2, overFetchFactor: 5 }).run({ queryVector: [1, 0, 0, 0] });
    // pure relevance → the three near-identical A's; high-diversity MMR → at least one of them swapped for B/C
    expect(relevance.passages.map((p) => p.text).sort()).toEqual(['A1', 'A2', 'A3']);
    const diverseSet = new Set(diverse.passages.map((p) => p.text));
    expect(diverseSet.has('B') || diverseSet.has('C')).toBe(true);
    await client.cleanup();
  });

  it('respects a custom text field', async () => {
    const client = db();
    await client.init();
    await client.store([1, 0, 0, 0], { passage: 'hello', text: 'should-not-be-used' });
    const r = new RetrieveModule({ client, k: 1, useMMR: false, textField: 'passage' });
    expect((await r.run({ queryVector: [1, 0, 0, 0] })).passages[0].text).toBe('hello');
    await client.cleanup();
  });

  it('throws when given neither query nor queryVector', async () => {
    const client = db();
    await client.init();
    await expect(new RetrieveModule({ client }).run({})).rejects.toThrow(/query.*queryVector/i);
    await client.cleanup();
  });
});
