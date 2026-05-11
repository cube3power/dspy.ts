/**
 * ReAct reflexion memory (P4) — episodic learning over AgentDB: recall lessons
 * before acting, record episodes after, promote repeated successful tool
 * sequences to skills; plus the `ReAct` integration hook.
 */
import { ReActReflexion } from '../../src/modules/react-reflexion';
import { ReAct, Tool } from '../../src/modules/react';
import { AgentDBClient } from '../../src/memory/agentdb/client';
import { Signature } from '../../src/core/signature';
import { configureLM, DummyLM } from '../../src/lm'; // ReAct reads the LM registry in src/lm/base

function store(): AgentDBClient {
  return new AgentDBClient({
    vectorDimension: 64,
    storage: { path: '.', inMemory: true },
    performance: { maxConcurrency: 1, cacheSize: 8, batchEnabled: true },
  });
}

describe('ReActReflexion', () => {
  let s: AgentDBClient;
  let refl: ReActReflexion;
  beforeEach(async () => {
    s = store();
    await s.init();
    refl = new ReActReflexion({ store: s, recallK: 3, skillThreshold: 2 });
  });
  afterEach(async () => {
    await s.cleanup();
  });

  it('a failed episode with a critique becomes a recallable lesson', async () => {
    const key = JSON.stringify({ task: 'lookup' });
    await refl.recordEpisode(key, { success: false, steps: ['search'], critique: 'call calculator after search' });
    const r = await refl.recall(key);
    expect(r.lessons).toContain('call calculator after search');
    expect(r.skills).toHaveLength(0);
    expect(refl.lessonsText(r.lessons)).toContain('call calculator after search');
    expect(refl.lessonsText([])).toBe('');
  });

  it('a successful tool sequence is promoted to a skill once it recurs skillThreshold times', async () => {
    const key = JSON.stringify({ task: 'compute' });
    const a = await refl.recordEpisode(key, { success: true, steps: ['search', 'calculator'] });
    const b = await refl.recordEpisode(key, { success: true, steps: ['search', 'calculator'] });
    expect(a.promotedSkill).toBe(false);
    expect(b.promotedSkill).toBe(true);
    const skills = await refl.getSkills(key);
    expect(skills.map((sk) => sk.steps.join('>'))).toContain('search>calculator');
  });

  it('does not cross-recall lessons from a different task', async () => {
    await refl.recordEpisode(JSON.stringify({ task: 'taskA' }), { success: false, steps: ['x'], critique: 'lesson-A' });
    const other = await refl.recall(JSON.stringify({ task: 'a-completely-different-task-name-here' }));
    expect(other.lessons).not.toContain('lesson-A');
  });
});

describe('ReAct + reflexion integration', () => {
  let lm: DummyLM;
  beforeEach(async () => {
    lm = new DummyLM();
    await lm.init();
    configureLM(lm); // configures the src/lm/base registry that ReAct uses
  });

  const tools: Tool[] = [
    { name: 'search', description: 'search the web', execute: async () => 'a result' },
    { name: 'calc', description: 'evaluate math', execute: async (i) => String(i.length) },
  ];
  const sig: Signature = {
    inputs: [{ name: 'task', type: 'string', required: true }],
    outputs: [{ name: 'answer', type: 'string', required: true }],
  };

  it('records an episode after a run and injects recalled lessons into the prompt', async () => {
    const s = store();
    await s.init();
    const refl = new ReActReflexion({ store: s });
    const key = JSON.stringify({ task: 'do the thing' });
    await refl.recordEpisode(key, { success: false, steps: ['search'], critique: 'always finish with Final Answer' });

    const seen: string[] = [];
    const orig = lm.generate.bind(lm);
    lm.generate = async (p: string) => {
      seen.push(p);
      return orig(p);
    };

    const agent = new ReAct({ name: 'agent', signature: sig, tools, maxIterations: 2, reflexion: refl });
    const before = s.getStats().totalVectors;
    const out = await agent.run({ task: 'do the thing' });
    expect(Array.isArray(out.steps)).toBe(true);
    // the recalled lesson reached the thought prompt
    expect(seen.some((p) => p.includes('always finish with Final Answer'))).toBe(true);
    // a new episode (and, since it didn't reach a Final Answer with DummyLM, a reflexion) was recorded
    expect(s.getStats().totalVectors).toBeGreaterThan(before);
    await s.cleanup();
  });

  it('works without reflexion (unchanged behaviour)', async () => {
    const agent = new ReAct({ name: 'agent', signature: sig, tools, maxIterations: 2 });
    const out = await agent.run({ task: 'plain run' });
    expect(Array.isArray(out.steps)).toBe(true);
  });
});
