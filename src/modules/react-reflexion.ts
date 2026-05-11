/**
 * ReAct reflexion memory — ReflexionMemory-style episodic learning over the
 * AgentDB store, wired into the `ReAct` module:
 *   - before acting, recall lessons from failed past attempts at a similar task
 *     (and any promoted skill plans) and inject them into the agent's prompt;
 *   - after a run, record the episode (tool sequence + outcome); a failed
 *     episode with a critique becomes a retrievable lesson;
 *   - a tool sub-strategy that succeeds `skillThreshold`+ times for a task gets
 *     promoted to a skill (mirrors `agentdb.SkillLibrary` consolidation).
 *
 * (Delegating to `agentdb.ReflexionMemory` / `SkillLibrary` directly needs a
 * sql.js `db` handle, which `AgentDBClient` doesn't currently expose — noted on
 * #8; this layers the same behaviour on the AgentDBClient vector store.)
 */
import { AgentDBClient } from '../memory/agentdb/client';

export interface ReActEpisode {
  success: boolean;
  /** The sequence of tool names invoked, in order. */
  steps: string[];
  /** A short critique / lesson — stored as a retrievable reflexion when the episode failed. */
  critique?: string;
}

export interface ReActSkill {
  steps: string[];
  uses: number;
}

export class ReActReflexion {
  private store: AgentDBClient;
  private recallK: number;
  private skillThreshold: number;

  constructor(opts: { store: AgentDBClient; recallK?: number; skillThreshold?: number }) {
    this.store = opts.store;
    this.recallK = Math.max(1, opts.recallK ?? 3);
    this.skillThreshold = Math.max(2, opts.skillThreshold ?? 2);
  }

  private key(taskKey: string): number[] {
    return this.store.hashEmbed(taskKey);
  }

  /** Lessons (from failed past episodes) and promoted skill plans for a similar task. */
  async recall(taskKey: string): Promise<{ lessons: string[]; skills: ReActSkill[] }> {
    try {
      const hits = await this.store.search(this.key(taskKey), { k: Math.max(this.recallK * 6, 16) });
      const lessons: string[] = [];
      const skills: ReActSkill[] = [];
      const seenL = new Set<string>();
      const seenS = new Set<string>();
      for (const h of hits) {
        const m = h.data.metadata;
        if (!m || h.score < 0.7) continue;
        if (m.type === 'react-reflexion' && typeof m.lesson === 'string' && m.lesson && !seenL.has(m.lesson)) {
          seenL.add(m.lesson);
          lessons.push(m.lesson);
        }
        if (m.type === 'react-skill' && Array.isArray(m.steps)) {
          const sk = m.steps.join('>');
          if (!seenS.has(sk)) {
            seenS.add(sk);
            skills.push({ steps: m.steps as string[], uses: Number(m.uses ?? 1) });
          }
        }
      }
      return { lessons: lessons.slice(0, this.recallK), skills: skills.slice(0, this.recallK) };
    } catch {
      return { lessons: [], skills: [] };
    }
  }

  /** Record an episode; promote a skill if the same successful tool sequence has now been seen `skillThreshold`+ times. */
  async recordEpisode(taskKey: string, ep: ReActEpisode): Promise<{ promotedSkill: boolean }> {
    const vec = this.key(taskKey);
    const seq = ep.steps.join('>');

    let promotedSkill = false;
    if (ep.success && ep.steps.length > 0) {
      let prior = 0;
      try {
        const hits = await this.store.search(vec, { k: 64 });
        for (const h of hits) {
          const m = h.data.metadata;
          if (m && h.score >= 0.95 && m.type === 'react-episode' && m.success && Array.isArray(m.steps) && (m.steps as string[]).join('>') === seq) prior++;
        }
      } catch {
        /* ignore */
      }
      promotedSkill = prior + 1 >= this.skillThreshold;
    }

    await this.store.store(vec, { type: 'react-episode', taskKey, success: ep.success, steps: ep.steps, critique: ep.critique ?? '', ts: Date.now() });
    if (!ep.success && ep.critique) {
      await this.store.store(vec, { type: 'react-reflexion', taskKey, lesson: ep.critique, ts: Date.now() });
    }
    if (promotedSkill) {
      await this.store.store(vec, { type: 'react-skill', taskKey, steps: ep.steps, uses: this.skillThreshold, ts: Date.now() });
    }
    return { promotedSkill };
  }

  /** Just the promoted skill plans for a similar task. */
  async getSkills(taskKey: string): Promise<ReActSkill[]> {
    return (await this.recall(taskKey)).skills;
  }

  /** Format recalled lessons as a prompt block (empty string when there are none). */
  lessonsText(lessons: string[]): string {
    if (lessons.length === 0) return '';
    return 'Lessons from past attempts:\n' + lessons.map((l, i) => `${i + 1}. ${l}`).join('\n');
  }
}
