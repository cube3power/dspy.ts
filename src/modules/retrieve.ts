/**
 * RetrieveModule — a first-class DSPy.ts retrieval module over the AgentDB
 * vector store. Embeds the query (or takes a query vector directly), runs an
 * over-fetched vector search, and optionally re-ranks the results for diversity
 * with `agentdb.MMRDiversityRanker` (Maximal Marginal Relevance). Drop it into a
 * `Pipeline` ahead of a `ChainOfThought`/`Predict` step for declarative RAG:
 *
 *   Retrieve → ChainOfThought("question, context -> answer")
 */
import { Module } from '../core/module';
import { Signature } from '../core/signature';
import { AgentDBClient } from '../memory/agentdb/client';

/** A single retrieved passage. */
export interface RetrievedPassage {
  id: string;
  text: string;
  score: number;
  metadata: Record<string, any>;
}

export interface RetrieveInput extends Record<string, any> {
  /** The query text. Embedded via the AgentDB EmbeddingService. */
  query?: string;
  /** A pre-computed query vector (skips embedding — useful for tests / cached embeddings). */
  queryVector?: number[];
  /** Override the configured top-k for this call. */
  k?: number;
}

export interface RetrieveOutput extends Record<string, any> {
  /** The top-k retrieved passages. */
  passages: RetrievedPassage[];
  /** The passage texts joined by blank lines — ready to feed into a downstream module's `context`. */
  context: string;
}

export interface RetrieveOptions {
  /** The AgentDB client backing this retriever. */
  client: AgentDBClient;
  name?: string;
  /** Default number of passages to return (default 5). */
  k?: number;
  /** Metadata field holding the passage text (default `'text'`). */
  textField?: string;
  /** Apply MMR diversity re-ranking (default true). */
  useMMR?: boolean;
  /** MMR λ — 1 = pure relevance, 0 = pure diversity (default 0.7). */
  mmrLambda?: number;
  /** How many candidates to fetch before re-ranking down to k (default 3 — i.e. fetch `3k`). */
  overFetchFactor?: number;
}

const retrieveSignature: Signature = {
  inputs: [{ name: 'query', type: 'string', required: true }],
  outputs: [
    { name: 'passages', type: 'object', required: true },
    { name: 'context', type: 'string', required: true },
  ],
};

export class RetrieveModule extends Module<RetrieveInput, RetrieveOutput> {
  private readonly client: AgentDBClient;
  private readonly k: number;
  private readonly textField: string;
  private readonly useMMR: boolean;
  private readonly mmrLambda: number;
  private readonly overFetchFactor: number;

  constructor(options: RetrieveOptions) {
    super({
      name: options.name ?? 'Retrieve',
      signature: retrieveSignature,
      strategy: 'Retrieve',
    });
    this.client = options.client;
    this.k = options.k ?? 5;
    this.textField = options.textField ?? 'text';
    this.useMMR = options.useMMR ?? true;
    this.mmrLambda = options.mmrLambda ?? 0.7;
    this.overFetchFactor = Math.max(1, options.overFetchFactor ?? 3);
  }

  public async run(input: RetrieveInput): Promise<RetrieveOutput> {
    try {
      if (!input.query && !input.queryVector) {
        throw new Error('Retrieve requires either `query` (text) or `queryVector` (number[])');
      }
      const k = input.k ?? this.k;
      const queryVector = input.queryVector ?? (await this.client.embed(input.query as string));

      const hits = await this.client.search(queryVector, {
        k: Math.max(k * this.overFetchFactor, k),
        includeVectors: true,
      });

      let ranked: typeof hits = hits;
      if (this.useMMR && hits.length > k) {
        ranked = this.mmrRerank(hits, k);
      }

      const passages: RetrievedPassage[] = ranked.slice(0, k).map((h) => ({
        id: h.id,
        text: String(h.data.metadata?.[this.textField] ?? h.data.metadata?.text ?? ''),
        score: h.score,
        metadata: h.data.metadata ?? {},
      }));

      const output: RetrieveOutput = { passages, context: passages.map((p) => p.text).join('\n\n') };
      this.validateOutput(output);
      return output;
    } catch (error: any) {
      throw new Error(`Error in ${this.name}: ${error.message}`);
    }
  }

  /**
   * Maximal Marginal Relevance re-rank — `mmr(c) = λ·relevance(c) − (1−λ)·max
   * cosine(c, s) over already-selected s`. Relevance is the search score; the
   * pairwise diversity term uses the stored vectors (`includeVectors: true` is
   * set on the search). Same algorithm as `agentdb.MMRDiversityRanker`, inlined
   * here so the module has no eager `agentdb` load.
   */
  private mmrRerank(hits: Awaited<ReturnType<AgentDBClient['search']>>, k: number): typeof hits {
    if (hits.length <= k) return hits;
    const vec = (id: string): number[] => hits.find((h) => h.id === id)?.data.vector ?? [];
    const pool = [...hits].sort((a, b) => b.score - a.score);
    const selected: typeof hits = [pool.shift()!]; // first pick = highest relevance
    while (selected.length < k && pool.length > 0) {
      let bestMmr = -Infinity;
      let bestIdx = 0;
      for (let i = 0; i < pool.length; i++) {
        let maxSim = 0;
        for (const s of selected) {
          const sim = AgentDBClient.cosineSimilarity(vec(pool[i].id), vec(s.id));
          if (sim > maxSim) maxSim = sim;
        }
        const mmr = this.mmrLambda * pool[i].score - (1 - this.mmrLambda) * maxSim;
        if (mmr > bestMmr) {
          bestMmr = mmr;
          bestIdx = i;
        }
      }
      selected.push(pool.splice(bestIdx, 1)[0]);
    }
    return selected;
  }
}
