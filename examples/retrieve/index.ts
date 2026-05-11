/**
 * RAG example: AgentDB-backed Retrieve → ChainOfThought.
 *
 *   npx ts-node examples/retrieve/index.ts
 */
import { AgentDBClient } from '../../src/memory/agentdb/client';
import { RetrieveModule } from '../../src/modules/retrieve';

export async function buildRetriever(): Promise<RetrieveModule> {
  const client = new AgentDBClient({
    vectorDimension: 384, // matches the default agentdb embedding model
    storage: { path: './data/agentdb', inMemory: true },
  });
  await client.init();

  // Index a small corpus (uses the AgentDB EmbeddingService for text → vector).
  const corpus = [
    'DSPy.ts compiles declarative LM programs instead of hand-tuning prompts.',
    'MIPROv2 is a Bayesian optimizer that searches over instructions and few-shot demos.',
    'GEPA evolves prompts on a Pareto frontier using reflective feedback.',
    'AgentDB provides HNSW vector search, RaBitQ quantization, and a learning system.',
    'BootstrapFewShot generates demonstrations from a training set automatically.',
  ];
  for (const text of corpus) {
    try {
      await client.storeText(text);
    } catch {
      // No embedding model in this environment — skip indexing.
    }
  }

  // MMR-diversified retrieval, top-3.
  return new RetrieveModule({ client, k: 3, useMMR: true, mmrLambda: 0.7 });
}

if (require.main === module) {
  buildRetriever()
    .then(async (retrieve) => {
      const out = await retrieve.run({ query: 'how do DSPy optimizers improve prompts?' });
      console.log('Retrieved passages:');
      for (const p of out.passages) console.log(`  [${p.score.toFixed(3)}] ${p.text}`);
      console.log('\nContext for a downstream ChainOfThought("question, context -> answer"):\n');
      console.log(out.context);
    })
    .catch((err) => {
      console.error('Example failed:', err.message);
    });
}
