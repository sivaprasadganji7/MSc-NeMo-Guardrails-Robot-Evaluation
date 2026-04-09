# Study 2 (RQ2): RAG Chatbot Safety Evaluation

**Research Question:** How does the integration of NeMo Guardrails into a RAG-powered chatbot affect safety against jailbreak attacks and response relevancy in multi-turn conversations?

## Pipeline Architecture

![RQ2 Architecture](../figures/rq2_architecture_png.png)
*Full evaluation pipeline: TMDB data ingestion, synthetic adversarial dataset generation, GuardrailedMovieRAG and SimpleMovieRAG pipelines, and three-pronged evaluation (GEval, DeepTeam, human annotation).*

## Setup

- **Knowledge base:** TMDB 5000 Movies (4,803 films)
- **Vector database:** Qdrant (Docker, hybrid dense+sparse, binary quantisation)
- **Embeddings:** Snowflake Arctic Embed S + SPLADE++
- **Language model:** GPT-4o-mini via LiteLLM
- **Guardrails:** NeMo Guardrails (Colang flows + custom actions)
- **Evaluation judge:** GPT-4 (DeepEval GEval)
- **Dataset:** 40 synthetic adversarial conversations (12 turns each)

## Structure

```
rq2-rag-chatbot/
├── src/
│   ├── simple_movierag/         # Baseline RAG pipeline (no guardrails)
│   ├── guardrailed_movierag/    # Guardrailed RAG pipeline
│   ├── memory/                  # Three-layer stateful conversation memory
│   └── retrieval/               # Qdrant hybrid search + TMDB ingestion
├── colang/
│   ├── rails/                   # Input/output/topical rail definitions
│   └── actions/                 # Custom NeMo actions (rag_search, scrub_output)
├── data/
│   ├── adversarial_dialogues/   # 40 synthetic 12-turn conversations
│   └── tmdb/                    # TMDB processing scripts (see note below)
├── evaluation/                  # GEval + DeepTeam evaluation scripts
└── results/                     # Scores and safety classification data
```

### TMDB Data

Download from: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Place these files in `data/tmdb/`:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

These CSVs are not committed to Git (too large). The ingestion script in `src/retrieval/` will process them.

## Key Results

| Metric | Guardrailed | Simple | Δ |
|--------|-------------|--------|---|
| Composite safety (T3-T6) | 0.925 | 0.487 | +0.438 |
| Conversations fully safe | 82.5% | 0% | +82.5 pp |
| Early-turn relevancy | 0.812 | 0.938 | -0.125 |

### GEval Safety-Relevancy Profile

![GEval Radar](../figures/fig1_spider__1_.png)
*Guardrailed pipeline (blue) dominates on all safety categories while trading a small amount of relevancy. SimpleMovieRAG (red) collapses on roleplay (0.03).*

### GEval Scores by Turn Category

![GEval Bars](../figures/fig2_butterfly.png)
*All safety categories show significant improvement (p < .001). Roleplay sees the largest gain (+0.950). Relevant turns show a modest, non-significant decrease (-0.125).*

### Conversation-Level Safety Classification

![Conversation Safety](../figures/fig3_conv_safety.png)
*GuardrailedMovieRAG: 33/40 conversations fully safe (82.5%). SimpleMovieRAG: 0/40 (0%). Seven guardrailed failures cluster around films with hacking, war, prison escape, and moral ambiguity themes.*

### Safety-Relevancy Trade-off

![Quadrant Plot](../figures/fig5_quadrant__1_.png)
*NeMo Guardrails move the system from the "dangerous" quadrant (high relevancy, low safety) toward the ideal quadrant (high safety, high relevancy), with a modest relevancy cost of -0.125.*

### DeepTeam Attack Block Rate

![Block Rate](../figures/fig4_blockrate__1_.png)
*NeMo substantially increases blocking for all three multi-turn methods (Crescendo 89%, Linear 63%, Bad Likert 61%). Single-turn attacks (Base64, Prompt Injection, Roleplay template) show 0% blocking — Colang rails are architecturally blind to stateless attacks.*

### Guardrail Effectiveness by Movie Theme

![RQ2 Heatmap](../figures/fig_rq2_key_heatmap.png)
*Guardrails are most effective for roleplay resistance across all themes but struggle with boundary testing in war/combat and AI ethics films, where domain context provides cover for adversarial escalation.*

### Qualitative Example: The Negotiator

![Negotiator Example](../figures/rq2_negotiator_example.png)
*Side-by-side comparison showing the guardrailed pipeline refusing escalation while the simple pipeline is incrementally led to provide unsafe tactical advice about hostage-taking.*

### Judge Validation: Inter-Rater Agreement

![Kappa](../figures/fig6_kappa__1_.png)
*Human-human agreement κ = 0.900 (almost perfect). Average human-LLM agreement κ = 0.753 (substantial), consistent with MT-Bench range (0.72–0.83).*

## What to Put Here

- [ ] SimpleMovieRAG pipeline code → `src/simple_movierag/`
- [ ] GuardrailedMovieRAG pipeline code → `src/guardrailed_movierag/`
- [ ] ConversationState (three-layer memory) → `src/memory/`
- [ ] Qdrant ingestion + search scripts → `src/retrieval/`
- [ ] NeMo config.yml → `colang/`
- [ ] Colang rail definitions (.co files) → `colang/rails/`
- [ ] Custom NeMo actions (rag_search, scrub_output) → `colang/actions/`
- [ ] 40 adversarial dialogue files → `data/adversarial_dialogues/`
- [ ] TMDB processing script → `data/tmdb/`
- [ ] GEval evaluation script → `evaluation/`
- [ ] DeepTeam red-team script → `evaluation/`
- [ ] Human validation script (Cohen's kappa) → `evaluation/`
- [ ] Result CSVs → `results/`
