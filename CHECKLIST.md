# Where to Put Your Local Files

Use this checklist to copy your files from your local machine into the right folders.

## RQ1 — LLM Safety (rq1-llm-safety/)

- [ ] Colab notebooks (.ipynb) → `rq1-llm-safety/notebooks/`
- [ ] Nemotron classification scripts (.py) → `rq1-llm-safety/scripts/`
- [ ] LLaMA response generation script → `rq1-llm-safety/scripts/`
- [ ] ASR computation script → `rq1-llm-safety/scripts/`
- [ ] Model/quantisation config files → `rq1-llm-safety/configs/`
- [ ] Result CSVs (ASR tables, radar plot data) → `rq1-llm-safety/results/`

## RQ2 — RAG Chatbot (rq2-rag-chatbot/)

- [ ] SimpleMovieRAG pipeline code → `rq2-rag-chatbot/src/simple_movierag/`
- [ ] GuardrailedMovieRAG pipeline code → `rq2-rag-chatbot/src/guardrailed_movierag/`
- [ ] ConversationState (three-layer memory) → `rq2-rag-chatbot/src/memory/`
- [ ] Qdrant ingestion script (TMDB→index) → `rq2-rag-chatbot/src/retrieval/`
- [ ] Search/retrieval module → `rq2-rag-chatbot/src/retrieval/`
- [ ] NeMo config.yml → `rq2-rag-chatbot/colang/`
- [ ] Colang .co rail definitions → `rq2-rag-chatbot/colang/rails/`
- [ ] Custom NeMo actions (rag_search, scrub_output) → `rq2-rag-chatbot/colang/actions/`
- [ ] 40 adversarial dialogue files (JSON/JSONL) → `rq2-rag-chatbot/data/adversarial_dialogues/`
- [ ] TMDB processing script → `rq2-rag-chatbot/data/tmdb/`
- [ ] GEval scoring script → `rq2-rag-chatbot/evaluation/`
- [ ] DeepTeam red-team script → `rq2-rag-chatbot/evaluation/`
- [ ] Human validation / Cohen's kappa script → `rq2-rag-chatbot/evaluation/`
- [ ] All result CSVs → `rq2-rag-chatbot/results/`

## RQ3 — Furhat Robot (rq3-furhat-robot/)

- [ ] Baseline Furhat main script → `rq3-furhat-robot/src/furhat_baseline/`
- [ ] Guardrailed Furhat main script → `rq3-furhat-robot/src/furhat_guardrailed/`
- [ ] Furhat utilities (ASR, TTS, gaze helpers) → `rq3-furhat-robot/src/furhat_utils/`
- [ ] ConversationState / memory module → `rq3-furhat-robot/src/memory/`
- [ ] ExperimentLogger class → `rq3-furhat-robot/src/`
- [ ] ChromaDB ingestion / load_tmdb script → `rq3-furhat-robot/src/`
- [ ] Furhat Colang config (.yml + .co files) → `rq3-furhat-robot/colang/`
- [ ] Questionnaire templates (S-TIAS, Godspeed) → `rq3-furhat-robot/survey/`
- [ ] Anonymised quantitative scores CSV → `rq3-furhat-robot/results/`
- [ ] Anonymised preference data CSV → `rq3-furhat-robot/results/`
- [ ] Anonymised session logs → `rq3-furhat-robot/results/session_logs/`

## Shared

- [ ] Architecture diagrams and plots → `figures/`
- [ ] LaTeX source (main.tex, references.bib, hwu_template/) → `dissertation/`

## DO NOT COMMIT

- Any `venv/` or virtual environment folder
- Any `chroma_db/` or `qdrant_storage/` database directories
- `.env` files or API keys
- `credentials.json` or service account files
- Raw TMDB CSV files (add download link in README instead)
- Model weight files (.bin, .pt, .safetensors)
- `__pycache__/` directories
- Any file with participant names or identifying information
