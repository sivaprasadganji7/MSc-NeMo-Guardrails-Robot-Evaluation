# NeMo Guardrails in LLM, RAG Chatbot, and Virtual Furhat Robot Systems

**A Comparative Analysis**

> MSc Robotics Dissertation — Heriot-Watt University, 2026
> **Author:** Siva Prasad Ganji (H00476424)
> **Supervisor:** Dr Idris Skloul Ibrahim
> **Course:** F21MP — Masters Project and Dissertation

---

## Overview

This repository contains all code, configurations, evaluation scripts, and results for a three-study dissertation evaluating NVIDIA NeMo Guardrails across progressively complex deployment contexts.

| Study | Focus | What it evaluates |
|-------|-------|-------------------|
| **RQ1** | LLM-Level Safety | Nemotron as input/output filter for LLaMA-3.1-8B-Instruct on SafeDialBench (2,037 dialogues) |
| **RQ2** | System-Level Safety | NeMo Guardrails in a movie-domain RAG chatbot under multi-turn adversarial pressure (40 conversations) |
| **RQ3** | Human-Level Safety | Within-subjects Furhat robot study (N=10) measuring trust, perceived safety, and likeability |

---

## Key Results

| Study | Metric | Without Guardrails | With Guardrails | Change |
|-------|--------|-------------------|-----------------|--------|
| RQ1 | Attack Success Rate | 40.2% | 13.0% | −27.2 pp |
| RQ2 | Composite Safety Score | 0.487 | 0.925 | +0.438 |
| RQ2 | Conversations Fully Safe | 0% | 82.5% | +82.5 pp |
| RQ3 | S-TIAS Trust (1–7) | 6.33 | 4.53 | −1.80 (p=.0098) |
| RQ3 | Likeability (1–5) | 4.48 | 3.74 | −0.74 (p=.010) |
| RQ3 | Perceived Safety (1–5) | 3.53 | 3.33 | n.s. |

> **Key finding:** Guardrails consistently improved automated safety metrics at every level, but did not improve — and in some cases significantly reduced — user perception of trust and likeability in the embodied robot setting.

---

## System Architectures

### RQ1: LLM-Level Safety Pipeline
![RQ1 Architecture](figures/rq1_architecture_png.png)
*Post-hoc Nemotron safety classification on SafeDialBench: 2,037 dialogues filtered through input and output classifiers around LLaMA-3.1-8B-Instruct, scored by GPT-3.5-Turbo judge.*

### RQ2: RAG Chatbot Evaluation Pipeline
![RQ2 Architecture](figures/rq2_architecture_png.png)
*GuardrailedMovieRAG vs SimpleMovieRAG: hybrid Qdrant retrieval, NeMo Colang rails, and three-pronged evaluation (GEval, DeepTeam, human annotation).*

### RQ3: Furhat Robot Architecture
![RQ3 Architecture](figures/rq3_architecture_png.png)
*Shared ChromaDB + RAG + memory core feeding baseline and NeMo-guardrailed paths, both connected to Furhat for within-subjects human evaluation.*

---

## Mind Map

![Mind Map](figures/Mind_Map.png)
*Overview of the dissertation structure and the relationships between the three studies.*

---

## Repository Structure

```
.
├── rq1-llm-safety/              # Study 1: Nemotron + LLaMA on SafeDialBench
│   ├── notebooks/               # Colab notebooks (.ipynb)
│   ├── scripts/                 # Classification & ASR computation scripts
│   ├── configs/                 # Model & quantisation settings
│   └── results/                 # Output JSONLs and result files
│
├── rq2-rag-chatbot/             # Study 2: MovieRAG ± NeMo Guardrails
│   ├── src/                     # Pipeline code (simple + guardrailed drivers)
│   ├── colang/                  # NeMo Colang flows & custom actions
│   ├── data/                    # Adversarial dialogues + TMDB scripts
│   ├── evaluation/              # GEval + DeepTeam evaluation scripts
│   └── results/                 # Scores and safety classification CSVs
│
├── rq3-furhat-robot/            # Study 3: Furhat robot human evaluation
│   ├── src/                     # Baseline & guardrailed Furhat scripts
│   ├── colang/                  # Furhat-specific Colang configs
│   ├── survey/                  # S-TIAS & Godspeed questionnaire files
│   └── results/                 # Session logs and anonymised scores
│
├── figures/                     # Architecture diagrams and result plots
├── dissertation/                # Submitted dissertation PDF
├── requirements.txt             # Python dependencies
├── CHECKLIST.md                 # Project checklist
└── LICENSE                      # Licence file
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Docker (for Qdrant in RQ2)
- Google Colab Pro+ with A100 GPU (for RQ1)
- Furhat SDK / `furhat_remote_api` (for RQ3)
- API keys: OpenAI, Google Cloud Speech-to-Text, Amazon Polly

### Installation

```bash
git clone https://github.com/sivaprasadganji7/MSc-NeMo-Guardrails-Robot-Evaluation.git
cd MSc-NeMo-Guardrails-Robot-Evaluation
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory — **never commit this file**:

```
OPENAI_API_KEY=your_key_here
GOOGLE_CLOUD_CREDENTIALS=path/to/credentials.json
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

---

## Running Each Study

See the README inside each study folder for detailed setup and execution instructions:

- [`rq1-llm-safety/README.md`](rq1-llm-safety/README.md)
- [`rq2-rag-chatbot/README.md`](rq2-rag-chatbot/README.md)
- [`rq3-furhat-robot/README.md`](rq3-furhat-robot/README.md)

---

## Figures

| Figure | Description |
|--------|-------------|
| ![](figures/rq1_architecture_png.png) | RQ1 three-stage evaluation pipeline |
| ![](figures/rq2_architecture_png.png) | RQ2 GuardrailedMovieRAG vs SimpleMovieRAG |
| ![](figures/rq3_architecture_png.png) | RQ3 Furhat robot system architecture |
| ![](figures/fig_rq1_key_heatmap.png) | RQ1 ASR reduction heatmap by dimension × attack method |
| ![](figures/rq2_geval_bars.png) | RQ2 GEval scores by turn category |
| ![](figures/rq2_conv_safety.png) | RQ2 conversation-level safety scores per movie |
| ![](figures/rq2_negotiator_example.png) | RQ2 qualitative pipeline comparison (The Negotiator) |

---

## Citation

```bibtex
@mastersthesis{ganji2026nemo,
  title     = {NeMo Guardrails in LLM, RAG Chatbot, and Virtual Furhat Robot Systems: A Comparative Analysis},
  author    = {Ganji, Siva Prasad},
  year      = {2026},
  school    = {Heriot-Watt University},
  type      = {MSc Dissertation},
  program   = {MSc Robotics}
}
```

---

## Ethics

This project involved human participants (RQ3). Ethical approval was obtained through Heriot-Watt University. All participant session logs in this repository are anonymised (PP01–PP10). No personally identifiable information is stored.

---

## License

See [LICENSE](LICENSE) for details.
