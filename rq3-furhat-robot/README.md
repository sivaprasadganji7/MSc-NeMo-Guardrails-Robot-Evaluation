# Study 3 (RQ3): Furhat Robot Human Evaluation

**Research Question:** How does the presence of NeMo Guardrails in a Furhat conversational robot affect user perception of trust, perceived safety, and likeability?

## System Architecture

![RQ3 Architecture](../figures/rq3_architecture_png.png)
*Shared ChromaDB + RAG + three-layer stateful memory core feeding a baseline GPT-4o-mini path and a NeMo-guardrailed GPT-4o-mini path, both connected to Furhat. Each participant interacts with both conditions (7 min each), followed by S-TIAS and Godspeed questionnaires.*

## Setup

- **Robot:** Furhat (virtual configuration via `furhat_remote_api`)
- **ASR:** Google Cloud Speech-to-Text
- **TTS:** Amazon Polly "Matthew" Neural US English
- **Knowledge base:** TMDB 5000 in ChromaDB with `text-embedding-ada-002`, top-3 retrieval
- **Language model:** GPT-4o-mini via LiteLLM
- **Guardrails:** NeMo Guardrails (Colang flows, input + output rails)
- **Participants:** N=10, within-subjects counterbalanced design
- **Measures:** S-TIAS, Godspeed Perceived Safety, Godspeed Likeability

## Structure

```
rq3-furhat-robot/
├── src/
│   ├── furhat_baseline/         # Baseline condition (GPT-4o-mini direct)
│   ├── furhat_guardrailed/      # Guardrailed condition (GPT-4o-mini via NeMo)
│   ├── furhat_utils/            # Shared Furhat API helpers (ASR, TTS, gaze)
│   └── memory/                  # ConversationState (reused from RQ2)
├── colang/                      # Furhat-specific Colang configs
├── survey/                      # Questionnaire templates
│   ├── (S-TIAS items)
│   ├── (Godspeed Perceived Safety)
│   └── (Godspeed Likeability)
└── results/
    ├── session_logs/            # Anonymised conversation transcripts
    ├── (quantitative scores CSV)
    └── (preference data CSV)
```

## Key Results

| Measure | Baseline | Guardrailed | p | r |
|---------|----------|-------------|---|---|
| S-TIAS Trust (1-7) | 6.33 | 4.53 | .0098 | .76 |
| Likeability (1-5) | 4.48 | 3.74 | .010 | .75 |
| Perceived Safety (1-5) | 3.53 | 3.33 | n.s. | ≈0 |

70% of participants preferred the baseline condition. Participants found the guardrailed condition slower (mean 16,130ms vs 5,921ms per turn) and less natural, with a single early false positive often colouring the entire session's perception.

## Study Procedure and Survey Instrument

![Consent Form and Survey](../figures/consent_form_png.png)
*Participant information sheet, consent form, demographics, and the full survey instrument used in the study — including S-TIAS (trust), Godspeed Perceived Safety and Likeability subscales, comparative preference ratings, and open-ended questions on naturalness and safety.*

## What to Put Here

- [ ] Baseline Furhat main script → `src/furhat_baseline/`
- [ ] Guardrailed Furhat main script → `src/furhat_guardrailed/`
- [ ] Furhat utility functions (ASR, TTS, gaze) → `src/furhat_utils/`
- [ ] ConversationState / memory module → `src/memory/`
- [ ] ExperimentLogger → `src/`
- [ ] ChromaDB ingestion script → `src/`
- [ ] Furhat-specific Colang configs (.yml, .co files) → `colang/`
- [ ] Questionnaire templates → `survey/`
- [ ] Anonymised quantitative scores → `results/`
- [ ] Anonymised session logs → `results/session_logs/`
- [ ] Preference data → `results/`

## Important Notes

- **Anonymise all participant data** before committing
- **Do not commit** API keys, credentials, or .env files
- **Do not commit** ChromaDB storage directory
