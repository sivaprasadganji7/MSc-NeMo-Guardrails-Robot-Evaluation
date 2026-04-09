# Study 1 (RQ1): LLM-Level Safety Filtering

**Research Question:** To what extent does integrating Nemotron as an input and output safety filter reduce the vulnerability of LLaMA-3.1-8B-Instruct to jailbreak attacks?

## Pipeline Architecture

![RQ1 Architecture](../figures/rq1_architecture_png.png)
*Three-stage evaluation pipeline: 2,037 SafeDialBench dialogues filtered through Nemotron input and output classifiers around LLaMA-3.1-8B-Instruct, scored by GPT-3.5-Turbo judge.*

## Setup

- **Environment:** Google Colab Pro+ with NVIDIA A100 GPU
- **Target model:** LLaMA-3.1-8B-Instruct
- **Safety classifier:** Nemotron Safety Guard 8B v3 (BitsAndBytes NF4 4-bit quantisation)
- **Benchmark:** SafeDialBench English subset (2,037 dialogues)
- **Judge:** GPT-3.5-Turbo (SafeDialBench framework)

## Structure

```
rq1-llm-safety/
├── notebooks/          # Colab notebooks for running SafeDialBench evaluation
├── scripts/            # Python scripts
│   ├── (Nemotron classification script)
│   ├── (ASR computation script)
│   └── (LLaMA response generation script)
├── configs/            # Model configs, quantisation settings
└── results/            # Output CSVs
    ├── (baseline ASR results)
    ├── (input filter ASR results)
    └── (combined filter ASR results)
```

## Key Results

| Condition | ASR (%) | Change |
|-----------|---------|--------|
| Baseline (no Nemotron) | 40.2 | — |
| Nemotron input only | 23.3 | -16.9 pp |
| Nemotron input+output | 13.0 | -27.2 pp |

### Nemotron Filter Effectiveness by Safety Dimension × Attack Method

![RQ1 Heatmap](../figures/fig_rq1_key_heatmap.png)
*Green = effective filtering, red = blind spots. Nemotron achieves 100% reduction for explicit attacks (Fallacy Attack, Ethics) but 0% for Fairness × Topic Change, where adversarial intent is distributed across benign turns.*

## What to Put Here

Place the following files from your local machine:

- [ ] Colab notebooks (.ipynb) → `notebooks/`
- [ ] Nemotron classification scripts → `scripts/`
- [ ] ASR computation scripts → `scripts/`
- [ ] LLaMA response generation scripts → `scripts/`
- [ ] Model config files → `configs/`
- [ ] Result CSVs (baseline, input filter, combined filter ASR) → `results/`
- [ ] Radar plot data → `results/`
