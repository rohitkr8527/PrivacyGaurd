# PrivacyGuard Architecture Diagrams

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Training Pipeline                           │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  OpenPII Dataset │
│   (HuggingFace)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Data Preparation                │
│  - Load & process OpenPII        │
│  - Format as instruction-tuning  │
│  - Split train/val/test          │
│  - Save to data/processed/       │
└─────────────┬────────────────────┘
              │
              ▼
┌──────────────────────────────────┐
│  Modal Training (GPU Cloud)      │
│  ┌────────────────────────────┐  │
│  │  Qwen3-4B Base Model       │  │
│  │  + QLoRA Fine-tuning       │  │
│  └────────────────────────────┘  │
│                                  │
│  Configuration:                  │
│  - GPU: A100 (40GB)              │
│  - LoRA rank: 16                 │
│  - Target modules: q,k,v,o,gate │
│  - Batch size: 2 per device      │
│  - Gradient accumulation: 8      │
│  - Max seq length: 2048          │
│  - Epochs: 3                     │
│  - Learning rate: 3e-4           │
│                                  │
│  Optimizations:                  │
│  - Flash Attention 2             │
│  - 4-bit quantization            │
│  - bfloat16 training             │
└─────────────┬────────────────────┘
              │
              ▼
┌──────────────────────────────────┐
│  Trained LoRA Adapter            │
│  - Pushed to HuggingFace Hub     │
│  - Model: rohitkmr8527/          │
│    privacyguard-qwen3-4b-qlora   │
└──────────────────────────────────┘
```

## Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Evaluation Pipeline                           │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Test Datasets   │     │  Test Datasets   │     │  External        │
│                  │     │                  │     │  Benchmarks      │
│  OpenPII Test    │     │  OpenPII Test    │     │                  │
│  (3000 samples)  │     │  (3000 samples)  │     │  - Gretel (1000) │
└────────┬─────────┘     └────────┬─────────┘     │  - Nemotron(1000)│
         │                        │                └────────┬─────────┘
         │                        │                         │
         ▼                        ▼                         ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Baseline Eval    │     │ Fine-tuned Eval  │     │ Generalization   │
│ (Modal + vLLM)   │     │ (Modal + vLLM)   │     │ Evaluation       │
│                  │     │                  │     │ (Modal + vLLM)   │
│ Base Qwen3-4B    │     │ Base + LoRA      │     │                  │
│ L4 GPU           │     │ Adapter          │     │ Both models on   │
│ bfloat16         │     │ L4 GPU           │     │ external data    │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                         │
         ▼                        ▼                         ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Predictions      │     │ Predictions      │     │ Predictions      │
│ baseline_        │     │ finetuned_       │     │ *_base_*.jsonl   │
│ predictions.jsonl│     │ predictions.jsonl│     │ *_finetuned_*    │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                         │
         └────────────┬───────────┴─────────────────────────┘
                      ▼
         ┌────────────────────────────┐
         │   Metrics Calculation      │
         │                            │
         │   - Precision/Recall/F1/F2 │
         │   - PII Leakage Rate       │
         │   - Over-redaction Rate    │
         │   - Entity-level Analysis  │
         │   - Error Analysis         │
         └──────────┬─────────────────┘
                    ▼
         ┌────────────────────────────┐
         │   Results & Visualizations │
         │                            │
         │   - JSON metrics files     │
         │   - Comparison charts      │
         │   - Entity F1 breakdown    │
         │   - Generalization plots   │
         └────────────────────────────┘
```

## Demo Application (Streamlit + Modal)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Demo Application Architecture                    │
└─────────────────────────────────────────────────────────────────────┘

        User Browser
             │
             │ HTTP
             ▼
┌────────────────────────┐
│   Streamlit Frontend   │
│   (Local/Deployed)     │
│                        │
│   UI Components:       │
│   - Text input         │
│   - Model selector     │
│   - Processing button  │
│   - Results display    │
│   - JSON viewer        │
└───────────┬────────────┘
            │
            │ API Call
            ▼
┌────────────────────────┐
│  Modal Inference API   │
│  (Serverless GPU)      │
│                        │
│  ┌──────────────────┐  │
│  │  Model Loading   │  │
│  │  - Base or       │  │
│  │  - Fine-tuned    │  │
│  └──────────────────┘  │
│           │            │
│           ▼            │
│  ┌──────────────────┐  │
│  │  vLLM Engine     │  │
│  │  - L4 GPU        │  │
│  │  - bfloat16      │  │
│  │  - Flash Attn 2  │  │
│  └──────────────────┘  │
│           │            │
│           ▼            │
│  ┌──────────────────┐  │
│  │  PII Detection   │  │
│  │  - Generate JSON │  │
│  │  - Parse entities│  │
│  └──────────────────┘  │
└───────────┬────────────┘
            │
            │ JSON Response
            ▼
┌────────────────────────┐
│   Results Display      │
│                        │
│   - Redacted text      │
│   - Detected entities  │
│   - Entity types       │
│   - Processing time    │
│   - Raw JSON           │
└────────────────────────┘
```

## System Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    PrivacyGuard System                         │
└────────────────────────────────────────────────────────────────┘

Data Layer:
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   OpenPII    │  │    Gretel    │  │   Nemotron   │
│   Dataset    │  │  Benchmark   │  │  Benchmark   │
└──────────────┘  └──────────────┘  └──────────────┘
       │                 │                  │
       └─────────────────┴──────────────────┘
                         │
                         ▼
Training Layer:
┌─────────────────────────────────────────────────────┐
│  Modal GPU Cloud (training/train_modal.py)          │
│  - A100 GPU for training                            │
│  - QLoRA fine-tuning pipeline                       │
│  - Model checkpoints to HuggingFace                 │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
Model Layer:
┌──────────────────┐          ┌──────────────────┐
│  Base Model      │          │  Fine-tuned      │
│  Qwen3-4B        │          │  + LoRA Adapter  │
└──────────────────┘          └──────────────────┘
       │                              │
       └──────────────┬───────────────┘
                      │
                      ▼
Evaluation Layer:
┌─────────────────────────────────────────────────────┐
│  Modal Inference (evaluation/*_modal.py)            │
│  - L4 GPU for inference                             │
│  - vLLM for fast batch processing                   │
│  - Metrics calculation & analysis                   │
└─────────────────────────────────────────────────────┘
                      │
                      ▼
Application Layer:
┌─────────────────────────────────────────────────────┐
│  Demo Application (app/streamlit_app.py)            │
│  - Interactive UI for PII detection                 │
│  - Real-time inference via Modal API                │
│  - Visual results display                           │
└─────────────────────────────────────────────────────┘
```

## Data Flow

```
Input Text
    │
    ▼
┌────────────────────┐
│  Prompt Template   │
│  + Instructions    │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Model Inference   │
│  (Qwen3-4B + LoRA) │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  JSON Generation   │
│  {                 │
│   "entities": [    │
│     {              │
│      "entity": str │
│      "label": str  │
│     }              │
│   ],               │
│   "redacted": str  │
│  }                 │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│  Post-processing   │
│  - Parse JSON      │
│  - Validate schema │
│  - Extract results │
└─────────┬──────────┘
          │
          ▼
    Output Results
```
