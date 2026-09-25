# PrivacyGuard — Architecture Diagrams

> All diagrams use [Mermaid](https://mermaid.js.org/) and render natively on GitHub and most modern Markdown viewers.

---

## 1. System Architecture Overview

High-level view of the four layers that make up PrivacyGuard.

```mermaid
flowchart TD
    subgraph DATA["Data Layer"]
        D1["OpenPII Dataset\n(HuggingFace)"]
        D2["Gretel Benchmark\n(1 000 samples)"]
        D3["Nemotron Benchmark\n(1 000 samples)"]
    end

    subgraph TRAIN["Training Layer"]
        T1["Modal GPU Cloud\ntrain_modal.py"]
        T2["Qwen3-4B Base Model\n+ QLoRA Fine-tuning\nL4 · 40 GB"]
        T3["LoRA Adapter\nHuggingFace Hub\nrohitkmr8527/privacyguard-qwen3-4b-qlora"]
    end

    subgraph MODEL["Model Layer"]
        M1["Base Model\nQwen3-4B"]
        M2["Fine-tuned Model\nQwen3-4B + LoRA Adapter"]
    end

    subgraph EVAL["Evaluation Layer"]
        E1["Modal Inference\nvLLM · L4 GPU"]
        E2["Metrics & Analysis\nPrecision · Recall · F1 · F2\nLeakage Rate · Over-redaction"]
    end

    subgraph APP["Application Layer"]
        A1["Streamlit UI\nstreamlit_app.py"]
        A2["Modal Serverless Inference\nmodal_inference.py"]
    end

    D1 --> T1
    T1 --> T2
    T2 --> T3
    T3 --> M2
    D1 --> M1

    M1 --> E1
    M2 --> E1
    D2 --> E1
    D3 --> E1
    E1 --> E2

    M1 --> A2
    M2 --> A2
    A1 --> A2
```

---

## 2. Training Pipeline

End-to-end flow from raw data to a published LoRA adapter.

```mermaid
flowchart TD
    A["OpenPII Dataset\nai4privacy/OpenPII · HuggingFace"]
    B["Data Preparation\nsrc/data/prepare_dataset.py\n─────────────────────\nLoad & format as instruction-tuning pairs\nSplit → train 10 000 / val 1 000 / test 3 000\nSave to data/processed/"]
    C["Modal Training Job\ntraining/train_modal.py\n─────────────────────\nGPU: L4 40 GB\nBase: Qwen/Qwen3-4B\nMethod: QLoRA 4-bit\nLoRA rank: 16\nTarget: q·k·v·o·gate·up·down proj\nBatch: 2 × 8 grad-accum = 16 effective\nLR: 3e-4 · Epochs: 3 · Max len: 2 048\nOptimiser: paged_adamw_8bit\nScheduler: cosine · warmup 3%\nExtra: Flash Attention 2 · bfloat16 · grad-ckpt"]
    D["LoRA Adapter Published\nrohitkmr8527/privacyguard-qwen3-4b-qlora\n~3 h training time · ~$3–5 compute cost"]

    A --> B --> C --> D
```

---

## 3. Evaluation Pipeline

Parallel evaluation of base and fine-tuned models across three datasets.

```mermaid
flowchart TD
    subgraph DATASETS["Test Datasets"]
        DS1["OpenPII Test\n3 000 samples"]
        DS2["Gretel Benchmark\n1 000 samples"]
        DS3["Nemotron Benchmark\n1 000 samples"]
    end

    subgraph INFERENCE["Modal Inference — L4 GPU · vLLM · bfloat16"]
        I1["Baseline Eval\nbaseline_modal.py\nBase Qwen3-4B"]
        I2["Fine-tuned Eval\nfinetuned_modal.py\nQwen3-4B + LoRA"]
        I3["Generalisation Eval\nexternal_eval_modal.py\nBoth models on external data"]
    end

    subgraph PREDS["Prediction Files"]
        P1["baseline_predictions.jsonl"]
        P2["finetuned_predictions.jsonl"]
        P3["*_base_*.jsonl\n*_finetuned_*.jsonl"]
    end

    subgraph METRICS["Metrics Calculation\nmetrics.py"]
        M1["Precision / Recall / F1 / F2"]
        M2["PII Leakage Rate"]
        M3["Over-redaction Rate"]
        M4["Entity-level F1 Breakdown"]
        M5["JSON Validity Rate"]
    end

    OUT["Results & Visualisations\nresults/ · results/figures/\nJSON metrics · comparison charts\nentity F1 breakdown · generalisation plots"]

    DS1 --> I1 & I2
    DS2 & DS3 --> I3

    I1 --> P1
    I2 --> P2
    I3 --> P3

    P1 & P2 & P3 --> M1 & M2 & M3 & M4 & M5
    M1 & M2 & M3 & M4 & M5 --> OUT
```

---

## 4. Demo Application Architecture

Request / response flow from the browser through to GPU inference.

```mermaid
flowchart TD
    U["User Browser"]

    subgraph FE["Streamlit Frontend\napp/streamlit_app.py · localhost:8501"]
        UI1["Text Input"]
        UI2["Model Selector\nBase vs Fine-tuned"]
        UI3["Process Button"]
        UI4["Results Display\nRedacted text · Entities · JSON viewer"]
    end

    subgraph MODAL["Modal Serverless Inference\napp/modal_inference.py"]
        ML["Model Loader\nBase Qwen3-4B\nor + LoRA Adapter"]
        VL["vLLM Engine\nL4 GPU · bfloat16 · Flash Attention 2"]
        PII["PII Detector\nGenerate JSON · Parse entities"]
    end

    U -->|"HTTP"| UI1
    UI1 & UI2 --> UI3
    UI3 -->|"API call"| ML
    ML --> VL --> PII
    PII -->|"JSON response"| UI4
    UI4 --> U
```

---

## 5. Inference Data Flow

Step-by-step transformation of raw input text into structured PII results.

```mermaid
flowchart TD
    IN["Raw Input Text\ne.g. 'Call John at 555-0100'"]
    PT["Prompt Construction\nSystem instruction + few-shot format\n+ input text → tokenised sequence"]
    INF["Model Inference\nQwen3-4B + LoRA Adapter\nvLLM · temperature 0.0 · max 384 tokens"]
    GEN["JSON Generation\n{\n  entities: [ {entity, label}, ... ],\n  redacted: 'Call {{PERSON}} at {{PHONENUMBER}}'\n}"]
    PP["Post-processing\nParse JSON · validate schema\nextract entities · handle failures"]
    OUT["Structured Output\nDetected entities + labels\nRedacted text · Processing stats"]

    IN --> PT --> INF --> GEN --> PP --> OUT
```

---

## 6. QLoRA Fine-tuning Detail

Internal mechanics of the quantisation and adapter training process.

```mermaid
flowchart TD
    BM["Qwen3-4B Base Weights\nFrozen — loaded in 4-bit NormalFloat"]

    subgraph QLORA["QLoRA Adapter Training"]
        direction TB
        Q["4-bit Quantisation\nNF4 · double quantisation\nbfloat16 compute dtype"]
        LA["Low-Rank Adapters\nRank 16 · Alpha 32\nTarget: q·k·v·o·gate·up·down proj"]
        GC["Gradient Checkpointing\n+ Flash Attention 2\npaged_adamw_8bit"]
    end

    HUB["HuggingFace Hub\nrohitkmr8527/privacyguard-qwen3-4b-qlora\n~1% of total parameters trained"]

    BM --> Q --> LA --> GC --> HUB
```

---

## 7. Metrics & Evaluation Framework

Relationships between predictions, ground truth, and computed metrics.

```mermaid
flowchart TD
    subgraph INPUTS["Inputs"]
        GT["Ground-truth Entities\n(true labels)"]
        PR["Model Predictions\n(predicted labels)"]
    end

    subgraph COUNTS["Token-level Matching"]
        TP["True Positives\ncorrectly detected PII"]
        FP["False Positives\nover-redaction"]
        FN["False Negatives\nPII leakage"]
    end

    subgraph METRICS["Derived Metrics"]
        F1["F1 Score\n2·P·R / (P+R)"]
        F2["F2 Score\nweights recall × 2"]
        LR["PII Leakage Rate\nFN / total true entities"]
        OR["Over-redaction Rate\nFP / total predicted entities"]
        JV["JSON Validity\n% parseable responses"]
    end

    GT & PR --> TP & FP & FN
    TP & FP --> F1 & F2 & OR
    TP & FN --> F1 & F2 & LR
    PR --> JV
```

---

## 8. Project Module Map

Relationship between source modules and their responsibilities.

```mermaid
flowchart LR
    subgraph SRC["src/"]
        SD["data/\ninspect_dataset.py\nprepare_dataset.py"]
        SP["privacygaurd/\n__init__.py"]
    end

    subgraph TRAIN["training/"]
        TR["train_modal.py\nL4 · QLoRA · HuggingFace push"]
    end

    subgraph EVAL["evaluation/"]
        EB["baseline_modal.py"]
        EF["finetuned_modal.py"]
        EX["external_eval_modal.py"]
        EP["prepare_external_benchmarks.py"]
        EM["metrics.py"]
        EA["analyze_external_results.py"]
        EE["error_analysis.py"]
        EL["audit_leakage.py"]
        EV["create_figures.py"]
    end

    subgraph APP["app/"]
        AS["streamlit_app.py"]
        AM["modal_inference.py"]
    end

    subgraph DATA["data/"]
        DP["processed/\ntrain · val · test .jsonl"]
        DE["external/\ngretel · nemotron .jsonl"]
    end

    subgraph RES["results/"]
        RM["*.json metrics"]
        RF["figures/*.png"]
        RE["external/*.json"]
    end

    SD --> DP
    EP --> DE
    DP --> TR
    TR --> EB & EF
    DE --> EX
    EB & EF & EX --> EM
    EM --> RM
    EM --> EA & EE & EL
    EA & EE --> RE
    RM & RE --> EV
    EV --> RF
    AM --> AS
```
