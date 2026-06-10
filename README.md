# LLM-Multilingual-Hallucination-Evaluation-Pipeline

# Multilingual LLM Hallucination Evaluation Pipeline

An end-to-end, production-grade validation and classification pipeline built to programmatically identify, score, and diagnose extrinsic hallucinations in Large Language Models (LLMs). This project evaluates model reliability across 5 state-of-the-art architectures, 7 languages (including morphologically rich, non-Latin scripts), and 8 diverse domain workflows.

## 📌 Project Overview & Motivation

Deploying Large Language Models into high-stakes production features (Medical, Legal, Financial) presents a critical engineering risk: **hallucinations**. 

This repository implements a modular Python engineering framework to systematically clean, process, evaluate, and flag hallucinations. Using a verified benchmark tracking 200 granular annotations from September 2024 to April 2025, this pipeline provides rigorous diagnostic metrics detailing where model architectures degrade, how languages impact stability, and which prompting frameworks effectively mitigate errors.

---

## 🛠️ System Architecture & Pipeline Phases

The repository treats data evaluation using a deterministic 4-phase architecture:

[llm_hallucination_dataset_v1.csv]
│
▼
┌───────────────────┐
│ Phase 1: Validate │ ──► Type enforcement, Schema guarantees
└───────────────────┘
│
▼
┌───────────────────┐
│ Phase 2: Process  │ ──► Unicode-aware cleaning (\w flags)
└───────────────────┘
│
▼
┌───────────────────┐
│ Phase 3: Evaluate │ ──► Sublinear TF-IDF + Weighted Class-Weight Base
└───────────────────┘
│
▼
┌───────────────────┐
│ Phase 4: Diagnose │ ──► Automated risk-matrix generation
└───────────────────┘

### 1. Schema Validation & Defensive Engineering
Before processing unstructured text metadata, the pipeline runs automated schema validation to guarantee feature availability and prevent downstream execution crashes.

### 2. Unicode-Aware Preprocessing (Multilingual Extraction)
Standard regex frameworks (`[^a-zA-Z]`) inadvertently strip out non-Latin scripts. This engine utilizes explicit Unicode-aware alphanumeric token parsing (`\w` flags). This guarantees that complex, morphologically rich languages like **Arabic, Hindi, and Mandarin** maintain their syntactic structure, boundary contexts, and integrity during tokenization.

### 3. Model Classification Mechanics
* **Sublinear TF-IDF Normalization:** LLM outputs fluctuate wildly based on generation style (e.g., direct answers vs. long Chain-of-Thought reasoning). The pipeline uses sublinear scaling ($1 + \log(\text{tf})$) to squash document length bias.
* **Class Balancing:** Addresses the baseline ~34.5% hallucination rate through balanced inverse-frequency class-weight distribution during training.

### 4. Operational Risk & Degradation Diagnostics
Compiles data aggregates to pinpoint architectural flaws:
* **Cross-Model Diagnostics:** Direct accuracy comparisons across `GPT-4o`, `Mistral-Large`, `Llama-3.1-70B`, `Gemini-1.5-Pro`, and `Claude-3.5-Sonnet`.
* **Language Degradation Metrics:** Tracking performance discrepancies between high-resource Western languages and non-Latin scripts.
* **High-Stakes Segmenting:** Isolating risk profiles within sensitive domains (`Medicine`, `Law`, `Finance`).

---

## 📊 Dataset Profile

The architecture ingests data from `llm_hallucination_dataset_v1.csv`, which covers:
* **Models Tracked:** GPT-4o, Mistral-Large, Llama-3.1-70B, Gemini-1.5-Pro, Claude-3.5-Sonnet.
* **Knowledge Domains:** Medicine, Technology, Science, Finance, Law, History, Politics, General.
* **Languages Evaluated:** English, Spanish, French, German, Mandarin, Arabic, Hindi.
* **Hallucination Phenotypes:** Factual-Contradiction, Overclaim, Entity-Error, Relation-Error, Outdatedness, Unverifiability, Incompleteness.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed along with the required analytical packages:
```bash
pip install pandas numpy scikit-learn
