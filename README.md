# ADE Extraction with ModernBERT

This project implements a pipeline for Adverse Drug Event (ADE) extraction from clinical text using modern BERT-based models and Large Language Models (LLMs) for data generation.

## Quick Start: How to Run

The pipeline is divided into three main steps:

1.  **Step 1: Generate Training Data (`run_step1.py`)**
    *   This step uses LLMs to process raw clinical notes and generate annotated data in NER (Named Entity Recognition) format.
    *   Run: `python run_step1.py`
    *   Configure data generation methods in `run_step1.py` via the `ENABLED_SOURCES` dictionary. See `Step_1_data_generation/README.md` for details on each generation approach.

2.  **Step 2: Train BERT Models (`run_step2.py`)**
    *   This step trains custom BERT models on the data generated in Step 1.
    *   Run: `python run_step2.py`
    *   Configure which models to train and training parameters in `run_step2.py`. You can choose between standard and enhanced training scripts.

3.  **Step 3: Evaluate Models (`run_step3.py`)**
    *   This step evaluates the trained BERT models and compares their performance against baseline LLM approaches and gold standard data.
    *   Run: `python run_step3.py`
    *   Configure evaluation parameters, data sources, and model paths in `run_step3.py`. Results are saved in the `analysis/comparison_results/` directory.

---

## Project Overview

This project aims to build and evaluate a robust system for extracting Adverse Drug Events (ADEs) and associated drug names from unstructured clinical text. It leverages:
*   **LLMs for Data Generation**: Various strategies (direct prompting, DSPy, multi-step pipelines) are employed to generate high-quality training data from raw text.
*   **BERT-based NER Models**: Custom BERT models (e.g., Bio_ClinicalBERT, ClinicalBERT-AE-NER) are fine-tuned for the NER task of identifying DRUG and ADE entities.
*   **Comprehensive Evaluation**: The system includes a thorough evaluation framework to compare the performance of different data generation methods and trained models against a gold standard dataset.

## Getting Started

### Prerequisites

*   Python 3.9+
*   Pip for package management
*   Access to OpenAI API (or other LLM providers compatible with LiteLLM) for data generation.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ADR_train_modernBERT
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    *   Create a `.env` file in the project root.
    *   Add your OpenAI API key:
        ```env
        OPENAI_API_KEY="your_openai_api_key_here"
        ```
    *   Other LLM provider keys can be added if you modify the `dspy` or `litellm` configurations.

## Usage Details

### Step 1: Data Generation

*   **Script**: `run_step1.py`
*   **Purpose**: To generate annotated training data for NER models using LLMs.
*   **Configuration**:
    *   Open `run_step1.py`.
    *   Modify the `ENABLED_SOURCES` dictionary to select which data generation approaches to run. Options include:
        *   `"direct"`: Direct LLM extraction.
        *   `"dspy"`: DSPy-based extraction framework.
        *   `"pipeline"`: Multi-step pipeline extraction.
        *   `"validator"`: Simple extraction with a validation step.
        *   `"structured"`: Structured DSPy-based extraction with position-aware entities.
    *   Adjust `base_config` parameters like `max_notes`, `overwrite_cache`, `model_name`, etc., as needed.
*   **Output**:
    *   `ner_data.jsonl`: Data formatted for NER model training (tokens and tags).
    *   `extracted_data.jsonl`: Raw LLM extractions with metadata.
    *   Outputs are saved in subdirectories within `Step_1_data_generation/data/` corresponding to the generation approach (e.g., `Step_1_data_generation/data/direct/`).
*   **Further Details**: For an in-depth explanation of each data generation approach, refer to `Step_1_data_generation/README.md`.

### Step 2: Train BERT Models

*   **Script**: `run_step2.py`
*   **Purpose**: To fine-tune BERT-based models for token classification (NER) using the data generated in Step 1.
*   **Configuration**:
    *   Open `run_step2.py`.
    *   `config["data_sources"]`: List of data sources (e.g., `["direct", "dspy"]`) from Step 1 to use for training. The script will look for `ner_data.jsonl` in the respective data source folder.
    *   `config["models_to_train"]`: List of model keys to train. These keys correspond to entries in `AVAILABLE_MODELS` dictionaries in `Step_2_train_BERT_models/train_bert_models.py` or `Step_2_train_BERT_models/enhanced_train_bert_models.py`.
        *   Example: `["MutazYoune_ClinicalBERT", "Bio_ClinicalBERT"]`
    *   `config["use_enhanced_training"]`: Set to `True` to use `enhanced_train_bert_models.py` (includes CRF layer, linear warmup, early stopping, etc.) or `False` to use `train_bert_models.py` (standard Hugging Face Trainer).
    *   Other training parameters like `batch_size`, `learning_rate`, `epochs`, `max_length` can be adjusted in the `config` dictionary.
*   **Model Definitions**:
    *   Standard models: `Step_2_train_BERT_models/train_bert_models.py` (see `AVAILABLE_MODELS`).
    *   Enhanced models: `Step_2_train_BERT_models/enhanced_train_bert_models.py` (see `AVAILABLE_MODELS`). You can add or modify model paths here.
*   **Output**:
    *   Trained model files (weights, config, tokenizer) are saved in subdirectories within `Step_2_train_BERT_models/trained_models/`.
    *   The directory structure is typically `<data_source>_<model_id>` or `<data_source>_<model_id>_enhanced`.
    *   A `training_config.json` and `training_summary.json` (or `training_summary_enhanced.json`) are also saved.

### Step 3: Model Evaluation

*   **Script**: `run_step3.py`
*   **Purpose**: To evaluate the performance of trained BERT models and compare them with LLM-based extraction methods.
*   **Configuration**:
    *   Open `run_step3.py`.
    *   `config["output_dir"]`: Directory where evaluation results and plots will be saved. A timestamped subdirectory is created here.
    *   `config["gold_data_path"]`: Path to the gold standard annotated dataset (e.g., `Step_1_data_generation/data/gold/gold_ner_data.jsonl`).
    *   `config["trained_models_dir"]`: Path to the directory containing the trained models from Step 2.
    *   `config["use_cache"]` and `config["overwrite_cache"]`: Control caching for LLM/DSPy evaluations to save costs.
    *   `config["skip_llm"]`: Set to `True` to skip evaluating LLM-based approaches.
    *   `config["max_test_notes"]`: Number of notes from the gold dataset to use for evaluation.
    *   `config["data_sources"]`: This list in `run_step3.py` refers to the *types of data generation methods* whose LLM performance you want to evaluate directly (if not skipping LLMs), not the training data sources for BERT models. The BERT models are found automatically from `trained_models_dir`.
*   **Evaluation Components**:
    *   The script `Step_3_model_evaluation/evaluate.py` contains the core evaluation logic.
    *   It calculates precision, recall, and F1-score for DRUG and ADE entities.
    *   Compares different BERT models (trained on various data sources) and direct LLM approaches.
*   **Output**:
    *   Detailed metrics, comparison plots, and raw results are saved in the `analysis/comparison_results/<timestamped_directory>/`.
    *   Includes `all_results.json`, `summary_metrics.csv`, and various plots like `f1_scores_by_model.png`.

## Project Structure

```
ADR_train_modernBERT/
├── Step_1_data_generation/       # Scripts and data for generating training data
│   ├── data/                     # Raw input data and generated datasets
│   │   ├── direct/
│   │   ├── dspy/
│   │   ├── gold/                 # Gold standard annotated data
│   │   ├── pipeline/
│   │   ├── structured/
│   │   └── validator/
│   ├── direct_llm_generator.py
│   ├── dspy_generator.py
│   ├── ... (other generator scripts)
│   └── README.md                 # Details on data generation approaches
├── Step_2_train_BERT_models/     # Scripts for training BERT models
│   ├── trained_models/           # Output directory for trained models
│   ├── train_bert_models.py      # Standard training script
│   └── enhanced_train_bert_models.py # Enhanced training script
├── Step_3_model_evaluation/      # Scripts for evaluating models
│   ├── evaluate.py               # Core evaluation logic
│   └── config.py                 # (Currently part of run_step3.py, might be separate)
├── analysis/                     # Output directory for evaluation results and analyses
│   └── comparison_results/
├── .env                          # Environment variables (e.g., API keys) - create this
├── requirements.txt              # Python dependencies
├── run_step1.py                  # Main script for Step 1
├── run_step2.py                  # Main script for Step 2
├── run_step3.py                  # Main script for Step 3
└── README.md                     # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

(Specify your project's license here, e.g., MIT, Apache 2.0, or leave blank if not yet decided.)
