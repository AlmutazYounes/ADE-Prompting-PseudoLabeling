#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for Step 3: Model Evaluation
All constants and paths needed for evaluation are defined here.
"""

import os

# Path to the gold standard NER data (relative to project root)
STEP_2_GOLD_NER_DATA = os.path.join("Step_1_data_generation", "data", "gold", "gold_ner_data.jsonl")

# Paths to generated data (for possible future use)
DIRECT_NER_DATA = os.path.join("Step_1_data_generation", "data", "direct", "ner_data.jsonl")
DSPY_NER_DATA = os.path.join("Step_1_data_generation", "data", "dspy", "ner_data.jsonl")

# Maximum number of test notes to evaluate (set to None for all)
MAX_TEST_NOTES = 10  # Change as needed

# Directory where trained models are stored
TRAINED_MODELS_DIR = os.path.join("Step_2_train_BERT_models", "trained_models")

# LLM cache directory (for direct and dspy approaches)
LLM_CACHE_DIR = os.path.join("analysis", "llm_cache")

# Output directory for evaluation results
COMPARISON_RESULTS_DIR = os.path.join("analysis", "comparison_results")

# Default BERT max length for tokenization
BERT_MAX_LENGTH = 256

# Default model name for tokenizer fallback (if needed)
DEFAULT_BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# Entity labels for NER (these should match the labels used in training)
NER_LABELS = ["ADE", "Drug", "Dosage", "Route", "Frequency", "Duration", "Reason", "Form"]

# ID to label mapping (used for model output conversion)
ID_TO_LABEL = {
    0: "O", 
    1: "B-ADE", 2: "I-ADE",
    3: "B-Drug", 4: "I-Drug",
    5: "B-Dosage", 6: "I-Dosage",
    7: "B-Route", 8: "I-Route",
    9: "B-Frequency", 10: "I-Frequency",
    11: "B-Duration", 12: "I-Duration",
    13: "B-Reason", 14: "I-Reason",
    15: "B-Form", 16: "I-Form"
}

# Visualization settings
VISUALIZATION = {
    "figsize_default": (12, 8),
    "colormap": "YlGnBu",
    "dpi": 100
}

# Cache settings
USE_CACHE_DEFAULT = True
OVERWRITE_CACHE_DEFAULT = False

# Available data sources for evaluation
AVAILABLE_DATA_SOURCES = ["direct", "dspy", "pipeline", "validator", "structured"] 