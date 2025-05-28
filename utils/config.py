#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration and constants for the ADE extraction pipeline
"""

import os
from dotenv import load_dotenv
import torch

# ==================== ENVIRONMENT ====================
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")

# ==================== BASE & DATA PATHS ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Method-specific data directories
DIRECT_DATA_DIR = os.path.join(DATA_DIR, "direct")
DSPY_DATA_DIR = os.path.join(DATA_DIR, "dspy")
GOLD_DATA_DIR = os.path.join(DATA_DIR, "gold")

# Create necessary directories
for directory in [DATA_DIR, DIRECT_DATA_DIR, DSPY_DATA_DIR, GOLD_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==================== DATA SETTINGS ====================
INPUT_FILE = os.path.join(DATA_DIR, 'train.txt')
MAX_NOTES = 5000000
MAX_TEST_NOTES = 1000

# Gold data processing steps
STEP_1_GOLD_STANDARD = os.path.join(GOLD_DATA_DIR, "gold_extracted_data.jsonl")
STEP_2_GOLD_NER_DATA = os.path.join(GOLD_DATA_DIR, "gold_ner_data.jsonl")

# Backward compatibility for gold standard path
GOLD_STANDARD_PATH = STEP_1_GOLD_STANDARD
GOLD_NER_DATA_PATH = STEP_2_GOLD_NER_DATA

# ==================== MODEL SETTINGS ====================
# --- LLM ---
LLM_MODEL_NAME = "gpt-4o-mini"

# --- BERT ---
BERT_MODEL_NAME = "MutazYoune/ClinicalBERT-AE-NER"  # emilyalsentzer/Bio_ClinicalBERT
BERT_MAX_LENGTH = 512
BERT_OUTPUT_DIR = os.path.join(BASE_DIR, "bert_ade_extractor")

# --- NER ---
NER_LABELS = {
    "O": 0,       # Outside any entity
    "B-DRUG": 1,  # Beginning of drug mention
    "I-DRUG": 2,  # Inside of drug mention
    "B-ADE": 3,   # Beginning of adverse event
    "I-ADE": 4    # Inside of adverse event
}
ID_TO_LABEL = {v: k for k, v in NER_LABELS.items()}

# ==================== TRAINING/EVAL SETTINGS ====================
BATCH_SIZE = 4
MAX_WORKERS = 5

# --- Training hyperparameters ---
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 5e-5

# --- HuggingFace Trainer Settings ---
TRAINING_ARGS = {
    "num_train_epochs": DEFAULT_EPOCHS,
    "per_device_train_batch_size": BATCH_SIZE,
    "per_device_eval_batch_size": BATCH_SIZE,
    "learning_rate": DEFAULT_LEARNING_RATE,
    "weight_decay": 0.01,
    "eval_strategy": "epoch",
    "save_strategy": "no",  # Disable checkpoint saving
    "logging_steps": 50,
    "load_best_model_at_end": False,  # Disable loading best model (requires checkpoints)
    "metric_for_best_model": "eval_overall_f1",
    "greater_is_better": True,
    "fp16": False,
    "report_to": "none",
    "gradient_accumulation_steps": 1,
    "warmup_steps": 100,
    "seed": 42,
    "disable_tqdm": False,
    "save_safetensors": True
}

# ==================== OUTPUT PATHS ====================
# Direct approach paths
STEP_1_LLM_DIRECT = os.path.join(DIRECT_DATA_DIR, 'extracted_data.jsonl')
STEP_2_NER_DATA_DIRECT = os.path.join(DIRECT_DATA_DIR, 'ner_data.jsonl')
GOLD_NER_DATA_DIRECT = os.path.join(DIRECT_DATA_DIR, 'gold_ner_data.jsonl')

# DSPy approach paths
STEP_1_LLM_DSPY = os.path.join(DSPY_DATA_DIR, 'extracted_data.jsonl')
STEP_2_NER_DATA_DSPY = os.path.join(DSPY_DATA_DIR, 'ner_data.jsonl')
GOLD_NER_DATA_DSPY = os.path.join(DSPY_DATA_DIR, 'gold_ner_data.jsonl')

# Train/val/test split paths (optional)
TRAIN_SPLIT_DIRECT = os.path.join(DIRECT_DATA_DIR, 'train_split.json')
VAL_SPLIT_DIRECT = os.path.join(DIRECT_DATA_DIR, 'val_split.json')
TEST_SPLIT_DIRECT = os.path.join(DIRECT_DATA_DIR, 'test_split.json')
TRAIN_SPLIT_DSPY = os.path.join(DSPY_DATA_DIR, 'train_split.json')
VAL_SPLIT_DSPY = os.path.join(DSPY_DATA_DIR, 'val_split.json')
TEST_SPLIT_DSPY = os.path.join(DSPY_DATA_DIR, 'test_split.json')

# ==================== LOGGING & ERROR HANDLING ====================
# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Error handling settings
MAX_RETRIES = 3  # Maximum number of retries for API calls
RETRY_DELAY = 1  # Seconds to wait between retries
API_TIMEOUT = 30  # Seconds to wait for API response before timeout

# ==================== BACKWARD COMPATIBILITY ====================
# These variables provided for backward compatibility
LLM_OUTPUT_PATH_DIRECT = STEP_1_LLM_DIRECT
LLM_OUTPUT_PATH_DSPY = STEP_1_LLM_DSPY
LLM_OUTPUT_PATH = STEP_1_LLM_DIRECT
