#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration and constants for the ADE extraction pipeline
"""

import os
from dotenv import load_dotenv

# ==================== ENVIRONMENT ====================
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")

# ==================== MODELS ====================
# LLM settings
LLM_MODEL_NAME = "gpt-4o-mini"

# ModernBERT settings
MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_TOKENIZER_LENGTH = 256

# ==================== NER LABELS ====================
NER_LABELS = {
    "O": 0,       # Outside any entity
    "B-DRUG": 1,  # Beginning of drug mention
    "I-DRUG": 2,  # Inside of drug mention
    "B-ADE": 3,   # Beginning of adverse event
    "I-ADE": 4    # Inside of adverse event
}
ID_TO_LABEL = {v: k for k, v in NER_LABELS.items()}

# ==================== DATA SETTINGS ====================
# Input files
INPUT_FILE = 'data/train.txt'   # Can be .jsonl or .txt
# TEST_FILE = 'data/test.txt'     # Can be .jsonl or .txt
MAX_NOTES = 10
MAX_TEST_NOTES = 50
GOLD_STANDARD_PATH = "data/gold_standard_annotations.json"

# ==================== TRAINING/EVAL SETTINGS ====================
BATCH_SIZE = 8
MAX_WORKERS = 5

# ==================== OUTPUT PATHS ====================
# Base output directory
OUTPUT_DIR = "./analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pipeline outputs organized by method
PIPELINE_OUTPUTS_DIR = os.path.join(OUTPUT_DIR, 'pipeline_outputs')
os.makedirs(PIPELINE_OUTPUTS_DIR, exist_ok=True)

# Direct LLM method outputs
DIRECT_OUTPUTS_DIR = os.path.join(PIPELINE_OUTPUTS_DIR, 'direct')
os.makedirs(DIRECT_OUTPUTS_DIR, exist_ok=True)

# DSPy method outputs
DSPY_OUTPUTS_DIR = os.path.join(PIPELINE_OUTPUTS_DIR, 'dspy')
os.makedirs(DSPY_OUTPUTS_DIR, exist_ok=True)

# LLM Extraction outputs
LLM_OUTPUT_PATH_DIRECT = os.path.join(DIRECT_OUTPUTS_DIR, 'extracted_data.jsonl')
LLM_OUTPUT_PATH_DSPY   = os.path.join(DSPY_OUTPUTS_DIR, 'extracted_data.jsonl')

# NER outputs
NER_OUTPUT_PATH_DIRECT = os.path.join(DIRECT_OUTPUTS_DIR, 'ner_data.jsonl')
NER_OUTPUT_PATH_DSPY   = os.path.join(DSPY_OUTPUTS_DIR, 'ner_data.jsonl')

# BIO dataset paths
BIO_TRAIN_PATH_DIRECT = os.path.join(DIRECT_OUTPUTS_DIR, 'bio_train.json')
BIO_VAL_PATH_DIRECT   = os.path.join(DIRECT_OUTPUTS_DIR, 'bio_val.json')
BIO_TEST_PATH_DIRECT  = os.path.join(DIRECT_OUTPUTS_DIR, 'bio_test.json')
BIO_TRAIN_PATH_DSPY   = os.path.join(DSPY_OUTPUTS_DIR, 'bio_train.json')
BIO_VAL_PATH_DSPY     = os.path.join(DSPY_OUTPUTS_DIR, 'bio_val.json')
BIO_TEST_PATH_DSPY    = os.path.join(DSPY_OUTPUTS_DIR, 'bio_test.json')

# Train/val/test split paths
TRAIN_SPLIT_PATH_DSPY = os.path.join(DSPY_OUTPUTS_DIR, 'train_split.json')
TRAIN_SPLIT_PATH_DIRECT = os.path.join(DIRECT_OUTPUTS_DIR, 'train_split.json')
VAL_SPLIT_PATH_DSPY = os.path.join(DSPY_OUTPUTS_DIR, 'val_split.json')
VAL_SPLIT_PATH_DIRECT = os.path.join(DIRECT_OUTPUTS_DIR, 'val_split.json')
TEST_SPLIT_PATH_DSPY = os.path.join(DSPY_OUTPUTS_DIR, 'test_split.json')
TEST_SPLIT_PATH_DIRECT = os.path.join(DIRECT_OUTPUTS_DIR, 'test_split.json')

# Model save paths
BEST_MODEL_PATH = "./best_modernbert_ade_model"
FINAL_MODEL_PATH = "./modernbert_ade_extractor_final"

# ==================== TRAINING HYPERPARAMETERS ====================
# Default hyperparameters for ModernBERT fine-tuning
DEFAULT_EPOCHS = 3  # Changed from 1 to more reasonable default
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_PATIENCE = 2
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
ONE_CYCLE_PCT_START = 0.1  # 10% warmup for OneCycleLR

# ==================== ERROR HANDLING & LOGGING ====================
# Error handling settings
MAX_RETRIES = 3  # Maximum number of retries for API calls
RETRY_DELAY = 1  # Seconds to wait between retries
API_TIMEOUT = 30  # Seconds to wait for API response before timeout

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# ==================== BACKWARD COMPATIBILITY ====================
# These variables provided for backward compatibility
LLM_OUTPUT_PATH = LLM_OUTPUT_PATH_DIRECT
NER_OUTPUT_PATH = NER_OUTPUT_PATH_DIRECT
BIO_TRAIN_PATH = BIO_TRAIN_PATH_DIRECT
BIO_VAL_PATH = BIO_VAL_PATH_DIRECT
BIO_TEST_PATH = BIO_TEST_PATH_DIRECT

# ==================== BIO_CLINICALBERT SETTINGS ====================
CLINICALBERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
BIO_CLINICALBERT_MAX_LENGTH = 128
BIO_CLINICALBERT_OUTPUT_DIR = "./bio_clinicalbert_ade_extractor"