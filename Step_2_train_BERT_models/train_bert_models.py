#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlined script for training BERT models for ADE extraction.
This script trains 5 BERT-based models for Named Entity Recognition.
"""

import os
import json
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any
import shutil
import logging
from dotenv import load_dotenv

import torch
from datasets import Dataset
import evaluate
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)

# Load environment variables
load_dotenv()

# =============== CONFIGURATION SETTINGS ===============

# Output directory for trained models
MODELS_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)

# Available BERT models to train
AVAILABLE_MODELS = {
    "Bio_ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "MutazYoune_ClinicalBERT": "MutazYoune/ClinicalBERT-AE-NER",
}

# Data sources to use for training (options: "direct", "dspy")
DATA_SOURCES = ["direct", "dspy"]

# Models to train (use keys from AVAILABLE_MODELS)
MODELS_TO_TRAIN = AVAILABLE_MODELS.keys()

# Training parameters
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_MAX_LENGTH = 128

# Data split parameters
VAL_SIZE = 0.1

# Training arguments for HuggingFace Trainer
TRAINING_ARGS = {
    "weight_decay": 0.01,
    "eval_strategy": "epoch",
    "save_strategy": "no",
    "logging_steps": 500,
    "load_best_model_at_end": False,
    "metric_for_best_model": "eval_overall_f1",
    "greater_is_better": True,
    "fp16": False,
    "report_to": "none",
    "gradient_accumulation_steps": 1,
    "warmup_steps": 100,
    "seed": 42,
    "disable_tqdm": False,
    "save_safetensors": True,
    "logging_strategy": "epoch",  # Only log at each epoch
    "logging_first_step": False,  # Don't log the first step
    "no_cuda": False,
    "log_level": "error",  # Reduce log verbosity
    "save_total_limit": 1,  # Only keep the last model (not used with save_strategy=no)
    "logging_dir": None  # Disable TensorBoard logging
}

# Suppress warnings
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# Suppress HuggingFace transformers info/warning logs
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# Set up logging
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise

def process_raw_ner_data(data: List[Dict], tokenizer) -> List[Dict]:
    """Process raw text and entity spans into tokens and tags"""
    processed_data = []
    for item in tqdm(data, desc="Processing data", ncols=100):
        text = item.get('text', '')
        entities = item.get('entities', [])
        
        # Tokenize the text and get offset mapping
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(encoding.input_ids)
        offset_mapping = encoding.offset_mapping
        
        # Initialize all tags as 'O' (outside)
        tags = ['O'] * len(tokens)
        
        # Assign BIO tags based on entity spans
        for entity in entities:
            entity_start = entity.get('start')
            entity_end = entity.get('end')
            entity_label = entity.get('label')
            
            # Find tokens that overlap with this entity
            for i, (start, end) in enumerate(offset_mapping):
                # Skip if token is outside entity boundaries
                if end <= entity_start or start >= entity_end:
                    continue
                
                # First token of entity gets B- prefix
                if i == 0 or offset_mapping[i-1][1] <= entity_start:
                    tags[i] = f'B-{entity_label}'
                else:
                    tags[i] = f'I-{entity_label}'
        
        processed_data.append({
            'tokens': tokens,
            'tags': tags
        })
    
    return processed_data

def prepare_labels(data: List[Dict]) -> Tuple[Dict, Dict, List[str]]:
    """Extract and prepare label mappings from training data"""
    # Collect all unique labels
    all_labels = set()
    
    # Process training data 
    for item in data:
        if "tags" in item and isinstance(item["tags"], list):
            all_labels.update(item["tags"])
        elif "entities" in item and isinstance(item["entities"], list):
            # For raw data with entities
            for entity in item["entities"]:
                label = entity.get("label")
                if label:
                    all_labels.add(f"B-{label}")
                    all_labels.add(f"I-{label}")
    
    # Add 'O' label if not already included
    all_labels.add("O")
    
    # Sort labels for deterministic behavior
    labels = sorted(list(all_labels))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    return label2id, id2label, labels

def create_datasets(data: List[Dict], val_size: float = 0.1) -> Dict[str, Dataset]:
    """Create train/validation datasets (no test split)"""
    # Split data into train and validation
    train_data, val_data = train_test_split(data, test_size=val_size, random_state=42)
    
    datasets = {}
    if "tokens" in data[0] and "tags" in data[0]:
        datasets["train"] = Dataset.from_dict({
            'tokens': [item['tokens'] for item in train_data],
            'tags': [item['tags'] for item in train_data]
        })
        datasets["validation"] = Dataset.from_dict({
            'tokens': [item['tokens'] for item in val_data],
            'tags': [item['tags'] for item in val_data]
        })
    else:
        datasets["train"] = Dataset.from_dict({
            'text': [item['text'] for item in train_data],
            'entities': [item['entities'] for item in train_data]
        })
        datasets["validation"] = Dataset.from_dict({
            'text': [item['text'] for item in val_data],
            'entities': [item['entities'] for item in val_data]
        })
    logger.info(f"Dataset split: {len(train_data)} train, {len(val_data)} validation")
    return datasets

def tokenize_and_align_labels(examples, tokenizer, label2id, max_length=DEFAULT_MAX_LENGTH):
    """Tokenize inputs and align labels for model training"""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length"
    )
    
    labels = []
    for i, tags in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 (ignored in loss calculation)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Only label the first token of a word
                label_ids.append(label2id[tags[word_idx]])
            else:
                # For other tokens of a word, copy the label or use -100
                label_ids.append(-100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def create_compute_metrics_fn(id2label):
    """Create a function to compute evaluation metrics"""
    seqeval = evaluate.load("seqeval")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        
        # Calculate overall metrics (micro avg)
        overall_precision = results.get('overall_precision', 0)
        overall_recall = results.get('overall_recall', 0)
        overall_f1 = results.get('overall_f1', 0)
        
        return {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1
        }
    
    return compute_metrics

class CustomTrainer(Trainer):
    """Custom trainer class with nicer evaluation output"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logged_eval_results = False
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to add custom logging"""
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Custom formatted output for evaluation results
        if not self._logged_eval_results and "eval_overall_f1" in metrics:
            # Fix the epoch formatting issue
            epoch_str = f"{self.state.epoch:.2f}" if self.state.epoch else "N/A"
            
            logger.info("\n" + "=" * 50)
            logger.info(f"Evaluation Results - Epoch {epoch_str}")
            logger.info("-" * 50)
            logger.info(f"F1 Score:    {metrics.get('eval_overall_f1', 0):.4f}")
            logger.info(f"Precision:   {metrics.get('eval_overall_precision', 0):.4f}")
            logger.info(f"Recall:      {metrics.get('eval_overall_recall', 0):.4f}")
            logger.info(f"Loss:        {metrics.get('eval_loss', 0):.4f}")
            logger.info("=" * 50 + "\n")
            self._logged_eval_results = True
        
        return metrics

def print_banner(model_id, data_source):
    term_width = shutil.get_terminal_size((80, 20)).columns
    banner_text = f" TRAINING: {model_id} ON {data_source.upper()} DATA "
    banner = f"\033[1;44m{banner_text.center(term_width)}\033[0m"
    logger.info("\n" + banner + "\n")

def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train_model(
    data_path: str,
    model_name: str,
    output_dir: str,
    data_source: str,
    config: Dict = None
) -> Dict[str, Any]:
    """Train a BERT model for NER on the specified data"""
    
    # Get settings from config or defaults
    batch_size = config.get("batch_size", DEFAULT_BATCH_SIZE)
    epochs = config.get("epochs", DEFAULT_EPOCHS)
    learning_rate = config.get("learning_rate", DEFAULT_LEARNING_RATE)
    max_length = config.get("max_length", DEFAULT_MAX_LENGTH)
    val_size = config.get("val_size", VAL_SIZE)
    
    # Get model identifier for nice naming
    model_id = model_name.split('/')[-1]
    
    # Create descriptive output directory
    model_output_dir = os.path.join(output_dir, f"{data_source}_{model_id}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    print_banner(model_id, data_source)
    
    device = get_best_device()
    logger.info(f"Using device: {device}")
    logger.info(f"\n{'='*70}")
    logger.info(f"Training: {model_id} on {data_source} data")
    logger.info(f"Output directory: {model_output_dir}")
    logger.info(f"{'='*70}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load and process data
    data = load_data(data_path)
    
    # Check if data already has tokens and tags
    needs_processing = "tokens" not in data[0] or "tags" not in data[0]
    
    if needs_processing:
        logger.info("Processing raw data into tokens and tags...")
        data = process_raw_ner_data(data, tokenizer)
    
    # Prepare labels and datasets
    label2id, id2label, labels = prepare_labels(data)
    
    datasets = create_datasets(data, val_size=val_size)
    
    # Load model with the correct number of labels
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    # Tokenize and align labels
    tokenized_datasets = {}
    for split, dataset in datasets.items():
        if "tokens" in dataset.column_names and "tags" in dataset.column_names:
            tokenized_datasets[split] = dataset.map(
                lambda examples: tokenize_and_align_labels(examples, tokenizer, label2id, max_length),
                batched=True
                )
        else:
            # Handle raw text data if needed
            logger.warning(f"Dataset {split} doesn't have tokens/tags - needs implementation")
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=TRAINING_ARGS["weight_decay"],
        eval_strategy=TRAINING_ARGS["eval_strategy"],
        save_strategy=TRAINING_ARGS["save_strategy"],
        logging_steps=TRAINING_ARGS["logging_steps"],
        load_best_model_at_end=TRAINING_ARGS["load_best_model_at_end"],
        metric_for_best_model=TRAINING_ARGS["metric_for_best_model"],
        greater_is_better=TRAINING_ARGS["greater_is_better"],
        fp16=TRAINING_ARGS["fp16"],
        report_to=TRAINING_ARGS["report_to"],
        gradient_accumulation_steps=TRAINING_ARGS["gradient_accumulation_steps"],
        warmup_steps=TRAINING_ARGS["warmup_steps"],
        seed=config.get("seed", TRAINING_ARGS["seed"]),
        disable_tqdm=TRAINING_ARGS["disable_tqdm"],
        save_safetensors=TRAINING_ARGS["save_safetensors"],
        no_cuda=(device.type == "cpu")
    )
    
    # Create data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Create compute metrics function
    compute_metrics = create_compute_metrics_fn(id2label)
    
    # Initialize custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation", None),
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    logger.info("\nStarting training...\n")
    train_result = trainer.train()
    
    # Save model, tokenizer, and configuration
    logger.info(f"Saving model to {model_output_dir}")
    trainer.save_model(model_output_dir)
    
    # Save training configuration
    config = {
        "model_name": model_name,
        "data_path": data_path,
        "data_source": data_source,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "val_size": val_size,
        "train_metrics": train_result.metrics,
        "labels": labels,
        "label2id": label2id,
        "id2label": id2label,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    with open(os.path.join(model_output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    return {
        "model_path": model_output_dir,
        "train_metrics": train_result.metrics
    }

def get_model_path(model_name: str) -> str:
    """Get the full model path for a given model name"""
    if model_name in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[model_name]
    else:
        return model_name

def run_training_pipeline(config: Dict = None):
    """Run the training pipeline for all models and data sources"""
    # Use default config if none provided
    if config is None:
        config = {}
    
    # Set output directory
    run_dir = config.get("output_dir", os.path.join(MODELS_OUTPUT_DIR, "trained_models"))
    os.makedirs(run_dir, exist_ok=True)
    
    # Get data sources and models to train
    data_sources = config.get("data_sources", DATA_SOURCES)
    models_to_train = config.get("models_to_train", MODELS_TO_TRAIN)
    overwrite_existing = config.get("overwrite_existing", False)
    
    logger.info("\n" + "="*70)
    logger.info("Starting BERT Model Training Pipeline")
    logger.info(f"Output directory: {run_dir}")
    logger.info(f"Models: {', '.join(models_to_train)}")
    logger.info(f"Data sources: {', '.join(data_sources)}")
    logger.info("="*70 + "\n")
    
    results = {}
    
    for data_source in data_sources:
        data_path = os.path.join("Step_1_data_generation", "data", data_source, "ner_data.jsonl")
        results[data_source] = {}
        
        for model_name in models_to_train:
            full_model_name = get_model_path(model_name)
            model_output_dir = os.path.join(run_dir, f"{data_source}_{model_name}")
            
            # Skip if model already exists and we're not overwriting
            if os.path.exists(model_output_dir) and not overwrite_existing:
                logger.info(f"Skipping {model_name} on {data_source} data - model already exists")
                results[data_source][model_name] = {"status": "skipped", "model_path": model_output_dir}
                continue
            
            try:
                training_results = train_model(
                    data_path=data_path,
                    model_name=full_model_name,
                    output_dir=run_dir,
                    data_source=data_source,
                    config=config
                )
                
                results[data_source][model_name] = {
                    "status": "success",
                    "model_path": training_results["model_path"],
                    "train_metrics": training_results["train_metrics"]
                }
                
                logger.info(f"Successfully trained {model_name} on {data_source} data")
                
            except Exception as e:
                logger.error(f"Error training {model_name} on {data_source} data: {e}")
                results[data_source][model_name] = {"status": "error", "error": str(e)}
    
    # Save summary of all results
    summary_path = os.path.join(run_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("Training pipeline completed")
    logger.info(f"Summary saved to {summary_path}")
    logger.info("="*70 + "\n")
    
    return results

def train_models(config: Dict):
    """Entry point for run_step2.py"""
    # Import numpy here to avoid issues
    import numpy as np
    return run_training_pipeline(config)

if __name__ == "__main__":
    # Import numpy here to avoid issues
    import numpy as np
    run_training_pipeline()