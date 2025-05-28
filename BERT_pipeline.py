#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
from transformers import logging as hf_logging
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
hf_logging.set_verbosity_error()

"""
Bio_ClinicalBERT pipeline for ADE extraction using HuggingFace Trainer
"""

import torch
import json
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from datasets import Dataset
import evaluate
from tabulate import tabulate

from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForTokenClassification
)
import warnings

# Suppress specific warning messages
warnings.filterwarnings("ignore", category=UserWarning, message="'pin_memory' argument is set as true")
warnings.filterwarnings("ignore", category=FutureWarning, message="`tokenizer` is deprecated and will be removed")

# Import utility functions from existing code
from utils.data_transformation import load_from_jsonl
from utils.config import (
    GOLD_STANDARD_PATH, MAX_TEST_NOTES,
    STEP_2_NER_DATA_DIRECT, STEP_2_NER_DATA_DSPY,
    BERT_MODEL_NAME, BERT_MAX_LENGTH, BERT_OUTPUT_DIR,
    TRAINING_ARGS, BASE_DIR, STEP_2_GOLD_NER_DATA
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> List[Dict]:
    """Load JSONL data from file"""
    logger.info(f"Loading data from {file_path}")
    try:
        return load_from_jsonl(file_path)
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise

def process_raw_ner_data(data: List[Dict], tokenizer) -> List[Dict]:
    """Process raw text and entity spans into tokens and tags"""
    logger.info("Processing raw data into tokens and tags...")
    
    processed_data = []
    for item in tqdm(data, desc="Processing records"):
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
    
    logger.info(f"Processed {len(processed_data)} records into tokens and tags")
    return processed_data

def prepare_labels(data: List[Dict], gold_data: List[Dict]=None) -> Tuple[Dict, Dict, List[str]]:
    """Extract and prepare label mappings from all available data"""
    logger.info("Preparing label mappings...")
    
    # Collect all unique labels from both train and gold sets
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
    
    # Also include labels from gold data if provided
    if gold_data:
        for item in gold_data:
            if "tags" in item and isinstance(item["tags"], list):
                all_labels.update(item["tags"])
            elif "entities" in item and isinstance(item["entities"], list):
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
    
    logger.info(f"Found {len(labels)} unique labels: {labels}")
    return label2id, id2label, labels

def create_datasets(train_data: List[Dict], val_data: List[Dict]=None) -> Dict[str, Dataset]:
    """Create train/validation datasets"""
    logger.info("Creating datasets...")
    
    datasets = {}
    
    # Check if data already has tokens and tags or needs processing
    if "tokens" in train_data[0] and "tags" in train_data[0]:
        # Already processed data
        datasets["train"] = Dataset.from_dict({
            'tokens': [item['tokens'] for item in train_data],
            'tags': [item['tags'] for item in train_data]
        })
    else:
        # Raw text and entities - create dataset with those fields
        datasets["train"] = Dataset.from_dict({
            'text': [item['text'] for item in train_data],
            'entities': [item['entities'] for item in train_data]
        })
    
    if val_data:
        if "tokens" in val_data[0] and "tags" in val_data[0]:
            datasets["validation"] = Dataset.from_dict({
                'tokens': [item['tokens'] for item in val_data],
                'tags': [item['tags'] for item in val_data]
            })
        else:
            datasets["validation"] = Dataset.from_dict({
                'text': [item['text'] for item in val_data],
                'entities': [item['entities'] for item in val_data]
            })
    
    logger.info(f"Created datasets: {len(train_data)} train" + 
               (f", {len(val_data)} validation" if val_data else ""))
    return datasets

def process_raw_examples(examples, tokenizer):
    """Process raw text and entities into tokens and tags"""
    batch_tokens = []
    batch_tags = []
    
    for i in range(len(examples["text"])):
        text = examples["text"][i]
        entities = examples["entities"][i]
        
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
            for j, (start, end) in enumerate(offset_mapping):
                # Skip if token is outside entity boundaries
                if end <= entity_start or start >= entity_end:
                    continue
                
                # First token of entity gets B- prefix
                if j == 0 or offset_mapping[j-1][1] <= entity_start:
                    tags[j] = f'B-{entity_label}'
                else:
                    tags[j] = f'I-{entity_label}'
        
        batch_tokens.append(tokens)
        batch_tags.append(tags)
    
    return {"tokens": batch_tokens, "tags": batch_tags}

def tokenize_and_align_labels(examples, tokenizer, label2id):
    """Tokenize text and align labels with tokens"""
    # Check if we're dealing with raw data or already tokenized data
    if "text" in examples and "entities" in examples:
        # Process raw text and entities first
        processed = process_raw_examples(examples, tokenizer)
        examples = processed
    
    tokenized = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True, 
        max_length=BERT_MAX_LENGTH, 
        padding="max_length"
    )
    labels_batch = []
    for i, tags in enumerate(examples["tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = [
            -100 if word_idx is None else label2id[tags[word_idx]] 
            for word_idx in word_ids
        ]
        labels_batch.append(label_ids)
    tokenized["labels"] = labels_batch
    return tokenized

def prepare_gold_test_data(tokenizer, label2id):
    """Prepare the gold standard test dataset for HuggingFace Trainer"""
    # Load the gold standard data
    gold_raw_data = load_from_jsonl(STEP_2_GOLD_NER_DATA)
    gold_raw_data = gold_raw_data[:MAX_TEST_NOTES]
    
    # Check if data is raw or already tokenized
    if "tokens" in gold_raw_data[0] and "tags" in gold_raw_data[0]:
        # Already processed data
        gold_dataset = Dataset.from_dict({
            'tokens': [item['tokens'] for item in gold_raw_data],
            'tags': [item['tags'] for item in gold_raw_data]
        })
    else:
        # Raw data with text and entities
        gold_processed = process_raw_ner_data(gold_raw_data, tokenizer)
        gold_dataset = Dataset.from_dict({
            'tokens': [item['tokens'] for item in gold_processed],
            'tags': [item['tags'] for item in gold_processed]
        })
    
    # Tokenize and align labels
    tokenized_gold = gold_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, label2id),
        batched=True,
        remove_columns=gold_dataset.column_names
    )
    
    logger.info(f"Prepared gold test dataset with {len(gold_raw_data)} samples")
    return tokenized_gold

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
        return results
    
    return compute_metrics

def train_bert_model(mode="direct"):
    """Train a BERT model using HuggingFace Trainer"""
    logger.info(f"Starting BERT model training with {mode} approach...")
    # Only keep error handling for critical model loading/training
    try:
        # Create output directory
        now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(BERT_OUTPUT_DIR, f"{mode}_approach_{now_str}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load raw data
        if mode == "direct":
            ner_data_path = STEP_2_NER_DATA_DIRECT
        else:
            ner_data_path = STEP_2_NER_DATA_DSPY
            
        data = load_data(ner_data_path)
        gold_data = load_from_jsonl(STEP_2_GOLD_NER_DATA)[:MAX_TEST_NOTES]
        train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
        
        # Load tokenizer 
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        
        # Check if we need to process raw data into tokens and tags
        if "text" in data[0] and "entities" in data[0]:
            logger.info("Processing raw data into tokens and tags...")
            train_data = process_raw_ner_data(train_data, tokenizer)
            val_data = process_raw_ner_data(val_data, tokenizer)
        
        label2id, id2label, labels = prepare_labels(data, gold_data)
        
        model = AutoModelForTokenClassification.from_pretrained(
            BERT_MODEL_NAME,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        
        datasets = create_datasets(train_data, val_data)
        
        logger.info("Tokenizing datasets...")
        tokenize_fn = lambda examples: tokenize_and_align_labels(examples, tokenizer, label2id)
        tokenized_datasets = {}
        for split, dataset in datasets.items():
            tokenized_datasets[split] = dataset.map(
                tokenize_fn, 
                batched=True, 
                remove_columns=dataset.column_names
            )
            
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        
        # Customize training arguments 
        args_dict = TRAINING_ARGS.copy()
        args_dict["output_dir"] = output_dir
        training_args = TrainingArguments(**args_dict)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=create_compute_metrics_fn(id2label)
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("Evaluating on gold standard test set...")
        gold_test_dataset = prepare_gold_test_data(tokenizer, label2id)
        gold_results = trainer.evaluate(gold_test_dataset)
        
        # Save the fine-tuned model and tokenizer
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save only the model weights, not the full trainer state
        try:
            logger.info("Saving minimal model representation...")
            model_config = {"id2label": id2label, "label2id": label2id}
            with open(os.path.join(output_dir, "model_config.json"), "w") as f:
                json.dump(model_config, f, indent=2)
                
            # Save metrics
            metrics = {
                "gold_results": {k.replace('eval_', ''): v for k, v in gold_results.items()}
            }
            
            with open(os.path.join(output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
                
            logger.info(f"Gold Standard Results - F1: {gold_results.get('eval_overall_f1', 0):.4f}, "
                  f"Precision: {gold_results.get('eval_overall_precision', 0):.4f}, "
                  f"Recall: {gold_results.get('eval_overall_recall', 0):.4f}")
            
            logger.info(f"Model config saved to {os.path.relpath(output_dir, BASE_DIR)}")
        except Exception as e:
            logger.warning(f"Could not save model: {e}")
              
        return model, tokenizer, metrics, output_dir
    except Exception as e:
        logger.error(f"Error in train_bert_model: {e}")
        raise

def display_results(results: Dict):
    """Display training results in a clean format"""
    logger.info("\n" + "="*80)
    logger.info("NER MODEL PERFORMANCE RESULTS".center(80))
    logger.info("="*80)

    # Overall Performance
    table = [
        ["F1-Score", f"{results.get('overall_f1', 0):.4f}", f"{results.get('overall_f1', 0):.1%}"],
        ["Precision", f"{results.get('overall_precision', 0):.4f}", f"{results.get('overall_precision', 0):.1%}"],
        ["Recall", f"{results.get('overall_recall', 0):.4f}", f"{results.get('overall_recall', 0):.1%}"],
        ["Accuracy", f"{results.get('overall_accuracy', 0):.4f}", f"{results.get('overall_accuracy', 0):.1%}"],
    ]
    logger.info("\n" + tabulate(table, headers=["Metric", "Score", "Percentage"], tablefmt="fancy_grid"))

    # Per-Entity Performance (if available)
    entity_metrics = {}
    for key, value in results.items():
        if isinstance(value, dict) and 'f1' in value:
            entity_metrics[key] = value
    if entity_metrics:
        entity_table = []
        for entity, metrics in entity_metrics.items():
            entity_table.append([
                entity,
                f"{metrics.get('f1', 0):.3f}",
                f"{metrics.get('precision', 0):.3f}",
                f"{metrics.get('recall', 0):.3f}",
                metrics.get('number', '-')
            ])
        logger.info("\n" + tabulate(
            entity_table,
            headers=["Entity Type", "F1", "Precision", "Recall", "Support"],
            tablefmt="fancy_grid"
        ))

    logger.info("\n" + "="*80)
    logger.info("Training and evaluation completed successfully!")
    logger.info("="*80)

def main():
    """Main function to run the BERT pipeline."""
    import argparse
    parser = argparse.ArgumentParser(description="BERT Pipeline for ADE Extraction")
    parser.add_argument("--mode", type=str, default="direct", choices=["direct", "dspy"],
                        help="Mode for data extraction (direct or dspy)")
    args = parser.parse_args()
    model, tokenizer, metrics, output_dir = train_bert_model(mode=args.mode)
    
    logger.info(f"BERT model training complete.")
    logger.info(f"Fine-tuned model metrics on gold standard - F1: {metrics['gold_results'].get('overall_f1', 0):.4f}, "
            f"Precision: {metrics['gold_results'].get('overall_precision', 0):.4f}, "
            f"Recall: {metrics['gold_results'].get('overall_recall', 0):.4f}")
    logger.info(f"Model saved to {os.path.relpath(output_dir, BASE_DIR)}")

if __name__ == "__main__":
    main() 
