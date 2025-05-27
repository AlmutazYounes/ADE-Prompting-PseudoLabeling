#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bio_ClinicalBERT pipeline for ADE extraction
"""

import torch
import os
import json
import logging
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

# Import utility functions from existing code
from utils.utils import load_llm_output, load_ner_data, load_bio_data, load_gold_standard
from utils.dataset import ADEDataset, calculate_class_weights, ADEDatasetProcessor
from utils.config import (
    NER_LABELS, ID_TO_LABEL, GOLD_STANDARD_PATH, MAX_TEST_NOTES,
    LLM_OUTPUT_PATH_DIRECT, NER_OUTPUT_PATH_DIRECT,
    BIO_TRAIN_PATH_DIRECT, BIO_VAL_PATH_DIRECT, BIO_TEST_PATH_DIRECT,
    LLM_OUTPUT_PATH_DSPY, NER_OUTPUT_PATH_DSPY,
    BIO_TRAIN_PATH_DSPY, BIO_VAL_PATH_DSPY, BIO_TEST_PATH_DSPY,
    CLINICALBERT_MODEL_NAME, BIO_CLINICALBERT_MAX_LENGTH, BIO_CLINICALBERT_OUTPUT_DIR
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, data_loader, device):
    """Evaluate the model on a dataset."""
    model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating Model"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            
            # Get predictions
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)
            
            # Only consider tokens that are part of the input (not padding)
            active_mask = attention_mask == 1
            active_labels = torch.where(
                active_mask, labels, torch.tensor(0).type_as(labels)
            )
            active_preds = torch.where(
                active_mask, predictions, torch.tensor(0).type_as(predictions)
            )
            
            # Collect for metrics calculation
            y_true.extend(active_labels.cpu().numpy().flatten())
            y_pred.extend(active_preds.cpu().numpy().flatten())
    
    # Calculate metrics (exclude padding tokens with label=0)
    metrics = {'f1': 0, 'precision': 0, 'recall': 0}
    mask = np.array(y_true) != 0
    
    if mask.sum() > 0:  # Ensure we have non-padding tokens
        y_true_masked = np.array(y_true)[mask]
        y_pred_masked = np.array(y_pred)[mask]
        
        # Calculate metrics
        metrics['f1'] = f1_score(y_true_masked, y_pred_masked, average='weighted')
        metrics['precision'] = precision_score(y_true_masked, y_pred_masked, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true_masked, y_pred_masked, average='weighted', zero_division=0)
        
        logger.info(f"Model Evaluation - F1: {metrics['f1']:.4f}, " 
              f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    else:
        logger.warning("No non-padding tokens found in evaluation set")
    
    return metrics

def train_bio_clinicalbert(all_texts, all_tags, all_input_ids, all_attention_masks, mode="direct"):
    """Train Bio_ClinicalBERT model using the prepared data."""
    logger.info("Starting Bio_ClinicalBERT training...")
    
    # Create output directory
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(BIO_CLINICALBERT_OUTPUT_DIR, f"direct_approach_{now_str}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data into train/val
    val_size = 0.1 if len(all_texts) > 10 else 0.5  # fallback for small data
    train_texts, val_texts, train_tags, val_tags, train_input_ids, val_input_ids, train_attention_masks, val_attention_masks = train_test_split(
        all_texts, all_tags, all_input_ids, all_attention_masks, test_size=val_size, random_state=42
    )
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(CLINICALBERT_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        CLINICALBERT_MODEL_NAME, 
        num_labels=len(NER_LABELS)
    )
    
    # Create datasets and data loaders
    train_dataset = ADEDataset(
        train_texts, train_tags, tokenizer,
        max_len=BIO_CLINICALBERT_MAX_LENGTH,
        input_ids=train_input_ids, 
        attention_masks=train_attention_masks
    )
    val_dataset = ADEDataset(
        val_texts, val_tags, tokenizer,
        max_len=BIO_CLINICALBERT_MAX_LENGTH,
        input_ids=val_input_ids, 
        attention_masks=val_attention_masks
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Calculate class weights for imbalanced data
    class_weights = calculate_class_weights(train_tags)
    
    # Setup training
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    
    # Setup loss function with class weights
    if class_weights:
        weights = torch.tensor([class_weights.get(i, 1.0) for i in range(len(NER_LABELS))]).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=0)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    # Training loop
    epochs = 3
    best_val_f1 = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}")):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Calculate loss
            logits = outputs.logits
            loss = criterion(logits.view(-1, len(NER_LABELS)), labels.view(-1))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            train_loss += loss.item()
            
            # Print progress
            if (step + 1) % 50 == 0 or step == len(train_loader) - 1:
                logger.info(f"  Step {step+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Training loss: {avg_train_loss:.4f}")
        
        # Evaluation phase
        model.eval()
        val_metrics = evaluate_model(model, val_loader, device)
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            
            # Save best model
            best_model_path = os.path.join(output_dir, "best_model")
            os.makedirs(best_model_path, exist_ok=True)
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            logger.info(f"Saved best model to {best_model_path}")
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model from training")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Evaluate on gold standard data
    gold_data = load_gold_standard(GOLD_STANDARD_PATH)
    gold_data = gold_data[:MAX_TEST_NOTES]
    
    # Convert gold data to NER and then BIO format
    processor = ADEDatasetProcessor(tokenizer=tokenizer)
    gold_ner = processor.prepare_ner_data(gold_data)
    _, gold_tags, _, _, _ = processor.prepare_bio_data(gold_ner, tokenizer)
    gold_texts = [record["text"] for record in gold_ner]
    test_dataset = ADEDataset(gold_texts, gold_tags, tokenizer, max_len=BIO_CLINICALBERT_MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Evaluate final model on gold standard
    final_metrics = evaluate_model(model, test_loader, device)
    
    # Save metrics
    metrics = {
        'best_val_f1': best_val_f1,
        'final_f1': final_metrics['f1'],
        'final_precision': final_metrics['precision'],
        'final_recall': final_metrics['recall']
    }
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    return model, tokenizer, metrics, output_dir

def main():
    """Main function to run the Bio_ClinicalBERT pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bio_ClinicalBERT Pipeline for ADE Extraction")
    parser.add_argument("--mode", type=str, default="direct", choices=["direct", "dspy"],
                        help="Mode for data extraction (direct or dspy)")
    args = parser.parse_args()
    
    # Configure paths based on mode
    if args.mode == "direct":
        llm_output_path = LLM_OUTPUT_PATH_DIRECT
        ner_output_path = NER_OUTPUT_PATH_DIRECT
        bio_train_path = BIO_TRAIN_PATH_DIRECT
    else:
        llm_output_path = LLM_OUTPUT_PATH_DSPY
        ner_output_path = NER_OUTPUT_PATH_DSPY
        bio_train_path = BIO_TRAIN_PATH_DSPY
    
    # Check if BIO data exists
    if not os.path.exists(bio_train_path):
        logger.error(f"BIO data not found at {bio_train_path}. Please run the main pipeline first.")
        return
    
    # Load BIO data
    logger.info(f"Loading BIO data from {bio_train_path}")
    bio_records = load_bio_data(bio_train_path)
    
    # Load NER data for text access
    ner_data = load_ner_data(ner_output_path)
    
    # Extract needed data
    all_texts = [record.get('text', '') for record in ner_data]
    all_tags = [record.get('tags', []) for record in bio_records]
    all_input_ids = [record.get('input_ids', []) for record in bio_records]
    all_attention_masks = [record.get('attention_mask', []) for record in bio_records]
    
    # Train the model
    model, tokenizer, metrics, output_dir = train_bio_clinicalbert(
        all_texts, all_tags, all_input_ids, all_attention_masks, mode=args.mode
    )
    
    logger.info(f"Bio_ClinicalBERT training complete. Final F1: {metrics['final_f1']:.4f}")
    logger.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main() 