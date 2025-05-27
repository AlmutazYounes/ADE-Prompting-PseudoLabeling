#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ModernBERT fine-tuning for ADE extraction
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import logging
import os
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split

from utils.config import (
    MODEL_NAME, NER_LABELS, ID_TO_LABEL, BEST_MODEL_PATH, FINAL_MODEL_PATH, MAX_TOKENIZER_LENGTH,
    DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_PATIENCE, DEFAULT_GRADIENT_ACCUMULATION_STEPS, ONE_CYCLE_PCT_START,
    LOG_FORMAT, LOG_LEVEL, GOLD_STANDARD_PATH, MAX_TEST_NOTES
)
from utils.dataset import ADEDataset, calculate_class_weights
from utils.dataset import ADEDatasetProcessor
from utils.utils import handle_extraction_errors, load_gold_standard

# Set up logging based on configuration
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class ModernBERTFineTuner:
    """Class for fine-tuning ModernBERT with advanced training features."""
    def __init__(self, model_name=MODEL_NAME, max_length=MAX_TOKENIZER_LENGTH):
        """Initialize the fine-tuner with model and tokenizer."""
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(NER_LABELS))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length  # Set tokenizer max length (can be up to 8k for ModernBERT)
        
    def evaluate_base_model(self, gold_texts, gold_tags):
        """Evaluate the base model on the test set."""
        test_dataset = ADEDataset(gold_texts, gold_tags, self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=16)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        return evaluate_model(self.model, test_loader, device)
        
    @handle_extraction_errors
    def train(self, train_texts, train_tags, val_texts=None, val_tags=None, 
              epochs=DEFAULT_EPOCHS, batch_size=16, learning_rate=DEFAULT_LEARNING_RATE, class_weights=None,
              patience=DEFAULT_PATIENCE, gradient_accumulation_steps=DEFAULT_GRADIENT_ACCUMULATION_STEPS, 
              input_ids=None, attention_masks=None, save_model=True, output_dir=None, mode="direct"):
        """Enhanced training function with regularization and learning rate scheduling. 'mode' determines output folder name."""
        logger.info("\n==================== TRAINING ModernBERT ====================")
        logger.info(f"Training on {len(train_texts)} examples for up to {epochs} epochs")
        
        # Set model save directory with mode in the name
        if output_dir is None:
            now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            if mode == "dspy":
                folder_name = f"modernbert_dspy_approach_{now_str}"
            else:
                folder_name = f"modernbert_direct_approach_{now_str}"
            output_dir = os.path.join(FINAL_MODEL_PATH, folder_name)
            os.makedirs(output_dir, exist_ok=True)
        
        # Check if dataset is too small
        if self._is_dataset_too_small(train_texts, batch_size):
            return self.model, self.tokenizer, {'error': 'Training dataset too small', 'train_size': len(train_texts)}
        
        # Create datasets and data loaders
        train_loader, val_loader = self._create_data_loaders(
            train_texts, train_tags, val_texts, val_tags, 
            input_ids, attention_masks, batch_size
        )
        
        # Setup training components (optimizer, scheduler, loss function)
        optimizer, scheduler, criterion, device = self._setup_training_components(
            train_loader, learning_rate, class_weights, gradient_accumulation_steps
        )
        
        # Initialize tracking variables
        best_val_f1 = 0.0
        best_model_state = None
        epochs_without_improvement = 0
        training_metrics = self._initialize_metrics_tracking()
        
        # Training loop
        for epoch in range(epochs):
            # Train for one epoch
            avg_train_loss = self._train_epoch(
                epoch, epochs, train_loader, device, optimizer, 
                scheduler, criterion, gradient_accumulation_steps, training_metrics
            )
            
            # Evaluate and potentially save model
            if val_texts and val_tags:
                early_stop, best_val_f1, best_model_state, epochs_without_improvement = self._evaluate_and_save_model(
                    val_loader, device, criterion, best_val_f1,
                    best_model_state, epochs_without_improvement,
                    training_metrics, save_model, output_dir, epoch
                )
                if early_stop:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break
            else:
                # If no validation set, evaluate on training set
                best_val_f1, best_model_state = self._evaluate_on_train_set(
                    train_loader, device, training_metrics, best_val_f1,
                    best_model_state, save_model, output_dir
                )
        
        # Finalize model (load best state and save final model)
        return self._finalize_model(best_model_state, save_model, output_dir, training_metrics)
    
    def evaluate(self, data_loader, device, criterion=None):
        """Evaluate the model on a dataset."""
        self.model.eval()
        total_loss = 0
        y_true = []
        y_pred = []
        
        # Use default criterion if none provided
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating Model"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Calculate loss
                logits = outputs.logits
                loss = criterion(logits.view(-1, len(NER_LABELS)), labels.view(-1))
                total_loss += loss.item()
                
                # Get predictions
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
        metrics = {'loss': total_loss / len(data_loader), 'f1': 0, 'precision': 0, 'recall': 0}
        mask = np.array(y_true) != 0
        
        if mask.sum() > 0:  # Ensure we have non-padding tokens
            y_true_masked = np.array(y_true)[mask]
            y_pred_masked = np.array(y_pred)[mask]
            
            # Calculate metrics
            metrics['f1'] = f1_score(y_true_masked, y_pred_masked, average='weighted')
            metrics['precision'] = precision_score(y_true_masked, y_pred_masked, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true_masked, y_pred_masked, average='weighted', zero_division=0)
        
        return metrics

    def _is_dataset_too_small(self, train_texts, batch_size):
        """Check if the dataset is too small to train meaningfully."""
        if len(train_texts) < 2:
            logger.error("ERROR: Training dataset too small (less than 2 examples). Training aborted.")
            return True
        
        # Adjust batch size if dataset is very small
        if len(train_texts) < batch_size:
            batch_size = max(1, len(train_texts) // 2)  # Ensure at least 1
            logger.warning(f"WARNING: Training set too small. Reducing batch size to {batch_size}")
        
        return False

    def _create_data_loaders(self, train_texts, train_tags, val_texts, val_tags, input_ids, attention_masks, batch_size):
        """Create datasets and data loaders for training and validation."""
        # Create training dataset
        if input_ids and attention_masks:
            train_dataset = ADEDataset(
                train_texts, train_tags, self.tokenizer,
                input_ids=input_ids, attention_masks=attention_masks
            )
        else:
            train_dataset = ADEDataset(train_texts, train_tags, self.tokenizer)
        
        # Create validation dataset if validation data provided
        val_dataset = None
        if val_texts and val_tags:
            val_dataset = ADEDataset(val_texts, val_tags, self.tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        
        return train_loader, val_loader

    def _setup_training_components(self, train_loader, learning_rate, class_weights, gradient_accumulation_steps):
        """Setup optimizer, scheduler, loss function, and device for training."""
        # Determine device
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on device: {device}")
        self.model.to(device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Setup scheduler
        scheduler = self._setup_lr_scheduler(optimizer, train_loader, gradient_accumulation_steps)
        
        # Setup loss function with class weights if provided
        criterion = self._setup_loss_function(class_weights, device)
        
        return optimizer, scheduler, criterion, device

    def _setup_lr_scheduler(self, optimizer, train_loader, gradient_accumulation_steps):
        """Setup learning rate scheduler with proper error handling."""
        # Calculate total steps with safety check - including total epochs to avoid "step beyond total_steps" error
        from utils.config import DEFAULT_EPOCHS
        total_steps = max(1, len(train_loader) // gradient_accumulation_steps * DEFAULT_EPOCHS)
        
        # Check if we have enough steps for OneCycleLR (at least 3 steps required)
        if len(train_loader) <= 1 or total_steps < 3:
            logger.warning(f"WARNING: Dataset too small for OneCycleLR (only {len(train_loader)} batches). Using constant learning rate.")
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        
        logger.info(f"Using OneCycleLR scheduler with {total_steps} total steps")
        try:
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=optimizer.param_groups[0]['lr'],
                total_steps=total_steps,
                pct_start=ONE_CYCLE_PCT_START,
                anneal_strategy='linear'
            )
        except Exception as e:
            logger.warning(f"Error creating OneCycleLR scheduler: {e}. Falling back to constant learning rate.")
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    def _setup_loss_function(self, class_weights, device):
        """Setup loss function with optional class weights."""
        if class_weights:
            logger.info("Using class weights for training")
            weights = torch.tensor([class_weights.get(i, 1.0) for i in range(len(NER_LABELS))]).to(device)
            return torch.nn.CrossEntropyLoss(weight=weights, ignore_index=0)
        else:
            return torch.nn.CrossEntropyLoss(ignore_index=0)

    def _initialize_metrics_tracking(self):
        """Initialize dictionaries for tracking metrics during training."""
        return {
            'epoch_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'f1_scores': [],
            'precision_scores': [],
            'recall_scores': []
        }

    def _train_epoch(self, epoch, epochs, train_loader, device, optimizer, scheduler, criterion, 
                    gradient_accumulation_steps, training_metrics):
        """Train the model for one epoch."""
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        self.model.train()
        train_loss = 0
        
        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        training_metrics['learning_rates'].append(current_lr)
        logger.info(f"Learning rate: {current_lr:.2e}")
        
        # Training phase
        optimizer.zero_grad()  # Zero gradients once at the beginning
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", total=len(train_loader))):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Calculate loss
            logits = outputs.logits
            loss = criterion(logits.view(-1, len(NER_LABELS)), labels.view(-1))
            loss = loss / gradient_accumulation_steps  # Scale loss for accumulation
            
            # Backward pass
            loss.backward()
            
            # Track loss
            train_loss += loss.item() * gradient_accumulation_steps
            
            # Update weights after accumulation or at the end of epoch
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Print progress
            if (step + 1) % (gradient_accumulation_steps * 5) == 0 or step == len(train_loader) - 1:
                logger.info(f"  Step {step+1}/{len(train_loader)} - Loss: {loss.item() * gradient_accumulation_steps:.4f}")
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        training_metrics['epoch_losses'].append(avg_train_loss)
        logger.info(f"Training loss: {avg_train_loss:.4f}")
        
        return avg_train_loss

    def _evaluate_and_save_model(self, val_loader, device, criterion, best_val_f1, best_model_state,
                                 epochs_without_improvement, training_metrics, save_model, output_dir, epoch):
        """Evaluate the model on validation data and save if improved."""
        val_metrics = self.evaluate(val_loader, device, criterion)
        val_loss = val_metrics['loss']
        f1 = val_metrics['f1']
        precision = val_metrics['precision']
        recall = val_metrics['recall']
        
        # Update training metrics
        training_metrics['val_losses'].append(val_loss)
        training_metrics['f1_scores'].append(f1)
        training_metrics['precision_scores'].append(precision)
        training_metrics['recall_scores'].append(recall)
        
        logger.info(f"Validation: Loss: {val_loss:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Check for improvement
        improved = f1 > best_val_f1
        
        if improved:
            best_val_f1 = f1
            best_model_state = self.model.state_dict().copy()
            epochs_without_improvement = 0
            
            # Save best model checkpoint during training
            if save_model:
                best_model_path = os.path.join(output_dir, "best_model")
                os.makedirs(best_model_path, exist_ok=True)
                self.model.save_pretrained(best_model_path)
                self.tokenizer.save_pretrained(best_model_path)
                logger.info(f"Saved best model to {best_model_path}")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epochs")
        
        # Check for early stopping
        if epochs_without_improvement >= DEFAULT_PATIENCE:
            logger.info(f"Early stopping after {epoch+1} epochs")
            return True, best_val_f1, best_model_state, epochs_without_improvement
        
        return False, best_val_f1, best_model_state, epochs_without_improvement

    def _evaluate_on_train_set(self, train_loader, device, training_metrics, best_val_f1, best_model_state, save_model, output_dir):
        """Evaluate model on training set when no validation set is available."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Training Metrics", total=len(train_loader)):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=2)
                
                # Filter out padding tokens (0)
                active_accuracy = labels.view(-1) != 0
                active_preds = preds.view(-1)[active_accuracy]
                active_labels = labels.view(-1)[active_accuracy]
                
                all_preds.extend(active_preds.cpu().numpy())
                all_labels.extend(active_labels.cpu().numpy())
        
        # Calculate metrics
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        train_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        train_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        training_metrics['f1_scores'].append(train_f1)
        training_metrics['precision_scores'].append(train_precision)
        training_metrics['recall_scores'].append(train_recall)
        
        logger.info(f"Training metrics: F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        
        # Track best model based on training F1 if no validation set
        improved = train_f1 > best_val_f1
        
        if improved:
            best_val_f1 = train_f1
            best_model_state = self.model.state_dict().copy()
            
            # Save best model checkpoint during training
            if save_model:
                best_model_path = os.path.join(output_dir, "best_model")
                os.makedirs(best_model_path, exist_ok=True)
                self.model.save_pretrained(best_model_path)
                self.tokenizer.save_pretrained(best_model_path)
                logger.info(f"Saved best model to {best_model_path}")
        
        return best_val_f1, best_model_state

    def _finalize_model(self, best_model_state, save_model, output_dir, training_metrics):
        """Load best model state and save the final model."""
        # Load the best model if available
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Loaded best model from training")
        
        # Save final model
        if save_model:
            final_model_path = os.path.join(output_dir, "final_model")
            os.makedirs(final_model_path, exist_ok=True)
            self.model.save_pretrained(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            # Save training metrics
            import json
            metrics_path = os.path.join(output_dir, "training_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(training_metrics, f, indent=2)
                
            logger.info(f"Saved final model to {final_model_path}")
            logger.info(f"Saved training metrics to {metrics_path}")
        
        return self.model, self.tokenizer, training_metrics


def find_optimal_learning_rate(model, train_loader, device, start_lr=1e-5, end_lr=1e-2, num_steps=100):
    """Improved version of the learning rate finder with better min rate handling."""
    logger.info("Running learning rate finder...")
    model.to(device)
    
    # Save initial model weights
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Setup
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)
    
    # Calculate learning rate multiplier
    mult = (end_lr / start_lr) ** (1 / num_steps)
    
    # Initialize lists to track learning rates and losses
    lrs = []
    losses = []
    
    # Only use a portion of the training data for speed
    max_steps = min(num_steps * 2, len(train_loader))
    
    # Iterate through batches
    for i, batch in enumerate(tqdm(train_loader, desc="LR Finder Progress", total=max_steps)):
        if i >= max_steps:
            break
            
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Record learning rate and loss
            if not torch.isnan(loss) and not torch.isinf(loss):
                lrs.append(optimizer.param_groups[0]['lr'])
                losses.append(loss.item())
                
                # Print progress (loss decreased or increased a lot)
                if i > 0 and (losses[-1] < losses[-2] * 0.8 or losses[-1] > losses[-2] * 1.5):
                    logger.info(f"  Step {i}/{max_steps}: lr={lrs[-1]:.1e}, loss={losses[-1]:.4f}")
            
            # Update weights
            optimizer.step()
            
            # Update learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= mult
            
        except Exception as e:
            logger.error(f"Error in learning rate search: {e}")
            continue
    
    # Restore initial model weights
    model.load_state_dict(initial_state)
    
    # If not enough data points, return a default value
    if len(losses) < 10:
        logger.warning("Not enough valid loss points to determine optimal learning rate. Using default.")
        return 5e-5
    
    # Smoothing losses
    logger.info(f"Collected {len(losses)} loss values across learning rates from {min(lrs):.1e} to {max(lrs):.1e}")
    
    # Find the point of the steepest decrease in the loss
    smoothed_losses = []
    window_size = min(5, len(losses) // 5)  # Adjust window size based on data
    
    for i in range(len(losses)):
        if i < window_size:
            smoothed_losses.append(losses[i])
        else:
            smoothed_losses.append(sum(losses[i-window_size:i]) / window_size)
    
    # Calculate gradients
    if len(smoothed_losses) <= 1:
        return 5e-5  # Default
        
    gradients = [(smoothed_losses[i+1] - smoothed_losses[i]) / (lrs[i+1] - lrs[i]) 
                 for i in range(len(smoothed_losses)-1)]
    
    # Skip the first few points (too noisy)
    start_idx = min(5, len(gradients) // 4)
    gradients = gradients[start_idx:]
    
    if not gradients:
        return 5e-5  # Default
    
    # Find the steepest descent
    try:
        steepest_idx = gradients.index(min(gradients))
        optimal_lr = lrs[steepest_idx + start_idx]
        
        # Typically we want a slightly lower learning rate than the steepest point
        optimal_lr = optimal_lr * 0.1
        logger.info(f"Identified optimal learning rate: {optimal_lr:.1e}")
        
        return max(optimal_lr, 1e-6)  # Lower bound to prevent extreme values
        
    except Exception as e:
        logger.error(f"Error finding optimal point: {e}")
        return 5e-5  # Default fallback


def evaluate_model(model, data_loader, device):
    """Evaluate a model on a dataset."""
    model.to(device)
    model.eval()
    test_loss = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating Model"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            test_loss += outputs.loss.item()
            
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
    metrics = {'f1': 0, 'precision': 0, 'recall': 0, 'loss': test_loss/len(data_loader)}
    mask = np.array(y_true) != 0
    
    if mask.sum() > 0:  # Ensure we have non-padding tokens
        y_true_masked = np.array(y_true)[mask]
        y_pred_masked = np.array(y_pred)[mask]
        
        # Calculate metrics
        metrics['f1'] = f1_score(y_true_masked, y_pred_masked, average='weighted')
        metrics['precision'] = precision_score(y_true_masked, y_pred_masked, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true_masked, y_pred_masked, average='weighted', zero_division=0)
        
        logger.info(f"Model Evaluation - Loss: {metrics['loss']:.4f}, F1: {metrics['f1']:.4f}, " 
              f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    else:
        logger.warning("No non-padding tokens found in evaluation set")
    
    return metrics


def train_and_evaluate(all_texts, all_tags, all_input_ids, all_attention_masks, tokenizer, mode="direct"):
    """Train and evaluate ModernBERT model. 'mode' determines output folder name."""
    from datetime import datetime
    from utils.config import GOLD_STANDARD_PATH
    from utils.utils import load_gold_standard

    # Split LLM-generated data into train/val
    val_size = 0.1 if len(all_texts) > 10 else 0.5  # fallback for small data
    train_texts, val_texts, train_tags, val_tags, train_input_ids, val_input_ids, train_attention_masks, val_attention_masks = train_test_split(
        all_texts, all_tags, all_input_ids, all_attention_masks, test_size=val_size, random_state=42
    )

    # Calculate class weights for imbalanced data
    class_weights = calculate_class_weights(train_tags)

    # Setup dimensions and device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and evaluate base model on validation set
    finetuner = ModernBERTFineTuner(max_length=MAX_TOKENIZER_LENGTH)
    base_metrics = finetuner.evaluate_base_model(val_texts, val_tags)

    # Create dataset and dataloader for learning rate finder
    if train_input_ids and train_attention_masks:
        train_dataset = ADEDataset(
            train_texts, train_tags, tokenizer,
            input_ids=train_input_ids, attention_masks=train_attention_masks
        )
    else:
        train_dataset = ADEDataset(train_texts, train_tags, tokenizer)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Find optimal learning rate with error handling
    try:
        optimal_lr = find_optimal_learning_rate(finetuner.model, train_loader, device)
        # Safeguard against extreme values
        if optimal_lr < 1e-6:
            optimal_lr = 5e-5
            logger.info(f"Learning rate too small, using default: {optimal_lr}")
        elif optimal_lr > 1e-3:
            optimal_lr = 1e-4
            logger.info(f"Learning rate too large, using safer value: {optimal_lr}")
    except Exception as e:
        logger.warning(f"Error finding optimal learning rate: {e}. Using default.")
        optimal_lr = 5e-5

    logger.info(f"Using learning rate: {optimal_lr}")

    # Set output_dir for this run (let train() handle naming if not provided)
    output_dir = None

    # Train the model with validation
    model, tokenizer, training_metrics = finetuner.train(
        train_texts, train_tags, val_texts, val_tags,
        epochs=DEFAULT_EPOCHS,
        batch_size=16,
        learning_rate=optimal_lr,
        class_weights=class_weights,
        patience=DEFAULT_PATIENCE,
        input_ids=train_input_ids,
        attention_masks=train_attention_masks,
        save_model=True,
        output_dir=output_dir,
        mode=mode
    )

    # Get the output directory used for this training run
    from utils.config import FINAL_MODEL_PATH
    now_str = datetime.now().strftime('%Y%m%d')
    if mode == "dspy":
        prefix = f"modernbert_dspy_approach_{now_str}"
    else:
        prefix = f"modernbert_direct_approach_{now_str}"
    possible_dirs = [d for d in os.listdir(FINAL_MODEL_PATH) if d.startswith(prefix)]
    possible_dirs.sort(reverse=True)
    if possible_dirs:
        output_dir = os.path.join(FINAL_MODEL_PATH, possible_dirs[0])
        logger.info(f"Using model output directory: {output_dir}")
    else:
        logger.warning("Could not determine model output directory. Using a default.")
        output_dir = os.path.join(FINAL_MODEL_PATH, f"{prefix}_default")
        os.makedirs(output_dir, exist_ok=True)

    # Final evaluation on gold standard data
    gold_data = load_gold_standard(GOLD_STANDARD_PATH)
    gold_data = gold_data[:MAX_TEST_NOTES]

    # Convert gold data to NER and then BIO format using pipeline utilities
    processor = ADEDatasetProcessor(tokenizer=tokenizer)
    gold_ner = processor.prepare_ner_data(gold_data)
    _, gold_tags, _, _, _ = processor.prepare_bio_data(gold_ner, tokenizer)
    gold_texts = [record["text"] for record in gold_ner]
    test_dataset = ADEDataset(gold_texts, gold_tags, tokenizer)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
    finetuned_metrics = evaluate_model(model, test_loader, device)

    return model, tokenizer, training_metrics, base_metrics, finetuned_metrics, output_dir 