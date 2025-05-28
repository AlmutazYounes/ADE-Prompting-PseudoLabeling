#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation functions for ADE extraction
"""

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os
import json
import logging
from tqdm import tqdm
from utils.extraction import DirectLLMExtractor
from utils.config import GOLD_STANDARD_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_and_visualize_results(pipeline_results, output_dir=None):
    """Generate comprehensive visualizations and analysis of model performance."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create an evaluation subdirectory
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Extract metrics
    base_metrics = pipeline_results["metrics"]["base_model"]
    finetuned_metrics = pipeline_results["metrics"]["finetuned_model"]
    training_metrics = pipeline_results["metrics"]["training"]
    
    # 1. Model Performance Comparison
    plt.figure(figsize=(12, 6))
    
    # Create bar positions
    models = ['Base BERT', 'Fine-tuned BERT']
    x = np.arange(len(models))
    width = 0.25
    
    # Plot bars for each metric
    plt.bar(x - width, [base_metrics['f1'], finetuned_metrics['f1']], width, label='F1 Score')
    plt.bar(x, [base_metrics['precision'], finetuned_metrics['precision']], width, label='Precision')
    plt.bar(x + width, [base_metrics['recall'], finetuned_metrics['recall']], width, label='Recall')
    
    # Customize plot
    plt.ylabel('Score')
    plt.title('Performance Comparison: Base vs. Fine-tuned BERT')
    plt.xticks(x, models)
    plt.legend()
    plt.ylim(0, 1.0)
    
    # Add value labels on the bars
    for i, model in enumerate(models):
        if i == 0:  # Base model
            plt.text(i - width, base_metrics['f1'] + 0.02, f"{base_metrics['f1']:.3f}", ha='center')
            plt.text(i, base_metrics['precision'] + 0.02, f"{base_metrics['precision']:.3f}", ha='center')
            plt.text(i + width, base_metrics['recall'] + 0.02, f"{base_metrics['recall']:.3f}", ha='center')
        else:  # Fine-tuned model
            plt.text(i - width, finetuned_metrics['f1'] + 0.02, f"{finetuned_metrics['f1']:.3f}", ha='center')
            plt.text(i, finetuned_metrics['precision'] + 0.02, f"{finetuned_metrics['precision']:.3f}", ha='center')
            plt.text(i + width, finetuned_metrics['recall'] + 0.02, f"{finetuned_metrics['recall']:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{eval_dir}/model_comparison.png")
    logger.info(f"Model comparison chart saved to '{eval_dir}/model_comparison.png'")
    
    # 2. Learning Curves
    if 'epoch_losses' in training_metrics and len(training_metrics['epoch_losses']) > 0:
        plt.figure(figsize=(14, 10))
        plt.subplot(2, 1, 1)
        
        epochs = range(1, len(training_metrics['epoch_losses']) + 1)
        
        plt.plot(epochs, training_metrics['epoch_losses'], 'b-', label='Training Loss')
        if 'val_losses' in training_metrics and len(training_metrics['val_losses']) > 0:
            plt.plot(epochs, training_metrics['val_losses'], 'r-', label='Validation Loss')
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot performance metrics
        if 'f1_scores' in training_metrics and len(training_metrics['f1_scores']) > 0:
            plt.subplot(2, 1, 2)
            plt.plot(epochs, training_metrics['f1_scores'], 'g-', label='F1 Score')
            plt.plot(epochs, training_metrics['precision_scores'], 'b-', label='Precision')
            plt.plot(epochs, training_metrics['recall_scores'], 'r-', label='Recall')
            
            plt.title('Training Metrics by Epoch')
            plt.xlabel('Epochs')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(f"{eval_dir}/learning_curves.png")
        logger.info(f"Learning curves saved to '{eval_dir}/learning_curves.png'")
    
    # 3. Extract and visualize entity distributions
    extract_data = pipeline_results["processed_data"]["extracted_data"]
    
    # Count drug and ADE frequencies
    drug_counts = {}
    ade_counts = {}
    
    for record in extract_data:
        for drug in record['drugs']:
            drug_counts[drug] = drug_counts.get(drug, 0) + 1
        for ade in record['adverse_events']:
            ade_counts[ade] = ade_counts.get(ade, 0) + 1
    
    # Sort by frequency
    sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    sorted_ades = sorted(ade_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Create bar charts for top entities
    plt.figure(figsize=(14, 8))
    plt.subplot(1, 2, 1)
    plt.barh([item[0][:25] for item in sorted_drugs[:10]], [item[1] for item in sorted_drugs[:10]])
    plt.title('Top 10 Most Frequent Drugs')
    plt.xlabel('Frequency')
    plt.tight_layout()
    
    plt.subplot(1, 2, 2)
    plt.barh([item[0][:25] for item in sorted_ades[:10]], [item[1] for item in sorted_ades[:10]])
    plt.title('Top 10 Most Frequent Adverse Events')
    plt.xlabel('Frequency')
    plt.tight_layout()
    
    plt.savefig(f"{eval_dir}/entity_frequencies.png")
    logger.info(f"Entity frequency charts saved to '{eval_dir}/entity_frequencies.png'")
    
    # 4. Create detailed metrics report
    with open(f"{eval_dir}/model_evaluation_report.json", "w") as f:
        json.dump({
            'base_model': base_metrics,
            'finetuned_model': finetuned_metrics,
            'improvements': {
                'f1': finetuned_metrics['f1'] - base_metrics['f1'],
                'precision': finetuned_metrics['precision'] - base_metrics['precision'],
                'recall': finetuned_metrics['recall'] - base_metrics['recall']
            },
            'training_metrics': training_metrics,
            'entities': {
                'drug_frequencies': {k: v for k, v in sorted_drugs},
                'ade_frequencies': {k: v for k, v in sorted_ades}
            }
        }, f, indent=2)
    logger.info(f"Detailed evaluation report saved to '{eval_dir}/model_evaluation_report.json'")
    
    # 5. Add interactive visualizations if Plotly is available
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Only create interactive visualizations if we have training metrics
        if 'epoch_losses' in training_metrics and len(training_metrics['epoch_losses']) > 0:
            # Define epochs variable (was previously defined in Learning Curves section)
            epochs = range(1, len(training_metrics['epoch_losses']) + 1)
            
            # Create interactive subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Training and Validation Loss", "Training Metrics by Epoch"),
                vertical_spacing=0.15
            )
            
            # Add loss curves
            fig.add_trace(
                go.Scatter(x=list(epochs), y=training_metrics['epoch_losses'], name="Training Loss", line=dict(color="blue")),
                row=1, col=1
            )
            
            if 'val_losses' in training_metrics and len(training_metrics['val_losses']) > 0:
                fig.add_trace(
                    go.Scatter(x=list(epochs), y=training_metrics['val_losses'], name="Validation Loss", line=dict(color="red")),
                    row=1, col=1
                )
            
            # Add performance metrics
            if 'f1_scores' in training_metrics and len(training_metrics['f1_scores']) > 0:
                fig.add_trace(
                    go.Scatter(x=list(epochs), y=training_metrics['f1_scores'], name="F1 Score", line=dict(color="green")),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=list(epochs), y=training_metrics['precision_scores'], name="Precision", line=dict(color="blue")),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=list(epochs), y=training_metrics['recall_scores'], name="Recall", line=dict(color="red")),
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(height=800, width=1000, title_text="Interactive Learning Curves")
            fig.update_xaxes(title_text="Epochs", row=1, col=1)
            fig.update_xaxes(title_text="Epochs", row=2, col=1)
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="Score", range=[0, 1], row=2, col=1)
            
            # Save as HTML
            fig.write_html(f"{eval_dir}/interactive_learning_curves.html")
            logger.info(f"Interactive learning curves saved to '{eval_dir}/interactive_learning_curves.html'")
        else:
            logger.info("Skipping interactive visualizations - no training metrics available")
    except ImportError:
        logger.warning("Plotly not installed. Skipping interactive visualizations.")
    
    return {
        'base_metrics': base_metrics,
        'finetuned_metrics': finetuned_metrics,
        'entity_counts': {
            'drugs': len(drug_counts),
            'adverse_events': len(ade_counts)
        }
    }

# def compare_extraction_methods(dspy_results, direct_llm_results):
#     """Compare the performance of DSPy-optimized and direct LLM extraction methods."""
#     # Extract metrics
#     dspy_metrics = dspy_results["evaluation"]
#     direct_metrics = direct_llm_results["evaluation"]
    
#     # Calculate improvements
#     f1_improvement = dspy_metrics['f1'] - direct_metrics['f1']
#     precision_improvement = dspy_metrics['precision'] - direct_metrics['precision']
#     recall_improvement = dspy_metrics['recall'] - direct_metrics['recall']
    
#     # Create visualization
#     plt.figure(figsize=(12, 6))
    
#     # Create bar positions
#     methods = ['DSPy-optimized', 'Direct LLM']
#     x = np.arange(len(methods))
#     width = 0.25
    
#     # Plot bars for each metric
#     plt.bar(x - width, [dspy_metrics['f1'], direct_metrics['f1']], width, label='F1 Score')
#     plt.bar(x, [dspy_metrics['precision'], direct_metrics['precision']], width, label='Precision')
#     plt.bar(x + width, [dspy_metrics['recall'], direct_metrics['recall']], width, label='Recall')
    
#     # Customize plot
#     plt.ylabel('Score')
#     plt.title('Extraction Performance: DSPy-optimized vs. Direct LLM')
#     plt.xticks(x, methods)
#     plt.legend()
#     plt.ylim(0, 1.0)
    
#     # Add value labels on the bars
#     for i, method in enumerate(methods):
#         if i == 0:  # DSPy
#             plt.text(i - width, dspy_metrics['f1'] + 0.02, f"{dspy_metrics['f1']:.3f}", ha='center')
#             plt.text(i, dspy_metrics['precision'] + 0.02, f"{dspy_metrics['precision']:.3f}", ha='center')
#             plt.text(i + width, dspy_metrics['recall'] + 0.02, f"{dspy_metrics['recall']:.3f}", ha='center')
#         else:  # Direct LLM
#             plt.text(i - width, direct_metrics['f1'] + 0.02, f"{direct_metrics['f1']:.3f}", ha='center')
#             plt.text(i, direct_metrics['precision'] + 0.02, f"{direct_metrics['precision']:.3f}", ha='center')
#             plt.text(i + width, direct_metrics['recall'] + 0.02, f"{direct_metrics['recall']:.3f}", ha='center')
    
#     plt.tight_layout()
#     plt.savefig(f"{eval_dir}/extraction_comparison.png")
#     print(f"Extraction comparison chart saved to '{eval_dir}/extraction_comparison.png'")
    
#     return {
#         'dspy_metrics': dspy_metrics,
#         'direct_llm_metrics': direct_metrics,
#         'improvements': {
#             'f1': f1_improvement,
#             'precision': precision_improvement,
#             'recall': recall_improvement
#         }
#     }

def calculate_entity_metrics(predicted, gold):
    """Calculate precision, recall, and F1 score for entity extraction."""
    # Convert to lowercase for case-insensitive comparison
    pred_drugs = set(drug.lower() for drug in predicted.get('drugs', []))
    pred_ades = set(ade.lower() for ade in predicted.get('adverse_events', []))
    
    gold_drugs = set(drug.lower() for drug in gold.get('drugs', []))
    gold_ades = set(ade.lower() for ade in gold.get('adverse_events', []))
    
    # Combined entities
    pred_entities = pred_drugs.union(pred_ades)
    gold_entities = gold_drugs.union(gold_ades)
    
    # Calculate metrics
    true_positives = len(pred_entities.intersection(gold_entities))
    false_positives = len(pred_entities - gold_entities)
    false_negatives = len(gold_entities - pred_entities)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate separate metrics for drugs and ADEs
    drug_true_positives = len(pred_drugs.intersection(gold_drugs))
    drug_precision = drug_true_positives / len(pred_drugs) if len(pred_drugs) > 0 else 0
    drug_recall = drug_true_positives / len(gold_drugs) if len(gold_drugs) > 0 else 0
    drug_f1 = 2 * drug_precision * drug_recall / (drug_precision + drug_recall) if (drug_precision + drug_recall) > 0 else 0
    
    ade_true_positives = len(pred_ades.intersection(gold_ades))
    ade_precision = ade_true_positives / len(pred_ades) if len(pred_ades) > 0 else 0
    ade_recall = ade_true_positives / len(gold_ades) if len(gold_ades) > 0 else 0
    ade_f1 = 2 * ade_precision * ade_recall / (ade_precision + ade_recall) if (ade_precision + ade_recall) > 0 else 0
    
    return {
        'overall': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        },
        'drug': {
            'precision': drug_precision,
            'recall': drug_recall,
            'f1': drug_f1
        },
        'ade': {
            'precision': ade_precision,
            'recall': ade_recall,
            'f1': ade_f1
        }
    }

def test_examples_with_evaluation(model, tokenizer, test_examples, gold_annotations):
    """Test the fine-tuned model with evaluation against gold standard."""
    from utils.extraction import extract_entities_improved
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    results = []
    overall_metrics = {
        'precision': 0, 'recall': 0, 'f1': 0,
        'drug_precision': 0, 'drug_recall': 0, 'drug_f1': 0,
        'ade_precision': 0, 'ade_recall': 0, 'ade_f1': 0
    }
    
    for i, (example, gold) in enumerate(zip(test_examples, gold_annotations)):
        # Extract entities using the fine-tuned model
        extracted = extract_entities_improved(example, model, tokenizer, device)
        
        # Calculate metrics
        metrics = calculate_entity_metrics(extracted, gold)
        
        # Add to results
        results.append({
            'text': example,
            'extracted': extracted,
            'gold': gold,
            'metrics': metrics
        })
        
        # Update overall metrics
        overall_metrics['precision'] += metrics['overall']['precision']
        overall_metrics['recall'] += metrics['overall']['recall']
        overall_metrics['f1'] += metrics['overall']['f1']
        overall_metrics['drug_precision'] += metrics['drug']['precision']
        overall_metrics['drug_recall'] += metrics['drug']['recall']
        overall_metrics['drug_f1'] += metrics['drug']['f1']
        overall_metrics['ade_precision'] += metrics['ade']['precision']
        overall_metrics['ade_recall'] += metrics['ade']['recall']
        overall_metrics['ade_f1'] += metrics['ade']['f1']
    
    # Calculate average metrics
    n = len(test_examples)
    if n > 0:
        for key in overall_metrics:
            overall_metrics[key] /= n
    
    # Log overall performance
    logger.info("\n=== OVERALL TEST RESULTS ===")
    logger.info(f"Overall metrics: F1={overall_metrics['f1']:.3f}, P={overall_metrics['precision']:.3f}, R={overall_metrics['recall']:.3f}")
    logger.info(f"Drug metrics: F1={overall_metrics['drug_f1']:.3f}, P={overall_metrics['drug_precision']:.3f}, R={overall_metrics['drug_recall']:.3f}")
    logger.info(f"ADE metrics: F1={overall_metrics['ade_f1']:.3f}, P={overall_metrics['ade_precision']:.3f}, R={overall_metrics['ade_recall']:.3f}")
    
    return results, overall_metrics

def generate_evaluation_report(test_results, overall_metrics, output_dir=None):
    """Generate an evaluation report with metrics and summary."""
    import os
    import json
    from datetime import datetime
    
    if output_dir is None:
        raise ValueError("Model output directory must be provided")
    
    # Create an evaluation subdirectory in the model directory
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
        
    # Save metrics to JSON
    with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
        json.dump(overall_metrics, f, indent=2)
    
    # Generate report
    report_path = os.path.join(eval_dir, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write("# ADE Extraction Evaluation Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overall Performance\n\n")
        f.write(f"| Metric | Overall | Drugs | ADEs |\n")
        f.write(f"|--------|---------|-------|------|\n")
        f.write(f"| F1 Score | {overall_metrics['f1']:.3f} | {overall_metrics['drug_f1']:.3f} | {overall_metrics['ade_f1']:.3f} |\n")
        f.write(f"| Precision | {overall_metrics['precision']:.3f} | {overall_metrics['drug_precision']:.3f} | {overall_metrics['ade_precision']:.3f} |\n")
        f.write(f"| Recall | {overall_metrics['recall']:.3f} | {overall_metrics['drug_recall']:.3f} | {overall_metrics['ade_recall']:.3f} |\n\n")
        
        f.write("## Example Results\n\n")
        for i, result in enumerate(test_results[:5]):  # Show first 5 examples
            f.write(f"### Example {i+1}\n\n")
            f.write(f"**Text:** {result['text'][:200]}...\n\n")
            
            f.write("**Gold Standard:**\n")
            f.write(f"- Drugs: {', '.join(result['gold']['drugs'])}\n")
            f.write(f"- ADEs: {', '.join(result['gold']['adverse_events'])}\n\n")
            
            f.write("**Model Extraction:**\n")
            f.write(f"- Drugs: {', '.join(result['extracted']['drugs'])}\n")
            f.write(f"- ADEs: {', '.join(result['extracted']['adverse_events'])}\n\n")
            
            f.write("**Metrics:**\n")
            f.write(f"- F1: {result['metrics']['overall']['f1']:.3f}\n")
            f.write(f"- Precision: {result['metrics']['overall']['precision']:.3f}\n")
            f.write(f"- Recall: {result['metrics']['overall']['recall']:.3f}\n\n")
    
    logger.info(f"Evaluation report generated at {report_path}")
    return eval_dir 