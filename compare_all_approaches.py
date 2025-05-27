#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive evaluation script to compare different ADE extraction approaches:
1. Direct LLM approach
2. DSPy approach 
3. ModernBERT trained on direct LLM output
4. ModernBERT trained on DSPy output

This script loads the gold standard dataset and evaluates each approach,
producing comparative metrics and visualizations.
"""

import os
import json
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import AutoModelForTokenClassification, AutoTokenizer
from tqdm import tqdm
from datetime import datetime

# Import utilities
from utils.config import GOLD_STANDARD_PATH, MAX_TEST_NOTES
from utils.utils import load_gold_standard
from utils.extraction import (
    initialize_extractor, 
    DirectLLMExtractor, 
    extract_entities_improved
)
from utils.evaluation import calculate_entity_metrics
from utils.config import FINAL_MODEL_PATH
from utils.dataset import ADEDatasetProcessor

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_modernbert_models():
    """Find and load the latest ModernBERT models for both approaches."""
    # Look for the latest model of each type
    direct_models = glob.glob(os.path.join(FINAL_MODEL_PATH, "modernbert_direct_approach_*"))
    dspy_models = glob.glob(os.path.join(FINAL_MODEL_PATH, "modernbert_dspy_approach_*"))
    
    models = {
        "direct": None,
        "dspy": None
    }
    
    if direct_models:
        # Sort by creation date (newest first)
        latest_direct = sorted(direct_models, key=os.path.getctime, reverse=True)[0]
        logger.info(f"Loading Direct ModernBERT model from: {latest_direct}")
        
        # First try to find final_model folder
        final_model_path = os.path.join(latest_direct, "final_model")
        best_model_path = os.path.join(latest_direct, "best_model")
        
        # Use final_model if it exists, otherwise use best_model
        model_path = None
        if os.path.exists(final_model_path):
            model_path = final_model_path
            logger.info(f"Using final_model from {latest_direct}")
        elif os.path.exists(best_model_path):
            model_path = best_model_path
            logger.info(f"Using best_model from {latest_direct} (final_model not found)")
        
        if model_path:
            try:
                model = AutoModelForTokenClassification.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                models["direct"] = (model, tokenizer, latest_direct)
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {e}")
        else:
            logger.error(f"No model directory found in {latest_direct}")
    else:
        logger.warning("No Direct ModernBERT models found")
    
    if dspy_models:
        # Sort by creation date (newest first)
        latest_dspy = sorted(dspy_models, key=os.path.getctime, reverse=True)[0]
        logger.info(f"Loading DSPy ModernBERT model from: {latest_dspy}")
        
        # First try to find final_model folder
        final_model_path = os.path.join(latest_dspy, "final_model")
        best_model_path = os.path.join(latest_dspy, "best_model")
        
        # Use final_model if it exists, otherwise use best_model
        model_path = None
        if os.path.exists(final_model_path):
            model_path = final_model_path
            logger.info(f"Using final_model from {latest_dspy}")
        elif os.path.exists(best_model_path):
            model_path = best_model_path
            logger.info(f"Using best_model from {latest_dspy} (final_model not found)")
        
        if model_path:
            try:
                model = AutoModelForTokenClassification.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                models["dspy"] = (model, tokenizer, latest_dspy)
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {e}")
        else:
            logger.error(f"No model directory found in {latest_dspy}")
    else:
        logger.warning("No DSPy ModernBERT models found")
    
    return models

def evaluate_llm_approaches(test_notes, gold_data):
    """Evaluate both LLM-based approaches: Direct and DSPy using pipeline logic."""
    results = {
        "direct": {"metrics": [], "examples": []},
        "dspy": {"metrics": [], "examples": []}
    }

    # Direct approach
    logger.info("Extracting ADEs using Direct approach (pipeline logic)...")
    direct_extractor = initialize_extractor(mode="direct")
    direct_processor = ADEDatasetProcessor(extractor=direct_extractor)
    direct_extracted = direct_processor.extract_ades_batched(test_notes)
    for i, (record, gold) in enumerate(zip(direct_extracted, gold_data)):
        direct_metrics = calculate_entity_metrics(record, gold)
        results["direct"]["metrics"].append(direct_metrics)
        if i < 10:
            results["direct"]["examples"].append({
                "text": record["text"],
                "extracted": record,
                "gold": gold,
                "metrics": direct_metrics
            })

    # DSPy approach
    logger.info("Extracting ADEs using DSPy approach (pipeline logic)...")
    dspy_extractor = initialize_extractor(mode="dspy")
    dspy_processor = ADEDatasetProcessor(extractor=dspy_extractor)
    dspy_extracted = dspy_processor.extract_ades_batched(test_notes)
    for i, (record, gold) in enumerate(zip(dspy_extracted, gold_data)):
        dspy_metrics = calculate_entity_metrics(record, gold)
        results["dspy"]["metrics"].append(dspy_metrics)
        if i < 10:
            results["dspy"]["examples"].append({
                "text": record["text"],
                "extracted": record,
                "gold": gold,
                "metrics": dspy_metrics
            })

    # Print a summary of the metrics
    direct_f1_avg = sum(m["overall"]["f1"] for m in results["direct"]["metrics"]) / len(results["direct"]["metrics"]) if results["direct"]["metrics"] else 0
    dspy_f1_avg = sum(m["overall"]["f1"] for m in results["dspy"]["metrics"]) / len(results["dspy"]["metrics"]) if results["dspy"]["metrics"] else 0

    logger.info(f"Direct LLM Approach - Average F1: {direct_f1_avg:.3f}")
    logger.info(f"DSPy Approach - Average F1: {dspy_f1_avg:.3f}")

    return results

def evaluate_modernbert_models(models, test_notes, gold_data):
    """Evaluate ModernBERT models trained on different approaches."""
    results = {
        "direct": {"metrics": [], "examples": []},
        "dspy": {"metrics": [], "examples": []}
    }
    
    # Set up device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Evaluate Direct-trained ModernBERT
    if models["direct"]:
        model, tokenizer, model_path = models["direct"]
        model.to(device)
        model.eval()
        
        logger.info(f"Evaluating ModernBERT trained on Direct LLM output: {model_path}")
        
        for i, (note, gold) in enumerate(tqdm(zip(test_notes, gold_data), total=len(test_notes))):
            # Extract entities
            extracted = extract_entities_improved(note, model, tokenizer, device)
            
            # Calculate metrics
            metrics = calculate_entity_metrics(extracted, gold)
            results["direct"]["metrics"].append(metrics)
            
            # Save example (first 10 only)
            if i < 10:
                results["direct"]["examples"].append({
                    "text": note[:200] + "..." if len(note) > 200 else note,
                    "extracted": extracted,
                    "gold": gold,
                    "metrics": metrics
                })
    else:
        logger.warning("Skipping Direct-trained ModernBERT evaluation (no model available)")
    
    # Evaluate DSPy-trained ModernBERT
    if models["dspy"]:
        model, tokenizer, model_path = models["dspy"]
        model.to(device)
        model.eval()
        
        logger.info(f"Evaluating ModernBERT trained on DSPy output: {model_path}")
        
        for i, (note, gold) in enumerate(tqdm(zip(test_notes, gold_data), total=len(test_notes))):
            # Extract entities
            extracted = extract_entities_improved(note, model, tokenizer, device)
            
            # Calculate metrics
            metrics = calculate_entity_metrics(extracted, gold)
            results["dspy"]["metrics"].append(metrics)
            
            # Save example (first 10 only)
            if i < 10:
                results["dspy"]["examples"].append({
                    "text": note[:200] + "..." if len(note) > 200 else note,
                    "extracted": extracted,
                    "gold": gold,
                    "metrics": metrics
                })
    else:
        logger.warning("Skipping DSPy-trained ModernBERT evaluation (no model available)")
    
    return results

def calculate_overall_metrics(metrics_list):
    """Calculate average metrics across all examples."""
    if not metrics_list:
        return {
            "overall": {"precision": 0, "recall": 0, "f1": 0},
            "drug": {"precision": 0, "recall": 0, "f1": 0},
            "ade": {"precision": 0, "recall": 0, "f1": 0}
        }
    
    overall_metrics = {
        "overall": {"precision": 0, "recall": 0, "f1": 0},
        "drug": {"precision": 0, "recall": 0, "f1": 0},
        "ade": {"precision": 0, "recall": 0, "f1": 0}
    }
    
    for metrics in metrics_list:
        # Overall metrics
        overall_metrics["overall"]["precision"] += metrics["overall"]["precision"]
        overall_metrics["overall"]["recall"] += metrics["overall"]["recall"]
        overall_metrics["overall"]["f1"] += metrics["overall"]["f1"]
        
        # Drug metrics
        overall_metrics["drug"]["precision"] += metrics["drug"]["precision"]
        overall_metrics["drug"]["recall"] += metrics["drug"]["recall"]
        overall_metrics["drug"]["f1"] += metrics["drug"]["f1"]
        
        # ADE metrics
        overall_metrics["ade"]["precision"] += metrics["ade"]["precision"]
        overall_metrics["ade"]["recall"] += metrics["ade"]["recall"]
        overall_metrics["ade"]["f1"] += metrics["ade"]["f1"]
    
    # Calculate averages
    count = len(metrics_list)
    for category in overall_metrics:
        for metric in overall_metrics[category]:
            overall_metrics[category][metric] /= count
    
    return overall_metrics

def visualize_results(llm_results, modernbert_results):
    """Create visualizations comparing all four approaches."""
    # Create results directory
    results_dir = os.path.join("analysis", "comparison_results", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(results_dir, exist_ok=True)
    
    # Calculate overall metrics for each approach
    metrics = {
        "Direct LLM": calculate_overall_metrics(llm_results["direct"]["metrics"]),
        "DSPy LLM": calculate_overall_metrics(llm_results["dspy"]["metrics"]),
        "ModernBERT (Direct)": calculate_overall_metrics(modernbert_results["direct"]["metrics"]),
        "ModernBERT (DSPy)": calculate_overall_metrics(modernbert_results["dspy"]["metrics"])
    }
    
    # Save metrics as JSON
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Create metrics DataFrame for plotting
    df_data = []
    for approach, approach_metrics in metrics.items():
        for entity_type in ["overall", "drug", "ade"]:
            for metric_name, metric_value in approach_metrics[entity_type].items():
                df_data.append({
                    "Approach": approach,
                    "Entity Type": entity_type.capitalize(),
                    "Metric": metric_name.capitalize(),
                    "Value": metric_value
                })
    
    df = pd.DataFrame(df_data)
    
    # Plot 1: Overall metrics comparison
    plt.figure(figsize=(12, 8))
    
    # Filter for overall metrics
    df_overall = df[df["Entity Type"] == "Overall"]
    
    # Create grouped bar chart
    sns.barplot(x="Approach", y="Value", hue="Metric", data=df_overall)
    plt.title("Overall Metrics Comparison Across Approaches", fontsize=16)
    plt.xlabel("Approach", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(title="Metric")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "overall_metrics.png"))
    
    # Plot 2: F1 scores by entity type
    plt.figure(figsize=(12, 8))
    
    # Filter for F1 scores
    df_f1 = df[df["Metric"] == "F1"]
    
    # Create grouped bar chart
    sns.barplot(x="Approach", y="Value", hue="Entity Type", data=df_f1)
    plt.title("F1 Scores by Entity Type Across Approaches", fontsize=16)
    plt.xlabel("Approach", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(title="Entity Type")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "f1_by_entity.png"))
    
    # Plot 3: Heatmap of all metrics
    plt.figure(figsize=(15, 10))
    
    # Pivot data for heatmap
    heatmap_data = df.pivot_table(
        index=["Approach", "Entity Type"],
        columns="Metric",
        values="Value"
    ).reset_index()
    
    # Create a properly shaped array for the heatmap
    approaches = df["Approach"].unique()
    entity_types = df["Entity Type"].unique()
    metrics_names = df["Metric"].unique()
    
    # Create labels for the heatmap
    row_labels = []
    for app in approaches:
        for ent in entity_types:
            row_labels.append(f"{app} - {ent}")
    
    # Create the data array
    heat_data = np.zeros((len(approaches) * len(entity_types), len(metrics_names)))
    
    for i, approach in enumerate(approaches):
        for j, entity_type in enumerate(entity_types):
            for k, metric_name in enumerate(metrics_names):
                filtered = df[(df["Approach"] == approach) & 
                             (df["Entity Type"] == entity_type) & 
                             (df["Metric"] == metric_name)]
                
                if not filtered.empty:
                    row_idx = i * len(entity_types) + j
                    heat_data[row_idx, k] = filtered["Value"].values[0]
    
    # Create the heatmap
    plt.figure(figsize=(10, 12))
    sns.heatmap(
        heat_data,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=metrics_names,
        yticklabels=row_labels,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Score"}
    )
    plt.title("Comprehensive Metrics Heatmap", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "metrics_heatmap.png"))
    
    # Create a summary report
    with open(os.path.join(results_dir, "summary_report.md"), "w") as f:
        f.write("# ADE Extraction Approaches Comparison\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary of Results\n\n")
        f.write("### Overall F1 Scores\n\n")
        f.write("| Approach | Overall F1 | Drug F1 | ADE F1 |\n")
        f.write("|----------|------------|---------|--------|\n")
        
        for approach, approach_metrics in metrics.items():
            overall_f1 = approach_metrics["overall"]["f1"]
            drug_f1 = approach_metrics["drug"]["f1"]
            ade_f1 = approach_metrics["ade"]["f1"]
            f.write(f"| {approach} | {overall_f1:.3f} | {drug_f1:.3f} | {ade_f1:.3f} |\n")
        
        f.write("\n### Detailed Metrics\n\n")
        
        for approach, approach_metrics in metrics.items():
            f.write(f"#### {approach}\n\n")
            f.write("| Entity Type | Precision | Recall | F1 |\n")
            f.write("|-------------|-----------|--------|----|\n")
            
            for entity_type, entity_metrics in approach_metrics.items():
                precision = entity_metrics["precision"]
                recall = entity_metrics["recall"]
                f1 = entity_metrics["f1"]
                f.write(f"| {entity_type.capitalize()} | {precision:.3f} | {recall:.3f} | {f1:.3f} |\n")
            
            f.write("\n")
    
    logger.info(f"Results saved to: {results_dir}")
    return results_dir

def main():
    # Load gold standard data only
    logger.info(f"Loading gold standard data from {GOLD_STANDARD_PATH}")
    gold_data = load_gold_standard(GOLD_STANDARD_PATH)
    gold_data = gold_data[:MAX_TEST_NOTES]
    
    # Use gold data as test notes and gold labels
    test_notes = [entry["text"] for entry in gold_data if "text" in entry]
    matched_gold = gold_data  # Already aligned
    
    logger.info(f"Using {len(test_notes)} gold standard notes for evaluation")
    
    # Evaluate LLM approaches
    logger.info("Evaluating LLM approaches (Direct and DSPy)...")
    llm_results = evaluate_llm_approaches(test_notes, matched_gold)
    
    # Load and evaluate ModernBERT models
    logger.info("Loading ModernBERT models...")
    modernbert_models = load_modernbert_models()
    
    logger.info("Evaluating ModernBERT models...")
    modernbert_results = evaluate_modernbert_models(modernbert_models, test_notes, matched_gold)
    
    # Visualize results
    logger.info("Generating visualizations and summary...")
    results_dir = visualize_results(llm_results, modernbert_results)
    
    logger.info(f"Evaluation complete. Results saved to: {results_dir}")

if __name__ == "__main__":
    main() 