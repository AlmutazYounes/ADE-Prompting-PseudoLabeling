#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: Model Evaluation
-----------------------
This script evaluates the trained BERT models against the gold standard data
and compares their performance with direct LLM and DSPy approaches.

The script produces performance metrics and visualizations that are saved to
the analysis folder.
"""

import os
import json
import shutil
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Import the separate evaluators
from .bert_evaluator import (
    load_from_jsonl, 
    print_banner, 
    entity_level_metrics,
    calculate_overall_metrics, 
    evaluate_bert_models
)
from .llm_evaluator import (
    evaluate_llm_approaches,
    get_display_name
)

# Constants
STEP_2_GOLD_NER_DATA = os.path.join("Step_1_data_generation", "data", "gold", "gold_ner_data.jsonl")
COMPARISON_RESULTS_DIR = os.path.join("analysis", "comparison_results")
USE_CACHE_DEFAULT = True
OVERWRITE_CACHE_DEFAULT = False
AVAILABLE_DATA_SOURCES = ["direct", "dspy", "pipeline", "validator", "structured"]
VISUALIZATION = {
    "figsize_default": (12, 8),
    "colormap": "YlGnBu",
    "dpi": 100
}

#############################
# Visualization Functions
#############################

def save_metrics_json(metrics, results_dir):
    """Save metrics as a JSON file."""
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics_path

def create_f1_comparison(metrics, results_dir):
    """Create and save a bar chart comparing F1 scores across approaches."""
    # Convert metrics to DataFrame
    df = pd.DataFrame([
        {"Approach": approach, "F1 Score": metrics_dict["f1"]} 
        for approach, metrics_dict in metrics.items()
    ])
    
    # Create plot
    plt.figure(figsize=VISUALIZATION["figsize_default"])
    sns.barplot(x="Approach", y="F1 Score", data=df)
    plt.title("F1 Score Comparison Across Approaches")
    plt.ylim(0, 1.0)
    plt.ylabel("F1 Score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    # Save figure
    f1_path = os.path.join(results_dir, "f1_comparison.png")
    plt.savefig(f1_path, dpi=VISUALIZATION["dpi"])
    plt.close()
    
    return f1_path

def create_prf_comparison(metrics, results_dir):
    """Create and save a grouped bar chart comparing precision, recall, and F1."""
    # Melt the data for grouped bars
    df_list = []
    for approach, metrics_dict in metrics.items():
        for metric, value in metrics_dict.items():
            df_list.append({
                "Approach": approach,
                "Metric": metric.capitalize(),
                "Score": value
            })
    
    df = pd.DataFrame(df_list)
    
    # Create plot
    plt.figure(figsize=VISUALIZATION["figsize_default"])
    sns.barplot(x="Approach", y="Score", hue="Metric", data=df)
    plt.title("Precision, Recall, F1 Comparison Across Approaches")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15)
    plt.legend(title="Metric")
    plt.tight_layout()
    
    # Save figure
    prf_path = os.path.join(results_dir, "prf_comparison.png")
    plt.savefig(prf_path, dpi=VISUALIZATION["dpi"])
    plt.close()
    
    return prf_path

def create_summary_report(metrics, results_dir):
    """Create and save a text report summarizing the evaluation results."""
    report_path = os.path.join(results_dir, "summary_report.txt")
    
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall performance table
        f.write("PERFORMANCE COMPARISON:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Approach':<25} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}\n")
        f.write("-" * 80 + "\n")
        
        for approach, metrics_dict in sorted(metrics.items()):
            precision = metrics_dict.get("precision", 0)
            recall = metrics_dict.get("recall", 0)
            f1 = metrics_dict.get("f1", 0)
            
            f.write(f"{approach:<25} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}\n")
        
        f.write("\n\n")
        
        # Best model
        best_approach = max(metrics.items(), key=lambda x: x[1]["f1"])[0]
        best_f1 = metrics[best_approach]["f1"]
        
        f.write("BEST PERFORMING MODEL:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Approach: {best_approach}\n")
        f.write(f"F1 Score: {best_f1:.4f}\n")
        f.write(f"Precision: {metrics[best_approach]['precision']:.4f}\n")
        f.write(f"Recall: {metrics[best_approach]['recall']:.4f}\n")
        
        f.write("\n\n")
        f.write("=" * 80 + "\n")
    
    return report_path

def analyze_error_patterns(predictions, gold_data, output_dir):
    """Analyze error patterns in model predictions."""
    # Initialize counters
    false_positives = Counter()
    false_negatives = Counter()
    true_positives = Counter()
    
    # Analyze each example
    for pred, gold in zip(predictions, gold_data):
        pred_entities = {(e["start"], e["end"], e["label"]) for e in pred.get("entities", [])}
        gold_entities = {(e["start"], e["end"], e["label"]) for e in gold.get("entities", [])}
        
        # Find false positives (in pred but not in gold)
        for entity in pred_entities:
            if entity not in gold_entities:
                label = entity[2]
                false_positives[label] += 1
        
        # Find false negatives (in gold but not in pred)
        for entity in gold_entities:
            if entity not in pred_entities:
                label = entity[2]
                false_negatives[label] += 1
            else:
                # True positives
                label = entity[2]
                true_positives[label] += 1
    
    # Create results dictionary
    results = {
        "false_positives": dict(false_positives),
        "false_negatives": dict(false_negatives),
        "true_positives": dict(true_positives)
    }
    
    # Calculate entity-level metrics
    entity_metrics = {}
    all_labels = set(list(true_positives.keys()) + 
                     list(false_positives.keys()) + 
                     list(false_negatives.keys()))
    
    for label in all_labels:
        tp = true_positives.get(label, 0)
        fp = false_positives.get(label, 0)
        fn = false_negatives.get(label, 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        entity_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn
        }
    
    results["entity_metrics"] = entity_metrics
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "error_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    plot_error_analysis(results, output_dir)
    
    return results

def plot_error_analysis(results, output_dir):
    """Create charts visualizing error analysis results."""
    # Entity-level metrics chart
    if "entity_metrics" in results:
        metrics = results["entity_metrics"]
        
        # Convert to DataFrame
        df_list = []
        for label, values in metrics.items():
            for metric_name, value in values.items():
                if metric_name != "support":  # Exclude support from the chart
                    df_list.append({
                        "Entity": label,
                        "Metric": metric_name.capitalize(),
                        "Value": value
                    })
        
        df = pd.DataFrame(df_list)
        
        # Create plot
        plt.figure(figsize=VISUALIZATION["figsize_default"])
        sns.barplot(x="Entity", y="Value", hue="Metric", data=df)
        plt.title("Performance Metrics by Entity Type")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "entity_metrics.png"), dpi=VISUALIZATION["dpi"])
        plt.close()
    
    # Error distribution chart
    fp = results.get("false_positives", {})
    fn = results.get("false_negatives", {})
    
    # Convert to DataFrame
    df_list = []
    for label in set(list(fp.keys()) + list(fn.keys())):
        df_list.append({
            "Entity": label,
            "Error Type": "False Positives",
            "Count": fp.get(label, 0)
        })
        df_list.append({
            "Entity": label,
            "Error Type": "False Negatives",
            "Count": fn.get(label, 0)
        })
    
    df = pd.DataFrame(df_list)
    
    # Create plot
    plt.figure(figsize=VISUALIZATION["figsize_default"])
    sns.barplot(x="Entity", y="Count", hue="Error Type", data=df)
    plt.title("Error Distribution by Entity Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "error_distribution.png"), dpi=VISUALIZATION["dpi"])
    plt.close()

#############################
# Main Function
#############################

def run_evaluation(config):
    """
    Run the complete model evaluation pipeline using a config dict.
    Keys expected in config:
        - gold_data_path
        - max_notes
        - output_dir
        - name
        - use_cache
        - overwrite_cache
        - skip_llm
        - detailed_analysis
        - data_sources (list of sources to evaluate)
    """
    # Print banner
    print_banner("STEP 3: MODEL EVALUATION")
    
    # Generate output directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = config.get("name")
    if name:
        dir_name = f"{timestamp}_{name}"
    else:
        dir_name = f"{timestamp}_evaluation"
    
    results_dir = config.get("output_dir") if config.get("output_dir") else os.path.join(COMPARISON_RESULTS_DIR, dir_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"📊 Results will be saved to: {results_dir}")
    
    # Load gold standard data
    gold_data_path = config.get("gold_data_path", STEP_2_GOLD_NER_DATA)
    # Only use max_notes/max_test_notes from config
    max_notes = config.get("max_notes")
    if max_notes is None:
        max_notes = config.get("max_test_notes")
    print(f"\n{'='*70}")
    print(f"📂 Loading gold standard data from {gold_data_path}")
    gold_data = load_from_jsonl(gold_data_path)
    if max_notes is not None:
        gold_data = gold_data[:max_notes]
    
    test_notes = [entry["text"] for entry in gold_data if "text" in entry]
    print(f"✅ Using {len(test_notes)} gold standard notes for evaluation")
    print(f"{'='*70}\n")
    
    # Get data sources to evaluate
    data_sources = config.get("data_sources", [])
    if not data_sources:
        print("⚠️ No data sources specified for evaluation. Exiting.")
        return 1
    print(f"🔎 Evaluating data sources: {data_sources}")
    
    # Initialize results dictionary
    all_results = {}
    
    # Evaluate BERT models
    bert_results = evaluate_bert_models(gold_data, data_sources)
    all_results.update(bert_results)
    
    # Evaluate LLM approaches if not skipped and if selected
    if not config.get("skip_llm", False):
        llm_data_sources = [source for source in data_sources if source in AVAILABLE_DATA_SOURCES]
        if llm_data_sources:
            print_banner(f"EVALUATING LLM APPROACHES: {', '.join(llm_data_sources)}")
            llm_results = evaluate_llm_approaches(
                test_notes, 
                gold_data,
                use_cache=config.get("use_cache", USE_CACHE_DEFAULT),
                overwrite_cache=config.get("overwrite_cache", OVERWRITE_CACHE_DEFAULT),
                data_sources=llm_data_sources
            )
            
            # Add results with descriptive names to all_results
            for source, results in llm_results.items():
                display_name = get_display_name(source)
                all_results[display_name] = results
    
    # Calculate overall metrics
    print_banner("CALCULATING OVERALL METRICS")
    metrics = {k: calculate_overall_metrics(v) for k, v in all_results.items()}
    
    # Save metrics as JSON
    print(f"💾 Saving metrics to {results_dir}")
    save_metrics_json(metrics, results_dir)
    
    # Create visualizations
    print_banner("GENERATING VISUALIZATIONS")
    create_f1_comparison(metrics, results_dir)
    create_prf_comparison(metrics, results_dir)
    create_summary_report(metrics, results_dir)
        
    # Detailed error analysis if requested
    if config.get("detailed_analysis", False):
        print_banner("PERFORMING ERROR ANALYSIS")
        # Implement error analysis here if needed
        print("⚠️ Detailed error analysis requested but not implemented in this version")
    
    # Print summary
    print_banner("EVALUATION SUMMARY")
    print("\n" + "=" * 80)
    print(f"{'Approach':<25} {'F1':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 80)
    for approach, metrics_dict in sorted(metrics.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"{approach:<25} {metrics_dict['f1']:<10.4f} {metrics_dict['precision']:<10.4f} {metrics_dict['recall']:<10.4f}")
    print("=" * 80 + "\n")
    
    print(f"✅ Evaluation complete. Results saved to: {results_dir}")
    
    return 0

if __name__ == "__main__":
    # Example config dict (edit as needed)
    config = {
        "gold_data_path": STEP_2_GOLD_NER_DATA,
        "max_notes": 100,  # Default test size
        "output_dir": None,  # or specify a path
        "name": None,        # or specify a name
        "use_cache": True,
        "overwrite_cache": False,
        "skip_llm": False,
        "detailed_analysis": False,
        "data_sources": ["direct", "dspy"]
    }
    exit(run_evaluation(config)) 