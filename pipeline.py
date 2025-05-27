#!/usr/bin/env python3
# Unified pipeline for ADE extraction with direct LLM and DSPy modes

import torch
import re
import json, os
import argparse
import logging
from tqdm import tqdm
import time
import dspy
from transformers import AutoModelForTokenClassification, AutoTokenizer

from utils.preprocessing import MedicalNoteProcessor, preprocess_notes
from utils.extraction import DirectLLMExtractor, extract_entities_improved, ADEExtractor, initialize_extractor, initialize_dspy_extractor, ADEOptimizer
from utils.dataset import ADEDataset, calculate_class_weights, ADEDatasetProcessor
from utils.training import ModernBERTFineTuner, find_optimal_learning_rate, evaluate_model, train_and_evaluate
from utils.utils import (
    load_medical_notes, create_bio_dataset, 
    save_llm_output, save_ner_data, save_bio_data, 
    load_llm_output, load_ner_data, load_bio_data, 
    load_gold_standard
)
from utils.evaluation import test_examples_with_evaluation, generate_evaluation_report, analyze_and_visualize_results
from utils.config import (
    LLM_OUTPUT_PATH_DIRECT, NER_OUTPUT_PATH_DIRECT, LLM_OUTPUT_PATH_DSPY, NER_OUTPUT_PATH_DSPY,
    BIO_TRAIN_PATH_DIRECT, BIO_VAL_PATH_DIRECT, BIO_TEST_PATH_DIRECT, 
    BIO_TRAIN_PATH_DSPY, BIO_VAL_PATH_DSPY, BIO_TEST_PATH_DSPY,
    ID_TO_LABEL, LLM_MODEL_NAME, GOLD_STANDARD_PATH, FINAL_MODEL_PATH,
    MAX_TOKENIZER_LENGTH, BATCH_SIZE, MAX_WORKERS, LOG_FORMAT, LOG_LEVEL, MAX_TEST_NOTES
)
from utils.logging_utils import configure_logging, disable_all_litellm_logs

# Set up logging and disable LiteLLM logs
configure_logging(level=LOG_LEVEL, format=LOG_FORMAT)
disable_all_litellm_logs()
logger = logging.getLogger(__name__)

class ADEExtractionPipeline:
    """
    A unified pipeline for extracting Adverse Drug Events (ADEs) from medical notes.
    
    This pipeline supports two extraction modes:
    - direct: Uses DirectLLMExtractor with OpenAI API directly
    - dspy: Uses DSPy for more sophisticated extraction with Chain-of-Thought
    """
    
    def __init__(self, mode="direct", force_extraction=False):
        """
        Initialize the ADE extraction pipeline.
        
        Args:
            mode (str): Extraction mode, either "direct" or "dspy"
            force_extraction (bool): Force re-extraction even if cached data exists
        """
        self.mode = mode
        self.force_extraction = force_extraction
        
        # Configure paths based on mode
        if mode == "direct":
            self.llm_output_path = LLM_OUTPUT_PATH_DIRECT
            self.ner_output_path = NER_OUTPUT_PATH_DIRECT
            self.bio_train_path = BIO_TRAIN_PATH_DIRECT
            self.bio_val_path = BIO_VAL_PATH_DIRECT
            self.bio_test_path = BIO_TEST_PATH_DIRECT
        elif mode == "dspy":
            self.llm_output_path = LLM_OUTPUT_PATH_DSPY
            self.ner_output_path = NER_OUTPUT_PATH_DSPY
            self.bio_train_path = BIO_TRAIN_PATH_DSPY
            self.bio_val_path = BIO_VAL_PATH_DSPY
            self.bio_test_path = BIO_TEST_PATH_DSPY
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Initialize pipeline components
        self._init_pipeline_components()
        
        logger.info(f"Initialized pipeline in {mode} mode")
    
    def _init_pipeline_components(self):
        """Initialize pipeline component variables."""
        # Data storage
        self.medical_notes = None
        self.test_notes = None
        self.processed_notes = None
        self.processed_test_notes = None
        self.extracted_data = None
        self.ner_data = None
        self.bio_records = None
        self.gold_annotations = None
        
        # Components
        self.extractor = None
        self.processor = None
        self.finetuner = None
        
        # Model and outputs
        self.model = None
        self.tokenizer = None
        self.model_output_dir = None
        self.training_metrics = None
        self.base_metrics = None
        self.finetuned_metrics = None
        self.test_results = None
        self.overall_metrics = None
        
        # Processing outputs
        self.all_texts = None
        self.all_tags = None
        self.all_input_ids = None
        self.all_attention_masks = None
    
    def load_data(self):
        """Load medical notes only (no test notes)."""
        logger.info("Loading data...")
        self.medical_notes = load_medical_notes()
        return self
    
    def preprocess_data(self):
        """Preprocess notes using MedicalNoteProcessor."""
        logger.info("Preprocessing notes...")
        train_processor = MedicalNoteProcessor(self.medical_notes)
        self.processed_notes = train_processor.preprocess_notes()
        return self
    
    def initialize_extraction(self):
        """Initialize the appropriate extractor based on the mode."""
        logger.info(f"Initializing {self.mode} extractor...")
        self.extractor = initialize_extractor(self.mode)
        self.processor = ADEDatasetProcessor(extractor=self.extractor)
        return self
    
    def extract_ades(self):
        """Extract ADEs from notes or load from cache."""
        if self.force_extraction or not os.path.exists(self.llm_output_path):
            logger.info(f"Extracting ADEs using {self.mode} approach...")
            self.extracted_data = self.processor.extract_ades_batched(self.processed_notes)
            save_llm_output(self.extracted_data, self.llm_output_path)
        else:
            logger.info(f"Loading existing LLM extraction output from {self.llm_output_path}")
            self.extracted_data = load_llm_output(self.llm_output_path)
            self.processor.extracted_data = self.extracted_data
        return self
    
    def optimize_extractor(self):
        """Optimize the DSPy extractor using human-annotated gold standard data if in DSPy mode."""
        if self.mode != "dspy":
            logger.info("Optimization is only available in DSPy mode. Skipping...")
            return self
            
        # Only optimize if the extractor is a DSPy extractor
        if not isinstance(self.extractor, ADEExtractor):
            return self
            
        logger.info("Optimizing DSPy extractor with ADEOptimizer using gold standard data...")
        try:
            # Create optimizer with current extractor
            optimizer = ADEOptimizer(self.extractor)
            with open(GOLD_STANDARD_PATH, 'r') as f:
                gold_data = json.load(f)
                
            examples = optimizer.prepare_examples(gold_data)
            
            # Run optimization with gold data for evaluation
            optimized_extractor = optimizer.optimize(examples)
            
            # Update extractors
            self.extractor = optimized_extractor
            self.processor.extractor = optimized_extractor
            
            logger.info("DSPy extractor optimization complete")
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            logger.warning("Continuing with non-optimized extractor")
            
        return self
    
    def prepare_ner_data(self):
        """Prepare NER data for training."""
        if not os.path.exists(self.ner_output_path) or self.force_extraction:
            logger.info("Preparing NER data...")
            self.ner_data = self.processor.prepare_ner_data(self.extracted_data)
            save_ner_data(self.ner_data, self.ner_output_path)
        else:
            logger.info(f"Loading existing NER data from {self.ner_output_path}")
            self.ner_data = load_ner_data(self.ner_output_path)
            self.processor.ner_data = self.ner_data
        return self
    
    def prepare_bio_data(self):
        """Prepare BIO data for ModernBERT fine-tuning."""
        # Initialize model and tokenizer
        self.finetuner = ModernBERTFineTuner(max_length=MAX_TOKENIZER_LENGTH)
        self.tokenizer = self.finetuner.tokenizer
        self.processor.tokenizer = self.tokenizer
        
        if not os.path.exists(self.bio_train_path) or self.force_extraction:
            logger.info("Preparing BIO data for ModernBERT fine-tuning...")
            bio_results = self.processor.prepare_bio_data(self.ner_data, self.tokenizer)
            self.all_texts, self.all_tags, self.all_input_ids, self.all_attention_masks, self.bio_records = bio_results
            save_bio_data(self.bio_records, self.bio_train_path)
        else:
            logger.info(f"Loading existing BIO data from {self.bio_train_path}")
            self.bio_records = load_bio_data(self.bio_train_path)
            self.all_texts = [record.get('text', '') for record in self.ner_data]
            self.all_tags = [record.get('tags', []) for record in self.bio_records]
            self.all_input_ids = [record.get('input_ids', []) for record in self.bio_records]
            self.all_attention_masks = [record.get('attention_mask', []) for record in self.bio_records]
        return self
    
    def train_model(self):
        """Train ModernBERT model on prepared BIO data."""
        # Use all_texts, all_tags, all_input_ids, all_attention_masks from BIO data
        # Evaluation will use gold standard data, not test_notes
        self.model, self.tokenizer, self.training_metrics, self.base_metrics, self.finetuned_metrics, self.model_output_dir = train_and_evaluate(
            self.all_texts, self.all_tags, self.all_input_ids, self.all_attention_masks, self.tokenizer, mode=self.mode
        )
        return self
    
    def evaluate_model(self):
        """Evaluate the model against gold standard annotations."""
        self.gold_annotations = load_gold_standard(GOLD_STANDARD_PATH)
        self.gold_annotations = self.gold_annotations[:MAX_TEST_NOTES]
        # Use the gold standard texts as input for evaluation
        test_texts = [entry["text"] for entry in self.gold_annotations if "text" in entry]
        logger.info("Testing model against gold standard annotations...")
        self.test_results, self.overall_metrics = test_examples_with_evaluation(
            self.model, self.tokenizer, test_texts, self.gold_annotations)
        return self
    
    def generate_reports(self):
        """Generate evaluation reports and visualizations."""
        # Create evaluation directory in model directory
        evaluation_dir = os.path.join(self.model_output_dir, "evaluation")
        os.makedirs(evaluation_dir, exist_ok=True)
        
        # Generate evaluation report if test results available
        if self.test_results and self.overall_metrics:
            generate_evaluation_report(self.test_results, self.overall_metrics, output_dir=self.model_output_dir)
        
        # Initialize base_metrics with defaults if missing (when using --skip-training)
        if not self.base_metrics:
            logger.info("No base model metrics available. Using defaults for visualizations.")
            self.base_metrics = {
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }
            
        # Ensure finetuned_metrics has proper keys
        if not self.finetuned_metrics:
            self.finetuned_metrics = self.overall_metrics if self.overall_metrics else {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        
        # Prepare pipeline results for visualizations
        pipeline_results = {
            "metrics": {
                "base_model": self.base_metrics,
                "finetuned_model": self.finetuned_metrics or {},
                "training": self.training_metrics or {}
            },
            "processed_data": {
                "extracted_data": self.extracted_data or []
            },
            "test_results": self.test_results or [],
            "evaluation": self.overall_metrics or {}
        }
        
        # Generate analysis visualizations
        analyze_and_visualize_results(pipeline_results, output_dir=self.model_output_dir)
        
        logger.info(f"Pipeline completed successfully. Results saved to {self.model_output_dir}")
        return self
    
    def load_best_model_for_mode(self):
        """Load the best or final ModernBERT model for the selected mode."""
        if self.mode == "dspy":
            prefix = "modernbert_dspy_approach_"
        else:
            prefix = "modernbert_direct_approach_"
        model_dirs = [os.path.join(FINAL_MODEL_PATH, d) for d in os.listdir(FINAL_MODEL_PATH) if d.startswith(prefix)]
        if not model_dirs:
            raise FileNotFoundError(f"No model directories found for mode {self.mode}")
        latest_dir = sorted(model_dirs, key=os.path.getctime, reverse=True)[0]
        final_model_path = os.path.join(latest_dir, "final_model")
        best_model_path = os.path.join(latest_dir, "best_model")
        if os.path.exists(final_model_path):
            model_path = final_model_path
        elif os.path.exists(best_model_path):
            model_path = best_model_path
        else:
            raise FileNotFoundError(f"No final_model or best_model found in {latest_dir}")
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model_output_dir = latest_dir
        logger.info(f"Loaded model and tokenizer from {model_path}")
    
    def run(self, skip_training=False):
        """Run the complete pipeline."""
        logger.info(f"Starting ADE extraction pipeline in {self.mode} mode")
        try:
            self.load_data()
            self.preprocess_data()
            self.initialize_extraction()
            self.extract_ades()
            self.optimize_extractor()
            self.prepare_ner_data()
            self.prepare_bio_data()
            if skip_training:
                logger.info("Skipping model training and loading best/final model for evaluation...")
                self.load_best_model_for_mode()
            else:
                self.train_model()
            self.evaluate_model()
            self.generate_reports()
            return self
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return self


def main():
    parser = argparse.ArgumentParser(description='ADE Extraction Pipeline')
    parser.add_argument('--mode', type=str, choices=['direct', 'dspy'], default='direct',
                      help='Pipeline mode: direct (default) or dspy')
    parser.add_argument('--force-extraction', action='store_true',
                      help='Force re-extraction even if cached data exists')
    parser.add_argument('--skip-training', action='store_true',
                      help='Skip model training and load best model for selected mode')
    args = parser.parse_args()
    
    # Create and run the pipeline with the selected mode
    pipeline = ADEExtractionPipeline(mode=args.mode, force_extraction=args.force_extraction)
    pipeline.run(skip_training=args.skip_training)

if __name__ == "__main__":
    main() 