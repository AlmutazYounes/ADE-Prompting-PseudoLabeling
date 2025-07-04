#!/usr/bin/env python3

import os
import json
import logging
import openai
import dspy
import ast
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from dotenv import load_dotenv
import shutil
import asyncio
import time

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose logging from external libraries
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('litellm').setLevel(logging.WARNING)
logging.getLogger('LiteLLM').setLevel(logging.WARNING)
logging.getLogger('dspy').setLevel(logging.WARNING)

# Configuration settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STRUCTURED_DATA_DIR = os.path.join(DATA_DIR, "structured")

# Ensure output directory exists
os.makedirs(STRUCTURED_DATA_DIR, exist_ok=True)

# ========= DEFINE STRUCTURED DSPy SIGNATURES =========

class StructuredEntityIdentification(dspy.Signature):
    """Identify all possible drug names and adverse events in the clinical text with detailed reasoning."""
    clinical_text = dspy.InputField(desc="Clinical note text that contains medical information")
    
    detailed_reasoning = dspy.OutputField(desc="Step-by-step analysis to identify every potential drug name and adverse event, considering context and medical terminology")
    
    identified_drugs = dspy.OutputField(
        desc="List of drug objects, each with name, start_index, end_index, and confidence level (high/medium/low)",
        prefix="Drugs identified with position information:"
    )
    
    identified_adverse_events = dspy.OutputField(
        desc="List of adverse event objects, each with name, start_index, end_index, and confidence level (high/medium/low)",
        prefix="Adverse events identified with position information:"
    )

class StructuredEntityRefinement(dspy.Signature):
    """Refine the identified entities by removing false positives and fixing boundaries."""
    clinical_text = dspy.InputField(desc="Original clinical text")
    
    candidate_drugs = dspy.InputField(desc="List of candidate drugs with position information")
    
    candidate_adverse_events = dspy.InputField(desc="List of candidate adverse events with position information")
    
    refinement_reasoning = dspy.OutputField(
        desc="Step-by-step reasoning to validate each entity, fix boundaries, and remove false positives"
    )
    
    refined_drugs = dspy.OutputField(
        desc="Final list of drug objects with corrected boundaries and high confidence"
    )
    
    refined_adverse_events = dspy.OutputField(
        desc="Final list of adverse event objects with corrected boundaries and high confidence"
    )

# ========= DEFINE DSPy MODULES =========

class StructuredDrugADEExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.identifier = dspy.ChainOfThought(StructuredEntityIdentification)
        self.refiner = dspy.ChainOfThought(StructuredEntityRefinement)
    
    def forward(self, clinical_text):
        # Step 1: Identify candidate entities with their positions
        identification_result = self.identifier(clinical_text=clinical_text)
        
        # Step 2: Refine entities (remove false positives, fix boundaries)
        refinement_result = self.refiner(
            clinical_text=clinical_text,
            candidate_drugs=identification_result.identified_drugs,
            candidate_adverse_events=identification_result.identified_adverse_events
        )
        
        # Extract the final drug and ADE names (without position info)
        try:
            drugs = self._extract_entity_names(refinement_result.refined_drugs)
            adverse_events = self._extract_entity_names(refinement_result.refined_adverse_events)
        except Exception as e:
            # Fallback if parsing fails
            logger.warning(f"Error parsing structured output: {e}")
            drugs = []
            adverse_events = []
            
            # Try to extract directly from strings
            drugs = self._extract_from_string(refinement_result.refined_drugs)
            adverse_events = self._extract_from_string(refinement_result.refined_adverse_events)
        
        # Store detailed reasoning for debugging
        self._detailed_reasoning = {
            'identification': identification_result.detailed_reasoning,
            'refinement': refinement_result.refinement_reasoning
        }
        
        return drugs, adverse_events
    
    def _extract_entity_names(self, entity_list_str):
        """Extract just the entity names from structured output."""
        if not entity_list_str:
            return []
        
        # Try multiple parsing approaches
        try:
            # First try: Direct JSON/list parsing with ast.literal_eval
            if entity_list_str.strip().startswith('[') and entity_list_str.strip().endswith(']'):
                parsed = ast.literal_eval(entity_list_str)
                if isinstance(parsed, list):
                    # Extract entity names from objects or strings
                    result = []
                    for item in parsed:
                        if isinstance(item, dict) and 'name' in item:
                            result.append(item['name'])
                        elif isinstance(item, str):
                            result.append(item)
                    return result
        except:
            pass
        
        # Second try: Extract names with regex-like approach
        return self._extract_from_string(entity_list_str)
    
    def _extract_from_string(self, text):
        """Extract entity names from any string format using simple text processing."""
        if not text:
            return []
        
        # Remove common formatting
        clean_text = text.replace('"', '').replace("'", "").strip()
        
        # Check for empty list indicators with comments
        if '[]' in clean_text and '//' in clean_text:
            return []
        
        # Check for explicit empty indicators
        if clean_text in ['[]', 'None', 'null', 'none', 'empty', '']:
            return []
        
        # Try to find items in a list-like format
        if "name:" in clean_text.lower() or "name :" in clean_text.lower():
            # Extract names from structured format with 'name:' labels
            result = []
            for line in clean_text.split('\n'):
                if "name:" in line.lower() or "name :" in line.lower():
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        name = parts[1].strip().strip(',').strip('"\'[]{}()').strip()
                        if name and not name.startswith('//') and name not in ['[]', 'None', 'null']:
                            result.append(name)
            return result
        
        # Try splitting by common separators
        for sep in [',', ';', '\n']:
            if sep in clean_text:
                items = [item.strip().strip('"\'[]{}()') for item in clean_text.split(sep)]
                items = [item for item in items if item and not item.startswith('start_') and not item.startswith('end_') and not item.startswith('//') and item not in ['[]', 'None', 'null']]
                if items:
                    return items
        
        # If we can't parse it, return the whole string as one item if it's not empty and not a comment
        if clean_text and clean_text not in ['[]', 'None', 'null'] and not clean_text.startswith('//'):
            return [clean_text]
        
        return []

def load_notes(input_file: str, max_notes: int) -> List[str]:
    """Load clinical notes from text file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        notes = [line.strip() for line in f if line.strip()]
    return notes[:max_notes]

def create_entities(text: str, drugs: List[str], adverse_events: List[str]) -> List[Dict[str, Any]]:
    """Create entity spans for NER data format."""
    entities = []
    
    # Find all drug entities
    for drug in drugs:
        if drug and drug in text:
            start_index = 0
            while True:
                start_pos = text.find(drug, start_index)
                if start_pos == -1:
                    break
                entities.append({
                    'start': start_pos,
                    'end': start_pos + len(drug),
                    'label': 'DRUG'
                })
                start_index = start_pos + 1
    
    # Find all adverse event entities
    for ade in adverse_events:
        if ade and ade in text:
            start_index = 0
            while True:
                start_pos = text.find(ade, start_index)
                if start_pos == -1:
                    break
                entities.append({
                    'start': start_pos,
                    'end': start_pos + len(ade),
                    'label': 'ADE'
                })
                start_index = start_pos + 1
    
    # Sort by start position
    entities.sort(key=lambda x: x['start'])
    return entities

def save_to_jsonl(data: List[Dict[str, Any]], output_path: str):
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Saved {len(data)} examples to {output_path}")

def print_banner(message):
    """Print a formatted banner message."""
    term_width = shutil.get_terminal_size((80, 20)).columns
    banner_text = f" {message} "
    banner = f"\033[1;44m{banner_text.center(term_width)}\033[0m"
    logger.info("\n" + banner + "\n")

async def process_note(note: str, extractor: StructuredDrugADEExtractor) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Process a single note with the structured DSPy extractor."""
    # Extract drugs and adverse events
    drugs, adverse_events = extractor(note)
    
    # Create NER format for training
    ner_data = {
        "text": note,
        "entities": create_entities(note, drugs, adverse_events)
    }
    
    # Create extracted format (clean format)
    extracted_data = {
        "text": note,
        "drugs": drugs,
        "adverse_events": adverse_events
    }
    
    return ner_data, extracted_data

async def process_batch(batch: List[str], extractor: StructuredDrugADEExtractor) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process a batch of notes."""
    # Create tasks for processing all notes in the batch
    tasks = [process_note(note, extractor) for note in batch]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Split results into ner_data and extracted_data
    ner_results = [item[0] for item in results]
    extracted_results = [item[1] for item in results]
    
    return ner_results, extracted_results

def run_structured_generation(config: Dict) -> Dict:
    """Run structured DSPy-based generation with the provided config."""
    print_banner("STRUCTURED DSPy-BASED ADE EXTRACTION")
    
    logger.info("=" * 70)
    logger.info(f"Model:        {config.get('model_name', 'gpt-4.1-nano')}")
    logger.info(f"Temperature:  {config.get('temperature', 0.1)}")
    logger.info(f"Max tokens:   {config.get('max_tokens', 2000)}")
    logger.info(f"Batch size:   {config.get('batch_size', 5)}")
    logger.info(f"Approach:     Structured entity extraction with DSPy")
    logger.info("=" * 70)
    
    # Setup DSPy configuration
    openai_api_key = os.getenv('OPENAI_API_KEY')
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Configure DSPy with model settings from config
    dspy.settings.configure(lm=dspy.LM(
        model=config.get('model_name', 'gpt-4.1-nano'),
        temperature=config.get('temperature', 0.1),
        max_tokens=config.get('max_tokens', 2000)
    ))
    
    # Set file paths from config
    input_file = config.get('input_file')
    output_file = config.get('structured_output_file')
    extracted_file = config.get('structured_extracted_file')
    max_notes = config.get('max_notes', 100)
    batch_size = config.get('batch_size', 5)  # Smaller batch size for structured approach
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(extracted_file), exist_ok=True)
    
    # Check if output files already exist and use_cache is enabled
    if not config.get('overwrite_cache', False) and os.path.exists(output_file) and os.path.exists(extracted_file):
        logger.info("-" * 70)
        logger.info(f"Using cached results from:")
        logger.info(f"  - {output_file}")
        logger.info(f"  - {extracted_file}")
        logger.info("-" * 70)
        return {"status": "cached"}
    
    notes = load_notes(input_file, max_notes)
    logger.info(f"Loaded {len(notes)} clinical notes from {input_file}")
    
    # Initialize the structured extractor
    extractor = StructuredDrugADEExtractor()
    
    # Statistics
    stats = {
        "total_drugs": 0,
        "total_ades": 0,
        "notes_with_drugs": 0,
        "notes_with_ades": 0
    }
    
    # Processing indicator
    logger.info("-" * 70)
    logger.info("Starting structured DSPy extraction...")
    logger.info(f"Processing {len(notes)} notes in batches of {batch_size}")
    
    # Start timing
    start_time = time.time()
    
    ner_results = []
    extracted_results = []
    
    # Process in batches
    loop = asyncio.get_event_loop()
    
    # Smaller processing batch size for progress tracking
    processing_batch_size = min(10, len(notes))
    
    for i in tqdm(range(0, len(notes), processing_batch_size), desc="Processing batches"):
        batch_notes = notes[i:i+processing_batch_size]
        batch_ner_results, batch_extracted_results = loop.run_until_complete(process_batch(batch_notes, extractor))
        
        # Update statistics
        for extracted_data in batch_extracted_results:
            drugs = extracted_data.get("drugs", [])
            ades = extracted_data.get("adverse_events", [])
            
            if drugs:
                stats["notes_with_drugs"] += 1
                stats["total_drugs"] += len(drugs)
            
            if ades:
                stats["notes_with_ades"] += 1
                stats["total_ades"] += len(ades)
        
        ner_results.extend(batch_ner_results)
        extracted_results.extend(batch_extracted_results)
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Save results
    save_to_jsonl(ner_results, output_file)
    save_to_jsonl(extracted_results, extracted_file)
    
    # Print summary statistics
    logger.info("-" * 70)
    logger.info("Structured DSPy Extraction Summary:")
    logger.info(f"  - Notes processed:      {len(notes)}")
    logger.info(f"  - Notes with drugs:     {stats['notes_with_drugs']} ({stats['notes_with_drugs']/len(notes)*100:.1f}%)")
    logger.info(f"  - Notes with ADEs:      {stats['notes_with_ades']} ({stats['notes_with_ades']/len(notes)*100:.1f}%)")
    logger.info(f"  - Total drugs found:    {stats['total_drugs']}")
    logger.info(f"  - Total ADEs found:     {stats['total_ades']}")
    logger.info(f"  - Avg drugs per note:   {stats['total_drugs']/len(notes):.2f}")
    logger.info(f"  - Avg ADEs per note:    {stats['total_ades']/len(notes):.2f}")
    logger.info(f"  - Processing time:      {processing_time:.1f} seconds")
    logger.info(f"  - Seconds per note:     {processing_time/len(notes):.2f}")
    logger.info("-" * 70)
    
    logger.info("Structured DSPy generation completed")
    return {
        "status": "completed",
        "ner_results_count": len(ner_results),
        "extracted_results_count": len(extracted_results),
        "notes_with_drugs": stats["notes_with_drugs"],
        "notes_with_ades": stats["notes_with_ades"],
        "total_drugs": stats["total_drugs"],
        "total_ades": stats["total_ades"],
        "processing_time_seconds": processing_time
    }

async def evaluate_structured_on_gold_async(config: Dict) -> Dict:
    """Evaluate the structured DSPy extraction approach against gold standard data."""
    # Load gold dataset
    gold_file = config.get('gold_file', os.path.join("Step_1_data_generation", "data", "gold", "gold_ner_data.jsonl"))
    max_eval_samples = config.get('max_eval_samples', 200)
    batch_size = config.get('batch_size', 5)  # Smaller batch size for structured approach
    
    # Load gold data
    gold_data = []
    with open(gold_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_eval_samples:
                break
            gold_data.append(json.loads(line.strip()))
    
    logger.info(f"Loaded {len(gold_data)} gold standard examples for evaluation")
    
    # Initialize the structured extractor
    extractor = StructuredDrugADEExtractor()
    
    # Initialize metrics
    metrics = {
        'DRUG': {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0},
        'ADE': {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}
    }
    
    # Process gold data in batches
    for i in tqdm(range(0, len(gold_data), batch_size), desc="Evaluating on gold data"):
        batch = gold_data[i:i+batch_size]
        
        # Extract texts from the batch
        batch_texts = [example['text'] for example in batch]
        
        # Process the batch
        batch_ner_results, _ = await process_batch(batch_texts, extractor)
        
        # Compare predictions with gold standards
        for j, example in enumerate(batch):
            gold_entities = example['entities']
            pred_entities = batch_ner_results[j]['entities']
            
            # Evaluate for each entity type
            for entity_type in ['DRUG', 'ADE']:
                gold_spans = set([(e['start'], e['end']) for e in gold_entities if e['label'] == entity_type])
                pred_spans = set([(e['start'], e['end']) for e in pred_entities if e['label'] == entity_type])
                
                # Calculate metrics
                true_positives = len(gold_spans.intersection(pred_spans))
                false_positives = len(pred_spans - gold_spans)
                false_negatives = len(gold_spans - pred_spans)
                
                # Update metrics
                metrics[entity_type]['true_positives'] += true_positives
                metrics[entity_type]['false_positives'] += false_positives
                metrics[entity_type]['false_negatives'] += false_negatives
    
    # Calculate precision, recall, F1 for each entity type
    results = {}
    for entity_type, counts in metrics.items():
        tp = counts['true_positives']
        fp = counts['false_positives']
        fn = counts['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[entity_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'counts': counts
        }
    
    # Calculate micro-average across all entity types
    total_tp = sum(metrics[et]['true_positives'] for et in metrics)
    total_fp = sum(metrics[et]['false_positives'] for et in metrics)
    total_fn = sum(metrics[et]['false_negatives'] for et in metrics)
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    results['overall'] = {
        'precision': micro_precision,
        'recall': micro_recall,
        'f1': micro_f1
    }
    
    return results

def evaluate_structured_on_gold(config: Dict) -> Dict:
    """Wrapper for async evaluation function."""
    print_banner("EVALUATING STRUCTURED DSPy EXTRACTION ON GOLD DATASET")
    
    # Setup DSPy configuration
    openai_api_key = os.getenv('OPENAI_API_KEY')
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Configure DSPy with model settings from config
    dspy.settings.configure(lm=dspy.LM(
        model=config.get('model_name', 'gpt-4.1-nano'),
        temperature=config.get('temperature', 0.1),
        max_tokens=config.get('max_tokens', 2000)
    ))
    
    # Need to properly handle asyncio
    try:
        # Get or create an event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the evaluation process
        results = loop.run_until_complete(
            evaluate_structured_on_gold_async(config)
        )
    except Exception as e:
        logger.error(f"Error during async evaluation: {e}")
        raise
    
    # Print results
    logger.info("-" * 70)
    logger.info("Evaluation Results (Structured DSPy Approach):")
    for entity_type, metrics in results.items():
        if entity_type != 'overall':
            logger.info(f"{entity_type} Metrics:")
            logger.info(f"  - Precision: {metrics['precision']:.4f}")
            logger.info(f"  - Recall:    {metrics['recall']:.4f}")
            logger.info(f"  - F1 Score:  {metrics['f1']:.4f}")
            logger.info(f"  - TP/FP/FN:  {metrics['counts']['true_positives']}/{metrics['counts']['false_positives']}/{metrics['counts']['false_negatives']}")
    
    logger.info("-" * 70)
    logger.info("Overall Metrics (Micro-Average):")
    logger.info(f"  - Precision: {results['overall']['precision']:.4f}")
    logger.info(f"  - Recall:    {results['overall']['recall']:.4f}")
    logger.info(f"  - F1 Score:  {results['overall']['f1']:.4f}")
    logger.info("-" * 70)
    
    return results

def main():
    # Default configuration
    config = {
        "input_file": os.path.join("Step_1_data_generation", "data", "train.txt"),
        "structured_output_file": os.path.join("Step_1_data_generation", "data", "structured", "ner_data.jsonl"),
        "structured_extracted_file": os.path.join("Step_1_data_generation", "data", "structured", "extracted_data.jsonl"),
        "gold_file": os.path.join("Step_1_data_generation", "data", "gold", "gold_ner_data.jsonl"),
        "max_notes": 20,
        "max_eval_samples": 200,
        "model_name": "gpt-4.1-nano-2025-04-14",
        "temperature": 0.1,
        "max_tokens": 2000,
        "batch_size": 5
    }
    
    # Uncomment one of these to run generation or evaluation
    # run_structured_generation(config)
    evaluate_structured_on_gold(config)

if __name__ == "__main__":
    main() 