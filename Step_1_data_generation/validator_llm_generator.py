#!/usr/bin/env python3

import os
import json
import openai
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from dotenv import load_dotenv
import asyncio
import logging
import shutil
import time

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose logging from external libraries
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# Configuration settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VALIDATOR_DATA_DIR = os.path.join(DATA_DIR, "validator")

# Prompt templates
# Step 1: Direct extraction (simple prompt)
DIRECT_PROMPT = "From the clinical text, extract only drug names and ADEs. Return a minimal JSON object like:\n{{\"drugs\": [...], \"adverse_events\": [...]}}\n\nClinical text: {text}"

# Step 2: Validation prompt to check for missing entities
VALIDATOR_PROMPT = """Check if any drug names or ADEs are missing.

Text: {{text}}
Drugs: {{drugs}}
ADEs: {{adverse_events}}

Return only missed ones:
{{"missed_drugs": [...], "missed_adverse_events": [...]}}

If none, return:
{{"missed_drugs": [], "missed_adverse_events": []}}"""


# Ensure output directory exists
os.makedirs(VALIDATOR_DATA_DIR, exist_ok=True)

async def call_openai_async(prompt: str, config: Dict, semaphore: asyncio.Semaphore) -> Dict:
    """Make an async API call to OpenAI with rate limiting"""
    async with semaphore:
        try:
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model=config.get('model_name', 'gpt-4.1-nano'),
                messages=[{"role": "user", "content": prompt}],
                temperature=config.get('temperature', 0.1),
                max_tokens=config.get('max_tokens', 2000)
            )
            return response
        except Exception as e:
            logger.error(f"API call error: {str(e)}")
            # Return empty response structure
            return {"choices": [{"message": {"content": "{\"drugs\": [], \"adverse_events\": []}"}}]}

def load_notes(input_file: str, max_notes: int) -> List[str]:
    with open(input_file, 'r', encoding='utf-8') as f:
        notes = [line.strip() for line in f if line.strip()]
    return notes[:max_notes]

async def process_note_with_validation(note: str, config: Dict, semaphore: asyncio.Semaphore) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Process a single note with the two-step validation approach"""
    # Step 1: Direct extraction
    direct_prompt = DIRECT_PROMPT.format(text=note)
    direct_response = await call_openai_async(direct_prompt, config, semaphore)
    direct_content = direct_response.choices[0].message.content
    
    try:
        direct_result = json.loads(direct_content)
        initial_drugs = direct_result.get('drugs', [])
        initial_adverse_events = direct_result.get('adverse_events', [])
    except:
        # Handle parsing errors
        initial_drugs = []
        initial_adverse_events = []
    
    # Step 2: Validation for missed entities
    validator_prompt = VALIDATOR_PROMPT.format(
        text=note,
        drugs=initial_drugs,
        adverse_events=initial_adverse_events
    )
    validator_response = await call_openai_async(validator_prompt, config, semaphore)
    validator_content = validator_response.choices[0].message.content
    
    try:
        validator_result = json.loads(validator_content)
        missed_drugs = validator_result.get('missed_drugs', [])
        missed_adverse_events = validator_result.get('missed_adverse_events', [])
    except:
        # Handle parsing errors
        missed_drugs = []
        missed_adverse_events = []
    
    # Combine initial and missed entities
    final_drugs = list(set(initial_drugs + missed_drugs))
    final_adverse_events = list(set(initial_adverse_events + missed_adverse_events))
    
    # Create NER format for training
    ner_data = {
        "text": note,
        "entities": create_entities(note, final_drugs, final_adverse_events)
    }
    
    # Create extracted format with validation metadata
    extracted_data = {
        "text": note,
        "drugs": final_drugs,
        "adverse_events": final_adverse_events,
        "validation_metadata": {
            "initial_extraction": {
                "drugs": initial_drugs,
                "adverse_events": initial_adverse_events
            },
            "missed_entities": {
                "drugs": missed_drugs,
                "adverse_events": missed_adverse_events
            },
            "added_by_validator": {
                "drugs_count": len(missed_drugs),
                "adverse_events_count": len(missed_adverse_events)
            }
        }
    }
    
    return ner_data, extracted_data

async def process_batch(batch: List[str], config: Dict) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process a batch of notes with controlled concurrency"""
    # Create a semaphore to limit concurrent API calls
    concurrent_requests = min(config.get('batch_size', 8), len(batch))
    semaphore = asyncio.Semaphore(concurrent_requests)
    
    # Create tasks for all notes in the batch
    tasks = [process_note_with_validation(note, config, semaphore) for note in batch]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Split results into ner_data and extracted_data
    ner_results = [item[0] for item in results]
    extracted_results = [item[1] for item in results]
    
    return ner_results, extracted_results

def create_entities(text: str, drugs: List[str], adverse_events: List[str]) -> List[Dict[str, Any]]:
    entities = []
    
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Saved {len(data)} examples to {output_path}")

def print_banner(message):
    term_width = shutil.get_terminal_size((80, 20)).columns
    banner_text = f" {message} "
    banner = f"\033[1;44m{banner_text.center(term_width)}\033[0m"
    logger.info("\n" + banner + "\n")

async def run_validator_extraction_async(notes: List[str], config: Dict) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """Run the validator extraction process asynchronously with proper batching"""
    batch_size = config.get('batch_size', 8) # Adjust batch size for 2 API calls per note
    processing_batch_size = min(30, len(notes))  # Process in larger chunks for progress tracking
    
    ner_results = []
    extracted_results = []
    
    # Statistics
    stats = {
        "total_drugs": 0,
        "total_ades": 0,
        "notes_with_drugs": 0,
        "notes_with_ades": 0,
        "drugs_added_by_validator": 0,
        "ades_added_by_validator": 0,
        "notes_improved_by_validator": 0
    }
    
    # Process in batches
    for i in tqdm(range(0, len(notes), processing_batch_size), desc="Processing batches"):
        process_batch_notes = notes[i:i+processing_batch_size]
        batch_ner_results, batch_extracted_results = await process_batch(process_batch_notes, config)
        
        # Update statistics for this batch
        for extracted_data in batch_extracted_results:
            drugs = extracted_data.get("drugs", [])
            ades = extracted_data.get("adverse_events", [])
            validation_metadata = extracted_data.get("validation_metadata", {})
            
            # Count final entities
            if drugs:
                stats["notes_with_drugs"] += 1
                stats["total_drugs"] += len(drugs)
            
            if ades:
                stats["notes_with_ades"] += 1
                stats["total_ades"] += len(ades)
            
            # Track validation statistics
            added_by_validator = validation_metadata.get("added_by_validator", {})
            drugs_added = added_by_validator.get("drugs_count", 0)
            ades_added = added_by_validator.get("adverse_events_count", 0)
            
            stats["drugs_added_by_validator"] += drugs_added
            stats["ades_added_by_validator"] += ades_added
            
            if drugs_added > 0 or ades_added > 0:
                stats["notes_improved_by_validator"] += 1
        
        ner_results.extend(batch_ner_results)
        extracted_results.extend(batch_extracted_results)
    
    return ner_results, extracted_results, stats

def run_validator_generation(config: Dict) -> Dict:
    """Run validator LLM generation with the provided config."""
    print_banner("VALIDATOR LLM-BASED ADE EXTRACTION")
    
    logger.info("=" * 70)
    logger.info(f"Model:        {config.get('model_name', 'gpt-4.1-nano')}")
    logger.info(f"Temperature:  {config.get('temperature', 0.1)}")
    logger.info(f"Max tokens:   {config.get('max_tokens', 2000)}")
    logger.info(f"Batch size:   {config.get('batch_size', 8)} (2x API calls per note)")
    logger.info(f"Approach:     Two-step extraction with validation")
    logger.info("=" * 70)
    
    # Set OpenAI API key
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    # Set file paths from config
    input_file = config.get('input_file')
    output_file = config.get('validator_output_file')
    extracted_file = config.get('validator_extracted_file')
    max_notes = config.get('max_notes', 100)
    
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
    
    # Processing indicator
    logger.info("-" * 70)
    logger.info("Starting two-step extraction with validation...")
    logger.info(f"Processing {len(notes)} notes")
    
    # Use asyncio to run the extraction process
    start_time = time.time()
    
    # Need to properly handle asyncio
    try:
        # Get or create an event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the extraction process
        ner_results, extracted_results, stats = loop.run_until_complete(
            run_validator_extraction_async(notes, config)
        )
    except Exception as e:
        logger.error(f"Error during async processing: {e}")
        raise
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Save results
    save_to_jsonl(ner_results, output_file)
    save_to_jsonl(extracted_results, extracted_file)
    
    # Print summary statistics
    logger.info("-" * 70)
    logger.info("Validator Extraction Summary:")
    logger.info(f"  - Notes processed:          {len(notes)}")
    logger.info(f"  - Notes with drugs:         {stats['notes_with_drugs']} ({stats['notes_with_drugs']/len(notes)*100:.1f}%)")
    logger.info(f"  - Notes with ADEs:          {stats['notes_with_ades']} ({stats['notes_with_ades']/len(notes)*100:.1f}%)")
    logger.info(f"  - Total drugs found:        {stats['total_drugs']}")
    logger.info(f"  - Total ADEs found:         {stats['total_ades']}")
    logger.info(f"  - Avg entities per note:    {(stats['total_drugs']+stats['total_ades'])/len(notes):.2f}")
    
    # Validator-specific statistics
    logger.info(f"  - Notes improved by validator: {stats['notes_improved_by_validator']} ({stats['notes_improved_by_validator']/len(notes)*100:.1f}%)")
    logger.info(f"  - Drugs added by validator:    {stats['drugs_added_by_validator']}")
    logger.info(f"  - ADEs added by validator:     {stats['ades_added_by_validator']}")
    logger.info(f"  - % entities from validation:  {(stats['drugs_added_by_validator']+stats['ades_added_by_validator'])/(stats['total_drugs']+stats['total_ades'])*100:.1f}%")
    
    logger.info(f"  - Processing time:          {processing_time:.1f} seconds ({len(notes)*2} API calls)")
    logger.info(f"  - Seconds per note:         {processing_time/len(notes):.2f} (2 steps per note)")
    logger.info("-" * 70)
    
    logger.info("Validator LLM generation completed")
    return {
        "status": "completed",
        "ner_results_count": len(ner_results),
        "extracted_results_count": len(extracted_results),
        "notes_with_drugs": stats["notes_with_drugs"],
        "notes_with_ades": stats["notes_with_ades"],
        "total_drugs": stats["total_drugs"],
        "total_ades": stats["total_ades"],
        "processing_time_seconds": processing_time,
        "validator_statistics": {
            "notes_improved": stats["notes_improved_by_validator"],
            "drugs_added": stats["drugs_added_by_validator"],
            "ades_added": stats["ades_added_by_validator"]
        }
    }

async def evaluate_validator_on_gold_async(config: Dict) -> Dict:
    """Evaluate the validator extraction approach against gold standard data asynchronously."""
    # Load gold dataset
    gold_file = config.get('gold_file', os.path.join("Step_1_data_generation", "data", "gold", "gold_ner_data.jsonl"))
    max_eval_samples = config.get('max_eval_samples', 200)
    batch_size = config.get('batch_size', 8)  # Adjust for 2x API calls
    
    # Load gold data
    gold_data = []
    with open(gold_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_eval_samples:
                break
            gold_data.append(json.loads(line.strip()))
    
    logger.info(f"Loaded {len(gold_data)} gold standard examples for evaluation")
    
    # Initialize metrics
    metrics = {
        'DRUG': {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0},
        'ADE': {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}
    }
    
    # Validator statistics
    validator_stats = {
        'initial_drugs': 0,
        'initial_ades': 0,
        'final_drugs': 0,
        'final_ades': 0,
        'drugs_added': 0,
        'ades_added': 0,
        'notes_improved': 0
    }
    
    # Process gold data in batches
    for i in tqdm(range(0, len(gold_data), batch_size), desc="Evaluating on gold data"):
        batch = gold_data[i:i+batch_size]
        
        # Extract texts from the batch
        batch_texts = [example['text'] for example in batch]
        
        # Process the batch using the validator approach
        # Create a semaphore to limit concurrent API calls
        concurrent_requests = min(config.get('batch_size', 8), len(batch_texts))
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        # Create tasks for all notes in the batch
        tasks = [process_note_with_validation(text, config, semaphore) for text in batch_texts]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Get ner_results and extracted_results
        batch_ner_results = [item[0] for item in results]
        batch_extracted_results = [item[1] for item in results]
        
        # Track validator statistics
        for extracted_data in batch_extracted_results:
            validation_metadata = extracted_data.get("validation_metadata", {})
            initial_extraction = validation_metadata.get("initial_extraction", {})
            added_by_validator = validation_metadata.get("added_by_validator", {})
            
            initial_drugs = len(initial_extraction.get("drugs", []))
            initial_ades = len(initial_extraction.get("adverse_events", []))
            drugs_added = added_by_validator.get("drugs_count", 0)
            ades_added = added_by_validator.get("adverse_events_count", 0)
            
            validator_stats['initial_drugs'] += initial_drugs
            validator_stats['initial_ades'] += initial_ades
            validator_stats['final_drugs'] += initial_drugs + drugs_added
            validator_stats['final_ades'] += initial_ades + ades_added
            validator_stats['drugs_added'] += drugs_added
            validator_stats['ades_added'] += ades_added
            
            if drugs_added > 0 or ades_added > 0:
                validator_stats['notes_improved'] += 1
        
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
    
    # Add validator statistics
    num_examples = len(gold_data)
    results['validator_stats'] = {
        'improvement_rate': validator_stats['notes_improved'] / num_examples,
        'entity_increase_rate': (validator_stats['drugs_added'] + validator_stats['ades_added']) / max(1, validator_stats['initial_drugs'] + validator_stats['initial_ades']),
        'initial_entities': validator_stats['initial_drugs'] + validator_stats['initial_ades'],
        'final_entities': validator_stats['final_drugs'] + validator_stats['final_ades'],
        'entities_added': validator_stats['drugs_added'] + validator_stats['ades_added'],
        'notes_improved': validator_stats['notes_improved']
    }
    
    return results

def evaluate_validator_on_gold(config: Dict) -> Dict:
    """Wrapper for async evaluation function for the validator approach."""
    print_banner("EVALUATING VALIDATOR EXTRACTION ON GOLD DATASET")
    
    # Set OpenAI API key
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
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
            evaluate_validator_on_gold_async(config)
        )
    except Exception as e:
        logger.error(f"Error during async evaluation: {e}")
        raise
    
    # Print results
    logger.info("-" * 70)
    logger.info("Evaluation Results (Validator Approach):")
    for entity_type, metrics in results.items():
        if entity_type != 'overall' and entity_type != 'validator_stats':
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
    
    # Print validator-specific statistics
    validator_stats = results.get('validator_stats', {})
    logger.info("-" * 70)
    logger.info("Validator Statistics:")
    logger.info(f"  - Notes improved by validator: {validator_stats.get('notes_improved', 0)} ({validator_stats.get('improvement_rate', 0)*100:.1f}%)")
    logger.info(f"  - Initial entities:            {validator_stats.get('initial_entities', 0)}")
    logger.info(f"  - Entities added:              {validator_stats.get('entities_added', 0)}")
    logger.info(f"  - Final entities:              {validator_stats.get('final_entities', 0)}")
    logger.info(f"  - Entity increase rate:        {validator_stats.get('entity_increase_rate', 0)*100:.1f}%")
    logger.info("-" * 70)
    
    return results

def main():
    # Default configuration
    config = {
        "input_file": os.path.join("Step_1_data_generation", "data", "train.txt"),
        "validator_output_file": os.path.join("Step_1_data_generation", "data", "validator", "ner_data.jsonl"),
        "validator_extracted_file": os.path.join("Step_1_data_generation", "data", "validator", "extracted_data.jsonl"),
        "gold_file": os.path.join("Step_1_data_generation", "data", "gold", "gold_ner_data.jsonl"),
        "max_notes": 20,
        "max_eval_samples": 200,
        "model_name": "gpt-4.1-nano-2025-04-14",
        "temperature": 0.1,
        "max_tokens": 2000,
        "batch_size": 8
    }
    
    # Uncomment one of these to run generation or evaluation
    # run_validator_generation(config)
    evaluate_validator_on_gold(config)

if __name__ == "__main__":
    main() 