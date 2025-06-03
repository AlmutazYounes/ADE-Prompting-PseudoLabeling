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
DIRECT_DATA_DIR = os.path.join(DATA_DIR, "direct")
PROMPT_TEMPLATE = "From the clinical text, extract only drug names and ADEs. Return a minimal JSON object like:\n{{\"drugs\": [...], \"adverse_events\": [...]}}\n\nClinical text: {text}"

# Ensure output directory exists
os.makedirs(DIRECT_DATA_DIR, exist_ok=True)

def load_notes(input_file: str, max_notes: int) -> List[str]:
    with open(input_file, 'r', encoding='utf-8') as f:
        notes = [line.strip() for line in f if line.strip()]
    return notes[:max_notes]

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

async def process_note(note: str, config: Dict, semaphore: asyncio.Semaphore) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Process a single note asynchronously"""
    prompt = PROMPT_TEMPLATE.format(text=note)
    response = await call_openai_async(prompt, config, semaphore)
    
    content = response.choices[0].message.content
    
    try:
        result = json.loads(content)
        drugs = result.get('drugs', [])
        adverse_events = result.get('adverse_events', [])
    except:
        # Handle parsing errors
        drugs = []
        adverse_events = []
    
    # Create NER format for training
    ner_data = {
        "text": note,
        "entities": create_entities(note, drugs, adverse_events)
    }
    
    # Create extracted format for human verification
    extracted_data = {
        "text": note,
        "drugs": drugs,
        "adverse_events": adverse_events
    }
    
    return ner_data, extracted_data

async def process_batch(batch: List[str], config: Dict) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process a batch of notes with controlled concurrency"""
    # Create a semaphore to limit concurrent API calls
    concurrent_requests = min(config.get('batch_size', 10), len(batch))
    semaphore = asyncio.Semaphore(concurrent_requests)
    
    # Create tasks for all notes in the batch
    tasks = [process_note(note, config, semaphore) for note in batch]
    
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

async def run_extraction_async(notes: List[str], config: Dict) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """Run the extraction process asynchronously with proper batching"""
    batch_size = config.get('batch_size', 10)
    processing_batch_size = min(50, len(notes))  # Process in larger chunks for progress tracking
    
    ner_results = []
    extracted_results = []
    
    # Statistics
    stats = {
        "total_drugs": 0,
        "total_ades": 0,
        "notes_with_drugs": 0,
        "notes_with_ades": 0
    }
    
    # Process in batches
    for i in tqdm(range(0, len(notes), processing_batch_size), desc="Processing batches"):
        process_batch_notes = notes[i:i+processing_batch_size]
        batch_ner_results, batch_extracted_results = await process_batch(process_batch_notes, config)
        
        # Update statistics for this batch
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
    
    return ner_results, extracted_results, stats

def run_direct_generation(config: Dict) -> Dict:
    """Run direct LLM generation with the provided config."""
    print_banner("DIRECT LLM-BASED ADE EXTRACTION")
    
    logger.info("=" * 70)
    logger.info(f"Model:        {config.get('model_name', 'gpt-4.1-nano')}")
    logger.info(f"Temperature:  {config.get('temperature', 0.1)}")
    logger.info(f"Max tokens:   {config.get('max_tokens', 2000)}")
    logger.info(f"Batch size:   {config.get('batch_size', 10)}")
    logger.info("=" * 70)
    
    # Set OpenAI API key
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    # Set file paths from config
    input_file = config.get('input_file')
    output_file = config.get('direct_output_file')
    extracted_file = config.get('direct_extracted_file')
    max_notes = config.get('max_notes', 100)
    batch_size = config.get('batch_size', 10)
    
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
    logger.info("Starting extraction process with parallel batches...")
    logger.info(f"Processing {len(notes)} notes with concurrency of {batch_size}")
    
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
            run_extraction_async(notes, config)
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
    total_drugs = stats["total_drugs"]
    total_ades = stats["total_ades"]
    notes_with_drugs = stats["notes_with_drugs"]
    notes_with_ades = stats["notes_with_ades"]
    
    logger.info("-" * 70)
    logger.info("Extraction Summary:")
    logger.info(f"  - Notes processed:     {len(notes)}")
    logger.info(f"  - Notes with drugs:    {notes_with_drugs} ({notes_with_drugs/len(notes)*100:.1f}%)")
    logger.info(f"  - Notes with ADEs:     {notes_with_ades} ({notes_with_ades/len(notes)*100:.1f}%)")
    logger.info(f"  - Total drugs found:   {total_drugs}")
    logger.info(f"  - Total ADEs found:    {total_ades}")
    logger.info(f"  - Avg drugs per note:  {total_drugs/len(notes):.2f}")
    logger.info(f"  - Avg ADEs per note:   {total_ades/len(notes):.2f}")
    logger.info(f"  - Processing time:     {processing_time:.1f} seconds")
    logger.info(f"  - Notes per second:    {len(notes)/processing_time:.2f}")
    logger.info("-" * 70)
    
    logger.info("Direct LLM generation completed")
    return {
        "status": "completed",
        "ner_results_count": len(ner_results),
        "extracted_results_count": len(extracted_results),
        "notes_with_drugs": notes_with_drugs,
        "notes_with_ades": notes_with_ades,
        "total_drugs": total_drugs,
        "total_ades": total_ades,
        "processing_time_seconds": processing_time
    }

async def evaluate_direct_on_gold_async(config: Dict) -> Dict:
    """Evaluate the direct LLM extraction approach against gold standard data asynchronously."""
    # Load gold dataset
    gold_file = config.get('gold_file', os.path.join("Step_1_data_generation", "data", "gold", "gold_ner_data.jsonl"))
    max_eval_samples = config.get('max_eval_samples', 200)
    batch_size = config.get('batch_size', 10)
    
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
    
    # Process gold data in batches
    for i in tqdm(range(0, len(gold_data), batch_size), desc="Evaluating on gold data"):
        batch = gold_data[i:i+batch_size]
        
        # Extract texts from the batch
        batch_texts = [example['text'] for example in batch]
        
        # Process the batch using direct LLM
        # Create a semaphore to limit concurrent API calls
        concurrent_requests = min(config.get('batch_size', 10), len(batch_texts))
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        # Create tasks for all notes in the batch
        tasks = [process_note(text, config, semaphore) for text in batch_texts]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Get ner_results
        batch_ner_results = [item[0] for item in results]
        
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

def evaluate_direct_on_gold(config: Dict) -> Dict:
    """Wrapper for async evaluation function."""
    print_banner("EVALUATING DIRECT LLM EXTRACTION ON GOLD DATASET")
    
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
            evaluate_direct_on_gold_async(config)
        )
    except Exception as e:
        logger.error(f"Error during async evaluation: {e}")
        raise
    
    # Print results
    logger.info("-" * 70)
    logger.info("Evaluation Results (Direct LLM Approach):")
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
        "direct_output_file": os.path.join("Step_1_data_generation", "data", "direct", "ner_data.jsonl"),
        "direct_extracted_file": os.path.join("Step_1_data_generation", "data", "direct", "extracted_data.jsonl"),
        "gold_file": os.path.join("Step_1_data_generation", "data", "gold", "gold_ner_data.jsonl"),
        "max_notes": 20,
        "max_eval_samples": 200,
        "model_name": "gpt-4.1-nano-2025-04-14",
        "temperature": 0.1,
        "max_tokens": 2000,
        "batch_size": 10
    }
    
    # Evaluate on gold dataset
    evaluate_direct_on_gold(config)

if __name__ == "__main__":
    main() 