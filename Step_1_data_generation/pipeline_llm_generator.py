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
PIPELINE_DATA_DIR = os.path.join(DATA_DIR, "pipeline")

# Pipeline-specific prompt templates
STEP1_PROMPT = """
First Step - High Recall Extraction:
From the clinical text, extract ALL possible drug names and adverse events, even if you're not completely sure.
Aim for high recall - it's better to extract too many candidates than to miss valid ones.
Return a JSON object with format:
{{"candidate_drugs": [...], "candidate_adverse_events": [...]}}

Clinical text: {text}
"""

STEP2_PROMPT = """
Second Step - Precision Filtering:
Review these candidate entities extracted from the clinical text:
Candidate drugs: {drugs}
Candidate adverse events: {adverse_events}

Original text: {text}

For each candidate, determine if it is actually a drug name or adverse event based on context.
Remove entities that are:
- Common medical terms that aren't drugs/adverse events
- General symptoms unrelated to medications
- Standard medical procedures
- Lab test names (unless abnormal results are adverse events)

Return a filtered JSON object with format:
{{"filtered_drugs": [...], "filtered_adverse_events": [...]}}
"""

STEP3_PROMPT = """
Final Step - Validation and Context:
Review these filtered entities from the clinical text:
Drugs: {drugs}
Adverse events: {adverse_events}

Original text: {text}

For each entity:
1. Verify it appears exactly as written in the text
2. Check for any misspellings or case issues
3. Ensure drug names include only the drug name, not dosage information
4. Ensure adverse events are actual side effects, not symptoms of the primary condition

Return the final validated JSON object:
{{"drugs": [...], "adverse_events": [...]}}
"""

# Ensure output directory exists
os.makedirs(PIPELINE_DATA_DIR, exist_ok=True)

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
            return {"choices": [{"message": {"content": "{}"}}]}

def load_notes(input_file: str, max_notes: int) -> List[str]:
    with open(input_file, 'r', encoding='utf-8') as f:
        notes = [line.strip() for line in f if line.strip()]
    return notes[:max_notes]

async def process_note_pipeline(note: str, config: Dict, semaphore: asyncio.Semaphore) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Process a single note through the three-step pipeline"""
    # Step 1: High-recall extraction
    step1_prompt = STEP1_PROMPT.format(text=note)
    step1_response = await call_openai_async(step1_prompt, config, semaphore)
    step1_content = step1_response.choices[0].message.content
    
    try:
        step1_result = json.loads(step1_content)
        candidate_drugs = step1_result.get('candidate_drugs', [])
        candidate_adverse_events = step1_result.get('candidate_adverse_events', [])
    except:
        # Handle parsing errors
        candidate_drugs = []
        candidate_adverse_events = []
    
    # Step 2: Precision filtering
    step2_prompt = STEP2_PROMPT.format(
        drugs=candidate_drugs,
        adverse_events=candidate_adverse_events,
        text=note
    )
    step2_response = await call_openai_async(step2_prompt, config, semaphore)
    step2_content = step2_response.choices[0].message.content
    
    try:
        step2_result = json.loads(step2_content)
        filtered_drugs = step2_result.get('filtered_drugs', [])
        filtered_adverse_events = step2_result.get('filtered_adverse_events', [])
    except:
        # Handle parsing errors
        filtered_drugs = candidate_drugs
        filtered_adverse_events = candidate_adverse_events
    
    # Step 3: Validation and context
    step3_prompt = STEP3_PROMPT.format(
        drugs=filtered_drugs,
        adverse_events=filtered_adverse_events,
        text=note
    )
    step3_response = await call_openai_async(step3_prompt, config, semaphore)
    step3_content = step3_response.choices[0].message.content
    
    try:
        step3_result = json.loads(step3_content)
        final_drugs = step3_result.get('drugs', [])
        final_adverse_events = step3_result.get('adverse_events', [])
    except:
        # Handle parsing errors
        final_drugs = filtered_drugs
        final_adverse_events = filtered_adverse_events
    
    # Create NER format for training
    ner_data = {
        "text": note,
        "entities": create_entities(note, final_drugs, final_adverse_events)
    }
    
    # Create extracted format (clean format)
    extracted_data = {
        "text": note,
        "drugs": final_drugs,
        "adverse_events": final_adverse_events
    }
    
    return ner_data, extracted_data

async def process_batch(batch: List[str], config: Dict) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process a batch of notes with controlled concurrency"""
    # Create a semaphore to limit concurrent API calls
    concurrent_requests = min(config.get('batch_size', 5), len(batch))
    semaphore = asyncio.Semaphore(concurrent_requests)
    
    # Create tasks for all notes in the batch
    tasks = [process_note_pipeline(note, config, semaphore) for note in batch]
    
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

async def run_pipeline_extraction_async(notes: List[str], config: Dict) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """Run the pipeline extraction process asynchronously with proper batching"""
    batch_size = config.get('batch_size', 5) # Smaller batch size due to 3x API calls
    processing_batch_size = min(20, len(notes))  # Process in larger chunks for progress tracking
    
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
            pipeline_metadata = extracted_data.get("pipeline_metadata", {})
            
            # Count final entities
            if drugs:
                stats["notes_with_drugs"] += 1
                stats["total_drugs"] += len(drugs)
            
            if ades:
                stats["notes_with_ades"] += 1
                stats["total_ades"] += len(ades)
            
            # Note: Pipeline statistics are simplified since we removed metadata
            # to keep the output format clean
        
        ner_results.extend(batch_ner_results)
        extracted_results.extend(batch_extracted_results)
    
    return ner_results, extracted_results, stats

def run_pipeline_generation(config: Dict) -> Dict:
    """Run pipeline LLM generation with the provided config."""
    print_banner("PIPELINE MULTI-STEP LLM-BASED ADE EXTRACTION")
    
    logger.info("=" * 70)
    logger.info(f"Model:        {config.get('model_name', 'gpt-4.1-nano')}")
    logger.info(f"Temperature:  {config.get('temperature', 0.1)}")
    logger.info(f"Max tokens:   {config.get('max_tokens', 2000)}")
    logger.info(f"Batch size:   {config.get('batch_size', 5)} (3x API calls per note)")
    logger.info(f"Approach:     Three-step pipeline (recall → precision → validation)")
    logger.info("=" * 70)
    
    # Set OpenAI API key
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    # Set file paths from config
    input_file = config.get('input_file')
    output_file = config.get('pipeline_output_file')
    extracted_file = config.get('pipeline_extracted_file')
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
    logger.info("Starting multi-step pipeline extraction process...")
    logger.info(f"Processing {len(notes)} notes with 3-step pipeline")
    
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
            run_pipeline_extraction_async(notes, config)
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
    logger.info("Pipeline Extraction Summary:")
    logger.info(f"  - Notes processed:          {len(notes)}")
    logger.info(f"  - Notes with drugs:         {stats['notes_with_drugs']} ({stats['notes_with_drugs']/len(notes)*100:.1f}%)")
    logger.info(f"  - Notes with ADEs:          {stats['notes_with_ades']} ({stats['notes_with_ades']/len(notes)*100:.1f}%)")
    logger.info(f"  - Total drugs found:        {stats['total_drugs']}")
    logger.info(f"  - Total ADEs found:         {stats['total_ades']}")
    logger.info(f"  - Avg entities per note:    {(stats['total_drugs']+stats['total_ades'])/len(notes):.2f}")
    
    # Pipeline statistics simplified (metadata removed for clean output format)
    
    logger.info(f"  - Processing time:          {processing_time:.1f} seconds ({len(notes)*3} API calls)")
    logger.info(f"  - Seconds per note:         {processing_time/len(notes):.2f} (3 steps per note)")
    logger.info("-" * 70)
    
    logger.info("Pipeline LLM generation completed")
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

async def evaluate_pipeline_on_gold_async(config: Dict) -> Dict:
    """Evaluate the pipeline extraction approach against gold standard data asynchronously."""
    # Load gold dataset
    gold_file = config.get('gold_file', os.path.join("Step_1_data_generation", "data", "gold", "gold_ner_data.jsonl"))
    max_eval_samples = config.get('max_eval_samples', 200)
    batch_size = config.get('batch_size', 5)  # Smaller batch size due to 3x API calls
    
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
    
    # Pipeline statistics
    pipeline_stats = {
        'step1_candidates': 0,
        'step2_filtered': 0,
        'final_entities': 0
    }
    
    # Process gold data in batches
    for i in tqdm(range(0, len(gold_data), batch_size), desc="Evaluating on gold data"):
        batch = gold_data[i:i+batch_size]
        
        # Extract texts from the batch
        batch_texts = [example['text'] for example in batch]
        
        # Process the batch using the pipeline
        # Create a semaphore to limit concurrent API calls
        concurrent_requests = min(config.get('batch_size', 5), len(batch_texts))
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        # Create tasks for all notes in the batch
        tasks = [process_note_pipeline(text, config, semaphore) for text in batch_texts]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Get ner_results and extracted_results
        batch_ner_results = [item[0] for item in results]
        batch_extracted_results = [item[1] for item in results]
        
        # Track pipeline statistics
        for extracted_data in batch_extracted_results:
            pipeline_metadata = extracted_data.get("pipeline_metadata", {})
            step1_candidates = pipeline_metadata.get("step1_candidates", {})
            step2_filtered = pipeline_metadata.get("step2_filtered", {})
            
            # Count entities at each pipeline stage
            pipeline_stats['step1_candidates'] += len(step1_candidates.get("drugs", [])) + len(step1_candidates.get("adverse_events", []))
            pipeline_stats['step2_filtered'] += len(step2_filtered.get("drugs", [])) + len(step2_filtered.get("adverse_events", []))
            pipeline_stats['final_entities'] += len(extracted_data.get("drugs", [])) + len(extracted_data.get("adverse_events", []))
        
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
    
    # Add pipeline statistics
    num_examples = len(gold_data)
    results['pipeline_stats'] = {
        'avg_step1_candidates': pipeline_stats['step1_candidates'] / num_examples,
        'avg_step2_filtered': pipeline_stats['step2_filtered'] / num_examples,
        'avg_final_entities': pipeline_stats['final_entities'] / num_examples,
    }
    
    return results

def evaluate_pipeline_on_gold(config: Dict) -> Dict:
    """Wrapper for async evaluation function for the pipeline approach."""
    print_banner("EVALUATING PIPELINE EXTRACTION ON GOLD DATASET")
    
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
            evaluate_pipeline_on_gold_async(config)
        )
    except Exception as e:
        logger.error(f"Error during async evaluation: {e}")
        raise
    
    # Print results
    logger.info("-" * 70)
    logger.info("Evaluation Results (Pipeline Approach):")
    for entity_type, metrics in results.items():
        if entity_type != 'overall' and entity_type != 'pipeline_stats':
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
    
    # Print pipeline-specific statistics
    pipeline_stats = results.get('pipeline_stats', {})
    logger.info("-" * 70)
    logger.info("Pipeline Statistics:")
    logger.info(f"  - Avg Step 1 candidates:  {pipeline_stats.get('avg_step1_candidates', 0):.2f}")
    logger.info(f"  - Avg Step 2 filtered:    {pipeline_stats.get('avg_step2_filtered', 0):.2f}")
    logger.info(f"  - Avg Final entities:     {pipeline_stats.get('avg_final_entities', 0):.2f}")
    logger.info(f"  - Filtering rate (1→2):   {(1 - pipeline_stats.get('avg_step2_filtered', 0) / max(1, pipeline_stats.get('avg_step1_candidates', 1))) * 100:.1f}%")
    logger.info(f"  - Validation rate (2→3):  {(1 - pipeline_stats.get('avg_final_entities', 0) / max(1, pipeline_stats.get('avg_step2_filtered', 1))) * 100:.1f}%")
    logger.info("-" * 70)
    
    return results

def main():
    # Default configuration
    config = {
        "input_file": os.path.join("Step_1_data_generation", "data", "train.txt"),
        "pipeline_output_file": os.path.join("Step_1_data_generation", "data", "pipeline", "ner_data.jsonl"),
        "pipeline_extracted_file": os.path.join("Step_1_data_generation", "data", "pipeline", "extracted_data.jsonl"),
        "gold_file": os.path.join("Step_1_data_generation", "data", "gold", "gold_ner_data.jsonl"),
        "max_notes": 20,
        "max_eval_samples": 200,
        "model_name": "gpt-4.1-nano-2025-04-14",
        "temperature": 0.1,
        "max_tokens": 2000,
        "batch_size": 5
    }
    
    # Uncomment one of these to run generation or evaluation
    # run_pipeline_generation(config)
    evaluate_pipeline_on_gold(config)

if __name__ == "__main__":
    main() 