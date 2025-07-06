#!/usr/bin/env python3

import os
import json
import dspy
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
DSPY_DATA_DIR = os.path.join(DATA_DIR, "dspy")

# Ensure output directory exists
os.makedirs(DSPY_DATA_DIR, exist_ok=True)

# DSPy Signature for Drug and ADE Extraction with Chain of Thought
class DrugADEExtraction(dspy.Signature):
    """Extract drugs and adverse drug events (ADEs) from clinical text using chain of thought reasoning."""
    
    clinical_text = dspy.InputField(desc="Clinical text to analyze for drugs and adverse events")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning process for identifying drugs and ADEs")
    drugs = dspy.OutputField(desc="List of drug names found in the text")
    adverse_events = dspy.OutputField(desc="List of adverse drug events found in the text")

class ChainOfThoughtExtractor(dspy.Module):
    """DSPy module for extracting drugs and ADEs using chain of thought reasoning."""
    
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(DrugADEExtraction)
    
    def forward(self, clinical_text: str):
        """Process clinical text and extract drugs and ADEs with reasoning."""
        return self.extract(clinical_text=clinical_text)

def load_notes(input_file: str, max_notes: int) -> List[str]:
    """Load clinical notes from file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        notes = [line.strip() for line in f if line.strip()]
    return notes[:max_notes]

async def call_dspy_async(extractor: ChainOfThoughtExtractor, note: str, semaphore: asyncio.Semaphore) -> Dict:
    """Make an async DSPy call with rate limiting"""
    async with semaphore:
        try:
            response = await asyncio.to_thread(
                extractor.forward,
                clinical_text=note
            )
            return response
        except Exception as e:
            logger.error(f"DSPy call error: {str(e)}")
            # Return empty response structure
            return {
                "reasoning": "Error occurred during processing",
                "drugs": [],
                "adverse_events": []
            }

def parse_dspy_output(response: Dict) -> Tuple[List[str], List[str]]:
    """Parse DSPy output to extract drugs and adverse events."""
    try:
        # Handle different possible response formats
        if hasattr(response, 'drugs') and hasattr(response, 'adverse_events'):
            drugs = response.drugs
            adverse_events = response.adverse_events
        else:
            drugs = response.get('drugs', [])
            adverse_events = response.get('adverse_events', [])
        
        # Ensure we have lists
        if isinstance(drugs, str):
            drugs = [d.strip() for d in drugs.split(',') if d.strip()]
        elif not isinstance(drugs, list):
            drugs = []
            
        if isinstance(adverse_events, str):
            adverse_events = [ae.strip() for ae in adverse_events.split(',') if ae.strip()]
        elif not isinstance(adverse_events, list):
            adverse_events = []
            
        return drugs, adverse_events
    except Exception as e:
        logger.error(f"Error parsing DSPy output: {e}")
        return [], []

async def process_note(note: str, extractor: ChainOfThoughtExtractor, semaphore: asyncio.Semaphore) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Process a single note asynchronously using DSPy."""
    response = await call_dspy_async(extractor, note, semaphore)
    
    drugs, adverse_events = parse_dspy_output(response)
    
    # Create NER format for training
    ner_data = {
        "text": note,
        "entities": create_entities(note, drugs, adverse_events)
    }
    
    # Create extracted format for human verification (including reasoning)
    extracted_data = {
        "text": note,
        "drugs": drugs,
        "adverse_events": adverse_events,
        "reasoning": getattr(response, 'reasoning', '') if hasattr(response, 'reasoning') else response.get('reasoning', '')
    }
    
    return ner_data, extracted_data

async def process_batch(batch: List[str], extractor: ChainOfThoughtExtractor, config: Dict) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process a batch of notes with controlled concurrency."""
    # Create a semaphore to limit concurrent API calls
    concurrent_requests = min(config.get('batch_size', 10), len(batch))
    semaphore = asyncio.Semaphore(concurrent_requests)
    
    # Create tasks for all notes in the batch
    tasks = [process_note(note, extractor, semaphore) for note in batch]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Split results into ner_data and extracted_data
    ner_results = [item[0] for item in results]
    extracted_results = [item[1] for item in results]
    
    return ner_results, extracted_results

def create_entities(text: str, drugs: List[str], adverse_events: List[str]) -> List[Dict[str, Any]]:
    """Create entity annotations for NER format."""
    entities = []
    
    def extract_key_terms(term: str) -> List[str]:
        """Extract key terms from descriptive names."""
        if not term or term.strip() in ['(empty)', 'None', 'none']:
            return []
        
        # Remove parenthetical information and split by common delimiters
        import re
        clean_term = re.sub(r'\([^)]*\)', '', term)
        key_terms = re.split(r'[,;/\s]+', clean_term.strip())
        return [t.strip() for t in key_terms if t.strip() and len(t.strip()) > 2]
    
    def find_term_in_text(text: str, term: str) -> List[Tuple[int, int]]:
        """Find all occurrences of a term in text (case-insensitive)."""
        positions = []
        text_lower = text.lower()
        term_lower = term.lower()
        
        start_index = 0
        while True:
            start_pos = text_lower.find(term_lower, start_index)
            if start_pos == -1:
                break
            positions.append((start_pos, start_pos + len(term)))
            start_index = start_pos + 1
        
        return positions
    
    # Process drugs
    for drug in drugs:
        if not drug or drug.strip() in ['(empty)', 'None', 'none']:
            continue
            
        # First try exact match
        positions = find_term_in_text(text, drug)
        if positions:
            for start_pos, end_pos in positions:
                entities.append({
                    'start': start_pos,
                    'end': end_pos,
                    'label': 'DRUG'
                })
        else:
            # Try key terms from descriptive names
            key_terms = extract_key_terms(drug)
            for term in key_terms:
                positions = find_term_in_text(text, term)
                for start_pos, end_pos in positions:
                    entities.append({
                        'start': start_pos,
                        'end': end_pos,
                        'label': 'DRUG'
                    })
    
    # Process adverse events
    for ade in adverse_events:
        if not ade or ade.strip() in ['(empty)', 'None', 'none']:
            continue
            
        # First try exact match
        positions = find_term_in_text(text, ade)
        if positions:
            for start_pos, end_pos in positions:
                entities.append({
                    'start': start_pos,
                    'end': end_pos,
                    'label': 'ADE'
                })
        else:
            # Try key terms from descriptive names
            key_terms = extract_key_terms(ade)
            for term in key_terms:
                positions = find_term_in_text(text, term)
                for start_pos, end_pos in positions:
                    entities.append({
                        'start': start_pos,
                        'end': end_pos,
                        'label': 'ADE'
                    })
    
    # Remove duplicates and sort by start position
    unique_entities = []
    seen_positions = set()
    
    for entity in entities:
        pos_key = (entity['start'], entity['end'], entity['label'])
        if pos_key not in seen_positions:
            seen_positions.add(pos_key)
            unique_entities.append(entity)
    
    return sorted(unique_entities, key=lambda x: x['start'])

def save_to_jsonl(data: List[Dict[str, Any]], output_path: str):
    """Save data to JSONL format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Saved {len(data)} examples to {output_path}")

def print_banner(message):
    """Print a formatted banner message."""
    term_width = shutil.get_terminal_size((80, 20)).columns
    banner_text = f" {message} "
    banner = f"\033[1;42m{banner_text.center(term_width)}\033[0m"
    logger.info("\n" + banner + "\n")

async def run_extraction_async(notes: List[str], extractor: ChainOfThoughtExtractor, config: Dict) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """Run the extraction process asynchronously with proper batching."""
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
        batch_ner_results, batch_extracted_results = await process_batch(process_batch_notes, extractor, config)
        
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

def setup_dspy_model(config: Dict):
    """Setup DSPy model configuration."""
    model_name = config.get('model_name', 'gpt-4.1-nano-2025-04-14')
    temperature = config.get('temperature', 0.1)
    max_tokens = config.get('max_tokens', 2000)
    
    # Configure DSPy with OpenAI - use the correct API format
    lm = dspy.LM(
        model=f"openai/{model_name}",
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=os.getenv('OPENAI_API_KEY')
    )
    
    dspy.settings.configure(lm=lm)
    logger.info(f"DSPy configured with model: openai/{model_name}")

def run_dspy_generation(config: Dict) -> Dict:
    """Run DSPy-based generation with chain of thought reasoning."""
    print_banner("DSPY CHAIN-OF-THOUGHT ADE EXTRACTION")
    
    logger.info("=" * 70)
    logger.info(f"Model:        {config.get('model_name', 'gpt-4.1-nano-2025-04-14')}")
    logger.info(f"Temperature:  {config.get('temperature', 0.1)}")
    logger.info(f"Max tokens:   {config.get('max_tokens', 2000)}")
    logger.info(f"Batch size:   {config.get('batch_size', 10)}")
    logger.info(f"Approach:     DSPy Chain of Thought")
    logger.info("=" * 70)
    
    # Setup DSPy model
    setup_dspy_model(config)
    
    # Initialize the extractor
    extractor = ChainOfThoughtExtractor()
    
    # Set file paths from config
    input_file = config.get('input_file')
    output_file = config.get('dspy_output_file')
    extracted_file = config.get('dspy_extracted_file')
    max_notes = config.get('max_notes', 100)
    batch_size = config.get('batch_size', 10)
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(extracted_file), exist_ok=True)
    
    notes = load_notes(input_file, max_notes)
    logger.info(f"Loaded {len(notes)} clinical notes from {input_file}")
    
    # Processing indicator
    logger.info("-" * 70)
    logger.info("Starting DSPy chain-of-thought extraction process...")
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
            run_extraction_async(notes, extractor, config)
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
    logger.info("DSPy Chain-of-Thought Extraction Summary:")
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
    
    logger.info("DSPy Chain-of-Thought generation completed")
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

def main():
    """Main function to run DSPy generation."""
    # Default configuration
    config = {
        "input_file": os.path.join("data", "train.txt"),
        "dspy_output_file": os.path.join("data", "dspy", "ner_data.jsonl"),
        "dspy_extracted_file": os.path.join("data", "dspy", "extracted_data.jsonl"),
        "max_notes": 5000,
        "model_name": "gpt-4.1-nano-2025-04-14",
        "temperature": 0.1,
        "max_tokens": 2000,
        "batch_size": 10
    }
    
    # Run DSPy generation
    run_dspy_generation(config)

if __name__ == "__main__":
    main() 