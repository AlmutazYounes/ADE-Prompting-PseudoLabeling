#!/usr/bin/env python3

import os
import json
import logging
import openai
import dspy
import ast
from dspy.clients.openai import OpenAIProvider
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from dotenv import load_dotenv
import shutil
import asyncio

load_dotenv()

# Set up logging - only show your app logs, suppress external library logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose logging from external libraries
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('litellm').setLevel(logging.WARNING)
logging.getLogger('LiteLLM').setLevel(logging.WARNING)
logging.getLogger('dspy').setLevel(logging.WARNING)

# Enhanced signatures with better prompting and reasoning
class EnhancedDrugExtractionThought(dspy.Signature):
    """Extract drug names from clinical text using step-by-step reasoning. Focus on precision and recall."""
    clinical_text = dspy.InputField(desc="Clinical note text that may contain drug mentions")
    reasoning = dspy.OutputField(desc="Detailed step-by-step analysis identifying potential drug mentions, considering generic names, brand names, abbreviations, and dosages")
    drugs = dspy.OutputField(desc="JSON array of exact drug names found in the text. Include generic names, brand names, and common abbreviations. Return [] if no drugs mentioned.")
    confidence = dspy.OutputField(desc="Confidence score (0-1) for drug extraction accuracy")

# Simpler drug extraction from old approach (adapted from dspy_generator_old.py)
class DrugExtractionThought(dspy.Signature):
    """From the clinical text, extract only drug names. Return a minimal list of drugs."""
    clinical_text = dspy.InputField()
    reasoning = dspy.OutputField(desc="Step-by-step reasoning to identify drug mentions")
    drugs = dspy.OutputField(desc="List of drug mentions found in the text. Return an empty list [] if no drugs are mentioned.")

class EnhancedADEExtractionThought(dspy.Signature):
    """Extract adverse drug events from clinical text using step-by-step reasoning. Focus on symptoms, side effects, and complications."""
    clinical_text = dspy.InputField(desc="Clinical note text that may contain adverse drug events")
    reasoning = dspy.OutputField(desc="Detailed step-by-step analysis identifying adverse events, symptoms, side effects, complications, and abnormal lab values")
    adverse_events = dspy.OutputField(desc="JSON array of adverse events found in the text. Include symptoms, side effects, lab abnormalities, and complications. Return [] if no adverse events mentioned.")
    confidence = dspy.OutputField(desc="Confidence score (0-1) for adverse event extraction accuracy")

class ValidationSignature(dspy.Signature):
    """Validate and refine extracted entities against the original clinical text."""
    original_text = dspy.InputField(desc="Original clinical text")
    extracted_drugs = dspy.InputField(desc="Previously extracted drug names")
    extracted_ades = dspy.InputField(desc="Previously extracted adverse events")
    validated_drugs = dspy.OutputField(desc="Validated and corrected drug names as JSON array")
    validated_ades = dspy.OutputField(desc="Validated and corrected adverse events as JSON array")
    validation_reasoning = dspy.OutputField(desc="Explanation of corrections made during validation")

# Enhanced ReAct Drug Extractor with Chain of Thought
class EnhancedReActDrugExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(EnhancedDrugExtractionThought)
    
    def forward(self, clinical_text):
        # Enhanced context with specific instructions
        enhanced_context = f"""
        DRUG EXTRACTION TASK:
        
        Analyze the following clinical text and extract ALL drug mentions including:
        - Generic drug names (e.g., metformin, lisinopril, atorvastatin)
        - Brand names (e.g., Lipitor, Glucophage, Prinivil)
        - Common abbreviations (e.g., ASA for aspirin, HCTZ for hydrochlorothiazide)
        - Drug classes when specific drugs are implied
        
        Consider context clues like dosages, routes of administration, and medical terminology.
        
        Clinical Text: {clinical_text}
        """
        
        result = self.extract(clinical_text=enhanced_context)
        return result

# Simple ReAct Drug Extractor from old approach (adapted from dspy_generator_old.py)
class ReActDrugExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(DrugExtractionThought)
    
    def forward(self, clinical_text):
        # Add the direct extraction instruction (simpler prompt from old approach)
        enhanced_text = f"From the clinical text, extract only drug names. Return a minimal list of drugs.\n\nClinical text: {clinical_text}"
        result = self.extract(clinical_text=enhanced_text)
        return result.drugs

# Enhanced Few-Shot ADE Extractor with better examples
class EnhancedFewShotADEExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(EnhancedADEExtractionThought)
        self.examples = self._create_enhanced_examples()
    
    def _create_enhanced_examples(self):
        """Create comprehensive examples covering various ADE scenarios"""
        return [
            {
                "clinical_text": "Patient started on lisinopril 10mg daily. After 2 days, developed a persistent dry cough and throat irritation.",
                "adverse_events": ["dry cough", "throat irritation"],
                "context": "ACE inhibitor side effect - cough"
            },
            {
                "clinical_text": "Metformin 500mg BID initiated. Patient reports nausea, diarrhea, and metallic taste since starting medication.",
                "adverse_events": ["nausea", "diarrhea", "metallic taste"],
                "context": "Common GI side effects of metformin"
            },
            {
                "clinical_text": "Atorvastatin therapy begun 3 weeks ago. Patient now complains of muscle pain and weakness. CK elevated at 450 U/L.",
                "adverse_events": ["muscle pain", "weakness", "elevated CK"],
                "context": "Statin-induced myopathy with lab abnormality"
            },
            {
                "clinical_text": "Warfarin dose stable. INR therapeutic at 2.3. Patient reports easy bruising and minor nosebleeds.",
                "adverse_events": ["easy bruising", "nosebleeds"],
                "context": "Anticoagulant bleeding effects"
            },
            {
                "clinical_text": "Amoxicillin course completed yesterday. Today patient presents with diffuse maculopapular rash and itching.",
                "adverse_events": ["maculopapular rash", "itching"],
                "context": "Antibiotic allergic reaction"
            },
            {
                "clinical_text": "Blood pressure well controlled on current medications. No side effects or adverse reactions reported.",
                "adverse_events": [],
                "context": "No adverse events example"
            },
            {
                "clinical_text": "Patient's diabetes management continues with current regimen. HbA1c improved to 7.2%. Tolerating medications well.",
                "adverse_events": [],
                "context": "Positive response, no adverse events"
            }
        ]
    
    def forward(self, clinical_text):
        # Create rich context with examples
        context = """
        ADVERSE DRUG EVENT EXTRACTION TASK:
        
        Extract adverse drug events (ADEs) from clinical text. ADEs include:
        - Physical symptoms (nausea, headache, rash, pain)
        - Laboratory abnormalities (elevated enzymes, low counts)
        - Functional impairments (weakness, dizziness, confusion)
        - Allergic reactions (rash, swelling, difficulty breathing)
        - Organ-specific effects (hepatotoxicity, nephrotoxicity)
        
        EXAMPLES:
        """
        
        for ex in self.examples:
            context += f"\nText: {ex['clinical_text']}\n"
            context += f"Adverse Events: {ex['adverse_events']}\n"
            context += f"Context: {ex['context']}\n"
        
        context += f"\nNow analyze this clinical text:\n{clinical_text}"
        
        result = self.extract(clinical_text=context)
        return result

# Validation Module for Quality Control
class ExtractionValidator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.validate = dspy.ChainOfThought(ValidationSignature)
    
    def forward(self, original_text, drugs, adverse_events):
        result = self.validate(
            original_text=original_text,
            extracted_drugs=str(drugs),
            extracted_ades=str(adverse_events)
        )
        return result

# Multi-Stage Ensemble Extractor with Validation
class EnhancedEnsembleADEExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        # Use the old, simpler drug extractor that had better performance
        self.drug_extractor = ReActDrugExtractor()
        # Keep the enhanced ADE extractor
        self.ade_extractor = EnhancedFewShotADEExtractor()
        self.validator = ExtractionValidator()
    
    def forward(self, clinical_text):
        # Stage 1: Primary extraction
        drug_result = self.drug_extractor(clinical_text)
        ade_result = self.ade_extractor(clinical_text)
        
        # Parse initial results - note drug_result is now directly the list of drugs
        drugs = self._parse_entity_list(drug_result)
        adverse_events = self._parse_entity_list(ade_result.adverse_events)
        
        # Stage 2: Validation and refinement
        try:
            validation_result = self.validator(clinical_text, drugs, adverse_events)
            validated_drugs = self._parse_entity_list(validation_result.validated_drugs)
            validated_ades = self._parse_entity_list(validation_result.validated_ades)
            
            # Use validated results if they seem reasonable, otherwise fall back to original
            final_drugs = validated_drugs if validated_drugs or not drugs else drugs
            final_ades = validated_ades if validated_ades or not adverse_events else adverse_events
            
        except Exception as e:
            logger.warning(f"Validation failed, using original extractions: {str(e)}")
            final_drugs = drugs
            final_ades = adverse_events
        
        # Store reasoning for debugging
        self._last_extraction_details = {
            'drug_reasoning': 'Using simpler drug extraction from old approach',
            'ade_reasoning': getattr(ade_result, 'reasoning', ''),
            'drug_confidence': 'N/A',
            'ade_confidence': getattr(ade_result, 'confidence', 'N/A'),
            'validation_reasoning': getattr(validation_result, 'validation_reasoning', '') if 'validation_result' in locals() else ''
        }
        
        return final_drugs, final_ades
    
    def _parse_entity_list(self, entities):
        """Enhanced parsing with better error handling."""
        # If already a clean list, return it
        if isinstance(entities, list) and all(isinstance(e, str) and e.strip() for e in entities):
            return [e.strip() for e in entities if e.strip()]
        
        # If it's a string representation of a list, try to parse it
        if isinstance(entities, str):
            entities = entities.strip()
            
            # Handle empty cases
            if entities in ["[]", "[", "]", "", "None", "null"]:
                return []
            
            # Try JSON parsing first
            try:
                if entities.startswith('[') and entities.endswith(']'):
                    parsed = json.loads(entities)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
            
            # Try ast.literal_eval
            try:
                if entities.startswith('[') and entities.endswith(']'):
                    parsed = ast.literal_eval(entities)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
            except (SyntaxError, ValueError):
                pass
            
            # Split by common delimiters
            for delimiter in [',', ';', '|', '\n']:
                if delimiter in entities:
                    items = [item.strip() for item in entities.split(delimiter)]
                    items = [item for item in items if item and item not in ['[', ']', 'and', 'or']]
                    if items:
                        return items
            
            # Single item case
            if entities and entities not in ['[', ']']:
                return [entities]
        
        return []

def load_notes(input_file: str, max_notes: int) -> List[str]:
    with open(input_file, 'r', encoding='utf-8') as f:
        notes = [line.strip() for line in f if line.strip()]
    return notes[:max_notes]

def extract_entities_with_dspy(text: str, config: Dict = None) -> tuple[Dict[str, Any], Dict[str, Any]]:
    extractor = EnhancedEnsembleADEExtractor()
    drugs, adverse_events = extractor(text)
    
    # Create NER format for training
    ner_data = {
        "text": text,
        "entities": create_entities(text, drugs, adverse_events)
    }
    
    # Create extracted format for human verification
    extracted_data = {
        "text": text,
        "drugs": drugs,
        "adverse_events": adverse_events
    }
    
    return ner_data, extracted_data

async def process_note(note: str, config: Dict, extractor: EnhancedEnsembleADEExtractor) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Process a single note asynchronously"""
    drugs, adverse_events = extractor(note)
    
    # Create NER format for training
    ner_data = {
        "text": note,
        "entities": create_entities(note, drugs, adverse_events)
    }
    
    # Create extracted format for human verification (clean format)
    extracted_data = {
        "text": note,
        "drugs": drugs,
        "adverse_events": adverse_events
    }
    
    return ner_data, extracted_data

async def process_batch(batch: List[str], config: Dict) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process a batch of notes in parallel"""
    extractor = EnhancedEnsembleADEExtractor()
    tasks = [process_note(note, config, extractor) for note in batch]
    
    results = await asyncio.gather(*tasks)
    
    # Split results into ner_data and extracted_data
    ner_results = [item[0] for item in results]
    extracted_results = [item[1] for item in results]
    
    return ner_results, extracted_results

def create_entities(text: str, drugs: List[str], adverse_events: List[str]) -> List[Dict[str, Any]]:
    entities = []
    text_lower = text.lower()
    
    # Find drug entities with case-insensitive matching
    for drug in drugs:
        if drug and len(drug.strip()) > 1:  # Avoid single character matches
            drug_lower = drug.lower()
            start_index = 0
            while True:
                start_pos = text_lower.find(drug_lower, start_index)
                if start_pos == -1:
                    break
                
                # Verify it's a word boundary match (not part of another word)
                if start_pos > 0 and text[start_pos-1].isalnum():
                    start_index = start_pos + 1
                    continue
                if start_pos + len(drug) < len(text) and text[start_pos + len(drug)].isalnum():
                    start_index = start_pos + 1
                    continue
                
                entities.append({
                    'start': start_pos,
                    'end': start_pos + len(drug),
                    'label': 'DRUG',
                    'text': text[start_pos:start_pos + len(drug)]
                })
                start_index = start_pos + 1
    
    # Find ADE entities with case-insensitive matching
    for ade in adverse_events:
        if ade and len(ade.strip()) > 2:  # Avoid very short matches
            ade_lower = ade.lower()
            start_index = 0
            while True:
                start_pos = text_lower.find(ade_lower, start_index)
                if start_pos == -1:
                    break
                
                entities.append({
                    'start': start_pos,
                    'end': start_pos + len(ade),
                    'label': 'ADE',
                    'text': text[start_pos:start_pos + len(ade)]
                })
                start_index = start_pos + 1
    
    # Sort entities by start position
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

def run_dspy_generation(config: Dict) -> Dict:
    """Run DSPy generation with the provided config."""
    print_banner("ENHANCED DSPy-BASED ADE EXTRACTION")
    
    logger.info("=" * 70)
    logger.info(f"Model:        {config.get('model_name', 'gpt-4.1-nano')}")
    logger.info(f"Temperature:  {config.get('temperature', 0.1)}")
    logger.info(f"Max tokens:   {config.get('max_tokens', 2000)}")
    logger.info(f"Batch size:   {config.get('batch_size', 10)}")
    logger.info("Enhancements: ChainOfThought, Validation, Enhanced Prompting")
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
    output_file = config.get('dspy_output_file')
    extracted_file = config.get('dspy_extracted_file')
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
    
    ner_results = []
    extracted_results = []
    
    # Initialize enhanced counters for statistics
    total_drugs = 0
    total_ades = 0
    notes_with_drugs = 0
    notes_with_ades = 0
    notes_with_temporal = 0
    high_confidence_extractions = 0
    
    # Processing indicator 
    logger.info("-" * 70)
    logger.info("Starting enhanced extraction with validation pipeline...")
    logger.info(f"Processing {len(notes)} notes in batches of {batch_size}")
    
    # Process in batches
    loop = asyncio.get_event_loop()
    for i in tqdm(range(0, len(notes), batch_size), desc="Processing batches"):
        batch = notes[i:i+batch_size]
        batch_ner_results, batch_extracted_results = loop.run_until_complete(process_batch(batch, config))
        
        # Update enhanced statistics for this batch
        for extracted_data in batch_extracted_results:
            drugs = extracted_data.get("drugs", [])
            ades = extracted_data.get("adverse_events", [])
            metadata = extracted_data.get("metadata", {})
            
            if drugs:
                notes_with_drugs += 1
                total_drugs += len(drugs)
            
            if ades:
                notes_with_ades += 1
                total_ades += len(ades)
            
            if metadata.get("has_temporal_indicators", False):
                notes_with_temporal += 1
        
        # Extend results
        ner_results.extend(batch_ner_results)
        extracted_results.extend(batch_extracted_results)
    
    # Save results
    save_to_jsonl(ner_results, output_file)
    save_to_jsonl(extracted_results, extracted_file)
    
    # Print enhanced summary statistics
    logger.info("-" * 70)
    logger.info("Enhanced Extraction Summary:")
    logger.info(f"  - Notes processed:        {len(notes)}")
    logger.info(f"  - Notes with drugs:       {notes_with_drugs} ({notes_with_drugs/len(notes)*100:.1f}%)")
    logger.info(f"  - Notes with ADEs:        {notes_with_ades} ({notes_with_ades/len(notes)*100:.1f}%)")
    logger.info(f"  - Notes with temporal:    {notes_with_temporal} ({notes_with_temporal/len(notes)*100:.1f}%)")
    logger.info(f"  - Total drugs found:      {total_drugs}")
    logger.info(f"  - Total ADEs found:       {total_ades}")
    logger.info(f"  - Avg drugs per note:     {total_drugs/len(notes):.2f}")
    logger.info(f"  - Avg ADEs per note:      {total_ades/len(notes):.2f}")
    logger.info(f"  - Notes with both D+A:    {len([r for r in extracted_results if r.get('drugs') and r.get('adverse_events')])}")
    logger.info("-" * 70)
    
    logger.info("Enhanced DSPy generation completed")
    return {
        "status": "completed",
        "ner_results_count": len(ner_results),
        "extracted_results_count": len(extracted_results),
        "notes_with_drugs": notes_with_drugs,
        "notes_with_ades": notes_with_ades,
        "total_drugs": total_drugs,
        "total_ades": total_ades,
        "notes_with_temporal": notes_with_temporal
    }

def evaluate_dspy_on_gold(config: Dict) -> Dict:
    """Evaluate the DSPy extraction approach against gold standard data."""
    print_banner("EVALUATING DSPy EXTRACTION ON GOLD DATASET")
    
    # Setup DSPy configuration
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return {"status": "error", "message": "API key not found"}
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Configure DSPy with model settings from config
    dspy.settings.configure(lm=dspy.LM(
        model=config.get('model_name', 'gpt-4.1-nano'),
        temperature=config.get('temperature', 0.1),
        max_tokens=config.get('max_tokens', 2000)
    ))
    
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
    
    # Process gold data in batches using the same batch processing approach
    loop = asyncio.get_event_loop()
    
    for i in tqdm(range(0, len(gold_data), batch_size), desc="Evaluating on gold data"):
        batch = gold_data[i:i+batch_size]
        
        # Extract texts from the batch
        batch_texts = [example['text'] for example in batch]
        
        # Process the batch using the same batch processing function
        batch_ner_results, _ = loop.run_until_complete(process_batch(batch_texts, config))
        
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
    
    # Print results
    logger.info("-" * 70)
    logger.info("Evaluation Results:")
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
        "dspy_output_file": os.path.join("Step_1_data_generation", "data", "dspy", "ner_data.jsonl"),
        "dspy_extracted_file": os.path.join("Step_1_data_generation", "data", "dspy", "extracted_data.jsonl"),
        "gold_file": os.path.join("Step_1_data_generation", "data", "gold", "gold_ner_data.jsonl"),
        "max_notes": 100,
        "max_eval_samples": 200,
        "model_name": "gpt-4.1-nano",
        "temperature": 0.1,
        "max_tokens": 2000,
        "batch_size": 10
    }
    
    # Run data generation with DSPy
    # run_dspy_generation(config)
    
    # Evaluate on gold dataset
    evaluate_dspy_on_gold(config)

if __name__ == "__main__":
    main()