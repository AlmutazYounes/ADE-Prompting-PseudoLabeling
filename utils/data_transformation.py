import os
import json
import logging
from tqdm import tqdm
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_to_jsonl(data: List[Dict], file_path: str) -> None:
    """Save data to a JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Saved {len(data)} records to {file_path}")

def load_from_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return []
        
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line: {line}")
    logger.info(f"Loaded {len(data)} records from {file_path}")
    return data

# ==================== STEP 1: Gold Standard Generation ====================
def generate_gold_standard(ade_relations, output_path: str) -> List[Dict]:
    """
    Step 1: Generate gold standard data from the ADE corpus v2 dataset.
    
    Args:
        ade_relations: Dataset with drug-ADE relations
        output_path: Path to save the gold standard data
        
    Returns:
        List of formatted gold standard records
    """
    logger.info("Step 1: Generating gold standard data from ADE corpus")
    
    entries = {}
    for row in ade_relations:
        text = row["text"]
        drug = row["drug"]
        effect = row["effect"]
        
        if text not in entries:
            entries[text] = {"drugs": set(), "adverse_events": set(), "drug_ade_pairs": set()}
            
        entries[text]["drugs"].add(drug)
        entries[text]["adverse_events"].add(effect)
        entries[text]["drug_ade_pairs"].add(f"{drug}: {effect}")
    
    # Convert sets to lists for JSON serialization
    formatted = []
    for text, data in entries.items():
        formatted.append({
            "text": text,
            "drugs": list(data["drugs"]),
            "adverse_events": list(data["adverse_events"]),
            "drug_ade_pairs": list(data["drug_ade_pairs"])
        })
    
    # Save to output
    # save_to_jsonl(formatted, output_path)
    # logger.info(f"Step 1: Saved {len(formatted)} gold standard records")
    
    return formatted

# ==================== STEP 2: NER Format Conversion ====================
def convert_to_ner_format(gold_data: List[Dict], output_path: str = None) -> List[Dict]:
    """
    Step 2: Convert gold data to NER format with entities having start/end/label fields.
    
    Args:
        gold_data: List of gold standard records
        output_path: Path to save the NER format data, if None will skip saving
        
    Returns:
        List of records in NER format
    """
    logger.info("Step 2: Converting gold standard data to NER format")
    ner_records = []
    
    for record in gold_data:
        text = record.get('text', '')
        drugs = record.get('drugs', [])
        adverse_events = record.get('adverse_events', [])
        
        entities = []
        
        # Process drugs
        for drug in drugs:
            if not drug or drug not in text:
                continue
                
            # Find all occurrences of the drug in the text
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
        
        # Process adverse events
        for ade in adverse_events:
            if not ade or ade not in text:
                continue
                
            # Find all occurrences of the adverse event in the text
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
        
        ner_records.append({
            'text': text,
            'entities': entities
        })
    
    # Save to output if path is provided
    if output_path:
        save_to_jsonl(ner_records, output_path)
        logger.info(f"Step 2: Saved {len(ner_records)} NER records")
    else:
        logger.info(f"Step 2: Generated {len(ner_records)} NER records (not saved)")
    
    return ner_records

# ==================== STEP 3: Generate BIO Tags and Tokens ====================
def generate_bio_tags(text: str, entities: List[Dict], tokenizer) -> tuple:
    """
    Generate BIO tags for a text with given entities.
    
    Args:
        text (str): The input text
        entities (list): List of entity dictionaries with 'start', 'end', and 'label'
        tokenizer: HuggingFace tokenizer
    
    Returns:
        tuple: (tokens, tags) where tokens are the tokenized words and tags are BIO tags
    """
    # Pre-tokenize the text with the tokenizer
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoding.input_ids)
    offset_mapping = encoding.offset_mapping
    
    # Initialize all tags as 'O' (outside)
    tags = ['O'] * len(tokens)
    
    # Assign tags to tokens based on entity positions
    for entity in entities:
        entity_start = entity['start']
        entity_end = entity['end']
        entity_label = entity['label']
        
        # Find tokens that overlap with this entity
        for i, (start, end) in enumerate(offset_mapping):
            # Skip if token is outside entity boundaries
            if end <= entity_start or start >= entity_end:
                continue
            
            # Token is part of the entity
            # First token of entity gets B- prefix
            if i == 0 or offset_mapping[i-1][1] <= entity_start:
                tags[i] = f'B-{entity_label}'
            else:
                tags[i] = f'I-{entity_label}'
    
    return tokens, tags

def save_raw_text_entities(records: List[Dict], output_path: str) -> List[Dict]:
    """
    Save raw text and entity spans for later tokenization in BERT pipeline.
    This avoids double tokenization by saving the text and entity positions
    instead of pre-tokenized data.
    
    Args:
        records: List of records with text and entities
        output_path: Path to save the raw text/entities data
    
    Returns:
        List of records with text and entity spans
    """
    logger.info(f"Saving {len(records)} records with raw text and entity spans")
    raw_data = []
    
    for record in tqdm(records, desc="Processing records"):
        try:
            text = record.get('text', '')
            
            # Use entities if present, otherwise build from drugs/adverse_events
            if 'entities' in record:
                entities = record['entities']
            else:
                drugs = record.get('drugs', [])
                adverse_events = record.get('adverse_events', [])
                entities = []
                
                # Process drugs
                for drug in drugs:
                    if not drug or drug not in text:
                        continue
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
                
                # Process adverse events
                for ade in adverse_events:
                    if not ade or ade not in text:
                        continue
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
            
            # Save raw text and entities
            raw_data.append({
                'text': text,
                'entities': entities
            })
            
        except Exception as e:
            logger.error(f"Error processing record: {e}")
    
    # Save to output
    save_to_jsonl(raw_data, output_path)
    logger.info(f"Saved {len(raw_data)} raw text/entity records to {output_path}")
    
    return raw_data