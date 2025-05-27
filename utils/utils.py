#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for the ADE extraction pipeline
"""

import re
from datasets import Dataset
import json
import os
import functools
import time
from utils.config import NER_LABELS, ID_TO_LABEL, INPUT_FILE, MAX_NOTES, MAX_TEST_NOTES, MAX_RETRIES, RETRY_DELAY, API_TIMEOUT, LOG_LEVEL, LOG_FORMAT
import logging

# Set up logging based on configuration
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ==================== ERROR HANDLING ====================

def handle_api_errors(func):
    """Decorator that handles API-related errors with retry logic."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "rate limit" in str(e).lower() or "timeout" in str(e).lower():
                    wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"API error: {e}. Retrying in {wait_time}s (attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                    continue
                else:
                    # For non-retriable errors, log and re-raise
                    module_name = func.__module__ if hasattr(func, '__module__') else 'unknown'
                    func_name = func.__name__ if hasattr(func, '__name__') else 'unknown'
                    logger.error(f"Error in {module_name}.{func_name}: {str(e)}")
                    raise
        
        # If we've exhausted retries
        logger.error(f"Maximum retries ({MAX_RETRIES}) exceeded.")
        raise Exception(f"Failed after {MAX_RETRIES} attempts")
    
    return wrapper

def handle_extraction_errors(func):
    """Decorator that handles common extraction errors with proper logging."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the error with appropriate context
            module_name = func.__module__ if hasattr(func, '__module__') else 'unknown'
            func_name = func.__name__ if hasattr(func, '__name__') else 'unknown'
            
            logger.error(f"Error in {module_name}.{func_name}: {str(e)}")
            
            # For extractors, return empty result
            if func.__name__ == '__call__' and hasattr(args[0], '_create_result_object'):
                return args[0]._create_result_object()
            
            # Re-raise the exception for other functions
            raise
    
    return wrapper

def safe_json_loads(json_str, default=None):
    """Safely load JSON string with error handling."""
    try:
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}")
        return default or {}

# ==================== FILE OPERATIONS ====================

def ensure_dir_exists(path):
    """Ensure directory exists for the given file path."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory

def save_to_json(data, path, indent=2):
    """Generic function to save data to JSON file."""
    ensure_dir_exists(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    logger.info(f"Data saved to {path}")
    return path

def save_to_jsonl(data_list, path):
    """Generic function to save list of records to JSONL file."""
    ensure_dir_exists(path)
    with open(path, 'w', encoding='utf-8') as f:
        for record in data_list:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Data saved to {path}")
    return path

def load_from_json(path, default=None):
    """Generic function to load data from JSON file."""
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return default or {}
    
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON from {path}: {e}")
            return default or {}

def load_from_jsonl(path, default=None):
    """Generic function to load records from JSONL file."""
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return default or []
    
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except Exception as e:
                logger.error(f"Error parsing line in {path}: {e}")
    
    return data

def load_texts_from_jsonl(jsonl_path):
    """Load the 'text' field from each line of a JSONL file."""
    texts = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                if 'text' in obj:
                    texts.append(obj['text'])
            except Exception as e:
                logger.error(f"Error parsing line: {e}")
    return texts 

def load_notes_from_file(file_path, max_notes):
    """Generic loader for notes from a file (supports .jsonl and .txt)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            notes = []
            for line in f:
                try:
                    obj = json.loads(line)
                    if 'text' in obj:
                        notes.append(obj['text'])
                except Exception:
                    continue
        return notes[:max_notes]
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            notes = [line.strip() for line in f if line.strip()]
        return notes[:max_notes]
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def load_medical_notes():
    """Load medical notes from the INPUT_FILE specified in config.py."""
    return load_notes_from_file(INPUT_FILE, MAX_NOTES)

# Use the generic functions for specific data types
def save_llm_output(extracted_data, output_path):
    """Save LLM extraction output to file (JSONL, one record per line)."""
    return save_to_jsonl(extracted_data, output_path)

def save_ner_data(ner_data, ner_output_path):
    """Save NER data to file (JSONL, one record per line)."""
    return save_to_jsonl(ner_data, ner_output_path)

def save_bio_data(bio_data, bio_output_path):
    """Save BIO data to file (JSONL, one record per line)."""
    return save_to_jsonl(bio_data, bio_output_path)

def load_llm_output(path):
    """Load LLM extraction output from a JSONL file."""
    return load_from_jsonl(path)

def load_ner_data(path):
    """Load NER data from a JSONL file."""
    return load_from_jsonl(path)

def load_bio_data(path):
    """Load BIO data from a JSONL file."""
    return load_from_jsonl(path)

def save_test_results(test_results, output_path="test_results.jsonl"):
    """Save test results to a JSONL file."""
    return save_to_jsonl(test_results, output_path)

def load_gold_standard(path):
    """Load gold standard annotations from file."""
    return load_from_json(path)

# ==================== NER & BIO TAGGING ====================

def token_level_bio_tagging(text, entities, tokenizer, max_len=128):
    """Create token-level BIO tags aligned with tokenizer outputs."""
    # Sort entities by start position to process them in order
    entities = sorted(entities, key=lambda e: e['start']) if entities else []
    
    # Tokenize the text with offsets to map characters to tokens
    encoding = tokenizer(text, return_offsets_mapping=True, 
                        truncation=True, max_length=max_len,
                        padding='max_length', return_tensors='pt')
    
    input_ids = encoding['input_ids'][0].tolist()
    attention_mask = encoding['attention_mask'][0].tolist()
    offset_mapping = encoding['offset_mapping'][0].tolist()
    
    # Initialize all tokens with 'O' tag (outside any entity)
    token_tags = ['O'] * len(input_ids)
    
    # Process each entity and align with tokens
    for entity in entities:
        start_char = entity['start']
        end_char = entity['end']
        entity_type = entity['label']  # 'DRUG' or 'ADE'
        
        # Flag to track if we've started tagging this entity
        entity_started = False
        
        # Check each token's character span to align with entity
        for token_idx, (token_start, token_end) in enumerate(offset_mapping):
            # Skip special tokens (have offset (0, 0))
            if token_start == 0 and token_end == 0:
                continue
                
            # If token is within entity span
            # A token overlaps with an entity if:
            # 1. The token starts within the entity
            # 2. OR the token contains the entity start boundary
            if (token_start >= start_char and token_start < end_char) or \
               (token_start <= start_char and token_end > start_char):
                
                # First token of the entity gets B- prefix
                if not entity_started:
                    token_tags[token_idx] = f'B-{entity_type}'
                    entity_started = True
                else:  # Subsequent tokens get I- prefix
                    token_tags[token_idx] = f'I-{entity_type}'
    
    # Convert tags to IDs using the NER_LABELS mapping
    tag_ids = [NER_LABELS.get(tag, 0) for tag in token_tags]
    
    # Return token IDs, attention mask, and tag IDs
    return (input_ids, attention_mask, tag_ids)

def create_bio_dataset(records, tokenizer, max_len=128):
    """Create a token-level BIO dataset from NER records."""
    bio_dataset = []
    
    for record in records:
        text = record.get('text', '')
        entities = record.get('entities', [])
        
        # Get token-aligned BIO tags
        input_ids, attention_mask, tag_ids = token_level_bio_tagging(
            text, entities, tokenizer, max_len)
            
        # Store as a single record
        bio_record = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tags': tag_ids
        }
        
        bio_dataset.append(bio_record)
        
    return bio_dataset
