#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset classes and data preparation for ADE extraction
"""

import re
import torch
from torch.utils.data import Dataset
from utils.config import NER_LABELS, ID_TO_LABEL, BERT_MAX_LENGTH, BATCH_SIZE, MAX_WORKERS
import numpy as np
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ADEDatasetProcessor:
    """Processor to extract and structure ADE data from medical notes."""
    
    def __init__(self, notes_processor=None, extractor=None, tokenizer=None):
        """Initialize the ADE dataset processor."""
        self.notes_processor = notes_processor
        self.extractor = extractor
        self.tokenizer = tokenizer
        self.extracted_data = []
        self.ner_data = []
        self.relation_data = []
        
    def extract_note(self, note):
        """Extract ADEs from a single note (used for parallelization)."""
        try:
            # Use the __call__ method which both extractor types implement
            extraction_result = self.extractor(note)
            
            # Create a standard record format
            record = {
                'text': note,
                'drugs': extraction_result.drugs,
                'adverse_events': extraction_result.adverse_events,
                'drug_ade_pairs': extraction_result.drug_ade_pairs
            }
        except Exception as e:
            logger.error(f"Extraction failed for note: {note[:30]}... Error: {e}")
            record = {'text': note, 'drugs': [], 'adverse_events': [], 'drug_ade_pairs': []}
        
        return record
        
    def extract_ades_batched(self, processed_notes, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS):
        """Extract ADEs from notes using parallel processing for improved throughput."""
        extracted_data = []
        
        # Process in batches to avoid overloading the API
        num_batches = (len(processed_notes) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(processed_notes), batch_size), desc="Extracting ADEs", total=num_batches):
            batch = processed_notes[i:i+batch_size]
            batch_results = []
            
            # Process each batch in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                batch_results = list(executor.map(
                    lambda note: self.extract_note(note), batch))
            
            extracted_data.extend(batch_results)
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        self.extracted_data = extracted_data
        return extracted_data
        
    def extract_ades_from_notes(self):
        """Extract ADEs from all processed notes."""
        if self.notes_processor is None or not hasattr(self.notes_processor, 'processed_notes'):
            logger.error("Cannot extract ADEs: notes_processor is not set or has no processed_notes.")
            return []
            
        if self.extractor is None:
            logger.error("Cannot extract ADEs: extractor is not set.")
            return []
            
        return self.extract_ades_batched(self.notes_processor.processed_notes)
        
    def prepare_ner_data(self, llm_output_records):
        """Generate NER data from LLM extraction output."""
        ner_data = []
        for record in llm_output_records:
            text = record.get('text', '')
            drugs = record.get('drugs', [])
            adverse_events = record.get('adverse_events', [])
            if not isinstance(drugs, list) or not isinstance(adverse_events, list):
                raise ValueError('drugs and adverse_events must be lists in LLM output')
            entities = []
            # Add drug entities
            for drug in drugs:
                if not drug:
                    continue
                for match in re.finditer(re.escape(drug), text):
                    entities.append({'start': match.start(), 'end': match.end(), 'label': 'DRUG'})
            # Add ADE entities
            for ade in adverse_events:
                if not ade:
                    continue
                for match in re.finditer(re.escape(ade), text):
                    entities.append({'start': match.start(), 'end': match.end(), 'label': 'ADE'})
            ner_data.append({'text': text, 'entities': entities})
        
        self.ner_data = ner_data
        return ner_data
        
    def prepare_bio_data(self, ner_data=None, tokenizer=None):
        """Prepare BIO data for BERT fine-tuning."""
        from utils.utils import create_bio_dataset
        
        if ner_data is None:
            ner_data = self.ner_data
            
        if not ner_data:
            logger.error("No NER data available to prepare BIO data.")
            return [], [], [], [], []
            
        tokenizer_to_use = tokenizer if tokenizer is not None else self.tokenizer
        if tokenizer_to_use is None:
            logger.error("No tokenizer available to prepare BIO data.")
            return [], [], [], [], []
            
        bio_records = create_bio_dataset(ner_data, tokenizer_to_use, max_len=BERT_MAX_LENGTH)
        all_texts = [record.get('text', '') for record in ner_data]
        all_input_ids = [record.get('input_ids', []) for record in bio_records]
        all_attention_masks = [record.get('attention_mask', []) for record in bio_records]
        all_tags = [record.get('tags', []) for record in bio_records]
        return all_texts, all_tags, all_input_ids, all_attention_masks, bio_records
        
    def prepare_relation_data(self):
        """Prepare data for Relation Extraction training."""
        relation_data = []
        
        for record in self.extracted_data:
            text = record['text']
            
            # Process each drug-ADE pair from the extracted data
            for pair in record['drug_ade_pairs']:
                # Pairs should be in format "drug: adverse_event"
                parts = pair.split(":")
                if len(parts) == 2:
                    drug, ade = parts[0].strip(), parts[1].strip()
                    
                    # Find positions of all occurrences in text
                    drug_positions = [(m.start(), m.end()) for m in re.finditer(re.escape(drug), text)]
                    ade_positions = [(m.start(), m.end()) for m in re.finditer(re.escape(ade), text)]
                    
                    # Add relation data for each drug-ADE occurrence
                    for drug_pos in drug_positions:
                        for ade_pos in ade_positions:
                            relation_data.append({
                                'text': text,
                                'drug': drug,
                                'drug_start': drug_pos[0],
                                'drug_end': drug_pos[1],
                                'ade': ade,
                                'ade_start': ade_pos[0],
                                'ade_end': ade_pos[1],
                                'relation': 'CAUSES'  # Relationship type
                            })
        
        self.relation_data = relation_data
        return relation_data
    
    def tag_to_id(self, tag):
        """Convert tag string to ID"""
        return NER_LABELS.get(tag, 0)


class ADEDataset(Dataset):
    """PyTorch Dataset for ADE extraction fine-tuning."""
    
    def __init__(self, texts, tags, tokenizer, max_len=BERT_MAX_LENGTH, input_ids=None, attention_masks=None):
        """Initialize the dataset."""
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.use_pretokenized = input_ids is not None and attention_masks is not None
        
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """Get a single example from the dataset."""
        if self.use_pretokenized:
            # Use pre-tokenized inputs (from token-level BIO tagging)
            input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
            attention_mask = torch.tensor(self.attention_masks[idx], dtype=torch.long)
            labels = torch.tensor(self.tags[idx], dtype=torch.long)
        else:
            # Legacy approach - tokenize text on the fly
            text = self.texts[idx]
            tags = self.tags[idx]
            
            # Tokenize the text
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_len,
                padding='max_length',
                return_tensors='pt'
            )
            
            # Get input_ids and attention_mask
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            
            # Convert tags to tensor and pad to match input length
            # Tags that don't align with tokens get label "O" (0)
            labels = torch.tensor(tags[:self.max_len] + [0] * (self.max_len - len(tags[:self.max_len])), dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def calculate_class_weights(tags_list):
    """Calculate class weights to handle imbalanced data."""
    # Flatten all tag sequences
    all_tags = []
    for tags in tags_list:
        all_tags.extend(tags)
    
    # Count occurrences
    tag_counts = {}
    for tag in all_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    # Calculate weights (inverse of frequency)
    total_tags = len(all_tags)
    class_weights = {}
    
    for tag_id, count in tag_counts.items():
        # Skip padding tag (0)
        if tag_id == 0:
            class_weights[tag_id] = 0.0
        else:
            class_weights[tag_id] = total_tags / (count * len(tag_counts))
    
    return class_weights 