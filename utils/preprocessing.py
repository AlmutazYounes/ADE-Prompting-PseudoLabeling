#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Medical note processing and preprocessing functions
"""

import re
import json
import os
from utils.config import INPUT_FILE, MAX_NOTES, MAX_TEST_NOTES, LOG_FORMAT, LOG_LEVEL
import logging
from utils.utils import handle_extraction_errors

# Set up logging based on configuration
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class MedicalNoteProcessor:
    """Class to handle loading and preprocessing of medical notes."""
    
    def __init__(self, notes=None):
        """Initialize the medical note processor."""
        self.notes = notes or []
        self.processed_notes = []
        
    def load_notes(self, notes_list):
        """Load notes from a list."""
        self.notes = notes_list
        return self
            
    @handle_extraction_errors
    def preprocess_notes(self):
        """Preprocess the medical notes to clean and standardize the text."""
        processed_notes = []
        for note in self.notes:
            # Remove multiple spaces and replace with single space
            note = re.sub(r'\s+', ' ', note)
            
            # Remove special characters except for allowed punctuation
            note = re.sub(r'[^\w\s.,;:?!()-]', '', note)
            
            # Standardize common abbreviations (expand as needed)
            note = re.sub(r'\bpt\b', 'patient', note, flags=re.IGNORECASE)
            note = re.sub(r'\brx\b', 'prescription', note, flags=re.IGNORECASE)
            
            # Add to processed notes
            processed_notes.append(note)
        
        self.processed_notes = processed_notes
        return processed_notes
    
    def tokenize_notes(self, tokenizer):
        """Tokenize the medical notes using the provided tokenizer."""
        tokenized_notes = []
        for note in self.processed_notes:
            # Tokenize with padding and truncation
            tokens = tokenizer(
                note, 
                padding="max_length", 
                truncation=True, 
                max_length=512,  # ModernBERT can handle up to 8192, but this is more efficient
                return_tensors="pt"
            )
            tokenized_notes.append(tokens)
        
        return tokenized_notes
    
    @classmethod
    def process_notes(cls, notes):
        """Class method to process a list of notes in a single step."""
        processor = cls(notes)
        return processor.preprocess_notes()


def process_notes_in_batches(notes, batch_size=100, processor_fn=None):
    """Process a large list of notes in batches to avoid memory issues."""
    results = []
    total_notes = len(notes)
    
    for i in range(0, total_notes, batch_size):
        # Get the current batch
        batch = notes[i:min(i+batch_size, total_notes)]
        # logger.info(f"Processing batch {i//batch_size + 1}/{(total_notes + batch_size - 1)//batch_size} ({len(batch)} notes)")
        
        # Process the batch
        if processor_fn:
            batch_results = processor_fn(batch)
            results.extend(batch_results)
        
    return results


def clean_note(note):
    """Clean and format a medical note for processing."""
    # Remove extra whitespace
    note = ' '.join(note.split())
    
    # Remove any special characters that might interfere with processing
    note = re.sub(r'[^\w\s.,;:?!()-]', '', note)
    
    return note


# For backward compatibility, use the class method directly
preprocess_notes = MedicalNoteProcessor.process_notes
