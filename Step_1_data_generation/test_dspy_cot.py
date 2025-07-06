#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv
from dspy_cot_llm_generator import run_dspy_generation

load_dotenv()

def test_dspy_cot():
    """Test the DSPy chain-of-thought approach with a small sample."""
    print("Testing DSPy Chain-of-Thought ADE Extraction...")
    
    # Create a test configuration with a small sample
    config = {
        "input_file": os.path.join("data", "train.txt"),
        "dspy_output_file": os.path.join("data", "dspy", "test_ner_data.jsonl"),
        "dspy_extracted_file": os.path.join("data", "dspy", "test_extracted_data.jsonl"),
        "max_notes": 3,  # Test with just 3 notes
        "model_name": "gpt-4.1-nano-2025-04-14",
        "temperature": 0.1,
        "max_tokens": 2000,
        "batch_size": 2
    }
    
    try:
        # Run the DSPy generation
        result = run_dspy_generation(config)
        
        print("\n" + "="*50)
        print("Test Results:")
        print(f"Status: {result['status']}")
        print(f"NER Results: {result['ner_results_count']}")
        print(f"Extracted Results: {result['extracted_results_count']}")
        print(f"Notes with drugs: {result['notes_with_drugs']}")
        print(f"Notes with ADEs: {result['notes_with_ades']}")
        print(f"Processing time: {result['processing_time_seconds']:.2f} seconds")
        print("="*50)
        
        # Verify files were created
        if os.path.exists(config['dspy_output_file']):
            print(f"✓ NER data file created: {config['dspy_output_file']}")
        else:
            print(f"✗ NER data file NOT created: {config['dspy_output_file']}")
            
        if os.path.exists(config['dspy_extracted_file']):
            print(f"✓ Extracted data file created: {config['dspy_extracted_file']}")
        else:
            print(f"✗ Extracted data file NOT created: {config['dspy_extracted_file']}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dspy_cot() 