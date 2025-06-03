# ADE Extraction Data Generation Approaches

This directory contains multiple implementations for generating training data for Adverse Drug Event (ADE) extraction from clinical text. Each approach uses different techniques to identify drug names and adverse events.

## Overview

The data generation step is the first phase in our ADE extraction pipeline. It takes raw clinical notes as input and produces:
1. NER-formatted data for model training (with entity spans)
2. Extracted entities in a structured format for analysis

## Available Approaches

### 1. Direct LLM Extraction (`direct_llm_generator.py`)

The simplest approach that uses a single LLM prompt to extract drugs and ADEs in one step.

- **Method**: Single prompt extraction
- **Advantages**: Fast, minimal API calls (1 call per note)
- **Best Performace**: Best performance so far

### 2. Validator Approach (`validator_llm_generator.py`)

A two-step approach that first extracts entities and then validates for missed entities.

- **Method**: Initial extraction + validation for missed entities
- **Advantages**: Higher recall by catching missed entities
- **Complexity**: Medium (2 API calls per note)
- **Best for**: Balancing thoroughness and cost

### 3. Pipeline Approach (`pipeline_llm_generator.py`)

A comprehensive three-step pipeline focusing on recall, precision, and validation.

- **Method**: High-recall extraction → precision filtering → validation/refinement
- **Advantages**: Most thorough, handles both recall and precision
- **Complexity**: High (3 API calls per note)
- **Best for**: When extraction quality is critical regardless of API cost

### 4. DSPy Framework (`dspy_generator.py`)

Uses the DSPy framework with an ensemble approach combining multiple extraction strategies.

- **Method**: Chain-of-thought reasoning with ensemble extraction and validation
- **Advantages**: Leverages DSPy's structured reasoning capabilities
- **Features**: Enhanced prompting, few-shot examples, validation module
- **Best for**: Complex extraction scenarios with varied entity mentions

### 5. Structured DSPy (`structured_dspy_generator.py`)

A position-aware structured approach using DSPy with entity boundary identification.

- **Method**: Two-step process with identification and refinement of entity boundaries
- **Advantages**: Precise entity boundary detection, position-aware extraction
- **Features**: Confidence scoring, structured entity output
- **Best for**: When entity boundary precision is critical

## Usage

Each generator can be run independently or via the main script `run_step1.py` in the root directory:

```python
# To run all enabled generators:
python run_step1.py

# To run a specific generator directly:
python -m Step_1_data_generation.direct_llm_generator
```

Configure which approaches to use by editing the `ENABLED_SOURCES` dictionary in `run_step1.py`.

## Output Files

Each approach generates two output files in its corresponding subdirectory:
- `ner_data.jsonl`: Entity spans for model training
- `extracted_data.jsonl`: Structured entity extractions with metadata

## Comparison Summary

| Approach     | API Calls/Note | Complexity | Recall | Precision | Special Features                      |
|--------------|----------------|------------|--------|-----------|--------------------------------------|
| Direct       | 1              | Low        | Best   | Best      | Simple baseline                      |
| Validator    | 2              | Medium     | High   | Medium    | Missed entity detection              |
| Pipeline     | 3              | High       | High   | High      | Multi-stage filtering                |
| DSPy         | 1+             | Medium     | High   | Medium    | Chain-of-thought reasoning           |
| Structured   | 2              | High       | High   | High      | Position-aware, confidence scoring   | 