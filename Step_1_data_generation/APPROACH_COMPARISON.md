# ADE Extraction Approach Comparison

This document compares two approaches for extracting drugs and adverse drug events (ADEs) from clinical notes using large language models.

## Overview

Both approaches aim to:
- Extract drug names and adverse events from clinical text
- Generate data in NER format for model training
- Provide extracted data for human verification
- Support batch processing with async operations

## Approach 1: Direct LLM Generation (`direct_llm_generator.py`)

### Key Features
- **Direct API calls** to OpenAI's API
- **Simple prompt template** with minimal instruction
- **JSON output format** with drugs and adverse_events lists
- **Async processing** with semaphore-based rate limiting
- **Batch processing** for efficiency

### Prompt Strategy
Uses a minimal, direct prompt:
```
From the clinical text, extract only drug names and ADEs. 
Return a minimal JSON object like:
{"drugs": [...], "adverse_events": [...]}

Clinical text: {text}
```

### Advantages
- Simple and straightforward implementation
- Fast processing with minimal overhead
- Direct control over API parameters
- Minimal token usage

### Limitations
- No reasoning process captured
- Limited ability to handle complex cases
- No intermediate steps for debugging
- Less interpretable results

## Approach 2: DSPy Chain-of-Thought (`dspy_cot_llm_generator.py`)

### Key Features
- **DSPy framework** with structured signatures
- **Chain-of-thought reasoning** for better accuracy
- **Structured output** with reasoning steps
- **Modular design** using DSPy components
- **Same async processing** architecture as direct approach

### DSPy Signature
```python
class DrugADEExtraction(dspy.Signature):
    clinical_text = dspy.InputField(desc="Clinical text to analyze")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning process")
    drugs = dspy.OutputField(desc="List of drug names found")
    adverse_events = dspy.OutputField(desc="List of adverse events found")
```

### Advantages
- **Reasoning captured**: Step-by-step thought process included
- **Better accuracy**: Chain-of-thought typically improves performance
- **Interpretable**: Can understand model's reasoning
- **Modular**: Easy to extend with additional components
- **Structured**: DSPy provides better prompt engineering

### Limitations
- More complex implementation
- Higher token usage (due to reasoning)
- Slightly slower processing
- Requires DSPy framework knowledge

## Output Format Comparison

### Direct Approach Output
```json
{
  "text": "Patient started on metformin and experienced nausea.",
  "drugs": ["metformin"],
  "adverse_events": ["nausea"]
}
```

### DSPy Chain-of-Thought Output
```json
{
  "text": "Patient started on metformin and experienced nausea.",
  "drugs": ["metformin"],
  "adverse_events": ["nausea"],
  "reasoning": "First, I identify medications mentioned: 'metformin' is a diabetes medication. Next, I look for adverse events: 'nausea' is described as experienced after starting medication, suggesting a potential adverse drug event."
}
```

## Performance Comparison

| Metric | Direct LLM | DSPy Chain-of-Thought |
|--------|------------|----------------------|
| **Speed** | Faster | Slower (due to reasoning) |
| **Token Usage** | Lower | Higher |
| **Accuracy** | Good | Potentially Better |
| **Interpretability** | Limited | High |
| **Debugging** | Difficult | Easier |

## Usage Instructions

### Running Direct LLM Approach
```bash
python direct_llm_generator.py
```

### Running DSPy Chain-of-Thought Approach
```bash
python dspy_cot_llm_generator.py
```

### Testing DSPy Approach
```bash
python test_dspy_cot.py
```

## Configuration

Both approaches support the same configuration parameters:

```python
config = {
    "input_file": "path/to/train.txt",
    "max_notes": 5000,
    "model_name": "gpt-4.1-nano-2025-04-14",
    "temperature": 0.1,
    "max_tokens": 2000,
    "batch_size": 10
}
```

## File Structure

```
Step_1_data_generation/
├── direct_llm_generator.py      # Direct LLM approach
├── dspy_cot_llm_generator.py    # DSPy chain-of-thought approach
├── test_dspy_cot.py             # Test script for DSPy approach
├── data/
│   ├── direct/                  # Direct approach outputs
│   │   ├── ner_data.jsonl
│   │   └── extracted_data.jsonl
│   └── dspy/                    # DSPy approach outputs
│       ├── ner_data.jsonl
│       └── extracted_data.jsonl
└── APPROACH_COMPARISON.md       # This file
```

## Recommendations

### Use Direct LLM Approach When:
- Speed is critical
- You need to process large volumes quickly
- Simple extraction is sufficient
- Token costs are a concern

### Use DSPy Chain-of-Thought Approach When:
- Accuracy is more important than speed
- You need interpretable results
- Complex clinical scenarios require reasoning
- You want to debug and improve the extraction process
- You plan to optimize prompts systematically

## Integration with Existing Pipeline

Both approaches generate the same output formats, making them interchangeable in your existing pipeline:

1. **NER Format**: For training BERT models (Step 2)
2. **Extracted Format**: For human verification and evaluation (Step 3)
3. **Statistics**: For analysis and comparison

The choice between approaches depends on your specific requirements for speed, accuracy, and interpretability. 