# ModernBERT ADE Extractor

A comprehensive pipeline for extracting Adverse Drug Events (ADEs) from medical notes using state-of-the-art language models.

## Overview

This project implements a complete end-to-end pipeline for identifying drugs, adverse events, and their relationships in medical texts. It combines the power of large language models for initial extraction with specialized fine-tuning of ModernBERT for optimized performance.

The pipeline supports two extraction approaches:

1. **Direct LLM extraction** using OpenAI's GPT-4o-mini - a straightforward approach that leverages the reasoning capabilities of large language models
2. **DSPy-optimized extraction** that enhances extraction using Chain-of-Thought reasoning and prompt optimization

The extracted entities are then used to fine-tune ModernBERT, transforming it into a specialized ADE extraction model that can be deployed in production environments.

## Features

- 🔍 **Dual extraction methods**: Compare direct LLM vs DSPy-optimized approaches
- 🩺 **Medical domain-specific**: Tailored for drug and adverse event extraction
- 🧠 **ModernBERT fine-tuning**: Adapt a state-of-the-art model for ADE extraction
- 📊 **Comprehensive evaluation**: Detailed metrics and visualizations
- 💾 **Efficient caching**: Intermediate results are saved to avoid redundant processing
- 🔄 **Parallelized processing**: Optimized for performance with multi-threading

## Pipeline Workflow

The pipeline follows a structured workflow to transform raw medical notes into a fine-tuned ADE extraction model:

1. **Data Loading**: Medical notes are loaded from text or JSONL files
2. **Preprocessing**: Notes are cleaned and standardized (removing extra whitespace, standardizing punctuation)
3. **Entity Extraction**: 
   - Raw notes are processed by either direct LLM calls or DSPy-optimized extraction
   - The system identifies drugs, adverse events, and their relationships
4. **NER Data Preparation**: 
   - Extractions are converted to token-level BIO tags (Beginning-Inside-Outside format)
   - This creates a structured dataset for sequence labeling
5. **ModernBERT Fine-tuning**: 
   - The base ModernBERT model is fine-tuned on the BIO-tagged dataset
   - Training includes learning rate optimization and early stopping
6. **Model Evaluation**: 
   - The fine-tuned model is evaluated against gold standard annotations
   - Comprehensive metrics and visualizations are generated

Each step is designed to be modular, with results cached to disk for efficiency and reproducibility.

## Architecture

```
┌─────────────────────┐
│    Medical Notes    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Preprocessing    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Entity Extraction  │◄───┐
│                     │    │
│  ┌───────┐ ┌───────┐│    │
│  │Direct │ │ DSPy  ││    │ DSPy
│  │ LLM   │ │Optim. ││    │ Optimization
│  └───┬───┘ └───┬───┘│    │
└──────┼─────────┼────┘    │
       │         └─────────┘
       │
       ▼
┌─────────────────────┐
│   NER Annotation    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  ModernBERT Model   │
│    Fine-tuning      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     Evaluation      │
└─────────────────────┘
```

## Project Structure

```
ADR_train_modernBERT/
├── analysis/                     # Analysis and visualizations
│   └── pipeline_outputs/         # Intermediate processing results
│       ├── direct/               # Direct LLM method outputs 
│       └── dspy/                 # DSPy method outputs
├── data/                         # Input data files
├── modernbert_ade_extractor_final/ # Fine-tuned models
│   └── modernbert_finetuned_*    # Individual model runs
│       ├── best_model/           # Best model checkpoint
│       ├── final_model/          # Final model checkpoint
│       └── evaluation/           # Evaluation results for this model
│           ├── metrics.json      # Performance metrics
│           ├── model_comparison.png # Visualization
│           └── evaluation_report.md # Detailed report
├── utils/                        # Utility modules
│   ├── config.py                 # Configuration settings
│   ├── dataset.py                # Dataset creation and processing
│   ├── evaluation.py             # Evaluation metrics and reporting
│   ├── extraction.py             # Entity extraction methods
│   ├── preprocessing.py          # Text preprocessing
│   ├── training.py               # Model training functions
│   └── utils.py                  # General utility functions
├── .env                          # Environment variables (OpenAI API key)
├── pipeline.py                   # Main pipeline script
└── requirements.txt              # Project dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ADR_train_modernBERT.git
   cd ADR_train_modernBERT
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Quickstart

1. **Generate Gold Data**

   Run the following command to generate the gold data:
   ```bash
   python generate_gold_data.py
   ```

2. **Train Models (Pipeline)**

   Run the pipeline for both modes (e.g., direct and dspy):
   ```bash
   python pipeline.py --mode direct
   python pipeline.py --mode dspy
   ```
   (Adjust the mode names if needed based on your use case.)

   This will train the models and save the outputs.

3. **Compare All Approaches**

   After training, compare all 4 approaches by running:
   ```bash
   python compare_all_approaches.py
   ```

This will produce a comparison of the results for all approaches.

### Running the Complete Pipeline

The main pipeline can be run with different extraction modes:

```bash
# Run with direct LLM extraction (default)
python pipeline.py --mode direct

# Run with DSPy-optimized extraction
python pipeline.py --mode dspy

# Force re-extraction (ignore cached results)
python pipeline.py --mode direct --force-extraction
```

### Understanding the Output

After running the pipeline, you'll find:

1. **Intermediate Results** in `analysis/pipeline_outputs/`:
   - Extracted entities in JSONL format
   - NER annotations 
   - BIO-tagged datasets

2. **Model Outputs** in `modernbert_ade_extractor_final/`:
   - Best and final model checkpoints
   - Training metrics and learning curves
   - Evaluation reports and visualizations

3. **Evaluation Metrics**:
   - Precision, recall, and F1 scores
   - Entity distribution analysis
   - Comparison between base and fine-tuned models

### Example Visualization

The pipeline generates visualizations to help interpret model performance:

- **Model Comparison**: Bar charts comparing base vs. fine-tuned performance
- **Learning Curves**: Training and validation loss/metrics over time
- **Entity Frequencies**: Distribution of drugs and adverse events in the dataset

## Customization

### Using Different LLMs

To use a different language model for extraction, modify `LLM_MODEL_NAME` in `utils/config.py`:

```python
# Change to another OpenAI model
LLM_MODEL_NAME = "gpt-4o" 
```

### Adjusting ModernBERT Fine-tuning

You can modify training hyperparameters in `utils/config.py`:

```python
# Fine-tuning hyperparameters
DEFAULT_EPOCHS = 5  # Increase for potentially better performance
DEFAULT_LEARNING_RATE = 3e-5  # Adjust based on your dataset
DEFAULT_PATIENCE = 3  # Increase for more training time before early stopping
```

### Input Data Format

The pipeline accepts medical notes in two formats:
- Plain text files (one note per line)
- JSONL files with a "text" field containing the note

Specify your input files in `utils/config.py`:

```python
INPUT_FILE = 'data/your_train_data.txt'  # or .jsonl
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) by Answer.AI
- [DSPy](https://github.com/stanfordnlp/dspy) by Stanford NLP
- [OpenAI](https://openai.com/) for API access

---

For questions or issues, please open an issue in this repository. 
