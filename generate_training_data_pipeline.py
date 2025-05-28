import logging
from transformers import AutoTokenizer

from utils.config import (
    INPUT_FILE, MAX_NOTES, LLM_MODEL_NAME,
    STEP_1_LLM_DIRECT, STEP_1_LLM_DSPY,
    STEP_2_NER_DATA_DIRECT, STEP_2_NER_DATA_DSPY,
    BERT_MODEL_NAME
)
from utils.extraction import initialize_extractor
from utils.dataset import ADEDatasetProcessor
from utils.utils import load_notes_from_file
from utils.data_transformation import save_to_jsonl, save_raw_text_entities

# ==================== LOGGING ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_data_pipeline(mode):
    assert mode in ("direct", "dspy")
    if mode == "direct":
        step_1_output = STEP_1_LLM_DIRECT
        step_2_output = STEP_2_NER_DATA_DIRECT
    else:
        step_1_output = STEP_1_LLM_DSPY
        step_2_output = STEP_2_NER_DATA_DSPY

    # Load notes (no preprocessing)
    notes = load_notes_from_file(INPUT_FILE, MAX_NOTES)
    logger.info(f"Loaded {len(notes)} notes from {INPUT_FILE}")

    # Step 1: Extract ADEs using LLM
    extractor = initialize_extractor(mode=mode, model_name=LLM_MODEL_NAME)
    ade_processor = ADEDatasetProcessor(extractor=extractor)
    extracted_data = ade_processor.extract_ades_batched(notes)
    logger.info(f"Step 1: Extracted ADEs for {len(extracted_data)} notes")
    save_to_jsonl(extracted_data, step_1_output)

    # Prepare NER data with raw text and entity spans
    ner_data = ade_processor.prepare_ner_data(extracted_data)
    save_raw_text_entities(ner_data, step_2_output)
    
    logger.info(f"\n=== Completed data pipeline for {mode} ===")

# ==================== MAIN ====================
def main():
    run_data_pipeline("direct")
    # run_data_pipeline("dspy")
    logger.info("\nAll training and extraction data generated successfully.")

if __name__ == "__main__":
    main() 