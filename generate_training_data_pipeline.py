import os
import json
import logging
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.config import (
    INPUT_FILE, MAX_NOTES, GOLD_STANDARD_PATH,
    LLM_MODEL_NAME, LLM_OUTPUT_PATH_DIRECT, LLM_OUTPUT_PATH_DSPY,
    NER_OUTPUT_PATH_DIRECT, NER_OUTPUT_PATH_DSPY, CLINICALBERT_MODEL_NAME
)
from utils.extraction import initialize_extractor
from utils.dataset import ADEDatasetProcessor
from utils.utils import (
    save_llm_output, save_ner_data, save_bio_data, load_notes_from_file
)
from datasets import load_dataset

# ==================== LOGGING ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== GOLD DATA GENERATION ====================
def generate_gold_data():
    """Generate gold standard data from the ADE corpus v2 and save to GOLD_STANDARD_PATH."""
    ade_relations = load_dataset("ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation")['train']
    entries = defaultdict(lambda: {"drugs": [], "adverse_events": [], "drug_ade_pairs": []})
    for row in ade_relations:
        text = row["text"]
        drug = row["drug"]
        effect = row["effect"]
        entries[text]["drugs"].append(drug)
        entries[text]["adverse_events"].append(effect)
        entries[text]["drug_ade_pairs"].append(f"{drug}: {effect}")
    for entry in entries.values():
        entry["drugs"] = list(set(entry["drugs"]))
        entry["adverse_events"] = list(set(entry["adverse_events"]))
        entry["drug_ade_pairs"] = list(set(entry["drug_ade_pairs"]))
    formatted = [{"text": text, **data} for text, data in entries.items()]
    os.makedirs(os.path.dirname(GOLD_STANDARD_PATH), exist_ok=True)
    with open(GOLD_STANDARD_PATH, "w") as f:
        json.dump(formatted, f, indent=2)
    logger.info(f"Gold standard data saved to {GOLD_STANDARD_PATH} ({len(formatted)} entries)")

# ==================== DATA PIPELINE ====================
def run_data_pipeline(mode):
    """
    Run extraction, NER, and BIO data preparation for a given mode ('direct' or 'dspy').
    LLM/NER outputs go to pipeline_outputs, BIO to data/{mode}/bio_train.json
    """
    assert mode in ("direct", "dspy")
    logger.info(f"\n=== Running data pipeline for mode: {mode} ===")

    # Set output paths
    if mode == "direct":
        llm_output_path = LLM_OUTPUT_PATH_DIRECT
        ner_output_path = NER_OUTPUT_PATH_DIRECT
    else:
        llm_output_path = LLM_OUTPUT_PATH_DSPY
        ner_output_path = NER_OUTPUT_PATH_DSPY
    data_dir = os.path.join("data", mode)
    os.makedirs(data_dir, exist_ok=True)
    bio_train_path = os.path.join(data_dir, "bio_train.json")

    # Load notes (no preprocessing)
    notes = load_notes_from_file(INPUT_FILE, MAX_NOTES)
    logger.info(f"Loaded {len(notes)} notes from {INPUT_FILE}")

    # Extract ADEs
    extractor = initialize_extractor(mode=mode, model_name=LLM_MODEL_NAME)
    ade_processor = ADEDatasetProcessor(extractor=extractor)
    extracted_data = ade_processor.extract_ades_batched(notes)
    logger.info(f"Extracted ADEs for {len(extracted_data)} notes")

    logger.info(f"\n=== Done running pipeline for {mode} saving data... ===")
    save_llm_output(extracted_data, llm_output_path)

    # Prepare and save NER data
    ner_data = ade_processor.prepare_ner_data(extracted_data)
    save_ner_data(ner_data, ner_output_path)

    # Prepare and save BIO data (in data/{mode}/)
    tokenizer = AutoTokenizer.from_pretrained(CLINICALBERT_MODEL_NAME)
    _, _, _, _, bio_records = ade_processor.prepare_bio_data(ner_data, tokenizer)
    save_bio_data(bio_records, bio_train_path)
    logger.info(f"Saved BIO data for {len(bio_records)} notes in {bio_train_path}")

# ==================== MAIN ====================
def main():
    generate_gold_data()
    run_data_pipeline("direct")
    run_data_pipeline("dspy")
    logger.info("\nAll training and extraction data generated successfully.")

if __name__ == "__main__":
    main() 