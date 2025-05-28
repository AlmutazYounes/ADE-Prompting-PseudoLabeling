from datasets import load_dataset

from utils.config import STEP_1_GOLD_STANDARD, STEP_2_GOLD_NER_DATA
from utils.data_transformation import (
    generate_gold_standard,
    save_raw_text_entities,
    convert_to_ner_format
)

def main():
    ade_relations = load_dataset("ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation")['train']
    gold_data = generate_gold_standard(ade_relations, STEP_1_GOLD_STANDARD)
    
    # Convert to NER format with entity spans
    ner_data = convert_to_ner_format(gold_data, output_path=None)
    
    # Save raw text and entity spans
    save_raw_text_entities(ner_data, STEP_2_GOLD_NER_DATA)
    
if __name__ == "__main__":
    main() 