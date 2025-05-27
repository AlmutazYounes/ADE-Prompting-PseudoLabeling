#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extraction of drugs and adverse events using LLM methods
"""

import os
import dspy
from utils.config import OPENAI_API_KEY, ID_TO_LABEL, API_TIMEOUT
import logging
from tqdm import tqdm
from utils.utils import safe_json_loads
from utils.logging_utils import disable_all_litellm_logs
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable LiteLLM logs
disable_all_litellm_logs()


class BaseExtractor:
    """Base extractor class that defines the common interface for ADE extraction."""
    
    def __init__(self, model_name):
        """Initialize the extractor with a model name."""
        self.model_name = model_name
        logger.info(f"Initializing extractor with {model_name}")
    
    def __call__(self, text):
        """Extract drugs and adverse events from text."""
        raise NotImplementedError("Subclasses must implement the __call__ method")
    
    def _create_result_object(self, drugs=None, adverse_events=None, drug_ade_pairs=None):
        """Create a standardized result object."""
        return type('ExtractorResult', (), {
            'drugs': drugs or [],
            'adverse_events': adverse_events or [],
            'drug_ade_pairs': drug_ade_pairs or []
        })


class DirectLLMExtractor(BaseExtractor):
    """Direct LLM-based extractor that uses OpenAI directly without DSPy optimization."""
    def __init__(self, model_name):
        """Initialize with the specified model name."""
        super().__init__(model_name)
        
        # Import OpenAI library
        import openai
        self.client = openai.OpenAI(timeout=API_TIMEOUT)
    
    def __call__(self, text):
        """Extract drugs and adverse events directly using the LLM."""
        # Create a prompt that asks for drug and adverse event extraction
        prompt = f"""
        Extract all medications (drugs) and adverse events/side effects from the following medical note.
        Format the output as JSON with three fields:
        1. "drugs": A list of drug names mentioned
        2. "adverse_events": A list of adverse events or side effects mentioned
        3. "drug_ade_pairs": A list of strings in the format "drug: adverse_event" for any drug and adverse event that appear to be related

        Medical Note:
        {text}

        Only include actual medications and genuine adverse events. Don't include normal medical conditions unless they appear to be side effects of medications.
        """
        
        # Call the OpenAI API directly
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a medical data extraction assistant that identifies drugs and adverse events in clinical notes."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        # Extract the generated response
        result_text = response.choices[0].message.content
        
        # Parse the JSON
        result = safe_json_loads(result_text, {})
        
        # Create standardized result object
        return self._create_result_object(
            drugs=result.get('drugs', []),
            adverse_events=result.get('adverse_events', []),
            drug_ade_pairs=result.get('drug_ade_pairs', [])
        )


class ADEExtractor(dspy.Module, BaseExtractor):
    """DSPy module for extracting adverse drug events from medical notes."""
    
    def __init__(self, model_name="gpt-4o-mini"):
        """Initialize the ADE extractor with appropriate signature."""
        dspy.Module.__init__(self)
        BaseExtractor.__init__(self, model_name)
        
        # Configure DSPy with a language model
        self._configure_dspy(model_name)
        
        # Flag to track if we're in optimization mode
        self._in_optimization = False
        
        # Define a proper signature class first
        class ExtractADESignature(dspy.Signature):
            """Extract medications and their adverse drug events from clinical text."""
            
            # Input field
            text = dspy.InputField(description="Clinical text from a medical note")
            
            # Output fields
            drugs = dspy.OutputField(description="List of medication/drug names mentioned in the text. Return as a comma-separated list or an actual list. Return an empty list or 'None' if no drugs are found.")
            adverse_events = dspy.OutputField(description="List of adverse drug events (side effects) found in the text. Return as a comma-separated list or an actual list. Return an empty list or 'None' if no adverse events are found.")
            drug_ade_pairs = dspy.OutputField(description="List of pairs mapping drugs to adverse events in format 'drug: adverse_event'. Return as a comma-separated list or an actual list. Return an empty list or 'None' if no relationships are found.")
        
        # Use the signature class with a system message that provides more context
        self.extract_ade = dspy.ChainOfThought(ExtractADESignature)
    
    def _configure_dspy(self, model_name):
        """Configure DSPy with the appropriate language model."""
        # Set the API key in the environment
        import os
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            
        # Configure DSPy with LM
        lm = dspy.LM(model=model_name)
        dspy.configure(lm=lm)
        logger.info("DSPy configured with OpenAI LM successfully")
    
    def _normalize_field(self, field_value):
        """Normalize a field value to ensure it's a properly formatted list."""
        if field_value is None or field_value == "None" or not field_value:
            return []
        elif isinstance(field_value, str) and ',' in field_value:
            # Convert comma-separated string to list
            return [item.strip() for item in field_value.split(',') if item.strip()]
        elif isinstance(field_value, str):
            return [field_value.strip()] if field_value.strip() else []
        elif isinstance(field_value, list):
            return [str(item).strip() for item in field_value if item and str(item).strip()]
        else:
            return []
    
    def forward(self, text):
        """Process text and extract ADE information, with special handling for optimization mode."""
        # During optimization, use a more direct approach to ensure consistent output
        if hasattr(self, '_in_optimization') and self._in_optimization:
            # For optimization, use simpler prompt with system message
            result = self.extract_ade(text=text)
            # Return the raw DSPy result for optimization evaluation
            return result
        else:
            # For normal operation, normalize the outputs
            result = self.extract_ade(text=text)
            
            # Normalize all fields to ensure consistent output format
            drugs = self._normalize_field(getattr(result, 'drugs', []))
            adverse_events = self._normalize_field(getattr(result, 'adverse_events', []))
            drug_ade_pairs = self._normalize_field(getattr(result, 'drug_ade_pairs', []))
            
            return self._create_result_object(
                drugs=drugs,
                adverse_events=adverse_events,
                drug_ade_pairs=drug_ade_pairs
            )
    
    def __call__(self, text):
        """Implement the BaseExtractor interface."""
        return self.forward(text)


class ADEOptimizer:
    """Uses DSPy optimization techniques to improve ADE extraction."""
    
    def __init__(self, ade_extractor):
        """Initialize the optimizer."""
        self.ade_extractor = ade_extractor
        
    def prepare_examples(self, extracted_data):
        """Prepare examples for DSPy optimization from extracted data or gold standard annotations."""
        examples = []
        logger.info(f"Preparing {len(extracted_data)} examples for DSPy optimization")
        
        for record in tqdm(extracted_data, desc="Preparing DSPy examples"):
            # Validate required fields
            if 'text' not in record:
                continue
            
            # Ensure fields exist with defaults if missing
            drugs = record.get('drugs', [])
            adverse_events = record.get('adverse_events', [])
            drug_ade_pairs = record.get('drug_ade_pairs', [])
            
            # Ensure all fields are lists
            if not isinstance(drugs, list):
                drugs = [drugs] if drugs else []
            if not isinstance(adverse_events, list):
                adverse_events = [adverse_events] if adverse_events else []
            if not isinstance(drug_ade_pairs, list):
                drug_ade_pairs = [drug_ade_pairs] if drug_ade_pairs else []
            
            # Create example with expected input/output for DSPy
            example = dspy.Example(
                text=record['text'],
                drugs=drugs,
                adverse_events=adverse_events,
                drug_ade_pairs=drug_ade_pairs
            ).with_inputs('text')  # Specify 'text' as the input field
            
            examples.append(example)
        
        return examples
    
    def evaluate_extraction(self, gold, pred, trace=None):
        """Evaluate extraction quality using F1 score with improved output handling."""
        # Extract gold standard data from DSPy Example
        gold_drugs = self._extract_field(gold, 'drugs')
        gold_ades = self._extract_field(gold, 'adverse_events')
        
        # Extract predicted data with flexible handling for different return types
        pred_drugs = self._extract_field(pred, 'drugs')
        pred_ades = self._extract_field(pred, 'adverse_events')
        
        # Calculate drug F1
        drug_f1 = self._calculate_f1(gold_drugs, pred_drugs)
        
        # Calculate ADE F1
        ade_f1 = self._calculate_f1(gold_ades, pred_ades)
        
        # Return average F1 score
        avg_f1 = (drug_f1 + ade_f1) / 2
        
        return avg_f1
    
    def _extract_field(self, obj, field_name):
        """Robustly extract and normalize a field from various object types."""
        values = []
        
        # Try different ways to access the field
        if hasattr(obj, field_name):
            # Direct attribute access (DSPy Example or Prediction)
            field_value = getattr(obj, field_name)
            values = self._normalize_field_value(field_value)
        elif hasattr(obj, 'data') and isinstance(obj.data, dict) and field_name in obj.data:
            # Result object from _create_result_object()
            field_value = obj.data.get(field_name)
            values = self._normalize_field_value(field_value)
        elif isinstance(obj, dict) and field_name in obj:
            # Direct dictionary
            field_value = obj.get(field_name)
            values = self._normalize_field_value(field_value)
        
        return values
    
    def _normalize_field_value(self, value):
        """Normalize field values to ensure consistent list format."""
        # Handle None, empty values, N/A, etc.
        if value is None or value == "None" or not value or value == "N/A":
            return []
            
        # Handle string values
        if isinstance(value, str):
            # Remove special characters like «» that DSPy might add for entity marking
            value = value.replace('«', '').replace('»', '')
            
            if ',' in value:
                # Comma-separated string
                items = [item.strip().lower() for item in value.split(',') if item.strip()]
                return items
            else:
                # Single string value
                return [value.strip().lower()] if value.strip() else []
                
        # Handle list values
        if isinstance(value, list):
            # Clean and normalize each item
            items = []
            for item in value:
                if item is None or not str(item).strip() or str(item).strip() == "N/A":
                    continue
                    
                item_str = str(item).strip().lower()
                # Remove special characters
                item_str = item_str.replace('«', '').replace('»', '')
                
                if item_str:
                    items.append(item_str)
            return items
            
        # Return empty list for unsupported types
        return []
    
    def _calculate_f1(self, gold_items, pred_items):
        """Calculate F1 score for a set of items."""
        # Perfect if both empty
        if not gold_items and not pred_items:
            return 1.0
        
        # One empty, one not
        if not pred_items or not gold_items:
            if not pred_items:
                logger.warning(f"pred_items is empty while gold_items={gold_items}")
            if not gold_items:
                logger.warning(f"gold_items is empty while pred_items={pred_items}")
            return 0.0
        
        # Calculate F1 using sets to handle duplicates
        intersection = len(set(gold_items) & set(pred_items))
        precision = intersection / len(pred_items) if pred_items else 0.0
        recall = intersection / len(gold_items) if gold_items else 0.0
        
        # Calculate F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1
    
    def optimize(self, examples):
        """Optimize the ADE extractor using DSPy's optimization techniques."""
        logger.info("Optimizing ADE extraction")
        
        # Set optimization context flag
        self.ade_extractor._in_optimization = True
        
        try:
            # Split examples into training and development sets
            from sklearn.model_selection import train_test_split
            train_examples, dev_examples = train_test_split(examples, test_size=0.2, random_state=42)
            
            # Evaluate before optimization on a few dev examples
            logger.info("Evaluating performance before optimization...")
            before_scores = []
            eval_sample = dev_examples[:min(30, len(dev_examples))]
            for example in eval_sample:
                pred = self.ade_extractor(text=example.text)
                score = self.evaluate_extraction(example, pred)
                before_scores.append(score)
            
            before_f1 = sum(before_scores) / len(before_scores) if before_scores else 0.0
            logger.info(f"F1 score before optimization: {before_f1:.3f}")
            
            # Enhance the ChainOfThought with a stronger system message
            self.ade_extractor.extract_ade.compiler_instruction = """
            You are a medical information extraction expert. Your task is to identify:
            1. All medication names (drugs) mentioned in the text
            2. All adverse events (side effects) mentioned in the text
            3. Relationships between medications and their adverse events
            
            Be thorough and careful in your extraction. Look for both generic and brand names of medications.
            Adverse events may be described in various ways, including medical terminology or patient-reported symptoms.
            
            If you don't find any items in a category, return an empty list or 'None'.
            """
            
            # Set up optimization parameters
            max_bootstrapped_demos = 5
            
            # Use BootstrapFewShot optimization
            teleprompter = dspy.BootstrapFewShot(
                metric=self.evaluate_extraction,
                max_bootstrapped_demos=max_bootstrapped_demos
            )
            
            # Compile the optimizer with training data
            optimized_extractor = teleprompter.compile(
                self.ade_extractor, 
                trainset=train_examples
            )
            
            # Copy important attributes
            optimized_extractor.model_name = self.ade_extractor.model_name
            
            # Set optimization context for the new extractor
            optimized_extractor._in_optimization = False  # Reset for normal use
            
            # Evaluate after optimization on the same dev examples
            logger.info("Evaluating performance after optimization...")
            after_scores = []
            for example in eval_sample:
                pred = optimized_extractor(text=example.text)
                score = self.evaluate_extraction(example, pred)
                after_scores.append(score)
            
            after_f1 = sum(after_scores) / len(after_scores) if after_scores else 0.0
            logger.info(f"F1 score after optimization: {after_f1:.3f}")
            
            # Show improvement
            improvement = (after_f1 - before_f1) * 100
            if after_f1 > before_f1:
                logger.info(f"Optimization improved F1 score from {before_f1:.3f} to {after_f1:.3f}. Returning optimized extractor.")
                optimized_extractor._in_optimization = False  # Ensure flag is reset
                return optimized_extractor
            else:
                logger.info(f"Optimization did not improve F1 score (Before: {before_f1:.3f}, After: {after_f1:.3f}). Reverting to original extractor.")
                self.ade_extractor._in_optimization = False  # Ensure flag is reset
                return self.ade_extractor
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return self.ade_extractor
        
        finally:
            # Always reset optimization context
            self.ade_extractor._in_optimization = False


def extract_entities_improved(text, model, tokenizer, device):
    """Extract drugs and adverse events with improved handling of subword tokens."""
    model.to(device)
    # Step 1: Tokenize text with offset mapping to track character positions
    encoding = tokenizer(
        text, 
        return_tensors="pt", 
        return_offsets_mapping=True,
        truncation=True, 
        padding=True
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offset_mapping = encoding["offset_mapping"][0].numpy()
    
    # Step 2: Get model predictions
    model.eval()
    import torch
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    
    # Step 3: Extract entities using character-level positions
    drugs = []
    adverse_events = []
    
    i = 0
    while i < len(predictions):
        # Skip paddings, CLS, SEP tokens which have offset (0,0)
        if i >= len(offset_mapping) or offset_mapping[i][0] == offset_mapping[i][1] == 0:
            i += 1
            continue
        
        # Check for entity starts
        pred_id = predictions[i]
        if pred_id == 1:  # B-DRUG
            # Find complete entity span
            start_char = offset_mapping[i][0]
            end_char = offset_mapping[i][1]
            j = i + 1
            
            # Continue until end of entity
            while j < len(predictions) and j < len(offset_mapping) and predictions[j] == 2:  # I-DRUG
                end_char = offset_mapping[j][1]
                j += 1
            
            # Extract drug entity
            drug = text[start_char:end_char].strip()
            if drug and len(drug) > 1:
                drugs.append(drug)
            
            i = j  # Move to next token after entity
            
        elif pred_id == 3:  # B-ADE
            # Find complete entity span
            start_char = offset_mapping[i][0]
            end_char = offset_mapping[i][1]
            j = i + 1
            
            # Continue until end of entity
            while j < len(predictions) and j < len(offset_mapping) and predictions[j] == 4:  # I-ADE
                end_char = offset_mapping[j][1]
                j += 1
            
            # Extract adverse event entity
            ade = text[start_char:end_char].strip()
            if ade and len(ade) > 1:
                adverse_events.append(ade)
            
            i = j  # Move to next token after entity
        else:
            i += 1
    
    # Step 4: Post-process to remove any duplicates and very common words
    stopwords = {'the', 'and', 'for', 'was', 'with', 'that', 'this', 'patient', 'reported'}
    drugs = [d for d in set(drugs) if d.lower() not in stopwords]
    adverse_events = [a for a in set(adverse_events) if a.lower() not in stopwords]
    
    return {
        "drugs": drugs,
        "adverse_events": adverse_events
    }


# Alias for backward compatibility
extract_ades_with_finetuned_model = extract_entities_improved


def initialize_extractor(mode="direct", model_name=None):
    """Initialize ADE extractor based on mode."""
    if model_name is None:
        from utils.config import LLM_MODEL_NAME
        model_name = LLM_MODEL_NAME
        
    if mode == "direct":
        return DirectLLMExtractor(model_name=model_name)
    elif mode == "dspy":
        return initialize_dspy_extractor(model_name=model_name)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def initialize_dspy_extractor(model_name=None):
    """Initialize DSPy ADE extractor."""
    if model_name is None:
        from utils.config import LLM_MODEL_NAME
        model_name = LLM_MODEL_NAME
    
    # Disable LiteLLM logs before initializing
    disable_all_litellm_logs()
    
    return ADEExtractor(model_name=model_name) 