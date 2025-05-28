#!/usr/bin/env python3
"""
Fixed Mistral Teacher Model for Resume Parser with Improved JSON Parsing
Includes logit saving capabilities for knowledge distillation
Consistent chain-of-thought prompting for all components
COMPLETELY REWRITTEN TO FIX SYNTAX ERRORS
"""

import os
import json
import re
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_extraction_instruction(resume_text: str) -> str:
    """
    Create the EXACT same instruction text for extracting structured information from a resume.
    This is the chain-of-thought prompt that should be used consistently across all components.
    
    Args:
        resume_text: The text content of the resume
        
    Returns:
        str: The instruction text
    """
    instruction = f"""Analyze the following resume text and extract information in a structured JSON format.

Resume text:
```
{resume_text}
```

Please think through this step-by-step:

Step 1: Identify the candidate's full name and their total years of experience.

Step 2: Extract the employment history. For each position, determine:
- Company name
- Position/title
- Duration (start year to end year or "Present")
- Key responsibilities or achievements (2-3 main points)

Step 3: Identify the highest level of education, including:
- Degree name and specialization
- Institution name
- Year of completion

Step 4: Identify 3-5 key skills relevant to finance or their professional domain.

Step 5: Extract any professional certifications mentioned.

Output the information in the following JSON structure:

```json
{{
  "candidate_info": {{
    "name": "Candidate's full name",
    "total_experience_years": years_of_experience
  }},
  "employment": [
    {{
      "company": "Most recent company name",
      "position": "Position title",
      "duration": "Start year-Present (or End year)",
      "responsibilities": [
        "Key responsibility 1",
        "Key responsibility 2",
        "Key responsibility 3"
      ]
    }},
    // Additional jobs in reverse chronological order
  ],
  "education": {{
    "highest_degree": "Degree name with specialization",
    "institution": "University/College name",
    "year": completion_year
  }},
  "key_skills": [
    "Skill 1",
    "Skill 2",
    "Skill 3",
    "Skill 4",
    "Skill 5"
  ],
  "certifications": [
    "Certification 1",
    "Certification 2"
  ]
}}
```

Output only the valid JSON object without any explanation.
"""
    return instruction

class ImprovedJSONExtractor:
    """Enhanced JSON extraction with multiple fallback strategies - SYNTAX ERROR FREE VERSION"""
    
    @staticmethod
    def clean_json_string(json_str: str) -> str:
        """Apply comprehensive JSON cleaning strategies with safe regex patterns"""
        logger.debug(f"Starting JSON cleaning on {len(json_str)} character string")
        
        # Step 1: Remove comment lines - using safe regex patterns
        # Remove single line comments starting with //
        json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
        # Remove multi-line comments /* ... */
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        logger.debug("Step 1: Removed comments")
        
        # Step 2: Fix trailing commas in objects and arrays
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        logger.debug("Step 2: Fixed trailing commas")
        
        # Step 3: Add missing quotes around unquoted keys
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*?)(\s*:)', r'\1"\2"\3', json_str)
        logger.debug("Step 3: Added quotes to unquoted keys")
        
        # Step 4: Convert single quotes to double quotes for strings
        json_str = re.sub(r"'([^']*?)'(\s*[:,\]}])", r'"\1"\2', json_str)
        logger.debug("Step 4: Converted single quotes to double quotes")
        
        # Step 5: Fix missing commas between array elements (KEY FIX FOR YOUR ISSUE)
        # Pattern 1: Missing comma between two quoted strings
        json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
        # Pattern 2: Missing comma between two objects
        json_str = re.sub(r'}\s*\n\s*{', '},\n{', json_str)
        # Pattern 3: Missing comma between two arrays
        json_str = re.sub(r']\s*\n\s*\[', '],\n[', json_str)
        # Pattern 4: Missing comma between string and object
        json_str = re.sub(r'"\s*\n\s*{', '",\n{', json_str)
        # Pattern 5: Missing comma between object and string
        json_str = re.sub(r'}\s*\n\s*"', '},\n"', json_str)
        logger.debug("Step 5: Fixed missing commas between array/object elements")
        
        # Step 6: Remove leading commas in arrays and objects
        json_str = re.sub(r'(\[|{)\s*,', r'\1', json_str)
        logger.debug("Step 6: Removed leading commas")
        
        # Step 7: Fix numbers that are accidentally quoted
        json_str = re.sub(r'"(\d+)"(\s*[,\]}])', r'\1\2', json_str)
        logger.debug("Step 7: Unquoted numeric values")
        
        # Step 8: Remove control characters that might break JSON parsing
        original_length = len(json_str)
        json_str = ''.join(char for char in json_str if ord(char) >= 32 or char in '\n\r\t')
        if len(json_str) != original_length:
            logger.debug(f"Step 8: Removed {original_length - len(json_str)} control characters")
        
        logger.debug(f"JSON cleaning complete. Final length: {len(json_str)}")
        return json_str
    
    @staticmethod
    def extract_json_block(text: str) -> Optional[str]:
        """Extract JSON block using proven working strategy - SYNTAX ERROR FREE"""
        # Strategy 1: Look for markdown json blocks first
        json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # Strategy 2: Look for content after Mistral chat format tag
        inst_end = text.find("[/INST]")
        if inst_end > 0:
            text = text[inst_end + 7:].strip()
        
        # Strategy 3: Simple and robust JSON extraction (like working student model)
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            # Basic validation - should be reasonable length
            if len(json_str) > 50:  # Minimum reasonable JSON size
                return json_str
        
        # Strategy 4: Look for JSON after common output markers
        markers = ['Output:', 'Result:', 'JSON:', 'output:']
        for marker in markers:
            idx = text.find(marker)
            if idx >= 0:
                text_after = text[idx + len(marker):].strip()
                start = text_after.find('{')
                if start >= 0:
                    # Find matching closing brace
                    brace_count = 0
                    end = start
                    for i, char in enumerate(text_after[start:], start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end = i + 1
                                break
                    if end > start:
                        return text_after[start:end]
        
        return None
    
    @staticmethod
    def validate_resume_structure(data: Dict) -> bool:
        """Validate that the parsed JSON has the expected resume structure"""
        required_keys = ['candidate_info', 'employment', 'education', 'key_skills']
        return all(key in data for key in required_keys)
    
    @staticmethod
    def fix_common_structure_issues(data: Dict) -> Dict:
        """Fix common structural issues in parsed resume data"""
        # Ensure all required keys exist with proper defaults
        if 'candidate_info' not in data:
            data['candidate_info'] = {'name': 'Unknown', 'total_experience_years': 0}
        
        if 'employment' not in data:
            data['employment'] = []
        elif not isinstance(data['employment'], list):
            data['employment'] = [data['employment']] if isinstance(data['employment'], dict) else []
        
        if 'education' not in data:
            data['education'] = {'highest_degree': 'Unknown', 'institution': 'Unknown', 'year': 0}
        
        if 'key_skills' not in data:
            data['key_skills'] = []
        elif not isinstance(data['key_skills'], list):
            data['key_skills'] = []
        
        if 'certifications' not in data:
            data['certifications'] = []
        
        return data

class MistralTeacherModel:
    """Teacher model class with comprehensive error handling and debugging"""
    
    def __init__(self, model_name: str, gpu_id: int):
        self.model_name = model_name
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 and torch.cuda.is_available() else "cpu"
        self.json_extractor = ImprovedJSONExtractor()
        
        logger.info(f"Loading model: {model_name} on {self.device}")
        
        # Load tokenizer and processor with proper error handling
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_auth_token=True
            )
            
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_auth_token=True
            )
            
            # Load model with appropriate settings
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_auth_token=True,
                torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
                device_map=self.device
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def create_prompt(self, resume_text: str) -> str:
        """Create the instruction prompt for resume parsing - USES EXACT SAME PROMPT AS ORIGINAL"""
        return create_extraction_instruction(resume_text)
    
    def extract_json_with_retries(self, response: str, max_attempts: int = 3) -> Tuple[Dict, bool]:
        """Extract JSON with multiple retry strategies - COMPREHENSIVE ERROR HANDLING"""
        logger.debug(f"Starting JSON extraction from response of length {len(response)}")
        
        for attempt in range(max_attempts):
            logger.debug(f"JSON extraction attempt {attempt + 1}/{max_attempts}")
            
            # Step 1: Extract JSON block
            json_str = self.json_extractor.extract_json_block(response)
            if not json_str:
                logger.warning(f"Attempt {attempt + 1}: No JSON block found")
                continue
            
            logger.debug(f"Extracted JSON string of length {len(json_str)}")
            
            # Step 2: Clean the JSON string
            try:
                cleaned_json = self.json_extractor.clean_json_string(json_str)
                logger.debug(f"Cleaned JSON string of length {len(cleaned_json)}")
            except Exception as e:
                logger.warning(f"Error during JSON cleaning: {e}")
                cleaned_json = json_str  # Use original if cleaning fails
            
            # Step 3: Try to parse the cleaned JSON
            try:
                data = json.loads(cleaned_json)
                logger.debug("✅ JSON parsing successful!")
                
                # Step 4: Validate structure
                if self.json_extractor.validate_resume_structure(data):
                    logger.debug("✅ JSON structure validation passed")
                    return data, True
                else:
                    logger.debug("⚠️ JSON structure validation failed, applying fixes")
                    fixed_data = self.json_extractor.fix_common_structure_issues(data)
                    return fixed_data, True
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt + 1}: JSON parsing failed: {e}")
                error_pos = getattr(e, 'pos', None)
                if error_pos:
                    logger.debug(f"Error position: {error_pos}")
                    error_context = cleaned_json[max(0, error_pos-50):error_pos+50]
                    logger.debug(f"Error context: '{error_context}'")
                
                # Step 5: Try specific error-based repairs
                try:
                    if "Expecting ',' delimiter" in str(e):
                        logger.debug("Attempting comma delimiter fix...")
                        # More aggressive comma fixing for array elements
                        repaired_json = re.sub(r'"\s*\n\s*"', '",\n"', cleaned_json)
                        repaired_json = re.sub(r'(\w)\s*\n\s*"', r'\1,\n"', repaired_json)
                        repaired_json = re.sub(r'"\s*\n\s*(\w)', r'",\n\1', repaired_json)
                        
                    elif "Expecting property name" in str(e):
                        logger.debug("Attempting property name fix...")
                        repaired_json = re.sub(r',\s*}', '}', cleaned_json)
                        repaired_json = re.sub(r',\s*]', ']', cleaned_json)
                        
                    else:
                        # Generic repair attempt
                        logger.debug("Attempting generic JSON repair...")
                        repaired_json = cleaned_json
                    
                    # Step 6: Try parsing the repaired JSON
                    data = json.loads(repaired_json)
                    logger.debug("✅ Repaired JSON parsing successful!")
                    fixed_data = self.json_extractor.fix_common_structure_issues(data)
                    return fixed_data, True
                    
                except json.JSONDecodeError as e2:
                    logger.debug(f"❌ Repaired JSON also failed: {e2}")
                    
                    # Step 7: Last resort - try to find valid JSON portion
                    if attempt == max_attempts - 1:  # Last attempt
                        logger.debug("Attempting to find valid JSON portion...")
                        try:
                            decoder = json.JSONDecoder()
                            valid_data, end_idx = decoder.raw_decode(cleaned_json)
                            logger.debug(f"✅ Found valid JSON portion ending at index {end_idx}")
                            remaining = cleaned_json[end_idx:].strip()
                            if remaining:
                                logger.debug(f"Ignored remaining text: '{remaining[:100]}...'")
                            
                            # Validate and fix the partial JSON
                            if self.json_extractor.validate_resume_structure(valid_data):
                                return valid_data, True
                            else:
                                fixed_data = self.json_extractor.fix_common_structure_issues(valid_data)
                                return fixed_data, True
                                
                        except Exception as e3:
                            logger.debug(f"❌ Could not find valid JSON portion: {e3}")
                            continue
                
                except Exception as repair_error:
                    logger.warning(f"Error during JSON repair: {repair_error}")
                    continue
        
        # If all attempts fail, return a comprehensive default structure
        logger.error("🔴 All JSON extraction attempts failed, returning default structure")
        logger.debug(f"Failed response preview (first 500 chars): {response[:500]}")
        
        return {
            "candidate_info": {"name": "PARSE_ERROR", "total_experience_years": 0},
            "employment": [],
            "education": {"highest_degree": "Unknown", "institution": "Unknown", "year": 0},
            "key_skills": [],
            "certifications": [],
            "parse_error": True,
            "error_details": f"Failed to parse JSON from response of length {len(response)}"
        }, False
    
    def process_resume(self, resume_text: str, save_logits: bool = True) -> Dict[str, Any]:
        """Process a resume and return both JSON and logits for KD - ERROR FREE VERSION"""
        prompt = self.create_prompt(resume_text)
        
        # Create conversation format
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are a resume information extraction expert."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        
        # Apply chat template
        formatted_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Tokenize
        inputs = self.processor(text=formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate with output scores for KD
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,  # Increased from 1024 based on debugging findings
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
                output_scores=save_logits,
                return_dict_in_generate=True,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode response
        generated_ids = outputs.sequences[0, inputs["input_ids"].shape[-1]:]
        response = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        # Debug: Print raw response for troubleshooting
        logger.debug(f"Raw model response (first 200 chars): {response[:200]}")
        
        # Extract JSON using improved method
        resume_data, success = self.extract_json_with_retries(response)
        
        result = {
            "resume_data": resume_data,
            "parse_success": success,
            "raw_response": response[:1000]  # Limit raw response size for storage
        }
        
        # Save logits more efficiently if requested
        if save_logits and hasattr(outputs, 'scores'):
            # OPTIMIZATION: Store logits more efficiently to reduce file size
            logits_list = []
            for score in outputs.scores:
                # Convert to half precision and only store top 1000 logits per position
                score_cpu = score.cpu().half()  # Use fp16 instead of fp32
                top_values, top_indices = torch.topk(score_cpu, k=min(1000, score_cpu.size(-1)), dim=-1)
                logits_list.append({
                    'top_values': top_values.numpy(),
                    'top_indices': top_indices.numpy()
                })
            
            result["teacher_logits"] = logits_list
            result["generated_token_ids"] = generated_ids.cpu().numpy()
            
            # Log cache file estimated size
            estimated_size = len(logits_list) * 1000 * 2 * 2  # positions * top_k * fp16 * (values+indices)
            logger.debug(f"Estimated cache size: {estimated_size / (1024*1024):.1f} MB")
        
        return result

# Standalone usage example
if __name__ == "__main__":
    import argparse
    import glob
    
    def main():
        parser = argparse.ArgumentParser(description="Standalone Teacher Model for Resume Parsing")
        parser.add_argument("--input_dir", required=True, help="Directory containing resume text files")
        parser.add_argument("--output_file", required=True, help="Output JSON file")
        parser.add_argument("--model", default="mistralai/Mistral-Small-3.1-24B-Instruct-2503", help="Model name")
        parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
        
        args = parser.parse_args()
        
        # Load teacher model
        teacher = MistralTeacherModel(args.model, args.gpu)
        
        # Get resume files
        resume_files = glob.glob(os.path.join(args.input_dir, "*.txt"))
        logger.info(f"Found {len(resume_files)} resume files")
        
        results = []
        
        for resume_file in resume_files:
            with open(resume_file, 'r', encoding='utf-8') as f:
                resume_text = f.read()
            
            result = teacher.process_resume(resume_text, save_logits=False)
            result['source_file'] = os.path.basename(resume_file)
            results.append(result)
            
            logger.info(f"Processed {resume_file}: {'Success' if result['parse_success'] else 'Failed'}")
        
        # Save results
        with open(args.output_file, 'w') as f:
            json.dump({"resumes": results}, f, indent=2)
        
        logger.info(f"Results saved to {args.output_file}")
    
    main()