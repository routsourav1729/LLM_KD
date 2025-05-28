#!/usr/bin/env python3
"""
Enhanced Debug script for testing teacher model on failing resumes
Now includes intelligent handling of truncated JSON responses
Educational approach to understanding and fixing generation issues
"""
import torch
import os
import json
import argparse
import logging
import re
from teacher import MistralTeacherModel, create_extraction_instruction

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def attempt_json_completion(truncated_json: str) -> str:
    """
    Educational approach to completing truncated JSON
    This teaches us about defensive programming and graceful error handling
    """
    print(f"\n📝 **Educational Moment: Attempting to Complete Truncated JSON**")
    print(f"Original length: {len(truncated_json)} characters")
    
    # Step 1: Detect what kind of truncation we have
    last_char = truncated_json.rstrip()[-1] if truncated_json.rstrip() else ''
    print(f"Last character before whitespace: '{last_char}'")
    
    # Step 2: Try to intelligently complete the JSON
    completed_json = truncated_json
    
    # If we're in the middle of a string, complete it
    if last_char != '"' and '"' in truncated_json and truncated_json.count('"') % 2 == 1:
        print("🔧 Detected incomplete string - adding closing quote")
        completed_json += '"'
    
    # Count unclosed brackets and braces
    open_braces = completed_json.count('{') - completed_json.count('}')
    open_brackets = completed_json.count('[') - completed_json.count(']')
    
    print(f"🔧 Unclosed braces: {open_braces}, Unclosed brackets: {open_brackets}")
    
    # Close any unclosed arrays first, then objects
    for _ in range(open_brackets):
        completed_json += ']'
        print("🔧 Added closing bracket ]")
    
    for _ in range(open_braces):
        completed_json += '}'
        print("🔧 Added closing brace }")
    
    print(f"Completed JSON length: {len(completed_json)} characters")
    return completed_json

def analyze_generation_parameters(teacher_model, resume_text: str):
    """
    Educational analysis of how different generation parameters affect output
    This teaches us about the relationship between model parameters and output quality
    """
    print(f"\n🧪 **Educational Experiment: Testing Different Generation Parameters**")
    
    prompt = create_extraction_instruction(resume_text)
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "You are a resume information extraction expert."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    
    formatted_prompt = teacher_model.processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )
    
    inputs = teacher_model.processor(text=formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(teacher_model.model.device) for k, v in inputs.items()}
    
    # Test different parameter combinations
    parameter_sets = [
        {"name": "Conservative (Original)", "max_new_tokens": 1024, "temperature": 0.1, "do_sample": True},
        {"name": "More Tokens", "max_new_tokens": 2048, "temperature": 0.1, "do_sample": True},
        {"name": "Deterministic", "max_new_tokens": 1024, "temperature": 0.0, "do_sample": False},
        {"name": "More Tokens + Deterministic", "max_new_tokens": 2048, "temperature": 0.0, "do_sample": False},
    ]
    
    results = []
    
    for params in parameter_sets:
        print(f"\n🔬 Testing: {params['name']}")
        print(f"   Parameters: max_tokens={params['max_new_tokens']}, temp={params['temperature']}, sample={params['do_sample']}")
        
        try:
            with torch.no_grad():
                outputs = teacher_model.model.generate(
                    **inputs,
                    max_new_tokens=params['max_new_tokens'],
                    temperature=params['temperature'],
                    do_sample=params['do_sample'],
                    top_p=0.95 if params['do_sample'] else None,
                    pad_token_id=teacher_model.processor.tokenizer.eos_token_id
                )
            
            generated_ids = outputs[0, inputs["input_ids"].shape[-1]:]
            response = teacher_model.processor.decode(generated_ids, skip_special_tokens=True)
            
            # Quick analysis
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                
                # Check if it's complete JSON
                try:
                    json.loads(json_str)
                    json_status = "✅ Valid JSON"
                except json.JSONDecodeError as e:
                    if "Expecting ',' delimiter" in str(e):
                        json_status = "❌ Missing comma (likely truncated)"
                    elif "Unterminated string" in str(e):
                        json_status = "❌ Unterminated string (definitely truncated)"
                    else:
                        json_status = f"❌ Other error: {str(e)[:50]}"
                
                print(f"   📊 Response length: {len(response)} chars")
                print(f"   📊 JSON length: {len(json_str)} chars")
                print(f"   📊 Status: {json_status}")
                
                results.append({
                    'params': params,
                    'response_length': len(response),
                    'json_length': len(json_str),
                    'status': json_status,
                    'json_str': json_str[:500]  # First 500 chars for comparison
                })
            else:
                print(f"   📊 No JSON found in response")
                results.append({
                    'params': params,
                    'response_length': len(response),
                    'json_length': 0,
                    'status': "❌ No JSON found",
                    'json_str': ''
                })
                
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            results.append({
                'params': params,
                'status': f"❌ Generation error: {e}",
                'response_length': 0,
                'json_length': 0,
                'json_str': ''
            })
    
    # Summary analysis
    print(f"\n📈 **Parameter Testing Summary**")
    print(f"{'Parameter Set':<25} {'Response Len':<12} {'JSON Len':<10} {'Status'}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['params']['name']:<25} {result['response_length']:<12} {result['json_length']:<10} {result['status']}")
    
    # Find best performing parameters
    valid_results = [r for r in results if 'Valid JSON' in r['status']]
    if valid_results:
        best = max(valid_results, key=lambda x: x['json_length'])
        print(f"\n🏆 **Best performing parameters:** {best['params']['name']}")
        print(f"   Generated {best['json_length']} character valid JSON")
        return best['params']
    else:
        print(f"\n⚠️ **No parameter set generated valid JSON - all responses were truncated**")
        longest = max(results, key=lambda x: x['json_length'])
        print(f"   Longest JSON was {longest['json_length']} chars with {longest['params']['name']}")
        return longest['params']

def test_single_resume_enhanced(teacher_model, resume_file: str):
    """Enhanced testing with educational insights about truncation and completion"""
    
    print(f"\n{'='*60}")
    print(f"🎓 **ENHANCED EDUCATIONAL TESTING**")
    print(f"Resume: {os.path.basename(resume_file)}")
    print(f"{'='*60}")
    
    # Read resume
    with open(resume_file, 'r', encoding='utf-8') as f:
        resume_text = f.read()
    
    print(f"📄 Resume length: {len(resume_text)} characters")
    
    # Step 1: Test with original parameters
    print(f"\n🔬 **Step 1: Testing with Original Parameters**")
    result = teacher_model.process_resume(resume_text, save_logits=False)
    
    print(f"Parse success: {result['parse_success']}")
    
    if not result['parse_success']:
        raw_response = result.get('raw_response', '')
        
        # Extract the JSON portion for analysis
        start_idx = raw_response.find('{')
        end_idx = raw_response.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = raw_response[start_idx:end_idx]
            
            print(f"\n🔍 **Detailed Truncation Analysis**")
            print(f"Raw response length: {len(raw_response)}")
            print(f"JSON portion length: {len(json_str)}")
            
            # Check if this looks like truncation
            if not json_str.rstrip().endswith('}'):
                print(f"🚨 **Truncation Detected!** JSON doesn't end with closing brace")
                print(f"Last 50 characters: '{json_str[-50:]}'")
                
                # Attempt completion
                print(f"\n🛠️ **Attempting Intelligent JSON Completion**")
                completed_json = attempt_json_completion(json_str)
                
                # Test if completion worked
                try:
                    completed_data = json.loads(completed_json)
                    print(f"✅ **Completion Successful!**")
                    print(f"Completed JSON keys: {list(completed_data.keys())}")
                    
                    # Validate structure
                    if 'candidate_info' in completed_data:
                        candidate_name = completed_data.get('candidate_info', {}).get('name', 'N/A')
                        print(f"📝 Candidate name: {candidate_name}")
                    
                    if 'employment' in completed_data:
                        employment_count = len(completed_data.get('employment', []))
                        print(f"💼 Employment entries: {employment_count}")
                    
                    return completed_data, True
                    
                except json.JSONDecodeError as e:
                    print(f"❌ **Completion Failed:** {e}")
        
        # Step 2: If truncation is the issue, test different parameters
        print(f"\n🔬 **Step 2: Testing Different Generation Parameters**")
        best_params = analyze_generation_parameters(teacher_model, resume_text)
        
        # Step 3: Try with best parameters
        print(f"\n🔬 **Step 3: Retrying with Best Parameters**")
        print(f"Optimal parameters found: {best_params}")
        
    else:
        print(f"✅ **Original parameters worked fine!**")
        candidate_name = result['resume_data'].get('candidate_info', {}).get('name', 'N/A')
        print(f"📝 Candidate name: {candidate_name}")
        employment_count = len(result['resume_data'].get('employment', []))
        print(f"💼 Employment entries: {employment_count}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Enhanced debug teacher model with truncation handling")
    parser.add_argument("--resume_file", help="Single resume file to test")
    parser.add_argument("--resume_dir", help="Directory of resumes to test")
    parser.add_argument("--gpu", type=int, default=7, help="GPU ID")
    parser.add_argument("--model", default="mistralai/Mistral-Small-3.1-24B-Instruct-2503", help="Model name")
    parser.add_argument("--max_files", type=int, default=3, help="Max files to test from directory")
    
    args = parser.parse_args()
    
    if not args.resume_file and not args.resume_dir:
        print("Please provide either --resume_file or --resume_dir")
        return
    
    # Load teacher model
    print(f"🤖 Loading teacher model: {args.model}")
    teacher_model = MistralTeacherModel(args.model, args.gpu)
    print(f"✅ Model loaded successfully!")
    
    # Import torch here since we need it for parameter testing
    import torch
    
    # Test files
    if args.resume_file:
        test_single_resume_enhanced(teacher_model, args.resume_file)
    
    elif args.resume_dir:
        import glob
        resume_files = glob.glob(os.path.join(args.resume_dir, "*.txt"))
        print(f"📁 Found {len(resume_files)} resume files")
        
        for i, resume_file in enumerate(resume_files[:args.max_files]):
            result = test_single_resume_enhanced(teacher_model, resume_file)

if __name__ == "__main__":
    main()