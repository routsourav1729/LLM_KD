#!/usr/bin/env python3
"""
Stage 1: Teacher Inference Only
Run this first to generate teacher outputs and cache logits for knowledge distillation.
This will create cached files in teacher_cache/ directory.
"""

import os
import argparse
import glob
import logging
from pathlib import Path

# Import from our files
from kd import KDConfig, TeacherInferenceManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_resume_files(input_dir: str) -> list:
    """Get list of resume files from directory"""
    resume_files = glob.glob(os.path.join(input_dir, "*.txt"))
    logger.info(f"Found {len(resume_files)} resume files")
    return resume_files

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Teacher Model Inference and Caching")
    parser.add_argument("--input_dir", required=True, help="Directory containing resume text files")
    parser.add_argument("--cache_dir", default="./teacher_cache", help="Directory to cache teacher outputs")
    parser.add_argument("--teacher_model", default="mistralai/Mistral-Small-3.1-24B-Instruct-2503", 
                        help="Teacher model name")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID for teacher model")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    # Create configuration for teacher only
    config = KDConfig(
        teacher_model_name=args.teacher_model,
        cache_dir=args.cache_dir,
        teacher_gpu=args.gpu
    )
    
    logger.info("=" * 60)
    logger.info("STAGE 1: TEACHER INFERENCE")
    logger.info("=" * 60)
    logger.info(f"Teacher Model: {config.teacher_model_name}")
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Cache Directory: {args.cache_dir}")
    logger.info(f"Input Directory: {args.input_dir}")
    
    # Get resume files
    resume_files = prepare_resume_files(args.input_dir)
    if not resume_files:
        logger.error("No .txt files found in input directory!")
        return
    
    # Create teacher manager and process resumes
    teacher_manager = TeacherInferenceManager(config)
    
    try:
        teacher_manager.process_and_cache_resumes(resume_files)
        
        # Check what was created
        cache_dir = Path(args.cache_dir)
        cached_files = list(cache_dir.glob("*.pkl"))
        
        logger.info("=" * 60)
        logger.info("TEACHER INFERENCE COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"Total resumes processed: {len(resume_files)}")
        logger.info(f"Cached files created: {len(cached_files)}")
        logger.info(f"Cache directory: {cache_dir.absolute()}")
        logger.info("\nNext step: Run student training with:")
        logger.info(f"python run_student.py --cache_dir {args.cache_dir}")
        
    except Exception as e:
        logger.error(f"Error during teacher inference: {e}")
        raise

if __name__ == "__main__":
    main()