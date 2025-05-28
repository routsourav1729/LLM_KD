#!/usr/bin/env python3
"""
Stage 2: Student Training Only
Run this after teacher inference is complete.
This will load cached teacher outputs and train the student model.
"""

import os
import argparse
import logging
import torch
from pathlib import Path

# Import from our files
from kd import KDConfig, StudentTrainer, ResumeKDDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_cache_directory(cache_dir: str) -> bool:
    """Check if cache directory has teacher outputs"""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        logger.error(f"Cache directory does not exist: {cache_dir}")
        return False
    
    cached_files = list(cache_path.glob("*.pkl"))
    if not cached_files:
        logger.error(f"No cached teacher outputs found in: {cache_dir}")
        logger.error("Run teacher inference first with: python run_teacher.py")
        return False
    
    logger.info(f"Found {len(cached_files)} cached teacher outputs")
    return True

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Student Model Training with Knowledge Distillation")
    parser.add_argument("--cache_dir", required=True, help="Directory with cached teacher outputs")
    parser.add_argument("--output_dir", default="./student_output", help="Directory to save student model")
    parser.add_argument("--student_model", default="mistralai/Mistral-7B-Instruct-v0.2", 
                        help="Student model name")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID for student model")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    
    # KD parameters
    parser.add_argument("--temperature", type=float, default=4.0, help="Knowledge distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for KD loss (0.5 = equal weight)")
    
    # Evaluation
    parser.add_argument("--eval_samples", type=int, default=50, help="Number of samples for evaluation")
    
    args = parser.parse_args()
    
    # Validate cache directory
    if not validate_cache_directory(args.cache_dir):
        return
    
    # Clear GPU memory from any previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create configuration for student training
    config = KDConfig(
        student_model_name=args.student_model,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        student_gpu=args.gpu,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        temperature=args.temperature,
        alpha=args.alpha
    )
    
    logger.info("=" * 60)
    logger.info("STAGE 2: STUDENT TRAINING")
    logger.info("=" * 60)
    logger.info(f"Student Model: {config.student_model_name}")
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Cache Directory: {args.cache_dir}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"KD Temperature: {args.temperature}")
    logger.info(f"KD Alpha: {args.alpha}")
    logger.info(f"Effective Batch Size: {args.batch_size * args.gradient_accumulation}")
    
    try:
        # Create student trainer
        logger.info("Loading student model...")
        trainer = StudentTrainer(config)
        
        # Create dataset from cached teacher outputs
        logger.info("Creating dataset from cached teacher outputs...")
        dataset = ResumeKDDataset(
            config.cache_dir,
            trainer.tokenizer,
            config.max_length
        )
        
        if len(dataset) == 0:
            logger.error("No valid cached examples found!")
            return
        
        logger.info(f"Dataset created with {len(dataset)} examples")
        
        # Train student model
        logger.info("Starting student training...")
        trainer.train(dataset)
        
        # Evaluate the trained model
        logger.info("Evaluating trained student model...")
        success_rate = trainer.evaluate_json_parsing(dataset, num_samples=args.eval_samples)
        
        # Final summary
        logger.info("=" * 60)
        logger.info("STUDENT TRAINING COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"Final JSON parsing success rate: {success_rate:.2%}")
        logger.info(f"Model saved to: {Path(args.output_dir).absolute()}")
        
        # Find the best checkpoint
        output_path = Path(args.output_dir)
        checkpoints = list(output_path.glob("checkpoint-*"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            logger.info(f"Best checkpoint: {latest_checkpoint}")
            logger.info(f"\nTo use the trained model:")
            logger.info(f"from transformers import AutoModelForCausalLM, AutoTokenizer")
            logger.info(f"model = AutoModelForCausalLM.from_pretrained('{latest_checkpoint}')")
            logger.info(f"tokenizer = AutoTokenizer.from_pretrained('{latest_checkpoint}')")
        
    except Exception as e:
        logger.error(f"Error during student training: {e}")
        raise

if __name__ == "__main__":
    main()