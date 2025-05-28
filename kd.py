#!/usr/bin/env python3
"""
Complete Knowledge Distillation Pipeline for Resume Parser
Implements two-stage process: Teacher inference + Student training
Imports teacher components from teacher.py
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import pickle
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

# Import teacher components - MAKE SURE teacher.py is in the same directory
from teacher import MistralTeacherModel, ImprovedJSONExtractor, create_extraction_instruction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class KDConfig:
    """Configuration for knowledge distillation"""
    # Model paths
    teacher_model_name: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    student_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Training parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_length: int = 2048  # Updated from 2048 based on our debugging findings
    
    # KD parameters
    temperature: float = 4.0
    alpha: float = 0.5  # Weight for KD loss
    beta: float = 0.5   # Weight for task loss
    
    # Paths
    cache_dir: str = "./teacher_cache"
    output_dir: str = "./student_output"
    
    # Device
    teacher_gpu: int = 0
    student_gpu: int = 0  # Same GPU, load one at a time

class TeacherInferenceManager:
    """Manages teacher model inference and caching"""
    
    def __init__(self, config: KDConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Use the teacher model from teacher.py
        self.teacher_model = MistralTeacherModel(
            config.teacher_model_name, 
            config.teacher_gpu
        )
    
    def process_and_cache_resumes(self, resume_files: List[str]):
        """Process all resumes with teacher and cache results"""
        logger.info(f"Processing {len(resume_files)} resumes with teacher model")
        
        for idx, resume_file in enumerate(tqdm(resume_files, desc="Teacher inference")):
            # Use original filename for cache file (remove .txt, add .pkl)
            original_name = os.path.basename(resume_file)
            cache_name = original_name.replace('.txt', '.pkl')
            cache_file = self.cache_dir / cache_name
            
            # Skip if already cached
            if cache_file.exists():
                logger.info(f"Skipping {resume_file} - already cached as {cache_name}")
                continue
            
            try:
                # Read resume
                with open(resume_file, 'r', encoding='utf-8') as f:
                    resume_text = f.read()
                
                # Process with teacher
                result = self.teacher_model.process_resume(resume_text, save_logits=True)
                
                # Add metadata
                result['resume_file'] = resume_file
                result['resume_text'] = resume_text
                result['index'] = idx
                
                # Save to cache
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                
                # Log file size and status for monitoring
                file_size_mb = cache_file.stat().st_size / (1024 * 1024)
                status = "✅ Success" if result['parse_success'] else "❌ Failed"
                logger.info(f"Cached {cache_name}: {file_size_mb:.1f} MB - {status}")
                
                # Clear GPU memory periodically
                if idx % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error processing {resume_file}: {e}")
                continue
        
        logger.info("Teacher inference completed")
        
        # Cleanup teacher model to free memory
        del self.teacher_model
        torch.cuda.empty_cache()

class ResumeKDDataset(Dataset):
    """Dataset for knowledge distillation from cached teacher outputs"""
    
    def __init__(self, cache_dir: str, tokenizer, max_length: int = 2048):
        self.cache_dir = Path(cache_dir)
        self.cache_files = sorted(list(self.cache_dir.glob("*.pkl")))
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Found {len(self.cache_files)} cached examples")
        
        # Log some example cache file names for verification
        if self.cache_files:
            logger.info(f"Example cache files: {[f.name for f in self.cache_files[:3]]}")
    
    def __len__(self):
        return len(self.cache_files)
    
    def __getitem__(self, idx):
        # Load cached data
        with open(self.cache_files[idx], 'rb') as f:
            data = pickle.load(f)
        
        # Prepare student input
        resume_text = data['resume_text']
        resume_json = data['resume_data']
        
        # Create the SAME prompt format as teacher - CRITICAL FIX
        prompt = self._create_prompt(resume_text)
        target = json.dumps(resume_json, indent=2)
        
        # Tokenize for student
        full_text = prompt + "\n" + target
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Find where the prompt ends
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'prompt_length': prompt_length,
            'teacher_logits': data.get('teacher_logits', None),
            'teacher_token_ids': data.get('generated_token_ids', None),
            'parse_success': data.get('parse_success', False),
            'resume_json': resume_json
        }
    
    def _create_prompt(self, resume_text: str) -> str:
        """Create the EXACT same prompt format as teacher - CRITICAL FIX"""
        # Use the EXACT same function as the teacher model
        return create_extraction_instruction(resume_text)

class KnowledgeDistillationLoss(nn.Module):
    """Combined loss for knowledge distillation"""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self, 
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined KD loss
        
        Args:
            student_logits: [batch_size, seq_len, vocab_size]
            teacher_logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        """
        # Task loss (cross-entropy)
        task_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction='none'
        )
        task_loss = (task_loss * attention_mask.view(-1)).sum() / attention_mask.sum()
        
        # KD loss (KL divergence)
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        kd_loss = self.kl_div(student_probs, teacher_probs) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * task_loss
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'kd_loss': kd_loss.item()
        }

class StudentTrainer:
    """Handles student model training with knowledge distillation"""
    
    def __init__(self, config: KDConfig):
        self.config = config
        self.device = f"cuda:{config.student_gpu}" if torch.cuda.is_available() else "cpu"
        
        # Load student model and tokenizer
        logger.info(f"Loading student model: {config.student_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.student_model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        
        # Initialize loss function
        self.kd_loss_fn = KnowledgeDistillationLoss(
            temperature=config.temperature,
            alpha=config.alpha
        )
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def train(self, train_dataset: ResumeKDDataset):
        """Train the student model using cached teacher outputs"""
        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        total_steps = len(train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        logger.info("Starting student training...")
        self.model.train()
        
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                prompt_length = batch['prompt_length']
                
                # Create labels (mask prompt tokens)
                labels = input_ids.clone()
                labels[:, :prompt_length] = -100  # Ignore prompt in loss
                
                # Forward pass
                with torch.amp.autocast('cuda'):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    student_logits = outputs.logits
                    
                    # Load teacher logits
                    if batch['teacher_logits'][0] is not None:
                        try:
                            # Handle both old and new logits formats
                            teacher_logits_list = batch['teacher_logits'][0]
                            
                            # Debug: Check what we actually received
                            if not teacher_logits_list:
                                logger.warning("Empty teacher_logits_list, falling back to task loss only")
                                raise ValueError("Empty logits list")
                            
                            first_item = teacher_logits_list[0]
                            logger.debug(f"First logits item type: {type(first_item)}")
                            
                            # Check if we have compressed logits (new format) or full logits (old format)
                            if isinstance(first_item, dict) and 'top_values' in first_item:
                                # New compressed format: reconstruct sparse tensors
                                logger.debug(f"Processing compressed logits: {len(teacher_logits_list)} steps")
                                teacher_logits_tensors = []
                                vocab_size = student_logits.size(-1)  # Get vocab size from student
                                
                                for step_logits in teacher_logits_list:
                                    if not isinstance(step_logits, dict):
                                        logger.warning(f"Expected dict, got {type(step_logits)}: {step_logits}")
                                        raise ValueError("Inconsistent logits format")
                                        
                                    # Create sparse tensor from top-k data
                                    top_values = torch.from_numpy(step_logits['top_values']).to(self.device)
                                    top_indices = torch.from_numpy(step_logits['top_indices']).to(self.device)
                                    
                                    # Create full logits tensor (initialized with very negative values)
                                    full_logits = torch.full((vocab_size,), -100.0, device=self.device, dtype=top_values.dtype)
                                    
                                    # Fill in the top-k values
                                    full_logits.scatter_(0, top_indices.flatten(), top_values.flatten())
                                    
                                    teacher_logits_tensors.append(full_logits.unsqueeze(0))  # Add batch dimension
                                
                                # Stack to create [batch_size, seq_len, vocab_size] tensor
                                teacher_logits = torch.stack(teacher_logits_tensors, dim=1)
                                logger.debug(f"Reconstructed teacher logits shape: {teacher_logits.shape}")
                                
                            elif isinstance(first_item, np.ndarray):
                                # Old full format: convert numpy arrays directly  
                                logger.debug(f"Processing full logits: {len(teacher_logits_list)} steps")
                                teacher_logits = torch.stack([
                                    torch.from_numpy(logits).to(self.device) 
                                    for logits in teacher_logits_list
                                ], dim=1)
                            else:
                                logger.error(f"Unexpected logits format. First item: {type(first_item)}, content: {first_item}")
                                raise ValueError(f"Unsupported logits format: {type(first_item)}")
                            
                            # Align dimensions with student logits
                            min_len = min(student_logits.size(1), teacher_logits.size(1))
                            student_logits_aligned = student_logits[:, :min_len]
                            teacher_logits_aligned = teacher_logits[:, :min_len]
                            labels_aligned = labels[:, :min_len]
                            mask_aligned = attention_mask[:, :min_len]
                            
                            logger.debug(f"Aligned shapes - Student: {student_logits_aligned.shape}, Teacher: {teacher_logits_aligned.shape}")
                            
                            # Compute loss
                            loss, loss_dict = self.kd_loss_fn(
                                student_logits_aligned,
                                teacher_logits_aligned,
                                labels_aligned,
                                mask_aligned
                            )
                            
                        except Exception as e:
                            logger.warning(f"Error processing teacher logits: {e}. Falling back to task loss only.")
                            # Fallback to regular cross-entropy if teacher logits fail
                            loss = F.cross_entropy(
                                student_logits.view(-1, student_logits.size(-1)),
                                labels.view(-1),
                                ignore_index=-100
                            )
                            loss_dict = {'total_loss': loss.item(), 'task_loss': loss.item(), 'kd_loss': 0.0}
                    else:
                        # Fallback to regular cross-entropy if no teacher logits
                        loss = F.cross_entropy(
                            student_logits.view(-1, student_logits.size(-1)),
                            labels.view(-1),
                            ignore_index=-100
                        )
                        loss_dict = {'total_loss': loss.item()}
                
                # Backward pass with gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                epoch_losses.append(loss_dict.get('total_loss', loss.item()))
                
                # Update weights
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Update progress bar
                    avg_loss = np.mean(epoch_losses[-100:])
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })
            
            # Save checkpoint at end of epoch
            avg_epoch_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch + 1} - Average loss: {avg_epoch_loss:.4f}")
            
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                self.save_checkpoint(epoch, global_step, best_loss)
    
    def save_checkpoint(self, epoch: int, step: int, loss: float):
        """Save model checkpoint"""
        checkpoint_dir = self.output_dir / f"checkpoint-epoch{epoch}-step{step}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training info
        with open(checkpoint_dir / "training_info.json", 'w') as f:
            json.dump({
                'epoch': epoch,
                'step': step,
                'loss': loss,
                'config': self.config.__dict__
            }, f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def evaluate_json_parsing(self, eval_dataset: ResumeKDDataset, num_samples: int = 50):
        """Evaluate student's JSON parsing accuracy"""
        logger.info(f"Evaluating JSON parsing on {num_samples} samples...")
        
        self.model.eval()
        success_count = 0
        results = []
        
        with torch.no_grad():
            for i in range(min(num_samples, len(eval_dataset))):
                batch = eval_dataset[i]
                
                # Generate response
                input_ids = batch['input_ids'][:batch['prompt_length']].unsqueeze(0).to(self.device)
                
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=1024,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Try to extract JSON using the same extraction logic as teacher
                try:
                    extractor = ImprovedJSONExtractor()
                    json_str = extractor.extract_json_block(response)
                    if json_str:
                        parsed_json = json.loads(extractor.clean_json_string(json_str))
                        success_count += 1
                        results.append({
                            'success': True,
                            'student_json': parsed_json,
                            'teacher_json': batch['resume_json']
                        })
                    else:
                        results.append({'success': False, 'error': 'No JSON found'})
                except Exception as e:
                    results.append({'success': False, 'error': str(e)})
        
        success_rate = success_count / num_samples
        logger.info(f"JSON parsing success rate: {success_rate:.2%}")
        
        # Save evaluation results
        with open(self.output_dir / "evaluation_results.json", 'w') as f:
            json.dump({
                'success_rate': success_rate,
                'total_samples': num_samples,
                'successful_parses': success_count,
                'results': results[:10]  # Save first 10 for inspection
            }, f, indent=2)
        
        return success_rate

class KnowledgeDistillationPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: KDConfig):
        self.config = config
    
    def run(self, resume_files: List[str]):
        """Run the complete KD pipeline"""
        logger.info("Starting Knowledge Distillation Pipeline")
        
        # Stage 1: Teacher Inference and Caching
        logger.info("Stage 1: Teacher Inference")
        teacher_manager = TeacherInferenceManager(self.config)
        teacher_manager.process_and_cache_resumes(resume_files)
        
        # Clear GPU memory before loading student
        torch.cuda.empty_cache()
        
        # Stage 2: Student Training
        logger.info("Stage 2: Student Training")
        trainer = StudentTrainer(self.config)
        
        # Create dataset from cached teacher outputs
        dataset = ResumeKDDataset(
            self.config.cache_dir,
            trainer.tokenizer,
            self.config.max_length
        )
        
        # Train student
        trainer.train(dataset)
        
        # Evaluate
        success_rate = trainer.evaluate_json_parsing(dataset)
        logger.info(f"Final JSON parsing success rate: {success_rate:.2%}")
        
        return success_rate

# Utility functions for running the pipeline

def prepare_resume_files(input_dir: str) -> List[str]:
    """Get list of resume files from directory"""
    import glob
    resume_files = glob.glob(os.path.join(input_dir, "*.txt"))
    logger.info(f"Found {len(resume_files)} resume files")
    return resume_files

def main():
    """Example usage of the KD pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge Distillation Pipeline for Resume Parser")
    parser.add_argument("--input_dir", required=True, help="Directory containing resume text files")
    parser.add_argument("--cache_dir", default="./teacher_cache", help="Directory to cache teacher outputs")
    parser.add_argument("--output_dir", default="./student_output", help="Directory to save student model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=4.0, help="KD temperature")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for KD loss")
    
    args = parser.parse_args()
    
    # Create configuration
    config = KDConfig(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        alpha=args.alpha
    )
    
    # Get resume files
    resume_files = prepare_resume_files(args.input_dir)
    
    # Run pipeline
    pipeline = KnowledgeDistillationPipeline(config)
    success_rate = pipeline.run(resume_files)
    
    logger.info(f"Pipeline completed! Final success rate: {success_rate:.2%}")

if __name__ == "__main__":
    main()