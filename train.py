import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
import argparse
from tqdm import tqdm
from datetime import datetime

# Import our modules
from models.biaffine_parser import BiaffineDependencyParser, ParserConfig
from models.bert_encoder import VietnameseBertTokenizer
from datasets.ud_dataset import create_dataloaders
from utils.mst_decoder import MSTDecoder, GreedyDecoder, evaluate_parsing_accuracy


class DependencyParserTrainer:
    def __init__(self, config: ParserConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        os.makedirs(config.output_dir, exist_ok=True)
        # Save config
        with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(os.path.join(config.output_dir, 'logs'))
        
        # Initialize tokenizer
        self.tokenizer = VietnameseBertTokenizer(config.bert_model_name)
        
        # Load data
        self.train_loader, self.dev_loader, self.test_loader, self.label_vocab = create_dataloaders(
            config.train_path,
            config.dev_path, 
            config.test_path,
            self.tokenizer,
            config.batch_size,
            config.max_sequence_length
        )
        
        # Update config with actual number of labels
        config.num_labels = self.label_vocab.size()
        
        # Initialize model
        self.model = BiaffineDependencyParser(
            bert_model_name=config.bert_model_name,
            num_labels=config.num_labels,
            arc_mlp_size=config.arc_mlp_size,
            label_mlp_size=config.label_mlp_size,
            dropout=config.dropout,
            freeze_bert=config.freeze_bert
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        total_steps = len(self.train_loader) * config.max_epochs // config.accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Initialize decoders
        self.mst_decoder = MSTDecoder()
        self.greedy_decoder = GreedyDecoder()
        
        # Training state
        self.global_step = 0
        self.best_dev_uas = 0.0
        self.patience_counter = 0
    
    def _create_optimizer(self):
        """Create optimizer with different learning rates for BERT and other parameters."""
        bert_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'bert_encoder.bert' in name:
                bert_params.append(param)
            else:
                other_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': bert_params, 'lr': self.config.bert_learning_rate},
            {'params': other_params, 'lr': self.config.learning_rate}
        ], weight_decay=self.config.weight_decay)
        
        return optimizer
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_arc_loss = 0.0
        total_label_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            arc_scores, label_scores = self.model(
                batch['input_ids'],
                batch['attention_mask'],
                batch['valid_positions'],
                batch['word_mask']
            )
            
            # Compute loss
            loss, arc_loss, label_loss = self.model.compute_loss(
                arc_scores,
                label_scores,
                batch['heads'],
                batch['deprels'],
                batch['word_mask']
            )
            
            # Scale loss by accumulation steps
            loss = loss / self.config.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulation_steps
            if (step + 1) % self.config.accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Log to tensorboard
                if self.global_step % 10 == 0:
                    self.writer.add_scalar('train/loss', loss.item() * self.config.accumulation_steps, self.global_step)
                    self.writer.add_scalar('train/arc_loss', arc_loss.item(), self.global_step)
                    self.writer.add_scalar('train/label_loss', label_loss.item(), self.global_step)
                    self.writer.add_scalar('train/learning_rate', self.scheduler.get_last_lr()[0], self.global_step)
            
            # Update progress bar
            total_loss += loss.item() * self.config.accumulation_steps
            total_arc_loss += arc_loss.item()
            total_label_loss += label_loss.item()
            
            avg_loss = total_loss / (step + 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'arc_loss': f'{total_arc_loss / (step + 1):.4f}',
                'label_loss': f'{total_label_loss / (step + 1):.4f}'
            })
            
            # Evaluate on dev set
            if self.global_step % self.config.eval_steps == 0:
                dev_metrics = self.evaluate(self.dev_loader, "dev")
                self._log_metrics(dev_metrics, "dev", self.global_step)
                
                # Check for improvement
                if dev_metrics['uas'] > self.best_dev_uas:
                    self.best_dev_uas = dev_metrics['uas']
                    self.patience_counter = 0
                    self._save_model('best_model.pt')
                else:
                    self.patience_counter += 1
                
                self.model.train()  # Back to training mode
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
    
    def evaluate(self, dataloader, split_name: str = "eval"):
        """Evaluate the model."""
        self.model.eval()
        
        total_loss = 0.0
        all_predicted_heads = []
        all_gold_heads = []
        all_predicted_labels = []
        all_gold_labels = []
        all_masks = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                arc_scores, label_scores = self.model(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch['valid_positions'],
                    batch['word_mask']
                )
                
                # Compute loss
                loss, _, _ = self.model.compute_loss(
                    arc_scores,
                    label_scores,
                    batch['heads'],
                    batch['deprels'],
                    batch['word_mask']
                )
                
                total_loss += loss.item()
                
                # Decode with MST
                predicted_heads = self.mst_decoder.decode(arc_scores, batch['word_mask'])
                
                # Get predicted labels
                predicted_labels = []
                for b in range(batch['batch_size']):
                    sent_labels = []
                    for i in range(batch['max_words']):
                        if batch['word_mask'][b, i]:
                            head = predicted_heads[b][i]
                            label_scores_arc = label_scores[b, head, i]
                            predicted_label = label_scores_arc.argmax().item()
                            sent_labels.append(predicted_label)
                        else:
                            sent_labels.append(0)
                    predicted_labels.append(sent_labels)
                
                # Collect results
                all_predicted_heads.extend(predicted_heads)
                all_predicted_labels.extend(predicted_labels)
                all_gold_heads.extend(batch['heads'].cpu().tolist())
                all_gold_labels.extend(batch['deprels'].cpu().tolist())
                all_masks.extend(batch['word_mask'].cpu().tolist())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        
        # UAS (Unlabeled Attachment Score)
        uas, uas_correct, uas_total = evaluate_parsing_accuracy(
            all_predicted_heads, all_gold_heads, torch.tensor(all_masks)
        )
        
        # LAS (Labeled Attachment Score) 
        las_correct = 0
        las_total = 0
        
        for pred_heads, gold_heads, pred_labels, gold_labels, mask in zip(
            all_predicted_heads, all_gold_heads, all_predicted_labels, all_gold_labels, all_masks
        ):
            for i, (ph, gh, pl, gl, m) in enumerate(zip(pred_heads, gold_heads, pred_labels, gold_labels, mask)):
                if m:  # Valid token
                    if ph == gh and pl == gl:
                        las_correct += 1
                    las_total += 1
        
        las = las_correct / las_total if las_total > 0 else 0.0
        
        metrics = {
            'loss': avg_loss,
            'uas': uas,
            'las': las,
            'uas_correct': uas_correct,
            'uas_total': uas_total,
            'las_correct': las_correct,
            'las_total': las_total
        }
        
        return metrics
    
    def _log_metrics(self, metrics: dict, split: str, step: int):
        """Log metrics to tensorboard and console."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'{split}/{key}', value, step)
        
        print(f"\n{split.upper()} Metrics at step {step}:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  UAS: {metrics['uas']:.4f} ({metrics['uas_correct']}/{metrics['uas_total']})")
        print(f"  LAS: {metrics['las']:.4f} ({metrics['las_correct']}/{metrics['las_total']})")
    
    def _save_model(self, filename: str):
        """Save model state dict."""
        torch.save(self.model.state_dict(), os.path.join(self.config.output_dir, filename))
    
    def _save_checkpoint(self, filename: str):
        """Save full checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_dev_uas': self.best_dev_uas,
            'config': self.config.to_dict()
        }
        torch.save(checkpoint, os.path.join(self.config.output_dir, filename))
    
    def train(self):
        """Main training loop."""
        print(f"Starting training...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Training on {len(self.train_loader)} batches")
        print(f"Evaluating on {len(self.dev_loader)} batches")
        print(f"Number of labels: {self.config.num_labels}")
        
        for epoch in range(self.config.max_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")
            
            self.train_epoch(epoch)
            
            # Evaluate on dev set at end of epoch
            dev_metrics = self.evaluate(self.dev_loader, "dev")
            self._log_metrics(dev_metrics, "dev", self.global_step)
            
            # Early stopping check
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping after {self.config.early_stopping_patience} epochs without improvement")
                break
        
        # Final evaluation on test set
        print("\nEvaluating on test set...")
        test_metrics = self.evaluate(self.test_loader, "test")
        self._log_metrics(test_metrics, "test", self.global_step)
        
        # Save final results
        results = {
            'best_dev_uas': self.best_dev_uas,
            'final_test_uas': test_metrics['uas'],
            'final_test_las': test_metrics['las']
        }
        
        with open(os.path.join(self.config.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Best dev UAS: {self.best_dev_uas:.4f}")
        print(f"Final test UAS: {test_metrics['uas']:.4f}")
        print(f"Final test LAS: {test_metrics['las']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Biaffine Dependency Parser')
    parser.add_argument('--config', type=str, help='Path to config file (JSON)')    
    args = parser.parse_args()
    
    # Create config
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        config = ParserConfig.from_dict(config_dict)
    else:
        raise ValueError("Config file path must be provided with --config")
    
    # Add timestamp to output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output_dir = os.path.join(config.output_dir, f"run_{timestamp}")
    os.makedirs(config.output_dir, exist_ok=True)

    # Create trainer and start training
    trainer = DependencyParserTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()