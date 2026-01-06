import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from models.bert_encoder import BertEncoder, extract_word_representations
from modules.biaffine_attention import BiaffineParser


class BiaffineDependencyParser(nn.Module):
    """
    Complete biaffine dependency parser with BERT backbone.
    
    Args:
        bert_model_name: Name of pretrained BERT model
        num_labels: Number of dependency relation labels
        arc_mlp_size: Hidden size for arc MLP
        label_mlp_size: Hidden size for label MLP
        dropout: Dropout rate
        freeze_bert: Whether to freeze BERT parameters
    """
    
    def __init__(self,
                 bert_model_name: str = "vinai/phobert-base-v2",
                 num_labels: int = 40,
                 arc_mlp_size: int = 500,
                 label_mlp_size: int = 100,
                 dropout: float = 0.33,
                 freeze_bert: bool = False):
        
        super(BiaffineDependencyParser, self).__init__()
        
        self.num_labels = num_labels
        
        # BERT encoder
        self.bert_encoder = BertEncoder(
            model_name=bert_model_name,
            dropout=dropout,
            freeze_bert=freeze_bert
        )
        
        # Biaffine parser
        self.biaffine_parser = BiaffineParser(
            input_size=self.bert_encoder.hidden_size,
            num_labels=num_labels,
            arc_mlp_size=arc_mlp_size,
            label_mlp_size=label_mlp_size,
            dropout=dropout
        )
        
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                valid_positions: torch.Tensor,
                word_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the dependency parser.
        
        Args:
            input_ids: [batch_size, seq_len] - BERT input tokens
            attention_mask: [batch_size, seq_len] - BERT attention mask  
            valid_positions: [batch_size, max_words] - positions of first subword for each word
            word_mask: [batch_size, max_words] - mask for valid words (optional)
            
        Returns:
            arc_scores: [batch_size, max_words, max_words] - arc attachment scores
            label_scores: [batch_size, max_words, max_words, num_labels] - label scores
        """
        # Encode with BERT
        encoded_output = self.bert_encoder(input_ids, attention_mask)
        
        # Extract word-level representations
        word_representations = extract_word_representations(encoded_output, valid_positions)
        
        # Generate arc and label scores
        arc_scores, label_scores = self.biaffine_parser(word_representations, word_mask)
        
        return arc_scores, label_scores
    
    def compute_loss(self,
                    arc_scores: torch.Tensor,
                    label_scores: torch.Tensor,
                    arc_targets: torch.Tensor,
                    label_targets: torch.Tensor,
                    mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute parsing loss.
        
        Args:
            arc_scores: [batch_size, seq_len, seq_len] - predicted arc scores
            label_scores: [batch_size, seq_len, seq_len, num_labels] - predicted label scores
            arc_targets: [batch_size, seq_len] - target head indices
            label_targets: [batch_size, seq_len] - target label indices
            mask: [batch_size, seq_len] - mask for valid tokens
            
        Returns:
            total_loss: Combined arc and label loss
            arc_loss: Arc attachment loss
            label_loss: Label classification loss
        """
        batch_size, seq_len = mask.shape
        
        # Create arc target matrix for cross-entropy
        # arc_targets contains head index for each word
        arc_target_matrix = torch.zeros_like(arc_scores, dtype=torch.bool)
        for b in range(batch_size):
            for i in range(seq_len):
                if mask[b, i]:
                    head_idx = arc_targets[b, i]
                    if 0 <= head_idx < seq_len:
                        arc_target_matrix[b, head_idx, i] = True
        
        # Arc loss - for each dependent, predict the correct head
        arc_loss = 0
        num_valid_tokens = 0
        
        for b in range(batch_size):
            for i in range(seq_len):
                if mask[b, i]:  # Valid token
                    head_idx = arc_targets[b, i]
                    if 0 <= head_idx < seq_len:
                        # Cross-entropy loss for this dependent
                        arc_loss += F.cross_entropy(
                            arc_scores[b, :, i].unsqueeze(0),  # Scores for all possible heads
                            torch.tensor([head_idx], device=arc_scores.device),
                            reduction='sum'
                        )
                        num_valid_tokens += 1
        
        if num_valid_tokens > 0:
            arc_loss = arc_loss / num_valid_tokens
        else:
            arc_loss = torch.tensor(0.0, device=arc_scores.device)
        
        # Label loss - for each (head, dependent) pair, predict the correct label
        label_loss = 0
        num_valid_arcs = 0
        
        for b in range(batch_size):
            for i in range(seq_len):
                if mask[b, i]:  # Valid dependent
                    head_idx = arc_targets[b, i]
                    if 0 <= head_idx < seq_len:
                        # Cross-entropy loss for this arc's label
                        label_loss += F.cross_entropy(
                            label_scores[b, head_idx, i].unsqueeze(0),  # Label scores for this arc
                            torch.tensor([label_targets[b, i]], device=label_scores.device),
                            reduction='sum'
                        )
                        num_valid_arcs += 1
        
        if num_valid_arcs > 0:
            label_loss = label_loss / num_valid_arcs
        else:
            label_loss = torch.tensor(0.0, device=label_scores.device)
        
        # Total loss
        total_loss = arc_loss + label_loss
        
        return total_loss, arc_loss, label_loss


class ParserConfig:
    """Configuration class for the dependency parser."""
    
    def __init__(self):
        # Model architecture
        self.bert_model_name = "vinai/phobert-base"
        self.num_labels = 40  # Will be set based on data
        self.arc_mlp_size = 500
        self.label_mlp_size = 100
        self.dropout = 0.33
        self.freeze_bert = False
        
        # Training
        self.learning_rate = 2e-5
        self.bert_learning_rate = 1e-5  # Lower LR for BERT
        self.weight_decay = 0.01
        self.warmup_steps = 1000
        self.max_epochs = 50
        self.early_stopping_patience = 5
        
        # Data
        self.max_sequence_length = 512
        self.batch_size = 16
        self.accumulation_steps = 1
        
        # Evaluation
        self.eval_steps = 500
        self.save_steps = 1000
        
        # Paths
        self.train_path = "data/vi_vtb-ud-train.conllu"
        self.dev_path = "data/vi_vtb-ud-dev.conllu"
        self.test_path = "data/vi_vtb-ud-test.conllu"
        self.output_dir = "experiments/biaffine_parser"
        
    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        config = cls()
        for k, v in config_dict.items():
            setattr(config, k, v)
        return config


if __name__ == "__main__":
    # Test the complete model
    print("Testing Biaffine Dependency Parser...")
    
    # Create config
    config = ParserConfig()
    
    # Create model
    model = BiaffineDependencyParser(
        bert_model_name=config.bert_model_name,
        num_labels=config.num_labels,
        arc_mlp_size=config.arc_mlp_size,
        label_mlp_size=config.label_mlp_size,
        dropout=config.dropout
    )
    
    # Test input
    batch_size, seq_len, max_words = 2, 20, 10
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    valid_positions = torch.randint(0, seq_len, (batch_size, max_words))
    word_mask = torch.ones(batch_size, max_words).bool()
    
    # Forward pass
    with torch.no_grad():
        arc_scores, label_scores = model(input_ids, attention_mask, valid_positions, word_mask)
    
    print(f"Arc scores shape: {arc_scores.shape}")
    print(f"Label scores shape: {label_scores.shape}")
    
    # Test loss computation
    arc_targets = torch.randint(0, max_words, (batch_size, max_words))
    label_targets = torch.randint(0, config.num_labels, (batch_size, max_words))
    
    total_loss, arc_loss, label_loss = model.compute_loss(
        arc_scores, label_scores, arc_targets, label_targets, word_mask
    )
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Arc loss: {arc_loss.item():.4f}")  
    print(f"Label loss: {label_loss.item():.4f}")
    
    print("Biaffine dependency parser test passed!")