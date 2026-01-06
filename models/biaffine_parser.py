"""
Biaffine Dependency Parser Model following supar architecture.

Reference: https://github.com/yzhangcs/parser
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List

from models.bert_encoder import BertEncoder, extract_word_representations
from modules.dropout import SharedDropout, IndependentDropout
from modules.mlp import MLP
from modules.biaffine_attention import Biaffine


class BiaffineDependencyParser(nn.Module):
    """
    Complete biaffine dependency parser with BERT backbone.
    Based on supar's BiaffineDependencyModel architecture.
    
    Args:
        bert_model_name: Name of pretrained BERT model
        num_labels: Number of dependency relation labels
        n_arc_mlp: Hidden size for arc MLP
        n_rel_mlp: Hidden size for relation MLP
        mlp_dropout: Dropout rate for MLP
        freeze_bert: Whether to freeze BERT parameters
        use_lexical_dropout: Whether to use independent dropout on embeddings
        scale: Scaling factor for biaffine scores
    """
    
    def __init__(self,
                 bert_model_name: str = "vinai/phobert-base-v2",
                 num_labels: int = 40,
                 n_arc_mlp: int = 500,
                 n_rel_mlp: int = 100,
                 mlp_dropout: float = 0.33,
                 freeze_bert: bool = False,
                 use_lexical_dropout: bool = True,
                 scale: float = 0.0):
        
        super(BiaffineDependencyParser, self).__init__()
        
        self.num_labels = num_labels
        self.n_arc_mlp = n_arc_mlp
        self.n_rel_mlp = n_rel_mlp
        self.scale = scale
        
        # BERT encoder
        self.bert_encoder = BertEncoder(
            model_name=bert_model_name,
            dropout=mlp_dropout,
            freeze_bert=freeze_bert
        )
        hidden_size = self.bert_encoder.hidden_size
        
        # Lexical dropout (supar style)
        if use_lexical_dropout:
            self.embed_dropout = IndependentDropout(p=mlp_dropout)
        else:
            self.embed_dropout = nn.Dropout(p=mlp_dropout)
        self.use_lexical_dropout = use_lexical_dropout
        
        # MLP layers for arc scoring
        # arc-dep: MLP to transform dependent representation
        # arc-head: MLP to transform head representation
        self.mlp_arc_d = MLP(n_in=hidden_size, n_out=n_arc_mlp, dropout=mlp_dropout)
        self.mlp_arc_h = MLP(n_in=hidden_size, n_out=n_arc_mlp, dropout=mlp_dropout)
        
        # MLP layers for relation scoring
        self.mlp_rel_d = MLP(n_in=hidden_size, n_out=n_rel_mlp, dropout=mlp_dropout)
        self.mlp_rel_h = MLP(n_in=hidden_size, n_out=n_rel_mlp, dropout=mlp_dropout)
        
        # Biaffine attention layers
        self.arc_attn = Biaffine(n_in=n_arc_mlp, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=n_rel_mlp, n_out=num_labels, bias_x=True, bias_y=True)
        
        # Criterion for computing loss
        self.criterion = nn.CrossEntropyLoss()
        
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
            s_arc: [batch_size, max_words, max_words] - arc attachment scores
            s_rel: [batch_size, max_words, max_words, num_labels] - relation scores
        """
        # Encode with BERT
        encoded_output = self.bert_encoder(input_ids, attention_mask)
        
        # Extract word-level representations
        x = extract_word_representations(encoded_output, valid_positions)
        
        # Apply embedding dropout if using independent dropout
        if self.use_lexical_dropout:
            x = self.embed_dropout(x)[0]
        else:
            x = self.embed_dropout(x)
        
        # Generate arc representations
        arc_d = self.mlp_arc_d(x)  # [batch, seq_len, n_arc_mlp]
        arc_h = self.mlp_arc_h(x)  # [batch, seq_len, n_arc_mlp]
        
        # Generate rel representations  
        rel_d = self.mlp_rel_d(x)  # [batch, seq_len, n_rel_mlp]
        rel_h = self.mlp_rel_h(x)  # [batch, seq_len, n_rel_mlp]
        
        # Compute scores using biaffine attention
        # s_arc[b, i, j] = score that word i depends on word j
        s_arc = self.arc_attn(arc_d, arc_h)  # [batch, seq_len, seq_len]
        
        # s_rel[b, i, j, r] = score that arc (j->i) has relation r
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)  # [batch, seq_len, seq_len, num_labels]
        
        return s_arc, s_rel
    
    def loss(self,
             s_arc: torch.Tensor,
             s_rel: torch.Tensor,
             arcs: torch.Tensor,
             rels: torch.Tensor,
             mask: torch.Tensor,
             partial: bool = False) -> torch.Tensor:
        """
        Compute the training loss.
        
        Args:
            s_arc: [batch_size, seq_len, seq_len] - arc scores
            s_rel: [batch_size, seq_len, seq_len, num_labels] - relation scores
            arcs: [batch_size, seq_len] - gold head indices
            rels: [batch_size, seq_len] - gold relation indices
            mask: [batch_size, seq_len] - mask for valid tokens
            partial: Whether using partial annotation
            
        Returns:
            loss: Combined arc and relation loss
        """
        # Don't predict for root position
        arc_mask = mask.clone()
        arc_mask[:, 0] = 0
        
        if partial:
            # For partially annotated data, only consider non-negative labels
            arc_mask = arc_mask & arcs.ge(0)
        
        # Flatten for loss computation
        s_arc = s_arc[arc_mask]  # [n_valid, seq_len]
        s_rel = s_rel[arc_mask]  # [n_valid, seq_len, num_labels]
        arcs = arcs[arc_mask]    # [n_valid]
        rels = rels[arc_mask]    # [n_valid]
        
        # Get relation scores for gold arcs
        s_rel = s_rel[torch.arange(len(arcs), device=arcs.device), arcs]  # [n_valid, num_labels]
        
        # Compute cross entropy loss
        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)
        
        return arc_loss + rel_loss
    
    def decode(self,
               s_arc: torch.Tensor,
               s_rel: torch.Tensor,
               mask: torch.Tensor,
               tree: bool = False,
               proj: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode predictions from scores.
        
        Args:
            s_arc: [batch_size, seq_len, seq_len] - arc scores
            s_rel: [batch_size, seq_len, seq_len, num_labels] - relation scores
            mask: [batch_size, seq_len] - mask for valid tokens
            tree: Whether to ensure tree structure via MST decoding
            proj: Whether to use projective parsing (Eisner)
            
        Returns:
            arc_preds: [batch_size, seq_len] - predicted heads
            rel_preds: [batch_size, seq_len] - predicted relations
        """
        batch_size, seq_len = mask.shape
        
        # Mask diagonal (can't depend on self)
        s_arc = s_arc.clone()
        s_arc.diagonal(0, 1, 2).fill_(float('-inf'))
        
        if tree:
            from utils.mst_decoder import mst, eisner
            
            # Create lens tensor - number of valid tokens per sentence
            lens = mask.sum(-1)
            
            if proj:
                arc_preds = eisner(s_arc, mask)
            else:
                arc_preds = mst(s_arc, mask)
        else:
            # Greedy decoding
            arc_preds = s_arc.argmax(-1)
        
        # Mask invalid positions
        arc_preds = arc_preds.masked_fill(~mask, -1)
        
        # Get relation predictions for predicted arcs
        # s_rel[b, i, arc_preds[b, i]] gives relation scores for predicted arcs
        batch_idx = torch.arange(batch_size, device=s_rel.device).unsqueeze(1).expand(-1, seq_len)
        token_idx = torch.arange(seq_len, device=s_rel.device).unsqueeze(0).expand(batch_size, -1)
        
        # Clamp arc_preds to valid range for indexing
        arc_preds_clamped = arc_preds.clamp(min=0)
        
        rel_scores = s_rel[batch_idx, token_idx, arc_preds_clamped]  # [batch, seq_len, num_labels]
        rel_preds = rel_scores.argmax(-1)  # [batch, seq_len]
        rel_preds = rel_preds.masked_fill(~mask, -1)
        
        return arc_preds, rel_preds
    
    # Keep old interface for compatibility
    def compute_loss(self,
                     arc_scores: torch.Tensor,
                     label_scores: torch.Tensor,
                     arc_targets: torch.Tensor,
                     label_targets: torch.Tensor,
                     mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Legacy interface for computing loss. 
        Wrapper around the new loss() method for backward compatibility.
        """
        # Compute combined loss using new method
        total_loss = self.loss(arc_scores, label_scores, arc_targets, label_targets, mask)
        
        # For backward compatibility, also compute individual losses
        arc_mask = mask.clone()
        arc_mask[:, 0] = 0
        
        s_arc = arc_scores[arc_mask]
        s_rel = label_scores[arc_mask]
        arcs = arc_targets[arc_mask]
        rels = label_targets[arc_mask]
        
        s_rel = s_rel[torch.arange(len(arcs), device=arcs.device), arcs]
        
        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)
        
        return total_loss, arc_loss, rel_loss


class ParserConfig:
    """Configuration class for the dependency parser following supar style."""
    
    def __init__(self):
        # Model architecture
        self.bert_model_name = "vinai/phobert-base-v2"
        self.num_labels = 40  # Will be set based on data
        self.n_arc_mlp = 500
        self.n_rel_mlp = 100
        self.mlp_dropout = 0.33
        self.freeze_bert = False
        self.use_lexical_dropout = True
        self.scale = 0.0
        
        # Training
        self.learning_rate = 2e-5
        self.bert_learning_rate = 1e-5  # Lower LR for BERT
        self.weight_decay = 0.01
        self.warmup_steps = 1000
        self.max_epochs = 50
        self.early_stopping_patience = 5
        
        # Decoding
        self.tree = True  # Use tree-structured decoding
        self.proj = False  # Use non-projective (MST) decoding
        
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
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load config from YAML file."""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


if __name__ == "__main__":
    # Test the complete model
    print("Testing Biaffine Dependency Parser...")
    
    # Create config
    config = ParserConfig()
    
    # Create model
    model = BiaffineDependencyParser(
        bert_model_name=config.bert_model_name,
        num_labels=config.num_labels,
        n_arc_mlp=config.n_arc_mlp,
        n_rel_mlp=config.n_rel_mlp,
        mlp_dropout=config.mlp_dropout
    )
    
    # Print model info
    print(f"Model architecture:")
    print(f"  - BERT: {config.bert_model_name}")
    print(f"  - Arc MLP size: {config.n_arc_mlp}")
    print(f"  - Rel MLP size: {config.n_rel_mlp}")
    print(f"  - Num labels: {config.num_labels}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")
    
    print("\nBiaffine dependency parser test passed!")
