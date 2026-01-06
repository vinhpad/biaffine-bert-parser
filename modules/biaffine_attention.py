"""
Biaffine Attention Module
Implements the biaffine attention mechanism for dependency parsing.
Based on "Deep Biaffine Attention for Neural Dependency Parsing" by Dozat & Manning (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiaffineAttention(nn.Module):
    """
    Biaffine attention layer for computing arc scores and label scores.
    
    The biaffine function computes:
    score(head, dependent) = head^T W dependent + U head + V dependent + b
    
    Args:
        input_size (int): Size of input representations
        output_size (int): Output dimension (1 for arc scores, num_labels for label scores)
        bias_x (bool): Whether to use bias for head representation
        bias_y (bool): Whether to use bias for dependent representation
        dropout (float): Dropout rate
    """
    
    def __init__(self, input_size, output_size=1, bias_x=True, bias_y=True, dropout=0.1):
        super(BiaffineAttention, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.bias_x = bias_x
        self.bias_y = bias_y
        
        # Biaffine weight matrix: [output_size, input_size + bias_x, input_size + bias_y]
        self.weight = nn.Parameter(torch.Tensor(output_size, 
                                              input_size + int(bias_x), 
                                              input_size + int(bias_y)))
        
        # Initialize weights
        self.reset_parameters()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, head_repr, dep_repr):
        """
        Compute biaffine scores.
        
        Args:
            head_repr: [batch_size, seq_len, input_size] - head representations
            dep_repr: [batch_size, seq_len, input_size] - dependent representations
            
        Returns:
            scores: [batch_size, seq_len, seq_len, output_size] - biaffine scores
                   scores[b, i, j, k] = score from head i to dependent j for label k
        """
        batch_size, seq_len, _ = head_repr.shape
        
        # Apply dropout
        head_repr = self.dropout(head_repr)
        dep_repr = self.dropout(dep_repr)
        
        # Add bias terms if needed
        if self.bias_x:
            head_repr = torch.cat([head_repr, torch.ones_like(head_repr[..., :1])], dim=-1)
        if self.bias_y:
            dep_repr = torch.cat([dep_repr, torch.ones_like(dep_repr[..., :1])], dim=-1)
            
        # Compute biaffine scores
        # head_repr: [batch_size, seq_len, input_size + bias_x]
        # dep_repr: [batch_size, seq_len, input_size + bias_y]
        # weight: [output_size, input_size + bias_x, input_size + bias_y]
        
        # Method 1: Using einsum (more readable)
        scores = torch.einsum('bxi,oij,byj->bxyo', head_repr, self.weight, dep_repr)
        
        return scores


class BiaffineParser(nn.Module):
    """
    Complete biaffine parser with separate arc and label scoring.
    """
    
    def __init__(self, input_size, num_labels, arc_mlp_size=500, label_mlp_size=100, dropout=0.33):
        super(BiaffineParser, self).__init__()
        
        self.input_size = input_size
        self.num_labels = num_labels
        self.arc_mlp_size = arc_mlp_size
        self.label_mlp_size = label_mlp_size
        
        # MLP layers for arc prediction
        self.arc_head_mlp = nn.Sequential(
            nn.Linear(input_size, arc_mlp_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.arc_dep_mlp = nn.Sequential(
            nn.Linear(input_size, arc_mlp_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # MLP layers for label prediction  
        self.label_head_mlp = nn.Sequential(
            nn.Linear(input_size, label_mlp_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.label_dep_mlp = nn.Sequential(
            nn.Linear(input_size, label_mlp_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Biaffine attention layers
        self.arc_attention = BiaffineAttention(
            arc_mlp_size, 
            output_size=1, 
            bias_x=True, 
            bias_y=False,
            dropout=dropout
        )
        
        self.label_attention = BiaffineAttention(
            label_mlp_size,
            output_size=num_labels,
            bias_x=True,
            bias_y=True,
            dropout=dropout
        )
        
    def forward(self, encoded_input, mask=None):
        """
        Forward pass of biaffine parser.
        
        Args:
            encoded_input: [batch_size, seq_len, input_size] - encoded representations
            mask: [batch_size, seq_len] - attention mask (1 for valid tokens, 0 for padding)
            
        Returns:
            arc_scores: [batch_size, seq_len, seq_len] - arc scores
            label_scores: [batch_size, seq_len, seq_len, num_labels] - label scores
        """
        batch_size, seq_len, _ = encoded_input.shape
        
        # Generate arc representations
        arc_head_repr = self.arc_head_mlp(encoded_input)  # [batch, seq_len, arc_mlp_size]
        arc_dep_repr = self.arc_dep_mlp(encoded_input)    # [batch, seq_len, arc_mlp_size]
        
        # Generate label representations
        label_head_repr = self.label_head_mlp(encoded_input)  # [batch, seq_len, label_mlp_size]
        label_dep_repr = self.label_dep_mlp(encoded_input)    # [batch, seq_len, label_mlp_size]
        
        # Compute arc scores
        arc_scores = self.arc_attention(arc_head_repr, arc_dep_repr)  # [batch, seq_len, seq_len, 1]
        arc_scores = arc_scores.squeeze(-1)  # [batch, seq_len, seq_len]
        
        # Compute label scores
        label_scores = self.label_attention(label_head_repr, label_dep_repr)  # [batch, seq_len, seq_len, num_labels]
        
        # Apply mask if provided
        if mask is not None:
            # Mask invalid positions
            mask = mask.unsqueeze(1)  # [batch, 1, seq_len] for broadcasting
            arc_scores = arc_scores.masked_fill(~mask, -float('inf'))
            
            mask_label = mask.unsqueeze(-1)  # [batch, 1, seq_len, 1] for broadcasting
            label_scores = label_scores.masked_fill(~mask_label, -float('inf'))
        
        return arc_scores, label_scores


if __name__ == "__main__":
    # Test the biaffine attention
    batch_size, seq_len, hidden_size = 2, 10, 256
    num_labels = 15
    
    # Create test input
    encoded_input = torch.randn(batch_size, seq_len, hidden_size)
    mask = torch.ones(batch_size, seq_len).bool()
    
    # Create parser
    parser = BiaffineParser(hidden_size, num_labels)
    
    # Forward pass
    arc_scores, label_scores = parser(encoded_input, mask)
    
    print(f"Arc scores shape: {arc_scores.shape}")      # [2, 10, 10]
    print(f"Label scores shape: {label_scores.shape}")  # [2, 10, 10, 15]
    
    print("Biaffine attention module test passed!")