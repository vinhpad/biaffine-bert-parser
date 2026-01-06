"""
Biaffine Attention Module
Implements the biaffine attention mechanism for dependency parsing.
Based on "Deep Biaffine Attention for Neural Dependency Parsing" by Dozat & Manning (2017)
and supar implementation.
"""

import torch
import torch.nn as nn

from modules.mlp import MLP
from modules.dropout import SharedDropout


class Biaffine(nn.Module):
    """
    Biaffine layer for first-order scoring.
    
    This function has a tensor of weights W and bias terms if needed.
    The score s(x, y) of the vector pair (x, y) is computed as x^T W y,
    in which x and y can be concatenated with bias terms.
    
    Args:
        n_in (int): Size of input features.
        n_out (int): Number of output channels. Default: 1
        bias_x (bool): If True, adds a bias term for tensor x. Default: True
        bias_y (bool): If True, adds a bias term for tensor y. Default: True
        
    References:
        - Timothy Dozat and Christopher D. Manning. 2017.
          `Deep Biaffine Attention for Neural Dependency Parsing`_.
    """
    
    def __init__(self, n_in: int, n_out: int = 1, bias_x: bool = True, bias_y: bool = True):
        super(Biaffine, self).__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        
        # Weight matrix: [n_out, n_in + bias_x, n_in + bias_y]
        self.weight = nn.Parameter(torch.zeros(n_out, n_in + int(bias_x), n_in + int(bias_y)))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(n_in={self.n_in}, n_out={self.n_out}, bias_x={self.bias_x}, bias_y={self.bias_y})"
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute biaffine scores.
        
        Args:
            x: [batch_size, seq_len, n_in] - head representations
            y: [batch_size, seq_len, n_in] - dependent representations
            
        Returns:
            scores: [batch_size, seq_len, seq_len] if n_out=1,
                   [batch_size, n_out, seq_len, seq_len] otherwise
        """
        # Add bias terms if needed
        if self.bias_x:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        if self.bias_y:
            y = torch.cat([y, torch.ones_like(y[..., :1])], dim=-1)
            
        # Compute biaffine scores using einsum
        # x: [batch_size, seq_len, n_in + bias_x]
        # y: [batch_size, seq_len, n_in + bias_y]
        # weight: [n_out, n_in + bias_x, n_in + bias_y]
        # result: [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        
        # Remove dim 1 if n_out == 1
        s = s.squeeze(1)
        
        return s


class Triaffine(nn.Module):
    """
    Triaffine layer for second-order scoring (sibling scoring).
    
    Args:
        n_in (int): Size of input features.
        n_out (int): Number of output channels. Default: 1
        bias_x (bool): If True, adds a bias term for tensor x. Default: False
        bias_y (bool): If True, adds a bias term for tensor y. Default: False
        
    References:
        - Yu Zhang, Zhenghua Li and Min Zhang. 2020.
          `Efficient Second-Order TreeCRF for Neural Dependency Parsing`_.
    """
    
    def __init__(self, n_in: int, n_out: int = 1, bias_x: bool = False, bias_y: bool = False):
        super(Triaffine, self).__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        
        # Weight matrix
        self.weight = nn.Parameter(torch.zeros(n_out, n_in + int(bias_x), n_in, n_in + int(bias_y)))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.zeros_(self.weight)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(n_in={self.n_in}, n_out={self.n_out})"
        
    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute triaffine scores.
        
        Args:
            x: [batch_size, seq_len, n_in]
            y: [batch_size, seq_len, n_in]
            z: [batch_size, seq_len, n_in]
            
        Returns:
            scores: [batch_size, seq_len, seq_len, seq_len] if n_out=1
        """
        if self.bias_x:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        if self.bias_y:
            z = torch.cat([z, torch.ones_like(z[..., :1])], dim=-1)
            
        # w: [n_out, n_in + bias_x, n_in, n_in + bias_y]
        # Compute: x @ W @ y @ z
        w = torch.einsum('bxi,oijk->boxjk', x, self.weight)
        w = torch.einsum('boxjk,byj->boxyk', w, y)
        s = torch.einsum('boxyk,bzk->boxyz', w, z)
        
        s = s.squeeze(1)
        
        return s


class BiaffineParser(nn.Module):
    """
    Complete biaffine parser with separate arc and label scoring.
    Following supar architecture.
    
    Args:
        input_size (int): Size of input representations (e.g., BERT hidden size)
        num_labels (int): Number of dependency relation labels
        n_arc_mlp (int): Arc MLP hidden size. Default: 500
        n_rel_mlp (int): Label/relation MLP hidden size. Default: 100
        dropout (float): Dropout rate. Default: 0.33
    """
    
    def __init__(self, 
                 input_size: int, 
                 num_labels: int, 
                 n_arc_mlp: int = 500, 
                 n_rel_mlp: int = 100, 
                 dropout: float = 0.33):
        super(BiaffineParser, self).__init__()
        
        self.input_size = input_size
        self.num_labels = num_labels
        self.n_arc_mlp = n_arc_mlp
        self.n_rel_mlp = n_rel_mlp
        
        # MLP layers for arc prediction (head and dependent)
        self.mlp_arc_d = MLP(n_in=input_size, n_out=n_arc_mlp, dropout=dropout)
        self.mlp_arc_h = MLP(n_in=input_size, n_out=n_arc_mlp, dropout=dropout)
        
        # MLP layers for label/relation prediction
        self.mlp_rel_d = MLP(n_in=input_size, n_out=n_rel_mlp, dropout=dropout)
        self.mlp_rel_h = MLP(n_in=input_size, n_out=n_rel_mlp, dropout=dropout)
        
        # Biaffine attention layers
        self.arc_attn = Biaffine(n_in=n_arc_mlp, n_out=1, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=n_rel_mlp, n_out=num_labels, bias_x=True, bias_y=True)
        
        # Criterion for loss computation
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass of biaffine parser.
        
        Args:
            x: [batch_size, seq_len, input_size] - encoded representations
            mask: [batch_size, seq_len] - mask for valid tokens (True for valid)
            
        Returns:
            s_arc: [batch_size, seq_len, seq_len] - arc scores
                  s_arc[b, h, d] = score of arc from head h to dependent d
            s_rel: [batch_size, seq_len, seq_len, num_labels] - relation scores
        """
        # Apply MLPs to get arc/rel representations
        arc_d = self.mlp_arc_d(x)  # [batch, seq_len, n_arc_mlp]
        arc_h = self.mlp_arc_h(x)  # [batch, seq_len, n_arc_mlp]
        rel_d = self.mlp_rel_d(x)  # [batch, seq_len, n_rel_mlp]
        rel_h = self.mlp_rel_h(x)  # [batch, seq_len, n_rel_mlp]
        
        # Compute arc scores: [batch, seq_len, seq_len]
        # s_arc[b, h, d] = score of arc from head h to dependent d
        s_arc = self.arc_attn(arc_h, arc_d)
        
        # Compute relation scores: [batch, num_labels, seq_len, seq_len]
        s_rel = self.rel_attn(rel_h, rel_d)
        # Permute to [batch, seq_len, seq_len, num_labels]
        s_rel = s_rel.permute(0, 2, 3, 1)
        
        # Apply mask if provided
        if mask is not None:
            # Mask invalid heads (set score to -inf so they won't be selected)
            s_arc = s_arc.masked_fill(~mask.unsqueeze(2), float('-inf'))
        
        return s_arc, s_rel


# Keep old class name for backward compatibility
BiaffineAttention = Biaffine


if __name__ == "__main__":
    # Test the biaffine modules
    print("Testing Biaffine Module...")
    
    batch_size, seq_len, hidden_size = 2, 10, 768
    num_labels = 40
    
    # Create test input
    x = torch.randn(batch_size, seq_len, hidden_size)
    mask = torch.ones(batch_size, seq_len).bool()
    
    # Create parser
    parser = BiaffineParser(hidden_size, num_labels)
    
    # Forward pass
    s_arc, s_rel = parser(x, mask)
    
    print(f"Arc scores shape: {s_arc.shape}")      # [2, 10, 10]
    print(f"Rel scores shape: {s_rel.shape}")      # [2, 10, 10, 40]
    
    # Test Biaffine separately
    print("\nTesting Biaffine layer...")
    biaffine = Biaffine(n_in=500, n_out=1)
    h = torch.randn(batch_size, seq_len, 500)
    d = torch.randn(batch_size, seq_len, 500)
    scores = biaffine(h, d)
    print(f"Biaffine output shape (n_out=1): {scores.shape}")
    
    biaffine_rel = Biaffine(n_in=100, n_out=num_labels)
    h_rel = torch.randn(batch_size, seq_len, 100)
    d_rel = torch.randn(batch_size, seq_len, 100)
    scores_rel = biaffine_rel(h_rel, d_rel)
    print(f"Biaffine output shape (n_out={num_labels}): {scores_rel.shape}")
    
    print("\nBiaffine attention module test passed!")
