"""
MLP Module for Dependency Parsing
Implements Multi-Layer Perceptron used in biaffine parser.
Based on supar implementation.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with one hidden layer.
    
    This is the standard MLP used in biaffine dependency parsing
    to project BERT/LSTM outputs to arc/label representations.
    
    Args:
        n_in (int): Size of input features.
        n_out (int): Size of output features.
        dropout (float): Dropout rate. Default: 0.33
        activation (bool): Whether to apply LeakyReLU activation. Default: True
        
    References:
        - Timothy Dozat and Christopher D. Manning. 2017.
          `Deep Biaffine Attention for Neural Dependency Parsing`_.
    """
    
    def __init__(self, 
                 n_in: int, 
                 n_out: int, 
                 dropout: float = 0.33,
                 activation: bool = True):
        super(MLP, self).__init__()
        
        self.n_in = n_in
        self.n_out = n_out
        self.dropout = dropout
        self.activation = activation
        
        # Linear transformation
        self.linear = nn.Linear(n_in, n_out)
        
        # Activation function (LeakyReLU as in original paper)
        self.activate = nn.LeakyReLU(negative_slope=0.1) if activation else nn.Identity()
        
        # Dropout
        self.drop = nn.Dropout(dropout)
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using orthogonal initialization."""
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def __repr__(self):
        return f"{self.__class__.__name__}(n_in={self.n_in}, n_out={self.n_out}, dropout={self.dropout})"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [..., n_in]
            
        Returns:
            Output tensor of shape [..., n_out]
        """
        x = self.linear(x)
        x = self.activate(x)
        x = self.drop(x)
        return x


class MLPPair(nn.Module):
    """
    Paired MLP for generating head and dependent representations.
    
    This is a convenience module that creates two MLPs (one for head, one for dependent)
    with the same configuration.
    
    Args:
        n_in (int): Size of input features.
        n_out (int): Size of output features.
        dropout (float): Dropout rate. Default: 0.33
    """
    
    def __init__(self, n_in: int, n_out: int, dropout: float = 0.33):
        super(MLPPair, self).__init__()
        
        self.mlp_head = MLP(n_in, n_out, dropout)
        self.mlp_dep = MLP(n_in, n_out, dropout)
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_in]
            
        Returns:
            Tuple of (head_repr, dep_repr) each of shape [batch_size, seq_len, n_out]
        """
        head_repr = self.mlp_head(x)
        dep_repr = self.mlp_dep(x)
        return head_repr, dep_repr


if __name__ == "__main__":
    # Test MLP
    print("Testing MLP...")
    
    batch_size, seq_len, n_in, n_out = 2, 10, 768, 500
    x = torch.randn(batch_size, seq_len, n_in)
    
    mlp = MLP(n_in, n_out, dropout=0.33)
    mlp.train()
    
    output = mlp(x)
    print(f"MLP input shape: {x.shape}")
    print(f"MLP output shape: {output.shape}")
    print(f"MLP: {mlp}")
    
    # Test MLPPair
    print("\nTesting MLPPair...")
    
    mlp_pair = MLPPair(n_in, n_out, dropout=0.33)
    head_repr, dep_repr = mlp_pair(x)
    
    print(f"MLPPair head output shape: {head_repr.shape}")
    print(f"MLPPair dep output shape: {dep_repr.shape}")
    
    print("\nMLP modules test passed!")
