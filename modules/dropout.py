"""
Dropout Modules for Dependency Parsing
Implements SharedDropout and IndependentDropout from supar.
Based on "Deep Biaffine Attention for Neural Dependency Parsing" by Dozat & Manning (2017)
"""

import torch
import torch.nn as nn


class SharedDropout(nn.Module):
    """
    SharedDropout differs from the vanilla dropout strategy in that the weights 
    are shared across one dimension (typically the time dimension).
    
    This means that the same positions are dropped for all time steps.
    
    Args:
        p (float): Probability of an element to be zeroed. Default: 0.5
        batch_first (bool): If True, the input tensor is of shape [batch_size, seq_len, hidden_size].
                           Default: True
    
    References:
        - Timothy Dozat and Christopher D. Manning. 2017.
          `Deep Biaffine Attention for Neural Dependency Parsing`_.
    """

    def __init__(self, p: float = 0.5, batch_first: bool = True):
        super(SharedDropout, self).__init__()
        
        self.p = p
        self.batch_first = batch_first

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply shared dropout.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size] if batch_first,
               or [seq_len, batch_size, hidden_size] otherwise.
               
        Returns:
            Tensor with dropout applied, same shape as input.
        """
        if not self.training or self.p == 0:
            return x
            
        if self.batch_first:
            # x: [batch_size, seq_len, hidden_size]
            # Create mask of shape [batch_size, 1, hidden_size] 
            # which will be broadcast across seq_len
            mask = self._get_mask(x[:, 0], self.p)
            mask = mask.unsqueeze(1)  # [batch_size, 1, hidden_size]
        else:
            # x: [seq_len, batch_size, hidden_size]
            mask = self._get_mask(x[0], self.p)
            mask = mask.unsqueeze(0)  # [1, batch_size, hidden_size]
            
        return x * mask
    
    @staticmethod
    def _get_mask(x: torch.Tensor, p: float) -> torch.Tensor:
        """Generate a dropout mask."""
        return x.new_empty(x.shape).bernoulli_(1 - p) / (1 - p)


class IndependentDropout(nn.Module):
    """
    IndependentDropout applies dropout to each input independently.
    
    For each input tensor, a separate dropout mask is sampled and applied,
    which encourages the model to exploit both representations equally.
    
    Args:
        p (float): Probability of an element to be zeroed. Default: 0.5
        
    References:
        - Timothy Dozat and Christopher D. Manning. 2017.
          `Deep Biaffine Attention for Neural Dependency Parsing`_.
    """

    def __init__(self, p: float = 0.5):
        super(IndependentDropout, self).__init__()
        
        self.p = p

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

    def forward(self, *items: torch.Tensor) -> tuple:
        """
        Apply independent dropout to each input.
        
        Args:
            *items: Variable number of input tensors.
                   All tensors should have the same shape.
                   
        Returns:
            Tuple of tensors with dropout applied independently to each.
        """
        if not self.training or self.p == 0:
            return items
            
        # Generate independent masks for each input
        masks = [x.new_ones(x.shape[:2]).bernoulli_(1 - self.p) for x in items]
        
        # Normalize masks so they sum to 1 at each position
        total = sum(masks)
        # Avoid division by zero
        total = total + (total == 0).float()
        masks = [mask / total for mask in masks]
        
        # Apply masks
        return tuple(x * mask.unsqueeze(-1) for x, mask in zip(items, masks))


if __name__ == "__main__":
    # Test SharedDropout
    print("Testing SharedDropout...")
    
    batch_size, seq_len, hidden_size = 2, 5, 10
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    shared_dropout = SharedDropout(p=0.5)
    shared_dropout.train()
    
    output = shared_dropout(x)
    print(f"SharedDropout input shape: {x.shape}")
    print(f"SharedDropout output shape: {output.shape}")
    
    # Test IndependentDropout
    print("\nTesting IndependentDropout...")
    
    x1 = torch.randn(batch_size, seq_len, hidden_size)
    x2 = torch.randn(batch_size, seq_len, hidden_size)
    
    independent_dropout = IndependentDropout(p=0.5)
    independent_dropout.train()
    
    out1, out2 = independent_dropout(x1, x2)
    print(f"IndependentDropout input shapes: {x1.shape}, {x2.shape}")
    print(f"IndependentDropout output shapes: {out1.shape}, {out2.shape}")
    
    print("\nDropout modules test passed!")
