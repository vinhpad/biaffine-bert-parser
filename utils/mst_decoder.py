"""
Decoding Algorithms for Dependency Parsing
Implements MST (Chu-Liu/Edmonds) and Eisner algorithms.
Based on supar implementation.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


def mst(scores: torch.Tensor, mask: torch.Tensor, multiroot: bool = False) -> torch.Tensor:
    """
    MST algorithm for non-projective dependency parsing using Chu-Liu/Edmonds.
    
    Args:
        scores: [batch_size, seq_len, seq_len] - arc scores
                scores[b, h, d] = score of arc from head h to dependent d
        mask: [batch_size, seq_len] - mask for valid tokens
        multiroot: If True, allows multiple roots. Default: False
        
    Returns:
        heads: [batch_size, seq_len] - predicted head indices
    """
    batch_size, seq_len, _ = scores.shape
    scores = scores.cpu().detach().numpy()
    mask = mask.cpu().numpy()
    
    heads = np.zeros((batch_size, seq_len), dtype=np.int64)
    
    for b in range(batch_size):
        length = int(mask[b].sum())
        if length <= 1:
            continue
            
        # Extract valid scores
        score_matrix = scores[b, :length, :length]
        
        # Run Chu-Liu/Edmonds
        heads[b, :length] = _chu_liu_edmonds(score_matrix, multiroot)
    
    return torch.from_numpy(heads)


def _chu_liu_edmonds(scores: np.ndarray, multiroot: bool = False) -> np.ndarray:
    """
    Chu-Liu/Edmonds algorithm for finding maximum spanning arborescence.
    
    Args:
        scores: [seq_len, seq_len] - arc scores
        multiroot: If True, allows multiple roots
        
    Returns:
        heads: [seq_len] - head indices
    """
    length = scores.shape[0]
    heads = np.zeros(length, dtype=np.int64)
    
    if length <= 1:
        return heads
    
    # For each node (except root), find the best incoming arc
    for d in range(1, length):
        # Find best head for dependent d
        best_head = 0
        best_score = scores[0, d]
        
        for h in range(length):
            if h != d and scores[h, d] > best_score:
                best_score = scores[h, d]
                best_head = h
        
        heads[d] = best_head
    
    # Check for cycles and break them
    heads = _break_cycles(heads, scores, length)
    
    return heads


def _break_cycles(heads: np.ndarray, scores: np.ndarray, length: int) -> np.ndarray:
    """
    Detect and break cycles in the dependency tree.
    
    Args:
        heads: Current head assignments
        scores: Arc scores
        length: Sequence length
        
    Returns:
        heads: Head assignments with cycles broken
    """
    # Find cycles
    visited = np.zeros(length, dtype=bool)
    in_cycle = np.zeros(length, dtype=bool)
    
    for start in range(1, length):
        if visited[start]:
            continue
            
        path = []
        current = start
        
        while current != 0 and current not in path and not visited[current]:
            path.append(current)
            current = heads[current]
        
        if current in path:
            # Found a cycle
            cycle_start = path.index(current)
            cycle = path[cycle_start:]
            
            # Break cycle by connecting one node to root
            # Choose the node with highest score from root
            best_node = cycle[0]
            best_score = scores[0, cycle[0]]
            
            for node in cycle[1:]:
                if scores[0, node] > best_score:
                    best_score = scores[0, node]
                    best_node = node
            
            heads[best_node] = 0
            
        for node in path:
            visited[node] = True
    
    return heads


def eisner(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Eisner algorithm for projective dependency parsing.
    
    Args:
        scores: [batch_size, seq_len, seq_len] - arc scores
        mask: [batch_size, seq_len] - mask for valid tokens
        
    Returns:
        heads: [batch_size, seq_len] - predicted head indices
    """
    batch_size, seq_len, _ = scores.shape
    scores = scores.cpu().detach().numpy()
    mask = mask.cpu().numpy()
    
    heads = np.zeros((batch_size, seq_len), dtype=np.int64)
    
    for b in range(batch_size):
        length = int(mask[b].sum())
        if length <= 1:
            continue
            
        score_matrix = scores[b, :length, :length]
        heads[b, :length] = _eisner_decode(score_matrix)
    
    return torch.from_numpy(heads)


def _eisner_decode(scores: np.ndarray) -> np.ndarray:
    """
    Eisner's algorithm for projective dependency parsing.
    
    Args:
        scores: [seq_len, seq_len] - arc scores
        
    Returns:
        heads: [seq_len] - head indices
    """
    length = scores.shape[0]
    heads = np.zeros(length, dtype=np.int64)
    
    if length <= 1:
        return heads
    
    # Initialize DP tables
    # complete[s][t][d] = best score for complete span [s,t] with direction d
    # incomplete[s][t][d] = best score for incomplete span [s,t] with direction d
    # d=0: right (head on left), d=1: left (head on right)
    
    complete = np.full((length, length, 2), -np.inf)
    incomplete = np.full((length, length, 2), -np.inf)
    complete_back = np.zeros((length, length, 2), dtype=np.int64)
    incomplete_back = np.zeros((length, length, 2), dtype=np.int64)
    
    # Base case: single word spans
    for i in range(length):
        complete[i, i, 0] = 0.0
        complete[i, i, 1] = 0.0
    
    # Fill DP tables
    for width in range(1, length):
        for s in range(length - width):
            t = s + width
            
            # Incomplete spans
            # Right: arc from s to t (s is head)
            for r in range(s, t):
                score = complete[s, r, 0] + complete[r + 1, t, 1] + scores[s, t]
                if score > incomplete[s, t, 0]:
                    incomplete[s, t, 0] = score
                    incomplete_back[s, t, 0] = r
            
            # Left: arc from t to s (t is head)
            for r in range(s, t):
                score = complete[s, r, 0] + complete[r + 1, t, 1] + scores[t, s]
                if score > incomplete[s, t, 1]:
                    incomplete[s, t, 1] = score
                    incomplete_back[s, t, 1] = r
            
            # Complete spans
            # Right: complete span with head on left
            for r in range(s, t):
                score = complete[s, r, 0] + incomplete[r, t, 0]
                if score > complete[s, t, 0]:
                    complete[s, t, 0] = score
                    complete_back[s, t, 0] = r
            
            # Left: complete span with head on right
            for r in range(s + 1, t + 1):
                score = incomplete[s, r, 1] + complete[r, t, 1]
                if score > complete[s, t, 1]:
                    complete[s, t, 1] = score
                    complete_back[s, t, 1] = r
    
    # Backtrack to find heads
    _backtrack(heads, incomplete_back, complete_back, 0, length - 1, 0, True)
    
    return heads


def _backtrack(heads, incomplete_back, complete_back, s, t, direction, complete):
    """
    Backtrack through DP tables to recover heads.
    """
    if s == t:
        return
    
    if complete:
        r = complete_back[s, t, direction]
        if direction == 0:  # Right
            _backtrack(heads, incomplete_back, complete_back, s, r, 0, True)
            _backtrack(heads, incomplete_back, complete_back, r, t, 0, False)
        else:  # Left
            _backtrack(heads, incomplete_back, complete_back, s, r, 1, False)
            _backtrack(heads, incomplete_back, complete_back, r, t, 1, True)
    else:
        r = incomplete_back[s, t, direction]
        if direction == 0:  # Right: s -> t
            heads[t] = s
            _backtrack(heads, incomplete_back, complete_back, s, r, 0, True)
            _backtrack(heads, incomplete_back, complete_back, r + 1, t, 1, True)
        else:  # Left: t -> s
            heads[s] = t
            _backtrack(heads, incomplete_back, complete_back, s, r, 0, True)
            _backtrack(heads, incomplete_back, complete_back, r + 1, t, 1, True)


class MSTDecoder:
    """
    MST decoder for non-projective dependency parsing.
    """
    
    def __init__(self, projective: bool = False):
        self.projective = projective
    
    def decode(self, arc_scores: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode dependency trees from arc scores.
        
        Args:
            arc_scores: [batch_size, seq_len, seq_len] - arc scores
            mask: [batch_size, seq_len] - mask for valid tokens
            
        Returns:
            heads: [batch_size, seq_len] - predicted head indices
        """
        if mask is None:
            batch_size, seq_len, _ = arc_scores.shape
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=arc_scores.device)
        
        if self.projective:
            return eisner(arc_scores, mask)
        else:
            return mst(arc_scores, mask)


class GreedyDecoder:
    """
    Simple greedy decoder that selects the highest scoring head for each dependent.
    """
    
    def __init__(self):
        pass
    
    def decode(self, arc_scores: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Greedy decoding: for each token, select the highest scoring head.
        
        Args:
            arc_scores: [batch_size, seq_len, seq_len] - arc scores
            mask: [batch_size, seq_len] - mask for valid tokens
            
        Returns:
            heads: [batch_size, seq_len] - predicted head indices
        """
        # arc_scores[b, h, d] = score of arc from h to d
        # For each dependent d, find the best head h
        heads = arc_scores.argmax(dim=1)  # [batch_size, seq_len]
        
        # Root should have head 0
        heads[:, 0] = 0
        
        return heads


def evaluate_parsing_accuracy(predicted_heads: torch.Tensor, 
                             gold_heads: torch.Tensor, 
                             predicted_rels: Optional[torch.Tensor] = None,
                             gold_rels: Optional[torch.Tensor] = None,
                             mask: Optional[torch.Tensor] = None) -> dict:
    """
    Evaluate parsing accuracy (UAS and LAS).
    
    Args:
        predicted_heads: [batch_size, seq_len] - predicted head indices
        gold_heads: [batch_size, seq_len] - gold head indices
        predicted_rels: [batch_size, seq_len] - predicted relation labels (optional)
        gold_rels: [batch_size, seq_len] - gold relation labels (optional)
        mask: [batch_size, seq_len] - mask for valid tokens
        
    Returns:
        dict with 'uas', 'las', 'uas_correct', 'las_correct', 'total'
    """
    if mask is None:
        mask = torch.ones_like(gold_heads, dtype=torch.bool)
    
    # Ignore first token (root/CLS)
    mask = mask.clone()
    mask[:, 0] = False
    
    # UAS
    correct_heads = (predicted_heads == gold_heads) & mask
    uas_correct = correct_heads.sum().item()
    total = mask.sum().item()
    
    # LAS
    if predicted_rels is not None and gold_rels is not None:
        correct_rels = (predicted_rels == gold_rels) & mask
        correct_both = correct_heads & correct_rels
        las_correct = correct_both.sum().item()
    else:
        las_correct = 0
    
    uas = uas_correct / total if total > 0 else 0.0
    las = las_correct / total if total > 0 else 0.0
    
    return {
        'uas': uas,
        'las': las,
        'uas_correct': uas_correct,
        'las_correct': las_correct,
        'total': total
    }


if __name__ == "__main__":
    # Test MST decoder
    print("Testing Decoding Algorithms...")
    
    # Create test data
    batch_size, seq_len = 2, 5
    
    # Create arc scores
    arc_scores = torch.randn(batch_size, seq_len, seq_len)
    arc_scores[:, 0, 1:] += 2.0  # Root has high scores
    
    # Create mask
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Test MST decoder
    mst_decoder = MSTDecoder(projective=False)
    mst_heads = mst_decoder.decode(arc_scores, mask)
    print(f"MST heads:\n{mst_heads}")
    
    # Test Eisner decoder
    eisner_decoder = MSTDecoder(projective=True)
    eisner_heads = eisner_decoder.decode(arc_scores, mask)
    print(f"Eisner heads:\n{eisner_heads}")
    
    # Test greedy decoder
    greedy_decoder = GreedyDecoder()
    greedy_heads = greedy_decoder.decode(arc_scores, mask)
    print(f"Greedy heads:\n{greedy_heads}")
    
    # Test evaluation
    gold_heads = torch.tensor([[0, 0, 1, 2, 3], [0, 0, 1, 0, 2]])
    metrics = evaluate_parsing_accuracy(mst_heads, gold_heads, mask=mask)
    print(f"\nMST UAS: {metrics['uas']:.4f} ({metrics['uas_correct']}/{metrics['total']})")
    
    print("\nDecoding algorithms test passed!")
