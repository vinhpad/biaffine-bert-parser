"""
Maximum Spanning Tree (MST) Decoder for Dependency Parsing
Implements Chu-Liu/Edmonds' algorithm for finding the optimal dependency tree.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from collections import defaultdict, deque


class MSTDecoder:
    """
    MST decoder using Chu-Liu/Edmonds' algorithm.
    
    This implementation finds the maximum spanning tree for dependency parsing,
    ensuring that the result is a valid dependency tree (single root, no cycles).
    """
    
    def __init__(self):
        pass
    
    def decode(self, arc_scores: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        Decode dependency trees from arc scores using MST algorithm.
        
        Args:
            arc_scores: [batch_size, seq_len, seq_len] - arc scores
                       arc_scores[b, i, j] = score of arc from head i to dependent j
            mask: [batch_size, seq_len] - mask for valid tokens
            
        Returns:
            List of dependency trees, where each tree is a list of head indices
        """
        batch_size, seq_len, _ = arc_scores.shape
        
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        trees = []
        
        for b in range(batch_size):
            # Get valid length for this sentence
            valid_length = mask[b].sum().item()
            
            if valid_length <= 1:
                trees.append([0])  # Single token points to root
                continue
            
            # Extract scores for valid tokens
            scores = arc_scores[b, :valid_length, :valid_length].detach().cpu().numpy()
            
            # Add artificial root with score 0 to all tokens except position 0
            # Position 0 will be the artificial root
            enhanced_scores = np.zeros((valid_length + 1, valid_length + 1))
            enhanced_scores[1:, 1:] = scores
            enhanced_scores[0, 1:] = 0  # Artificial root can point to any token
            enhanced_scores[:, 0] = -np.inf  # No token can point to artificial root
            enhanced_scores[0, 0] = 0  # Root points to itself
            
            # Run MST algorithm
            tree = self._chu_liu_edmonds(enhanced_scores)
            
            # Convert back to original indexing (remove artificial root)
            final_tree = []
            for i in range(1, len(tree)):
                head = tree[i]
                if head == 0:  # Points to artificial root
                    final_tree.append(0)  # Convert to ROOT (index 0)
                else:
                    final_tree.append(head - 1)  # Adjust for removed artificial root
            
            # Pad with zeros if necessary
            while len(final_tree) < seq_len:
                final_tree.append(0)
            
            trees.append(final_tree)
        
        return trees
    
    def _chu_liu_edmonds(self, scores: np.ndarray) -> List[int]:
        """
        Chu-Liu/Edmonds' algorithm for maximum spanning tree.
        
        Args:
            scores: [n+1, n+1] - arc scores (including artificial root at index 0)
            
        Returns:
            tree: List of head indices for each node
        """
        n = scores.shape[0]
        
        # Step 1: For each node, find the maximum incoming arc
        max_incoming = np.zeros(n, dtype=int)
        max_scores = np.full(n, -np.inf)
        
        for j in range(n):  # For each dependent
            for i in range(n):  # For each potential head
                if i != j and scores[i, j] > max_scores[j]:
                    max_scores[j] = scores[i, j]
                    max_incoming[j] = i
        
        # Step 2: Check if this forms a valid tree (no cycles)
        visited = np.zeros(n, dtype=bool)
        in_cycle = np.zeros(n, dtype=bool)
        cycle_id = np.full(n, -1, dtype=int)
        
        cycles = []
        
        for v in range(n):
            if visited[v]:
                continue
                
            # Follow the path from v
            path = []
            current = v
            
            while not visited[current]:
                visited[current] = True
                path.append(current)
                current = max_incoming[current]
                
                # If we've seen this node in the current path, we found a cycle
                if current in path:
                    cycle_start_idx = path.index(current)
                    cycle = path[cycle_start_idx:]
                    cycles.append(cycle)
                    
                    # Mark cycle nodes
                    for node in cycle:
                        in_cycle[node] = True
                        cycle_id[node] = len(cycles) - 1
                    break
        
        # Step 3: If no cycles, return the tree
        if not cycles:
            return max_incoming.tolist()
        
        # Step 4: Contract cycles and recursively solve
        # Create contracted graph
        num_cycles = len(cycles)
        cycle_nodes = set()
        for cycle in cycles:
            cycle_nodes.update(cycle)
        
        # Map old nodes to new nodes
        old_to_new = {}
        new_to_old = {}
        new_node_id = 0
        
        # Non-cycle nodes keep their relative order
        for i in range(n):
            if not in_cycle[i]:
                old_to_new[i] = new_node_id
                new_to_old[new_node_id] = i
                new_node_id += 1
        
        # Each cycle becomes one node
        cycle_to_new = {}
        for i, cycle in enumerate(cycles):
            cycle_to_new[i] = new_node_id
            new_to_old[new_node_id] = cycle[0]  # Representative node
            for node in cycle:
                old_to_new[node] = new_node_id
            new_node_id += 1
        
        # Create contracted scores matrix
        contracted_n = new_node_id
        contracted_scores = np.full((contracted_n, contracted_n), -np.inf)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                new_i = old_to_new[i]
                new_j = old_to_new[j]
                
                if new_i == new_j:  # Both in same cycle
                    continue
                
                # Adjust score if j is in a cycle
                adjusted_score = scores[i, j]
                if in_cycle[j]:
                    # Subtract the score of the edge that j would lose in the cycle
                    adjusted_score -= max_scores[j]
                
                if adjusted_score > contracted_scores[new_i, new_j]:
                    contracted_scores[new_i, new_j] = adjusted_score
        
        # Recursively solve on contracted graph
        contracted_tree = self._chu_liu_edmonds(contracted_scores)
        
        # Step 5: Expand the solution
        expanded_tree = [0] * n
        
        # First, set the tree edges within cycles
        for cycle in cycles:
            for node in cycle:
                expanded_tree[node] = max_incoming[node]
        
        # Then, set the tree edges between contracted nodes
        for new_node in range(contracted_n):
            if new_node == 0:  # Skip root
                continue
                
            new_head = contracted_tree[new_node]
            
            if new_head == new_node:  # Self-loop (shouldn't happen)
                continue
            
            # Find the best edge from new_head to new_node
            best_score = -np.inf
            best_head = -1
            best_dep = -1
            
            # Find nodes in the head component
            if new_head in new_to_old:
                head_nodes = [new_to_old[new_head]]
            else:
                # Find cycle containing this new_head
                head_nodes = []
                for cycle_idx, cycle in enumerate(cycles):
                    if cycle_to_new[cycle_idx] == new_head:
                        head_nodes = cycle
                        break
            
            # Find nodes in the dependent component  
            if new_node in new_to_old:
                dep_nodes = [new_to_old[new_node]]
            else:
                dep_nodes = []
                for cycle_idx, cycle in enumerate(cycles):
                    if cycle_to_new[cycle_idx] == new_node:
                        dep_nodes = cycle
                        break
            
            # Find best edge between components
            for h in head_nodes:
                for d in dep_nodes:
                    if scores[h, d] > best_score:
                        best_score = scores[h, d]
                        best_head = h
                        best_dep = d
            
            # If new_node represents a cycle, break the cycle at best_dep
            if len(dep_nodes) > 1:  # It's a cycle
                expanded_tree[best_dep] = best_head
        
        return expanded_tree


class GreedyDecoder:
    """
    Simple greedy decoder that selects the highest scoring head for each dependent.
    This is much faster but may produce invalid trees with cycles.
    """
    
    def __init__(self):
        pass
    
    def decode(self, arc_scores: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        Greedy decoding: for each token, select the highest scoring head.
        
        Args:
            arc_scores: [batch_size, seq_len, seq_len] - arc scores
            mask: [batch_size, seq_len] - mask for valid tokens
            
        Returns:
            List of dependency trees (may contain cycles)
        """
        batch_size, seq_len, _ = arc_scores.shape
        
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        trees = []
        
        for b in range(batch_size):
            tree = []
            
            for i in range(seq_len):
                if mask[b, i]:
                    # Find the head with maximum score
                    head = arc_scores[b, :, i].argmax().item()
                    tree.append(head)
                else:
                    tree.append(0)  # Invalid tokens point to root
            
            trees.append(tree)
        
        return trees


def evaluate_parsing_accuracy(predicted_heads: List[List[int]], 
                            gold_heads: List[List[int]], 
                            mask: Optional[torch.Tensor] = None) -> Tuple[float, int, int]:
    """
    Evaluate parsing accuracy (UAS - Unlabeled Attachment Score).
    
    Args:
        predicted_heads: List of predicted head sequences
        gold_heads: List of gold head sequences  
        mask: Optional mask for valid tokens
        
    Returns:
        accuracy: Unlabeled attachment score
        correct: Number of correct attachments
        total: Total number of valid attachments
    """
    correct = 0
    total = 0
    
    for i, (pred, gold) in enumerate(zip(predicted_heads, gold_heads)):
        seq_len = len(pred)
        
        for j in range(seq_len):
            if mask is None or mask[i, j]:
                if pred[j] == gold[j]:
                    correct += 1
                total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


if __name__ == "__main__":
    # Test MST decoder
    print("Testing MST Decoder...")
    
    # Create test data
    batch_size, seq_len = 2, 5
    
    # Create arc scores (higher scores for valid arcs)
    arc_scores = torch.randn(batch_size, seq_len, seq_len)
    
    # Make root (index 0) have high scores to other tokens
    arc_scores[:, 0, 1:] += 2.0
    
    # Create mask
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Test MST decoder
    mst_decoder = MSTDecoder()
    mst_trees = mst_decoder.decode(arc_scores, mask)
    
    print("MST Trees:")
    for i, tree in enumerate(mst_trees):
        print(f"Sentence {i}: {tree}")
    
    # Test greedy decoder
    greedy_decoder = GreedyDecoder()
    greedy_trees = greedy_decoder.decode(arc_scores, mask)
    
    print("\nGreedy Trees:")
    for i, tree in enumerate(greedy_trees):
        print(f"Sentence {i}: {tree}")
    
    # Test evaluation
    gold_trees = [[0, 0, 1, 2, 3], [0, 0, 1, 0, 2]]
    uas, correct, total = evaluate_parsing_accuracy(mst_trees, gold_trees, mask)
    print(f"\nMST UAS: {uas:.4f} ({correct}/{total})")
    
    uas_greedy, correct_greedy, total_greedy = evaluate_parsing_accuracy(greedy_trees, gold_trees, mask)
    print(f"Greedy UAS: {uas_greedy:.4f} ({correct_greedy}/{total_greedy})")
    
    print("MST decoder test passed!")