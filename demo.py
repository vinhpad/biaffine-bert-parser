"""
Script to run quick test/demo of the biaffine parser
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.biaffine_parser import BiaffineDependencyParser, ParserConfig
from src.models.bert_encoder import VietnameseBertTokenizer
from src.utils.mst_decoder import MSTDecoder


def demo_parser():
    """Demo the biaffine parser with sample Vietnamese text."""
    
    print("=== Biaffine Dependency Parser Demo ===")
    
    # Create config
    config = ParserConfig()
    config.num_labels = 20  # Simplified for demo
    
    # Create model
    print("Loading model...")
    model = BiaffineDependencyParser(
        bert_model_name=config.bert_model_name,
        num_labels=config.num_labels
    )
    model.eval()
    
    # Create tokenizer
    tokenizer = VietnameseBertTokenizer()
    
    # Sample Vietnamese sentences
    test_sentences = [
        ["Tôi", "thích", "học", "tiếng", "Việt"],
        ["Hôm", "nay", "trời", "đẹp", "quá"],
        ["Chúng", "tôi", "đang", "làm", "bài", "tập"]
    ]
    
    # Create MST decoder
    mst_decoder = MSTDecoder()
    
    print("\nProcessing sentences...")
    
    with torch.no_grad():
        for i, words in enumerate(test_sentences):
            print(f"\nSentence {i+1}: {' '.join(words)}")
            
            # Tokenize
            encoded = tokenizer.batch_encode_sentences([words], return_tensors="pt")
            
            # Forward pass
            arc_scores, label_scores = model(
                encoded['input_ids'],
                encoded['attention_mask'], 
                encoded['valid_positions']
            )
            
            # Decode with MST
            predicted_heads = mst_decoder.decode(arc_scores)
            
            # Print dependency tree
            print("Dependency structure:")
            for j, (word, head) in enumerate(zip(words, predicted_heads[0][:len(words)])):
                head_word = "ROOT" if head == 0 else words[head-1] if head > 0 else "?"
                print(f"  {word} -> {head_word} (head index: {head})")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    try:
        demo_parser()
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Make sure you have installed all requirements: pip install -r requirements.txt")