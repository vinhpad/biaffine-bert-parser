

import torch
from models.bert_encoder import BertEncoder, VietnameseBertTokenizer, extract_word_representations


if __name__ == "__main__":
    # Test the BERT encoder
    print("Testing Vietnamese BERT Encoder...")
    
    # Test tokenizer
    tokenizer = VietnameseBertTokenizer("vinai/phobert-base-v2")
    
    # Test sentences
    sentences = [
        ["Tôi", "thích", "học", "tiếng", "Việt"],
        ["Hôm", "nay", "trời", "đẹp"]
    ]
    
    # Encode sentences
    encoded = tokenizer.batch_encode_sentences(sentences)
    print(f"Input IDs shape: {encoded['input_ids'].shape}")
    print(f"Attention mask shape: {encoded['attention_mask'].shape}")
    print(f"Valid positions shape: {encoded['valid_positions'].shape}")
    
    # Test BERT encoder
    encoder = BertEncoder()
    print(f"BERT hidden size: {encoder.hidden_size}")
    
    # Forward pass
    with torch.no_grad():
        bert_output = encoder(encoded['input_ids'], encoded['attention_mask'])
        print(f"BERT output shape: {bert_output.shape}")
        
        # Extract word representations
        word_repr = extract_word_representations(bert_output, encoded['valid_positions'])
        print(f"Word representations shape: {word_repr.shape}")
    
    print("Vietnamese BERT encoder test passed!")