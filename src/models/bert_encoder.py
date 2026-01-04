"""
BERT Encoder for Vietnamese Dependency Parsing
Wraps pretrained BERT models for encoding Vietnamese text.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional


class BertEncoder(nn.Module):
    """
    BERT encoder for Vietnamese text processing.
    
    Args:
        model_name: Name of pretrained BERT model
        dropout: Dropout rate for final layer
        freeze_bert: Whether to freeze BERT parameters during training
    """
    
    def __init__(self, 
                 model_name: str = "vinai/phobert-base", 
                 dropout: float = 0.1,
                 freeze_bert: bool = False):
        super(BertEncoder, self).__init__()
        
        self.model_name = model_name
        self.freeze_bert = freeze_bert
        
        # Load pretrained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        self.dropout = nn.Dropout(dropout)
        
        # Get hidden size from BERT config
        self.hidden_size = self.bert.config.hidden_size
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode input sequences using BERT.
        
        Args:
            input_ids: [batch_size, seq_len] - tokenized input
            attention_mask: [batch_size, seq_len] - attention mask
            token_type_ids: [batch_size, seq_len] - token type ids (optional)
            
        Returns:
            encoded: [batch_size, seq_len, hidden_size] - contextualized representations
        """
        # BERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Get last hidden states
        encoded = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply dropout
        encoded = self.dropout(encoded)
        
        return encoded


class VietnameseBertTokenizer:
    """
    Wrapper for Vietnamese BERT tokenizer with utilities for dependency parsing.
    """
    
    def __init__(self, model_name: str = "vinai/phobert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Special tokens
        self.pad_token = self.tokenizer.pad_token
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.unk_token = self.tokenizer.unk_token
        
        # Token IDs
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        
    def tokenize_conllu_sentence(self, words: List[str], max_length: int = 512) -> Dict:
        """
        Tokenize a sentence from CoNLL-U format for dependency parsing.
        
        Args:
            words: List of words from CoNLL-U sentence
            max_length: Maximum sequence length
            
        Returns:
            Dict containing:
                - input_ids: Tokenized input
                - attention_mask: Attention mask
                - word_ids: Mapping from subword tokens to original words
                - valid_positions: Positions of first subword for each word
        """
        # Add [CLS] token at the beginning
        input_ids = [self.cls_token_id]
        word_ids = [None]  # [CLS] doesn't correspond to any word
        valid_positions = []
        
        for word_idx, word in enumerate(words):
            # Tokenize word (without adding special tokens)
            word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
            
            if len(input_ids) + len(word_tokens) + 1 > max_length:  # +1 for [SEP]
                break
                
            # Mark the position of first subword token for this word
            valid_positions.append(len(input_ids))
            
            # Add tokens
            input_ids.extend(word_tokens)
            
            # Track which original word each subword belongs to
            word_ids.extend([word_idx] * len(word_tokens))
        
        # Add [SEP] token
        input_ids.append(self.sep_token_id)
        word_ids.append(None)  # [SEP] doesn't correspond to any word
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'word_ids': word_ids,
            'valid_positions': valid_positions,
            'num_words': len(words)
        }
    
    def batch_encode_sentences(self, 
                             sentences: List[List[str]], 
                             max_length: int = 512,
                             return_tensors: str = "pt") -> Dict:
        """
        Batch encode multiple sentences.
        
        Args:
            sentences: List of sentences (each sentence is list of words)
            max_length: Maximum sequence length
            return_tensors: Format of returned tensors
            
        Returns:
            Batch of encoded sentences with padding
        """
        batch_encoded = []
        max_seq_len = 0
        max_words = 0
        
        # Encode each sentence
        for words in sentences:
            encoded = self.tokenize_conllu_sentence(words, max_length)
            batch_encoded.append(encoded)
            max_seq_len = max(max_seq_len, len(encoded['input_ids']))
            max_words = max(max_words, encoded['num_words'])
        
        # Pad sequences
        batch_input_ids = []
        batch_attention_mask = []
        batch_valid_positions = []
        
        for encoded in batch_encoded:
            # Pad input_ids and attention_mask
            seq_len = len(encoded['input_ids'])
            padded_input_ids = encoded['input_ids'] + [self.pad_token_id] * (max_seq_len - seq_len)
            padded_attention_mask = encoded['attention_mask'] + [0] * (max_seq_len - seq_len)
            
            # Pad valid_positions
            padded_valid_positions = encoded['valid_positions'] + [-1] * (max_words - len(encoded['valid_positions']))
            
            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_attention_mask)
            batch_valid_positions.append(padded_valid_positions)
        
        result = {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'valid_positions': batch_valid_positions
        }
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch
            result['input_ids'] = torch.tensor(result['input_ids'], dtype=torch.long)
            result['attention_mask'] = torch.tensor(result['attention_mask'], dtype=torch.long)
            result['valid_positions'] = torch.tensor(result['valid_positions'], dtype=torch.long)
        
        return result


def extract_word_representations(encoded_output: torch.Tensor,
                               valid_positions: torch.Tensor) -> torch.Tensor:
    """
    Extract word-level representations from subword-level BERT output.
    
    Args:
        encoded_output: [batch_size, seq_len, hidden_size] - BERT output
        valid_positions: [batch_size, max_words] - positions of first subword for each word
        
    Returns:
        word_repr: [batch_size, max_words, hidden_size] - word-level representations
    """
    batch_size, seq_len, hidden_size = encoded_output.shape
    max_words = valid_positions.shape[1]
    
    # Create output tensor
    word_repr = torch.zeros(batch_size, max_words, hidden_size, 
                           dtype=encoded_output.dtype, 
                           device=encoded_output.device)
    
    # Extract representations
    for batch_idx in range(batch_size):
        for word_idx in range(max_words):
            pos = valid_positions[batch_idx, word_idx]
            if pos >= 0:  # Valid position
                word_repr[batch_idx, word_idx] = encoded_output[batch_idx, pos]
    
    return word_repr


if __name__ == "__main__":
    # Test the BERT encoder
    print("Testing Vietnamese BERT Encoder...")
    
    # Test tokenizer
    tokenizer = VietnameseBertTokenizer()
    
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