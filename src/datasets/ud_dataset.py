"""
Dataset loader for CoNLL-U format for Vietnamese dependency parsing.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import re
from collections import Counter, defaultdict

from ..models.bert_encoder import VietnameseBertTokenizer


class ConlluSentence:
    """Represents a single sentence from CoNLL-U format."""
    
    def __init__(self):
        self.sent_id: str = ""
        self.text: str = ""
        self.tokens: List[Dict] = []
    
    def add_token(self, token_data: Dict):
        """Add a token to the sentence."""
        self.tokens.append(token_data)
    
    def get_words(self) -> List[str]:
        """Get list of word forms."""
        return [token['form'] for token in self.tokens if '-' not in str(token['id'])]
    
    def get_heads(self) -> List[int]:
        """Get list of head indices."""
        return [int(token['head']) for token in self.tokens if '-' not in str(token['id'])]
    
    def get_deprels(self) -> List[str]:
        """Get list of dependency relations."""
        return [token['deprel'] for token in self.tokens if '-' not in str(token['id'])]
    
    def get_upos(self) -> List[str]:
        """Get list of UPOS tags."""
        return [token['upos'] for token in self.tokens if '-' not in str(token['id'])]


class ConlluReader:
    """Reader for CoNLL-U format files."""
    
    def __init__(self):
        pass
    
    def read_file(self, file_path: str) -> List[ConlluSentence]:
        """
        Read a CoNLL-U file and return list of sentences.
        
        Args:
            file_path: Path to CoNLL-U file
            
        Returns:
            List of ConlluSentence objects
        """
        sentences = []
        current_sentence = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line:  # Empty line indicates end of sentence
                    if current_sentence and current_sentence.tokens:
                        sentences.append(current_sentence)
                    current_sentence = None
                    continue
                
                if line.startswith('#'):  # Comment line
                    if current_sentence is None:
                        current_sentence = ConlluSentence()
                    
                    if line.startswith('# sent_id ='):
                        current_sentence.sent_id = line.split('=', 1)[1].strip()
                    elif line.startswith('# text ='):
                        current_sentence.text = line.split('=', 1)[1].strip()
                    continue
                
                # Token line
                if current_sentence is None:
                    current_sentence = ConlluSentence()
                
                fields = line.split('\t')
                if len(fields) == 10:
                    token_data = {
                        'id': fields[0],
                        'form': fields[1],
                        'lemma': fields[2],
                        'upos': fields[3],
                        'xpos': fields[4],
                        'feats': fields[5],
                        'head': fields[6],
                        'deprel': fields[7],
                        'deps': fields[8],
                        'misc': fields[9]
                    }
                    current_sentence.add_token(token_data)
        
        # Add the last sentence if exists
        if current_sentence and current_sentence.tokens:
            sentences.append(current_sentence)
        
        return sentences


class LabelVocabulary:
    """Vocabulary for dependency relation labels."""
    
    def __init__(self):
        self.label_to_id: Dict[str, int] = {}
        self.id_to_label: Dict[int, str] = {}
        self.label_counts: Counter = Counter()
    
    def add_label(self, label: str):
        """Add a label to vocabulary."""
        self.label_counts[label] += 1
        if label not in self.label_to_id:
            label_id = len(self.label_to_id)
            self.label_to_id[label] = label_id
            self.id_to_label[label_id] = label
    
    def get_id(self, label: str) -> int:
        """Get ID for a label."""
        return self.label_to_id.get(label, 0)  # Return 0 for unknown labels
    
    def get_label(self, label_id: int) -> str:
        """Get label for an ID."""
        return self.id_to_label.get(label_id, "UNK")
    
    def size(self) -> int:
        """Get vocabulary size."""
        return len(self.label_to_id)
    
    def most_common_labels(self, n: int = 20) -> List[Tuple[str, int]]:
        """Get most common labels."""
        return self.label_counts.most_common(n)


class DependencyParsingDataset(Dataset):
    """
    Dataset for dependency parsing with BERT tokenization.
    """
    
    def __init__(self, 
                 conllu_file: str,
                 tokenizer: VietnameseBertTokenizer,
                 label_vocab: Optional[LabelVocabulary] = None,
                 max_length: int = 512,
                 is_training: bool = True):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        
        # Read sentences
        reader = ConlluReader()
        self.sentences = reader.read_file(conllu_file)
        
        # Build or use label vocabulary
        if label_vocab is None:
            self.label_vocab = LabelVocabulary()
            # Add special labels
            self.label_vocab.add_label("ROOT")  # For root attachments
            self.label_vocab.add_label("UNK")   # For unknown labels
            
            # Build vocabulary from data
            for sentence in self.sentences:
                for deprel in sentence.get_deprels():
                    self.label_vocab.add_label(deprel)
        else:
            self.label_vocab = label_vocab
        
        # Preprocess sentences
        self.processed_sentences = []
        for sentence in self.sentences:
            processed = self._preprocess_sentence(sentence)
            if processed is not None:
                self.processed_sentences.append(processed)
    
    def _preprocess_sentence(self, sentence: ConlluSentence) -> Optional[Dict]:
        """
        Preprocess a sentence for training/evaluation.
        
        Args:
            sentence: ConlluSentence object
            
        Returns:
            Preprocessed data dict or None if sentence is too long
        """
        words = sentence.get_words()
        heads = sentence.get_heads()
        deprels = sentence.get_deprels()
        
        # Skip empty sentences
        if not words:
            return None
        
        # Skip very long sentences
        if len(words) > self.max_length - 2:  # Account for [CLS] and [SEP]
            return None
        
        # Tokenize with BERT tokenizer
        tokenized = self.tokenizer.tokenize_conllu_sentence(words, self.max_length)
        
        # Convert heads (1-indexed to 0-indexed)
        processed_heads = []
        for head in heads:
            if head == 0:  # Root
                processed_heads.append(0)
            else:
                processed_heads.append(head - 1)  # Convert to 0-indexed
        
        # Convert deprels to IDs
        processed_deprels = []
        for deprel in deprels:
            if deprel == "root":
                processed_deprels.append(self.label_vocab.get_id("ROOT"))
            else:
                processed_deprels.append(self.label_vocab.get_id(deprel))
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'valid_positions': tokenized['valid_positions'],
            'num_words': tokenized['num_words'],
            'heads': processed_heads,
            'deprels': processed_deprels,
            'words': words,
            'sent_id': sentence.sent_id
        }
    
    def __len__(self) -> int:
        return len(self.processed_sentences)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.processed_sentences[idx]


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching dependency parsing data.
    
    Args:
        batch: List of data samples
        
    Returns:
        Batched data dictionary
    """
    batch_size = len(batch)
    
    # Get maximum sequence length and word count
    max_seq_len = max(len(sample['input_ids']) for sample in batch)
    max_words = max(sample['num_words'] for sample in batch)
    
    # Initialize batch tensors
    input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    valid_positions = torch.full((batch_size, max_words), -1, dtype=torch.long)
    heads = torch.zeros(batch_size, max_words, dtype=torch.long)
    deprels = torch.zeros(batch_size, max_words, dtype=torch.long)
    word_mask = torch.zeros(batch_size, max_words, dtype=torch.bool)
    
    # Fill tensors
    for i, sample in enumerate(batch):
        seq_len = len(sample['input_ids'])
        num_words = sample['num_words']
        
        # Input sequences
        input_ids[i, :seq_len] = torch.tensor(sample['input_ids'])
        attention_mask[i, :seq_len] = torch.tensor(sample['attention_mask'])
        
        # Valid positions and targets
        valid_positions[i, :len(sample['valid_positions'])] = torch.tensor(sample['valid_positions'])
        heads[i, :num_words] = torch.tensor(sample['heads'])
        deprels[i, :num_words] = torch.tensor(sample['deprels'])
        word_mask[i, :num_words] = True
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'valid_positions': valid_positions,
        'heads': heads,
        'deprels': deprels,
        'word_mask': word_mask,
        'batch_size': batch_size,
        'max_words': max_words
    }


def create_dataloaders(train_file: str,
                      dev_file: str,
                      test_file: str,
                      tokenizer: VietnameseBertTokenizer,
                      batch_size: int = 16,
                      max_length: int = 512,
                      num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader, LabelVocabulary]:
    """
    Create data loaders for training, development, and test sets.
    
    Args:
        train_file: Path to training CoNLL-U file
        dev_file: Path to development CoNLL-U file  
        test_file: Path to test CoNLL-U file
        tokenizer: BERT tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loader workers
        
    Returns:
        train_loader, dev_loader, test_loader, label_vocabulary
    """
    # Create training dataset and build vocabulary
    train_dataset = DependencyParsingDataset(
        train_file, tokenizer, max_length=max_length, is_training=True
    )
    
    label_vocab = train_dataset.label_vocab
    
    # Create dev and test datasets using the same vocabulary
    dev_dataset = DependencyParsingDataset(
        dev_file, tokenizer, label_vocab, max_length=max_length, is_training=False
    )
    
    test_dataset = DependencyParsingDataset(
        test_file, tokenizer, label_vocab, max_length=max_length, is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, dev_loader, test_loader, label_vocab


if __name__ == "__main__":
    # Test the dataset
    print("Testing Dependency Parsing Dataset...")
    
    # Create tokenizer
    tokenizer = VietnameseBertTokenizer()
    
    # Test with a sample file (you'll need to adjust the path)
    train_file = "../../vi_vtb-ud-train.conllu"
    
    try:
        # Create dataset
        dataset = DependencyParsingDataset(train_file, tokenizer, max_length=128)
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Label vocabulary size: {dataset.label_vocab.size()}")
        
        # Print most common labels
        print("\nMost common dependency relations:")
        for label, count in dataset.label_vocab.most_common_labels(10):
            print(f"  {label}: {count}")
        
        # Test a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample data keys: {list(sample.keys())}")
            print(f"Number of words: {sample['num_words']}")
            print(f"Words: {sample['words'][:5]}...")  # First 5 words
            print(f"Heads: {sample['heads'][:5]}")     # First 5 heads
            
            # Test collate function
            batch = [sample, dataset[min(1, len(dataset)-1)]]
            batched = collate_fn(batch)
            print(f"\nBatched data keys: {list(batched.keys())}")
            print(f"Input IDs shape: {batched['input_ids'].shape}")
            print(f"Heads shape: {batched['heads'].shape}")
        
        print("Dataset test passed!")
        
    except FileNotFoundError:
        print(f"Test file not found: {train_file}")
        print("Please adjust the path or run from the correct directory.")
    except Exception as e:
        print(f"Error during testing: {e}")