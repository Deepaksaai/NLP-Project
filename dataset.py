import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
import re
import pickle
import os


# -------------------------------------------------------
# Special tokens
# -------------------------------------------------------
PAD_TOKEN = '<PAD>'   # index 0 — padding
SOS_TOKEN = '<SOS>'   # index 1 — start of sequence
EOS_TOKEN = '<EOS>'   # index 2 — end of sequence
UNK_TOKEN = '<UNK>'   # index 3 — unknown token

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class Vocabulary:
    """
    Builds and manages vocabulary from training data.
    Converts between words and indices.
    """

    def __init__(self, max_vocab_size=30000):
        self.max_vocab_size = max_vocab_size

        # Initialize with special tokens
        self.word2idx = {
            PAD_TOKEN: 0,
            SOS_TOKEN: 1,
            EOS_TOKEN: 2,
            UNK_TOKEN: 3
        }
        self.idx2word = {
            0: PAD_TOKEN,
            1: SOS_TOKEN,
            2: EOS_TOKEN,
            3: UNK_TOKEN
        }
        self.word_freq = Counter()

    def build_vocab(self, texts):
        """
        Builds vocabulary from list of texts.
        Input: list of strings
        """
        print("Building vocabulary...")

        # Count word frequencies
        for text in texts:
            tokens = self.tokenize(text)
            self.word_freq.update(tokens)

        # Take top max_vocab_size - 4 words
        # (subtract 4 for special tokens)
        most_common = self.word_freq.most_common(
            self.max_vocab_size - len(SPECIAL_TOKENS)
        )

        # Add to vocabulary
        for word, freq in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"Vocabulary built — size: {len(self.word2idx)}")

    def tokenize(self, text):
        """
        Simple whitespace tokenizer with basic cleaning.
        Lowercases and removes special characters.
        """
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:\-]', '', text)
        tokens = text.split()
        return tokens

    def encode(self, text, max_len=None):
        """
        Converts text string to list of token indices.
        Adds SOS and EOS tokens.
        """
        tokens = self.tokenize(text)

        # Truncate if needed
        if max_len is not None:
            tokens = tokens[:max_len - 2]  # -2 for SOS and EOS

        # Convert to indices
        indices = [self.word2idx.get(token, 3) for token in tokens]

        # Add SOS and EOS
        indices = [1] + indices + [2]

        return indices

    def decode(self, indices):
        """
        Converts list of token indices back to text string.
        Stops at EOS token.
        """
        words = []
        for idx in indices:
            if idx == 2:  # EOS
                break
            if idx not in [0, 1]:  # Skip PAD and SOS
                words.append(self.idx2word.get(idx, UNK_TOKEN))
        return ' '.join(words)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Vocabulary saved to {path}")

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Vocabulary loaded from {path}")
        return vocab

    def __len__(self):
        return len(self.word2idx)


class BillSumDataset(Dataset):
    """
    PyTorch Dataset for BillSum.
    Each example is a (bill text, summary) pair.
    """

    def __init__(self, data, vocab, src_max_len=400, trg_max_len=100):
        self.data = data
        self.vocab = vocab
        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # Encode source and target
        src = self.vocab.encode(example['text'], max_len=self.src_max_len)
        trg = self.vocab.encode(example['summary'], max_len=self.trg_max_len)

        return {
            'src': torch.tensor(src, dtype=torch.long),
            'trg': torch.tensor(trg, dtype=torch.long)
        }


def collate_fn(batch):
    """
    Pads sequences in a batch to the same length.
    Called automatically by DataLoader.
    """

    src_batch = [item['src'] for item in batch]
    trg_batch = [item['trg'] for item in batch]

    # Pad sequences to max length in batch
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=0
    )
    trg_padded = torch.nn.utils.rnn.pad_sequence(
        trg_batch, batch_first=True, padding_value=0
    )

    return {
        'src': src_padded,
        'trg': trg_padded
    }


def load_billsum(vocab_path='vocab.pkl', max_vocab_size=30000):
    """
    Loads BillSum dataset, builds vocabulary, returns dataloaders.
    """

    print("Loading BillSum dataset...",flush = True)
    dataset = load_dataset("FiscalNote/billsum")

    train_data = list(dataset['train'])
    test_data = list(dataset['test'])

    print(f"Train examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")

    # Build or load vocabulary
    if os.path.exists(vocab_path):
        vocab = Vocabulary.load(vocab_path)
    else:
        vocab = Vocabulary(max_vocab_size=max_vocab_size)

        # Build vocab from training texts and summaries
        all_texts = (
            [ex['text'] for ex in train_data] +
            [ex['summary'] for ex in train_data]
        )
        vocab.build_vocab(all_texts)
        vocab.save(vocab_path)

    # Create datasets
    train_dataset = BillSumDataset(
        train_data, vocab,
        src_max_len=400,
        trg_max_len=100
    )

    test_dataset = BillSumDataset(
        test_data, vocab,
        src_max_len=400,
        trg_max_len=100
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, test_dataset, vocab


# Test it
if __name__ == "__main__":

    train_dataset, test_dataset, vocab = load_billsum()

    print(f"\nVocabulary size: {len(vocab)}")

    # Test one example
    example = train_dataset[0]
    print(f"\nFirst example:")
    print(f"Source tokens: {example['src'].shape}")
    print(f"Target tokens: {example['trg'].shape}")

    # Decode back to text
    print(f"\nSource preview:")
    print(vocab.decode(example['src'].tolist())[:200])
    print(f"\nTarget preview:")
    print(vocab.decode(example['trg'].tolist())[:200])

    # Test dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )

    batch = next(iter(train_loader))
    print(f"\nBatch source shape: {batch['src'].shape}")
    print(f"Batch target shape: {batch['trg'].shape}")
    print("\nDataset test passed successfully")


