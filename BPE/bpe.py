import pickle
import tqdm

from collections import Counter
from typing import List, Tuple


class BPETokenizer:
    """
    Classic Byte Pair Encoding Tokenizer for ASCII text
    This tokenizer builds a vocabulary of character pairs and merges them iteratively to create a fixed-size vocabulary.
    """
    type = 'BPETokenizer'
    
    def __init__(self):
        self.vocab = {}   # idx -> token
        self.merges = {}  # (idx, idx) -> idx


    def train(self, text: str, vocab_size: int, verbose: bool = False):
        assert text.isascii(), "Text must be ASCII, or you should use BPEByteTokenizer for utf-8 supporting" \
            + ','.join([w for w in text if not w.isascii()])
        self.vocab = {i: chr(i) for i in range(128)}  # all ASCII chars
        
        num_merges = vocab_size - len(self.vocab)
        assert num_merges >= 0, "Please input a larger vocab_size."
        

        ids = [ord(c) for c in text]
        idx = len(self.vocab)
        for i in (range(num_merges) if verbose else tqdm.trange(num_merges, desc="Training BPE Tokenizer")):
            # Given a list of integers, return a dictionary of counts of consecutive pairs
            # Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
            # Optionally allows to update an existing dictionary of counts
            pairs_count = Counter(zip(ids, ids[1:]))
            pair, n = pairs_count.most_common(1)[0]  # Get the most common pair
            ids = self._merge(ids, pair, idx)
            
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            idx += 1

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({self.vocab[idx]}) had {n} occurrences")


    def encode(self, text: str) -> List[int]:
        """"text to ids"""
        ids = [ord(c) for c in text]

        while len(ids) >= 2:
            pairs_count = Counter(zip(ids, ids[1:]))
            pair = min(pairs_count, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = self._merge(ids, pair, idx)

        return ids


    def decode(self, ids: List[int]) -> str:
        """ids to text"""
        return "".join(self.vocab[idx] for idx in ids)

    @staticmethod
    def _merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        """
        In the list of integers (ids), replace all consecutive occurrences
        of pair with the new integer token idx
        Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
        """
        newids = []
        i = 0
        
        while i < len(ids):
            # if not at the very last position AND the pair matches, replace it
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        
        return newids

    def save(self, path):
        dump = {
            'type': self.type,
            'vocab': self.vocab,
            'merges': self.merges,
        }
        with open(path, 'wb') as f:
            pickle.dump(dump, f)
    
    def save_vocab(self, path='vocab.txt'):
        with open(path, 'w') as f:
                f.write("\n".join(f"{key} {item}" for key, item in self.vocab.items()))

    @classmethod
    def from_pretrained(cls, path):
        with open(path, 'rb') as f:
            dump = pickle.load(f)
        assert dump['type'] == cls.type, f"Unmatched tokenizer type: {dump['type']}"
        tokenizer = cls()
        tokenizer.vocab = dump['vocab']
        tokenizer.merges = dump['merges']
        return tokenizer
    
    @property
    def vocab_size(self):
        return len(self.vocab)
    



def filter_non_ascii(text: str) -> str:
    """Filter out non-ASCII and control characters from the text."""
    return ''.join(c for c in text if c.isascii())


if __name__ == "__main__":
    # Example usage
    tokenizer = BPETokenizer()

    with open("./data/hp.txt", "r") as f:
        data = f.read()
    data = filter_non_ascii(data)

    tokenizer.train(data, vocab_size=1000)
    tokenizer.save("tokenizer.pkl")
    tokenizer.save_vocab("vocab.txt")
    
    tokenizer = BPETokenizer.from_pretrained("tokenizer.pkl")
    
    encoded = tokenizer.encode("Hello, world! This is a test for the BPE tokenizer.")
    print("Encoded:", encoded, len(encoded), "Chars")
    
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded, len(decoded), "Chars")

    
    tokenizer = BPETokenizer.from_pretrained("tokenizer.pkl")

