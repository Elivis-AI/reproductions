import tqdm
from collections import Counter
from typing import List, Tuple

from bpe import BPETokenizer


class BPEByteTokenizer(BPETokenizer):
    """
    Byte Pair Encoding Tokenizer for UTF-8 text
    This tokenizer is designed to handle UTF-8 encoded text by treating each byte as a token.
    It builds a vocabulary of byte pairs and merges them iteratively to create a fixed-size vocabulary.
    It is suitable for text that may contain non-ASCII characters, as it operates on byte-level encoding.
    """
    type = 'BPEByteTokenizer'

    def train(self, text: str, vocab_size: int, verbose: bool = False):

        self.vocab = {i: bytes([i]) for i in range(256)}  # all Byte
        
        num_merges = vocab_size - len(self.vocab)
        assert num_merges >= 0, "Please input a larger vocab_size."
        

        ids = list(text.encode('utf-8'))  # convert text to list of byte integers
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
        ids = list(text.encode('utf-8'))  # convert text to list of byte integers

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
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


    def save_vocab(self, path='vocab.txt'):
        """
        Copy from `bytes_to_unicode` by Andrey Karpathy.
        Every possible byte (really an integer 0..255) gets mapped by OpenAI to a unicode
        character that represents it visually. Some bytes have their appearance preserved
        because they don't cause any trouble. These are defined in list bs. For example:
        chr(33) returns "!", so in the returned dictionary we simply have d[33] -> "!".
        However, chr(0), for example, is '\x00', which looks ugly. So OpenAI maps these
        bytes, into new characters in a range where chr() returns a single nice character.
        So in the final dictionary we have d[0] -> 'Ā' instead, which is just chr(0 + 2**8).
        In particular, the space character is 32, which we can see by ord(' '). Instead,
        this function will shift space (32) by 256 to 288, so d[32] -> 'Ġ'.
        So this is just a simple one-to-one mapping of bytes 0..255 into unicode characters
        that "look nice", either in their original form, or a funny shifted character
        like 'Ā', or 'Ġ', etc.
        """
        # the 188 integers that render fine in their original form and need no shifting
        bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:] # all integers b in bs will simply map to chr(b) in the output dict
        # now get the representations of the other 68 integers that do need shifting
        # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
        n = 0
        for b in range(2**8):
            if b not in bs:
                # if this byte is "ugly" then map it to the next available "nice" character
                bs.append(b)
                cs.append(2**8+n)
                n += 1
        cs = [chr(n) for n in cs]
        d = dict(zip(bs, cs))

        with open(path, 'w', encoding='utf-8') as f:
                f.write("\n".join(f"{key} {''.join(d[c] for c in item)}" for key, item in self.vocab.items()))



if __name__ == '__main__':    
    tokenizer = BPEByteTokenizer()

    with open("./data/天龙八部.txt", "r", encoding='utf-8') as f:
        data = f.read()

    tokenizer.train(data, vocab_size=1000)
    tokenizer.save("byte_tokenizer.pkl")
    tokenizer.save_vocab("byte_vocab.txt")

    
    tokenizer = BPEByteTokenizer.from_pretrained("byte_tokenizer.pkl")

    encoded = tokenizer.encode("青光闪动，一柄青钢剑倏地刺出，指向在年汉子左肩，使剑少年不等招用老，腕抖剑斜，剑锋已削向那汉子右颈。")
    
    print("Encoded:", encoded, len(encoded), "Bytes")
    
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded, len(decoded.encode("utf-8")), "Bytes")
    
# Encoded: [919, 688, 339, 170, 631, 532, 784, 132, 919, 528, 162, 632, 525, 143, 493, 293, 186,
#  355, 320, 140, 135, 457, 334, 577, 818, 329, 796, 534, 169, 257, 751, 632, 509, 577, 271, 640, 
# 768, 706, 405, 337, 133, 149, 397, 150, 632, 377, 156, 262, 137, 145, 716, 139, 467, 387, 138, 457,
#  331, 818, 329, 981, 601, 136, 264] 63 Bytes
# Decoded: 青光闪动，一柄青钢剑倏地刺出，指向在年汉子左肩，使剑少年不等招用老，腕抖剑斜，剑锋已削向那汉子右颈。 150 Bytes