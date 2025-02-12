import json
import regex as re
from typing import List, Dict, Union, Optional
from pathlib import Path
import string

class GPT2Tokenizer:
    """GPT-2 BPE tokenizer implementation."""
    
    def __init__(
        self,
        vocab_file: Optional[Union[str, Path]] = None,
        merges_file: Optional[Union[str, Path]] = None,
        errors: str = "replace",
        max_len: int = None,
    ):
        self.max_len = max_len if max_len is not None else int(1e12)
        self.errors = errors
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache = {}
        
        # Load pre-trained vocab and merges if provided
        if vocab_file is not None and merges_file is not None:
            self.load_vocab(vocab_file)
            self.load_merges(merges_file)
        else:
            # Initialize with base vocabulary (all ASCII characters)
            vocab = ["<|endoftext|>"]
            # Add all ASCII characters
            vocab.extend(list(string.printable))
            # Add common words
            vocab.extend([
                "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
                "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
                "Once", "upon", "time", "was", "were", "will", "would", "could", "should",
                "there", "their", "they", "this", "those", "these", "then", "than",
                "what", "when", "where", "which", "who", "whom", "whose", "why", "how"
            ])
            self.encoder = {token: i for i, token in enumerate(vocab)}
            self.encoder["<|endoftext|>"] = len(vocab)  # Special token at the end
            self.decoder = {v: k for k, v in self.encoder.items()}
        
        # Regex for tokenization
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    def load_vocab(self, vocab_file: Union[str, Path]):
        """Load vocabulary from file."""
        vocab_file = Path(vocab_file)
        with vocab_file.open('r', encoding='utf-8') as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
    
    def load_merges(self, merges_file: Union[str, Path]):
        """Load BPE merge operations from file."""
        merges_file = Path(merges_file)
        self.bpe_ranks = {}
        with merges_file.open('r', encoding='utf-8') as f:
            for i, merge in enumerate(f):
                if merge.strip():
                    pair = tuple(merge.split())
                    self.bpe_ranks[pair] = i
    
    def bpe(self, token: str) -> str:
        """Apply Byte-Pair Encoding to token."""
        if token in self.cache:
            return self.cache[token]
            
        if token in self.encoder:
            return token
            
        word = tuple(token)
        pairs = get_pairs(word)
        
        if not pairs:
            return token
            
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
                
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    if word[j + 1] == second:
                        new_word.append(first + second)
                        i = j + 2
                    else:
                        new_word.append(word[j])
                        i = j + 1
                except ValueError:
                    new_word.extend(word[i:])
                    break
                    
            word = tuple(new_word)
            if len(word) == 1:
                break
                
            pairs = get_pairs(word)
            
        word = ' '.join(word)
        self.cache[token] = word
        return word
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            if token in self.encoder:
                bpe_tokens.append(self.encoder[token])
            else:
                # Fallback to character-level tokenization
                for char in token:
                    if char in self.encoder:
                        bpe_tokens.append(self.encoder[char])
                    else:
                        # Skip unknown characters
                        continue
        return bpe_tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token ids to text."""
        text = ''.join([self.decoder.get(token, '') for token in tokens])
        return text
    
    def save_pretrained(self, save_directory: Union[str, Path]):
        """Save tokenizer vocabulary and merges files."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        vocab_file = save_directory / "vocab.json"
        merges_file = save_directory / "merges.txt"
        
        with vocab_file.open('w', encoding='utf-8') as f:
            json.dump(self.encoder, f, ensure_ascii=False)
            
        if hasattr(self, 'bpe_ranks'):
            with merges_file.open('w', encoding='utf-8') as f:
                for pair, rank in sorted(self.bpe_ranks.items(), key=lambda x: x[1]):
                    f.write(f"{pair[0]} {pair[1]}\n")

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings.
    Specifically avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs
