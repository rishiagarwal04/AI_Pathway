"""
Tokenizer setup using tiktoken (GPT-2 BPE)
"""
import tiktoken

# Global tokenizer instance
_tokenizer = None

VOCAB_SIZE = 50257
EOT_TOKEN = 50256  # End of text token


def get_tokenizer():
    """Get or create the GPT-2 tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("gpt2")
        print(f"Tokenizer: GPT-2 BPE, {VOCAB_SIZE} tokens")
    return _tokenizer


def encode(text: str) -> list[int]:
    """Encode text to token IDs."""
    enc = get_tokenizer()
    return enc.encode(text)


def encode_ordinary(text: str) -> list[int]:
    """Encode text without special tokens."""
    enc = get_tokenizer()
    return enc.encode_ordinary(text)


def decode(tokens: list[int]) -> str:
    """Decode token IDs to text."""
    enc = get_tokenizer()
    return enc.decode(tokens)
