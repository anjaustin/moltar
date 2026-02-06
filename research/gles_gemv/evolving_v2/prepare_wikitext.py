"""
Pre-download and cache WikiText-2 data.
Run this first to avoid timeout issues during training.
"""

import os
import pickle


def main():
    print("Loading libraries...")
    from datasets import load_dataset
    from transformers import GPT2Tokenizer

    cache_path = "wikitext2_tokens.pkl"

    if os.path.exists(cache_path):
        print(f"Cache exists at {cache_path}")
        with open(cache_path, "rb") as f:
            tokens = pickle.load(f)
        print(f"Loaded {len(tokens):,} tokens from cache")
        return tokens

    print("Downloading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)

    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print("Concatenating text...")
    texts = dataset["train"]["text"]
    full_text = "\n".join([t for t in texts if t.strip()])

    print(f"Total characters: {len(full_text):,}")

    print("Tokenizing...")
    tokens = tokenizer.encode(full_text)

    print(f"Total tokens: {len(tokens):,}")

    print(f"Caching to {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump(tokens, f)

    print("Done!")

    # Show some stats
    print(f"\nDataset stats:")
    print(f"  Tokens: {len(tokens):,}")
    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    print(f"  First 20 tokens: {tokens[:20]}")
    print(f"  Decoded: '{tokenizer.decode(tokens[:20])}'")

    return tokens


if __name__ == "__main__":
    main()
