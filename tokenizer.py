import tiktoken

class GPTTokenizer:

    def __init__(self, encoding_name='gpt2'):
        
        self.tokenizer  =tiktoken.get_encoding(encoding_name=encoding_name)

        self.special_tokens = {"<|endoftext|>" : self.tokenizer.eot_token}


    def encode(self, text, allowed_special=None):

        if allowed_special is None:
            allowed_special = set()

        return self.tokenizer.encode(text, allowed_special=allowed_special)
    
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
    
    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab
    

if __name__ == "__main__":

    tokenizer = GPTTokenizer()
    sample_text = "Hello world, this is atokenizer of GPT 2 . <|endoftext|>"

    encoded = tokenizer.encode(sample_text, allowed_special={"<|endoftext|>"})
    print(f"Sample text: '{sample_text}'")
    print(f"Encoded token IDs: {encoded}")
    print(f"Number of tokens: {len(encoded)}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded text: '{decoded}'")
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")