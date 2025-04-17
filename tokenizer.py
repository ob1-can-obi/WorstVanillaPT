import tiktoken

class Tokenizer:
    def __init__(self, model_name="cl100k_base"):
        """
        Initialize a tokenizer with the specified model.
        
        Args:
            model_name (str): The encoding model to use
                - "cl100k_base" (default): Used for ChatGPT/GPT-4 models
                - "p50k_base": Used for text-davinci-003 and many earlier models
                - "r50k_base": Used for first-generation models like davinci
        """
        self.encoding = tiktoken.get_encoding(model_name)
    
    def encode(self, text):
        """
        Tokenize the input text into token IDs.
        
        Args:
            text (str): The text to tokenize
            
        Returns:
            list: List of token IDs
        """
        return self.encoding.encode(text)
    
    def decode(self, tokens):
        """
        Convert token IDs back to text.
        
        Args:
            tokens (list): List of token IDs
            
        Returns:
            str: Decoded text
        """
        return self.encoding.decode(tokens)
    
    def token_count(self, text):
        """
        Count the number of tokens in the text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            int: Number of tokens
        """
        return len(self.encode(text))

def main():
    # Example usage
    tokenizer = Tokenizer()
    sample_text = "Hello, world! This is a test of the tiktoken tokenizer."
    
    # Encode text to tokens
    tokens = tokenizer.encode(sample_text)
    print(f"Tokens: {tokens}")
    
    # Count tokens
    token_count = tokenizer.token_count(sample_text)
    print(f"Token count: {token_count}")
    
    # Decode tokens back to text
    decoded_text = tokenizer.decode(tokens)
    print(f"Decoded text: {decoded_text}")

if __name__ == "__main__":
    main() 