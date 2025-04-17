import torch
from tokenizer import Tokenizer

class Data:
    def __init__(self, model_name="cl100k_base"):
        self.tokenizer = Tokenizer(model_name)
    
    def getTokens(self, text_file=None, text=None):
        if text_file:
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
        elif text is None:
            raise ValueError("Either text_file or text must be provided")
            
        return self.tokenizer.encode(text)
    
    def getTrainData(self, tokens, context_window):
        x_train = []
        y_train = []
        for i in range(len(tokens) - context_window):
            x = tokens[i: i + context_window]   # context_window-token input
            y = tokens[i + context_window]      # 1-token target
            x_train.append(x)
            y_train.append(y)
        x_train = torch.tensor(x_train)
        y_train = torch.tensor(y_train)
        
        return x_train, y_train
    
    def getTestData(self, tokens, context_window):
        return self.getTrainData(tokens, context_window)
    
    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)