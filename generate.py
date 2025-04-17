import torch
import torch.nn.functional as F
import math
from data import Data

def generate_sequence(model, data_processor, start_text=None, start_tokens=None, max_length=100, temperature=1.0, use_multi_head=False):
    """
    Generate a sequence using the trained model.
    """
    device = next(param for param in model.parameters if isinstance(param, torch.Tensor)).device
    
    # Get parameters from the model
    c = model.c
    positions = model.positions
    context_window = model.context_window
    vocab_size = model.vocab_size
    number_of_dimensions = model.number_of_dimensions
    
    if use_multi_head:
        # Multi-head attention parameters
        W_O = model.W_O
        head_weights = model.head_weights
        num_heads = model.num_heads
        head_dim = model.head_dim
    else:
        # Single-head attention parameters
        QW = model.QW
        KW = model.KW
        VW = model.VW
    
    W1 = model.W1
    b1 = model.b1
    W2 = model.W2
    b2 = model.b2
    ln1 = model.ln1
    
    # Get start tokens if text is provided
    if start_tokens is None and start_text is not None:
        start_tokens = torch.tensor(data_processor.getTokens(text=start_text), device=device)
    elif start_tokens is None:
        # Default to a single token if nothing is provided
        start_tokens = torch.tensor([1], device=device)
    
    # Ensure start_tokens is a tensor
    if not isinstance(start_tokens, torch.Tensor):
        start_tokens = torch.tensor(start_tokens, device=device)
    
    # Move to correct device if needed
    if start_tokens.device != device:
        start_tokens = start_tokens.to(device)
    
    # Ensure tokens are within vocabulary size
    start_tokens = torch.clamp(start_tokens, 0, vocab_size-1)
    
    # Initialize sequence with start tokens
    sequence = start_tokens.clone()
    
    # Generate tokens one at a time
    for _ in range(max_length - len(start_tokens)):
        # Get the last context_window tokens or pad if needed
        if len(sequence) < context_window:
            # Pad with zeros if sequence is shorter than context_window
            context = torch.zeros(context_window, dtype=torch.long, device=device)
            context[-len(sequence):] = sequence
        else:
            # Take last context_window tokens
            context = sequence[-context_window:]
        
        # Clamp context values to be within vocab size
        context = torch.clamp(context, 0, vocab_size-1)
        
        # Get embeddings
        Embeddings = c[context]
        pos_expand = positions[:len(context)].clone()
        if len(pos_expand) < len(Embeddings):
            pos_expand = torch.zeros_like(Embeddings)
            pos_expand[:len(positions)] = positions[:len(positions)]
        Embeddings = Embeddings + pos_expand
        
        # Expand dimensions for batch processing
        Embeddings = Embeddings.unsqueeze(0)  # [1, seq_len, dim]
        
        if use_multi_head:
            # Process each attention head
            seq_len = Embeddings.size(1)
            multi_head_outputs = []
            
            for i in range(num_heads):
                # Get the weights for this head
                head_idx = i * 3
                QW_head = head_weights[head_idx]
                KW_head = head_weights[head_idx + 1]
                VW_head = head_weights[head_idx + 2]
                
                # Apply linear projections
                Q = Embeddings @ QW_head  # [1, seq_len, head_dim]
                K = Embeddings @ KW_head
                V = Embeddings @ VW_head
                
                # Scale dot-product attention
                scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(head_dim)
                
                # Apply causal mask
                mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
                scores = scores.masked_fill(mask == 0, float('-inf'))
                
                # Apply softmax
                attention = F.softmax(scores, dim=-1)
                
                # Apply attention to values
                head_output = torch.bmm(attention, V)  # [1, seq_len, head_dim]
                multi_head_outputs.append(head_output)
            
            # Concatenate heads and apply output projection
            multi_head_concat = torch.cat(multi_head_outputs, dim=-1)  # [1, seq_len, num_heads*head_dim]
            AttentionOutput = multi_head_concat @ W_O  # [1, seq_len, number_of_dimensions]
            
            # Add & Norm (residual connection)
            Add = AttentionOutput + Embeddings
        else:
            # Single-head attention
            Q = Embeddings @ QW.unsqueeze(0)
            K = Embeddings @ KW.unsqueeze(0)
            V = Embeddings @ VW.unsqueeze(0)
            KT = torch.transpose(K, -1, -2)
            
            d_k = Q.size(-1)
            scores = (Q @ KT) / math.sqrt(d_k)
            
            # Create attention mask (causal)
            mask = torch.tril(torch.ones(context_window, context_window, device=device))
            mask = mask.unsqueeze(0)  # Add batch dimension
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
            attention = F.softmax(scores, dim=-1)
            AttentionHead = attention @ V
            
            # Add & Norm
            Add = AttentionHead + Embeddings
        
        # Layer normalization
        Add = ln1(Add)
        
        # Feed Forward
        mean = Add.mean(dim=-1, keepdim=True)
        std = Add.std(dim=-1, keepdim=True)
        normalized = (Add - mean) / (std + 1e-6)
        
        # Flatten for the feed-forward network
        X_in = normalized.view(normalized.shape[0], -1)
        
        # Feed-forward network
        hidden = torch.tanh(X_in @ W1 + b1)
        logits = hidden @ W2 + b2
        
        # We only care about the last token's predictions
        next_token_logits = logits[0, :vocab_size] / temperature
        
        # Sample from the logits
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Ensure token is within vocabulary range
        next_token = torch.clamp(next_token, 0, vocab_size-1)
        
        # Append the new token to the sequence
        sequence = torch.cat([sequence, next_token])
        
    return sequence

def generate_text(model, data_processor, start_text=None, max_length=100, temperature=0.8, use_multi_head=False):
    """
    Generate text using the model.
    """
    if start_text:
        # Tokenize the start text
        start_tokens = data_processor.getTokens(text=start_text)
        # Ensure tokens are within vocab size
        start_tokens = [min(t, model.vocab_size-1) for t in start_tokens]
        start_tokens = torch.tensor(start_tokens)
    else:
        # Start with a simple token if no text provided
        start_tokens = torch.tensor([1])
    
    # Generate sequence of tokens
    generated_tokens = generate_sequence(
        model, 
        data_processor, 
        start_tokens=start_tokens, 
        max_length=max_length, 
        temperature=temperature,
        use_multi_head=use_multi_head
    )
    
    # Convert tokens to text
    generated_text = data_processor.decode(generated_tokens.tolist())
    
    return generated_text

if __name__ == "__main__":
    # Example usage
    import torch
    from data import Data
    from train import Train
    import os
    
    # Check if loss plot exists to confirm model was trained
    if not os.path.exists('training_loss.png'):
        print("Warning: No training_loss.png found. Has the model been trained?")
    
    # Initialize data processor
    data_processor = Data()
    
    # Set hyperparameters (should match those used in training)
    context_window = 10
    num_dimensions = 64
    vocab_size = 50257
    hidden_layer = 256
    
    # Initialize model
    model = Train()
    model.vocab_size = vocab_size
    model.number_of_dimensions = num_dimensions
    model.context_window = context_window
    model.hidden_layer = hidden_layer
    
    # Get parameters
    model.getParams()
    
    # Generate text
    start_text = "The transformer architecture"
    generated_text = generate_text(model, data_processor, start_text=start_text, 
                                  max_length=50, temperature=0.8)
    
    print(f"Start: {start_text}")
    print(f"Generated: {generated_text}") 