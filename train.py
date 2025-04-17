import torch
import torch.nn.functional as F
import math
from parameters import Tuner
import matplotlib.pyplot as plt


class Train:
    # number_of_dimensions = 300
    # vocab_size = 5001
    # context_window = 300
    batch_size = 32
    num_epochs = 100
    hidden_layer = 7000
    num_heads = 2  # Number of attention heads

    def getParams(self):
        self.parameters =  Tuner.tune(self,vocab_size = self.vocab_size, number_of_dimensions = self.number_of_dimensions,context_window = self.context_window, hidden_layer = self.hidden_layer)
        self.c, self.positions, self.QW, self.KW, self.VW, self.W1, self.b1, self.W2, self.b2, self.ln1, self.ln2 = self.parameters
        print("Parameters Loaded")
    
    def getParamsMultiHead(self):
        # Initialize parameters for multi-head attention
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        g = torch.Generator(device=device)
        g.manual_seed(0)
        
        # Calculate head dimension
        self.head_dim = self.number_of_dimensions // self.num_heads
        
        # Initialize weights with Xavier/Glorot
        def init_weights(m):
            if isinstance(m, torch.Tensor) and m.dim() > 1:
                torch.nn.init.xavier_uniform_(m)
        
        # Create model parameters for each head
        self.head_weights = []
        for _ in range(self.num_heads):
            # Create Q, K, V weights for each head
            QW = torch.randn(self.number_of_dimensions, self.head_dim, generator=g, requires_grad=True, device=device)
            KW = torch.randn(self.number_of_dimensions, self.head_dim, generator=g, requires_grad=True, device=device)
            VW = torch.randn(self.number_of_dimensions, self.head_dim, generator=g, requires_grad=True, device=device)
            self.head_weights.extend([QW, KW, VW])
        
        # Create output projection weights
        self.W_O = torch.randn(self.num_heads * self.head_dim, self.number_of_dimensions, generator=g, requires_grad=True, device=device)
        
        # Create other model parameters
        self.c = torch.randn([self.vocab_size, self.number_of_dimensions], generator=g, requires_grad=True, device=device)
        self.positions = torch.rand(self.context_window, self.number_of_dimensions, generator=g, requires_grad=True, device=device)
        self.W1 = torch.randn(self.number_of_dimensions * self.context_window, self.hidden_layer, requires_grad=True, device=device)
        self.b1 = torch.randn(self.hidden_layer, requires_grad=True, device=device)
        self.W2 = torch.randn(self.hidden_layer, self.vocab_size, requires_grad=True, device=device)
        self.b2 = torch.randn(self.vocab_size, requires_grad=True, device=device)
        
        # Apply Xavier initialization
        for p in [self.c, self.W_O, self.W1, self.W2] + self.head_weights:
            init_weights(p)
        
        # Layer normalization parameters
        self.ln1 = torch.nn.LayerNorm(self.number_of_dimensions, device=device)
        self.ln2 = torch.nn.LayerNorm(self.number_of_dimensions, device=device)
        
        # Combine all parameters
        self.parameters = [self.c, self.positions, self.W_O, self.W1, self.b1, self.W2, self.b2] + self.head_weights
        
        print("Multi-head Parameters Loaded")
        return self.parameters

    def train(self, x_train, y_train):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        g = torch.Generator(device=device)
        g.manual_seed(0)
        
        # Create parameters list excluding LayerNorm modules
        optimizer_params = [self.c, self.positions, self.QW, self.KW, self.VW, self.W1, self.b1, self.W2, self.b2]
        # Higher learning rate for faster convergence
        optimizer = torch.optim.AdamW(optimizer_params, lr=1e-3)
        
        # Initialize losses list to collect loss values
        losses = []

        for epoch in range(self.num_epochs):

            batch_indices = torch.randint(0, x_train.shape[0], (self.batch_size,), generator=g, device=device)
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Forward pass
            Embeddings = self.c[x_batch]
            pos_expand = self.positions.expand(Embeddings.shape[0], -1, -1)
            Embeddings = pos_expand + Embeddings

            # Attention
            Q = Embeddings @ self.QW.unsqueeze(0)
            K = Embeddings @ self.KW.unsqueeze(0)
            V = Embeddings @ self.VW.unsqueeze(0)
            KT = torch.transpose(K, -1, -2)

            d_k = Q.size(-1)
            scores = (Q @ KT) / math.sqrt(d_k)

            mask = torch.tril(torch.ones(self.context_window, self.context_window, device=device))
            mask = mask.unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))

            attention = F.softmax(scores, dim=-1)
            AttentionHead = attention @ V

            # Add & Norm
            Add = AttentionHead + Embeddings
            Add = self.ln1(Add)  # Layer normalization

            # Feed Forward
            mean = Add.mean(dim=-1, keepdim=True)
            std = Add.std(dim=-1, keepdim=True)
            normalized = (Add - mean) / (std + 1e-6)

            X_in = normalized.view(normalized.shape[0], -1)
            hidden = torch.tanh(X_in @ self.W1 + self.b1)
            logits = hidden @ self.W2 + self.b2

            # Compute loss
            loss = F.cross_entropy(logits, y_batch)
            
            # Store loss
            losses.append(loss.item())

            # Backward pass with stronger gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm=1.0)
            optimizer.step()

            if epoch % 5 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
        
        # Plot losses
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('training_loss.png')
        print(f"Training complete. Loss plot saved to training_loss.png")
        
        # Store losses for later analysis
        self.losses = losses
        
        return optimizer_params
    
    def train_multi_head(self, x_train, y_train):
        """Train using multi-head attention"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        g = torch.Generator(device=device)
        g.manual_seed(0)
        
        # Create parameters list excluding LayerNorm modules
        optimizer_params = [self.c, self.positions, self.W_O, self.W1, self.b1, self.W2, self.b2] + self.head_weights
        # Higher learning rate for faster convergence
        optimizer = torch.optim.AdamW(optimizer_params, lr=1e-3)
        
        # Initialize losses list to collect loss values
        losses = []

        for epoch in range(self.num_epochs):
            batch_indices = torch.randint(0, x_train.shape[0], (self.batch_size,), generator=g, device=device)
            x_batch = x_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Forward pass
            Embeddings = self.c[x_batch]  # [batch_size, context_window, number_of_dimensions]
            pos_expand = self.positions.expand(Embeddings.shape[0], -1, -1)
            Embeddings = pos_expand + Embeddings

            # Process each attention head
            batch_size, seq_len, _ = Embeddings.shape
            multi_head_outputs = []
            
            for i in range(self.num_heads):
                # Get the weights for this head
                head_idx = i * 3
                QW_head = self.head_weights[head_idx]
                KW_head = self.head_weights[head_idx + 1]
                VW_head = self.head_weights[head_idx + 2]
                
                # Apply linear projections
                Q = Embeddings @ QW_head  # [batch_size, seq_len, head_dim]
                K = Embeddings @ KW_head
                V = Embeddings @ VW_head
                
                # Scale dot-product attention
                scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.head_dim)
                
                # Apply causal mask
                mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
                scores = scores.masked_fill(mask == 0, float('-inf'))
                
                # Apply softmax
                attention = F.softmax(scores, dim=-1)
                
                # Apply attention to values
                head_output = torch.bmm(attention, V)  # [batch_size, seq_len, head_dim]
                multi_head_outputs.append(head_output)
            
            # Concatenate heads and apply output projection
            multi_head_concat = torch.cat(multi_head_outputs, dim=-1)  # [batch_size, seq_len, num_heads*head_dim]
            AttentionOutput = multi_head_concat @ self.W_O  # [batch_size, seq_len, number_of_dimensions]
            
            # Add & Norm (residual connection)
            Add = AttentionOutput + Embeddings
            Add = self.ln1(Add)  # Layer normalization
            
            # Feed Forward Network
            # Reshape for the feed-forward network
            X_in = Add.view(batch_size, -1)  # [batch_size, seq_len*number_of_dimensions]
            hidden = torch.tanh(X_in @ self.W1 + self.b1)
            logits = hidden @ self.W2 + self.b2
            
            # Compute loss
            loss = F.cross_entropy(logits, y_batch)
            
            # Store loss
            losses.append(loss.item())
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm=1.0)
            optimizer.step()
            
            if epoch % 5 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
        
        # Plot losses
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Multi-Head Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('multi_head_training_loss.png')
        print(f"Multi-head training complete. Loss plot saved to multi_head_training_loss.png")
        
        # Store losses for later analysis
        self.losses = losses
        
        return optimizer_params
    
    def test(self,x_test, y_test):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        g = torch.Generator(device=device)
        g.manual_seed(0)

        Embeddings = self.c[x_test]
        pos_expand = self.positions.expand(Embeddings.shape[0], -1, -1)
        Embeddings = pos_expand + Embeddings

        # Attention
        Q = Embeddings @ self.QW.unsqueeze(0)
        K = Embeddings @ self.KW.unsqueeze(0)
        V = Embeddings @ self.VW.unsqueeze(0)
        KT = torch.transpose(K, -1, -2)

        d_k = Q.size(-1)
        scores = (Q @ KT) / math.sqrt(d_k)

        mask = torch.tril(torch.ones(self.context_window, self.context_window, device=device))
        mask = mask.unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)
        AttentionHead = attention @ V

        # Add & Norm
        Add = AttentionHead + Embeddings
        Add = self.ln1(Add)  # Layer normalization

        # Feed Forward
        mean = Add.mean(dim=-1, keepdim=True)
        std = Add.std(dim=-1, keepdim=True)
        normalized = (Add - mean) / (std + 1e-6)

        X_in = normalized.view(normalized.shape[0], -1)
        hidden = torch.tanh(X_in @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2

        # Compute loss
        loss = F.cross_entropy(logits, y_test)
        return loss
    
    def test_multi_head(self, x_test, y_test):
        """Test using multi-head attention"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Forward pass
        Embeddings = self.c[x_test]  # [batch_size, context_window, number_of_dimensions]
        pos_expand = self.positions.expand(Embeddings.shape[0], -1, -1)
        Embeddings = pos_expand + Embeddings
        
        # Process each attention head
        batch_size, seq_len, _ = Embeddings.shape
        multi_head_outputs = []
        
        for i in range(self.num_heads):
            # Get the weights for this head
            head_idx = i * 3
            QW_head = self.head_weights[head_idx]
            KW_head = self.head_weights[head_idx + 1]
            VW_head = self.head_weights[head_idx + 2]
            
            # Apply linear projections
            Q = Embeddings @ QW_head  # [batch_size, seq_len, head_dim]
            K = Embeddings @ KW_head
            V = Embeddings @ VW_head
            
            # Scale dot-product attention
            scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.head_dim)
            
            # Apply causal mask
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
            # Apply softmax
            attention = F.softmax(scores, dim=-1)
            
            # Apply attention to values
            head_output = torch.bmm(attention, V)  # [batch_size, seq_len, head_dim]
            multi_head_outputs.append(head_output)
        
        # Concatenate heads and apply output projection
        multi_head_concat = torch.cat(multi_head_outputs, dim=-1)  # [batch_size, seq_len, num_heads*head_dim]
        AttentionOutput = multi_head_concat @ self.W_O  # [batch_size, seq_len, number_of_dimensions]
        
        # Add & Norm (residual connection)
        Add = AttentionOutput + Embeddings
        Add = self.ln1(Add)  # Layer normalization
        
        # Feed Forward Network
        # Reshape for the feed-forward network
        X_in = Add.view(batch_size, -1)  # [batch_size, seq_len*number_of_dimensions]
        hidden = torch.tanh(X_in @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2
        
        # Compute loss
        loss = F.cross_entropy(logits, y_test)
        return loss