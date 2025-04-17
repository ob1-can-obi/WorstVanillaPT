import torch
import math
import torch.nn.functional as F


class Tuner:
    def tune(self, number_of_dimensions,vocab_size, context_window, batch_size, num_epochs, hidden_layer):
# Hyperparameters
        number_of_dimensions = number_of_dimensions
        vocab_size = vocab_size
        context_window = context_window
        batch_size = batch_size
        num_epochs = num_epochs
        hidden_layer  = hidden_layer

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        g = torch.Generator(device=device)
        g.manual_seed(0)

        # Initialize weights with Xavier/Glorot
        def init_weights(m):
            if isinstance(m, torch.Tensor) and m.dim() > 1:
                torch.nn.init.xavier_uniform_(m)

        # Create model parameters
        c = torch.randn([vocab_size, number_of_dimensions], generator=g, requires_grad=True, device=device)
        QW = torch.randn(number_of_dimensions, number_of_dimensions, generator=g, requires_grad=True, device=device)
        KW = torch.randn(number_of_dimensions, number_of_dimensions, generator=g, requires_grad=True, device=device)
        VW = torch.randn(number_of_dimensions, number_of_dimensions, generator=g, requires_grad=True, device=device)
        positions = torch.rand(context_window, number_of_dimensions, generator=g, requires_grad=True, device=device)
        W1 = torch.randn(number_of_dimensions * context_window, hidden_layer, requires_grad=True, device=device)
        b1 = torch.randn(hidden_layer, requires_grad=True, device=device)
        W2 = torch.randn(hidden_layer, vocab_size, requires_grad=True, device=device)
        b2 = torch.randn(vocab_size, requires_grad=True, device=device)

        # Apply Xavier initialization
        for p in [c, QW, KW, VW, W1, W2]:
            init_weights(p)


        # Layer normalization parameters
        ln1 = torch.nn.LayerNorm(number_of_dimensions, device=device)
        ln2 = torch.nn.LayerNorm(number_of_dimensions, device=device)

        parameters = [c, positions, QW, KW, VW, W1, b1, W2, b2, ln1, ln2]

        return parameters



