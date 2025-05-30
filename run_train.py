import torch
from data import Data
from train import Train

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set hyperparameters
context_window = 10  # Smaller context window
num_dimensions = 64  # Smaller embeddings
vocab_size = 50257   # Default for tiktoken
hidden_layer = 256   # Smaller hidden layer
num_epochs = 100     # More epochs for better convergence

# Initialize data processor
data_processor = Data()

# Load text from file
print("Loading ChatGPTWiki.txt...")
text_file = "ChatGPTWiki.txt"

# Tokenize the text from file
print("Tokenizing text...")
tokens = data_processor.getTokens(text_file=text_file)
print(f"Number of tokens: {len(tokens)}")

# Ensure all tokens are within vocabulary size
max_token = max(tokens)
if max_token >= vocab_size:
    print(f"Warning: Found token {max_token} which is outside vocabulary size {vocab_size}")
    print("Clipping tokens to vocabulary size...")
    tokens = [min(t, vocab_size-1) for t in tokens]

# Create training data
print("Creating training data...")
x_train, y_train = data_processor.getTrainData(tokens, context_window)
print(f"Training data shape: {x_train.shape}, {y_train.shape}")

# Move data to device
x_train = x_train.to(device)
y_train = y_train.to(device)

# Create a simple test split
split_idx = int(len(x_train) * 0.8)
x_test = x_train[split_idx:]
y_test = y_train[split_idx:]
x_train = x_train[:split_idx]
y_train = y_train[:split_idx]
print(f"Train/test split: {len(x_train)}/{len(x_test)}")

# Initialize model
print("Initializing model...")
model = Train()
model.vocab_size = vocab_size
model.number_of_dimensions = num_dimensions
model.context_window = context_window
model.hidden_layer = hidden_layer
model.num_epochs = num_epochs
model.batch_size = 16  # Smaller batch size

# Get model parameters
model.getParams()

# Train model
print("Starting training...")
model.train(x_train, y_train)

# Test model
test_loss = model.test(x_test, y_test)
print(f"Test loss: {test_loss:.4f}")

print("Training complete!") 