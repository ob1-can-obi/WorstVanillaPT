import torch
from data import Data
from train import Train
from generate import generate_text
import os

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    print("Initializing model...")
    model = Train()
    model.vocab_size = vocab_size
    model.number_of_dimensions = num_dimensions
    model.context_window = context_window
    model.hidden_layer = hidden_layer
    
    # Get parameters
    model.getParams()
    
    # Move parameters to device
    for i in range(len(model.parameters)):
        if isinstance(model.parameters[i], torch.Tensor):
            model.parameters[i] = model.parameters[i].to(device)
    
    # Try different prompts with different temperatures
    prompts = [
        "The transformer architecture",
        "Attention is all you need",
        "ChatGPT is a",
        "Venice is known for"
    ]
    
    temperatures = [0.7, 0.8, 1.0, 1.2]
    
    print("\n" + "="*50)
    print("GENERATING TEXT SAMPLES")
    print("="*50 + "\n")
    
    for temp in temperatures:
        print(f"\nTemperature: {temp:.1f}")
        print("-" * 30)
        
        for prompt in prompts:
            # Generate text with current prompt and temperature
            generated_text = generate_text(model, data_processor, 
                                         start_text=prompt, 
                                         max_length=50, 
                                         temperature=temp)
            
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated_text}")
            print("-" * 30)

if __name__ == "__main__":
    main() 