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
    if not os.path.exists('multi_head_training_loss.png'):
        print("Warning: No multi_head_training_loss.png found. Has the multi-head model been trained?")
    
    # Initialize data processor
    data_processor = Data()
    
    # Set hyperparameters (matching those in run_train_multi_head.py)
    context_window = 50
    num_dimensions = 256
    vocab_size = 50257
    hidden_layer = 5000
    num_heads = 4
    
    # Initialize model
    print("Initializing model...")
    model = Train()
    model.vocab_size = vocab_size
    model.number_of_dimensions = num_dimensions
    model.context_window = context_window
    model.hidden_layer = hidden_layer
    model.num_heads = num_heads
    
    # Get parameters for multi-head attention
    model.getParamsMultiHead()
    
    # Move parameters to device
    for i in range(len(model.parameters)):
        if isinstance(model.parameters[i], torch.Tensor):
            model.parameters[i] = model.parameters[i].to(device)
    
    # Try different prompts with different temperatures
    prompts = [
        "The transformer architecture",
        "Attention is all you need",
        "ChatGPT is a",
        "Venice is known for",
        "Deep learning techniques",
        "The future of AI will"
    ]
    
    temperatures = [0.7, 0.8, 1.0, 1.2]
    
    print("\n" + "="*50)
    print("GENERATING TEXT SAMPLES WITH MULTI-HEAD ATTENTION")
    print("="*50 + "\n")
    
    for temp in temperatures:
        print(f"\nTemperature: {temp:.1f}")
        print("-" * 30)
        
        for prompt in prompts:
            # Generate text with current prompt and temperature using multi-head attention
            generated_text = generate_text(
                model, 
                data_processor, 
                start_text=prompt, 
                max_length=150,  # Increased max length for generation
                temperature=temp,
                use_multi_head=True
            )
            
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated_text}")
            print("-" * 30)

if __name__ == "__main__":
    main() 