import torch
import numpy as np
from dqn import DQN  # Import your DQN model class

def load_model(model_path, state_size, action_size):
    """Load a DQN model from a specified path."""
    # Create an instance of the DQN model
    model = DQN(state_size, action_size)
    
    # Load the state dictionary from the specified file
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

if __name__ == "__main__":
    # Example parameters, modify as needed
    state_size = 4  # Number of state features (Open, High, Low, Close)
    action_size = 3  # Number of actions (Buy, Sell, Hold)
    
    model_path = 'dqn_model.pth'  # Path to your saved model
    loaded_agent = load_model(model_path, state_size, action_size)
    
    print("Model loaded successfully!")

    # Example state for prediction
    state = np.array([238.206, 238.206, 238.206, 238.206], dtype=np.float32)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Shape: (1, state_size)

    # Predict the action
    with torch.no_grad():  # No gradient calculation needed
        action_values = loaded_agent(state_tensor)  # Get action values
        action = torch.argmax(action_values[0]).item()  # Get the action with the highest Q-value

    # Action mapping for clarity
    action_map = {0: "Buy", 1: "Sell", 2: "Hold"}
    print(f"Predicted action: {action_map[action]}")
