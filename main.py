import torch
import numpy as np
from dqn import DQN  # Import your DQN model class

def load_model(model_path, state_size, action_size):
    """Load a DQN model from a specified path."""
    model = DQN(state_size, action_size)
    try:
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    model.eval()
    return model

def predict_action(model, state):
    """Predict the action given a state."""
    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Shape: (1, state_size)
    with torch.no_grad():  # No gradient calculation needed
        action_values = model(state_tensor)  # Get action values
        action = torch.argmax(action_values[0]).item()  # Get the action with the highest Q-value
    return action

if __name__ == "__main__":
    # Example parameters
    state_size = 17  # Ensure this matches your model
    action_size = 3  # Number of actions (Buy, Sell, Hold)
    
    model_path = 'dqn_model.pth'  # Path to your saved model
    loaded_agent = load_model(model_path, state_size, action_size)

    # Example state for prediction (ensure the state has `state_size` features)
    # Replace this with your actual state representation
    example_state = np.random.rand(state_size).astype(np.float32)  # Random state for demonstration

    # Predict the action
    action = predict_action(loaded_agent, example_state)

    # Action mapping for clarity
    action_map = {0: "Buy", 1: "Sell", 2: "Hold"}
    print(f"Predicted action: {action_map[action]}")
