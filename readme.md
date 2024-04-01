# Deep Q-Network (DQN) Agent for CartPole Control

This code implements a Deep Q-Network (DQN) agent to learn and control the CartPole environment in OpenAI Gym. 

## Key Features:

- **Epsilon-greedy exploration:** Balances exploration and exploitation during training.
- **Experience Replay:** Utilizes a replay buffer to improve training efficiency.
- **Target Network:** Separates the training network (primary) from the target network, enhancing stability.
- **TensorFlow implementation:** Leverages TensorFlow for efficient deep learning operations.

## Hyperparameters:

- `REPLAY_MEMORY_SIZE`: Size of the experience replay buffer (default: 2000)
- `EPSILON`: Exploration rate (default: 1.0)
- `EPSILON_DECAY`: Decay rate for epsilon (default: 0.95)
- `EPSILON_DECAY_FREQ`: Frequency of epsilon decay updates (default: 10)
- `HIDDEN1_SIZE`, `HIDDEN2_SIZE`, `HIDDEN3_SIZE`: Sizes of hidden layers in the Q-network (default: 128)
- `EPISODES_NUM`: Number of training episodes (default: 50)
- `MAX_STEPS`: Maximum steps per episode (default: 200)
- `LEARNING_RATE`: Learning rate for the optimizer (default: 1e-4)
- `MINIBATCH_SIZE`: Size of the minibatch sampled from the replay buffer (default: 256)
- `DISCOUNT_FACTOR`: Discount factor (gamma) for future rewards (default: 0.9)
- `TARGET_UPDATE_FREQ`: Frequency of updating the target network (default: 10)
- `LOG_DIR`: Directory for logging data (default: './logs')
- `REPLAY`: Boolean flag to enable/disable experience replay (default: True)
- `EPS_MIN`: Minimum epsilon value (default: 1e-5)

## Code Structure:

1. **Dependency Installation:** Checks and installs required dependencies using `pip`.
2. **Environment Setup:**
   - Imports necessary libraries (TensorFlow, Gym, etc.)
   - Creates a DQN agent class.
3. **DQN Class:**
   - `__init__`: Initializes the environment, input/output sizes, and hyperparameters.
   - `initialize_network`: Defines the Q-network architecture with hidden layers.
   - `train`: Main training loop performing:
     - Action selection (epsilon-greedy)
     - Environment interaction (step, reward)
     - Experience replay buffer update
     - Minibatch sampling and Q-learning update
     - Target network update
   - `playPolicy`: Evaluates the learned policy by playing the game for a certain number of steps.
   - `closeenv`: Closes the environment after training.

## Running the Script:

1. Save the code as `dqn_cartpole.py`.
2. Install required libraries (`pip install gym tensorflow numpy matplotlib`).
3. Run the script: `python dqn_cartpole.py`.

## Expected Output:

- Training progress will be displayed with episode number, episode length, and global step.
- After training, the script will showcase the agent's performance by playing the game for several episodes.

## Further Enhancements:

- Implement reward shaping for faster learning.
- Explore different network architectures and hyperparameter tuning.
- Visualize training progress using TensorBoard.
