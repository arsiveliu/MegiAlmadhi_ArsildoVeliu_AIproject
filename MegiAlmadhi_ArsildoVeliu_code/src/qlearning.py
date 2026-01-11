"""
Q-Learning Agent for Campus Navigation

This implements Q-Learning for path planning. The agent learns
through experience and improves over time.

Reference implementations:
- tayfunkscu/QLearning-path-planning
- zerosansan/dqn_qlearning_sarsa_mobile_robot_navigation
"""
import numpy as np
import random
import pickle
from typing import Tuple, List, Optional
from campus_environment import CampusEnvironment


class QLearningAgent:
    """
    Q-Learning agent that learns to navigate the campus environment.
    
    Q-Learning builds a table (Q-table) that stores values for each
    (state, action) pair. The agent uses this to decide what action to take.
    
    Update formula (Bellman equation):
    Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
    
    Where:
    s = current state (position)
    a = action taken
    r = reward received
    s' = next state
    α = learning rate (how much to update)
    γ = discount factor (importance of future rewards)
    
    The agent balances exploration (trying new actions) and exploitation
    (using known good actions).
    """
    
    # Action space: [up, down, left, right]
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ACTION_NAMES = ['Up', 'Down', 'Left', 'Right']
    
    def __init__(self, environment: CampusEnvironment,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Set up the agent with all the parameters.
        
        learning_rate (α): how much to update Q-values (0.1 is pretty standard)
        discount_factor (γ): how much future rewards matter (0.99 = cares about long-term)
        epsilon (ε): exploration rate - 1.0 means fully random at start
        epsilon_decay: reduce epsilon each episode so it gets less random
        epsilon_min: don't go below this (keep some randomness)
        """
        self.env = environment
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: rows = states, columns = actions, values = expected rewards
        num_states = self.env.get_num_states()
        num_actions = len(self.ACTIONS)
        self.q_table = np.zeros((num_states, num_actions))
        
        # keep track of training progress
        self.episode_rewards = []    # Total reward per episode
        self.episode_lengths = []    # Number of steps per episode
        self.training_losses = []    # TD-error magnitude per episode
        
    def get_state(self, position: Tuple[int, int]) -> int:
        """Convert position to state index."""
        return self.env.get_state_representation(position)
    
    def get_position(self, state: int) -> Tuple[int, int]:
        """Convert state index to position."""
        return self.env.get_position_from_state(state)
    
    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        With probability epsilon: choose random action (exploration)
        With probability 1-epsilon: choose action with highest Q-value (exploitation)
        
        This balances trying new actions versus using known good ones.
        """
        if training and random.random() < self.epsilon:
            # exploration - try a random action
            return random.randint(0, len(self.ACTIONS) - 1)
        else:
            # exploitation - use best known action
            return np.argmax(self.q_table[state])
    
    def get_valid_actions(self, position: Tuple[int, int]) -> List[int]:
        """
        Get list of valid actions from the current position.
        
        Valid actions don't move outside bounds or into static obstacles.
        Congested areas are allowed during training (agent learns to avoid them).
        """
        valid_actions = []
        for i, action in enumerate(self.ACTIONS):
            next_pos = (position[0] + action[0], position[1] + action[1])
            # Allow congestion during training (agent learns to avoid it)
            if self.env.is_walkable(next_pos, allow_congestion=True):
                valid_actions.append(i)
        return valid_actions if valid_actions else [0]  # Default to first action if stuck
    
    def take_action(self, position: Tuple[int, int], action_idx: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Execute the action and calculate the resulting state, reward, and termination.
        
        Reward structure (tuned through experimentation):
        - Hit edge of grid: -10
        - Hit obstacle: -50 (large penalty)
        - Go through congestion: -5 (discouraged but allowed)
        - Reach goal: +100 (success)
        - Move closer to goal: -0.5 (small penalty to encourage speed)
        - Move away: -1.0 (larger penalty)
        
        Negative rewards for normal moves encourage the agent to find shorter paths.
        """
        action = self.ACTIONS[action_idx]
        next_pos = (position[0] + action[0], position[1] + action[1])
        
        # check if we're trying to move outside the grid
        if not self.env.is_valid_position(next_pos):
            return position, -10.0, False
        
        cell_value = self.env.grid[next_pos[0]][next_pos[1]]
        
        # calculate the reward based on cell type
        if cell_value == 1:  # static obstacle
            return position, -50.0, False
        
        elif cell_value == 2:  # congestion
            reward = -5.0
            done = False
        
        elif next_pos == self.env.goal:  # reached goal
            reward = 100.0
            done = True
        
        else:  # normal free space
            # reward based on progress toward goal
            distance_to_goal = self.env.euclidean_distance(next_pos, self.env.goal)
            old_distance = self.env.euclidean_distance(position, self.env.goal)
            
            # Reward structure encourages moving toward the goal
            if distance_to_goal < old_distance:
                reward = -0.5  # Moving in right direction (still negative to encourage speed)
            else:
                reward = -1.0  # Moving away from goal (higher penalty)
            
            done = False
        
        return next_pos, reward, done
    
    def update_q_value(self, state: int, action: int, reward: float, next_state: int):
        """
        Update Q-value using the Q-learning update rule (Bellman equation).
        
        This is the core of the Q-Learning algorithm. The update rule is:
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        
        Breaking down the formula:
        - Q(s,a): current Q-value for this state-action pair
        - α (alpha): learning rate - how much to update
        - r: immediate reward received
        - γ (gamma): discount factor - importance of future rewards
        - max Q(s',a'): best possible future Q-value from next state
        - [r + γ max Q(s',a') - Q(s,a)]: TD-error (temporal difference)
        
        The TD-error represents how much better/worse the action was
        compared to our previous estimate.
        
        Args:
            state: Current state index
            action: Action taken
            reward: Immediate reward received
            next_state: Resulting state after action
            
        Returns:
            Absolute TD-error (for monitoring convergence)
        """
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])  # Best possible Q-value from next state
        
        # Apply Q-learning update rule (Bellman equation)
        # TD target = r + γ * max Q(s',a')
        # TD error = TD target - current Q-value
        # New Q-value = old Q-value + α * TD error
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        # Track the magnitude of the update for monitoring convergence
        # Large values mean agent is still learning; small values mean convergence
        loss = abs(reward + self.gamma * max_next_q - current_q)
        return loss
    
    def train_episode(self, max_steps: int = 200) -> Tuple[float, int, bool]:
        """
        Train agent for one episode.
        
        Args:
            max_steps: Maximum steps per episode
            
        Returns:
            Tuple of (total_reward, steps_taken, success)
        """
        # Reset environment
        position = self.env.reset()
        
        # Randomly add dynamic obstacles
        self.env.add_dynamic_obstacles(count=random.randint(3, 8))
        
        total_reward = 0.0
        episode_loss = 0.0
        
        for step in range(max_steps):
            # Get current state
            state = self.get_state(position)
            
            # Select action
            action = self.select_action(state, training=True)
            
            # Take action
            next_position, reward, done = self.take_action(position, action)
            next_state = self.get_state(next_position)
            
            # Update Q-table
            loss = self.update_q_value(state, action, reward, next_state)
            episode_loss += loss
            
            # Update statistics
            total_reward += reward
            position = next_position
            
            if done:
                # Reached goal
                return total_reward, step + 1, True
        
        # Max steps reached without finding goal
        self.training_losses.append(episode_loss / max_steps)
        return total_reward, max_steps, False
    
    def train(self, num_episodes: int = 1000, max_steps: int = 200, 
              verbose: bool = True, save_freq: int = 1000):
        """
        Train the agent for multiple episodes using Q-Learning.
        
        Training Process:
        1. For each episode, start from a random valid position
        2. Add random dynamic obstacles to vary the environment
        3. Agent takes actions using epsilon-greedy policy
        4. Q-table is updated after each action using Bellman equation
        5. Epsilon gradually decays to shift from exploration to exploitation
        6. Track statistics for monitoring learning progress
        
        Args:
            num_episodes: Total number of training episodes to run
            max_steps: Maximum steps allowed per episode before timeout
            verbose: If True, prints progress updates every 100 episodes
            save_freq: Save checkpoint every N episodes (currently unused)
        """
        success_count = 0
        
        for episode in range(num_episodes):
            # Start from a random position for each episode
            # This ensures agent learns to navigate from anywhere, not just one start point
            position = (random.randint(0, self.env.height - 1), 
                       random.randint(0, self.env.width - 1))
            
            # Ensure start position is valid (not on obstacle or goal)
            while not self.env.is_walkable(position) or position == self.env.goal:
                position = (random.randint(0, self.env.height - 1), 
                           random.randint(0, self.env.width - 1))
            
            # Reset environment and add random dynamic obstacles
            # This creates variability so agent learns robust navigation
            self.env.reset()
            self.env.add_dynamic_obstacles(count=random.randint(3, 8))
            
            # Initialize episode tracking variables
            total_reward = 0.0
            episode_loss = 0.0
            steps = 0
            done = False
            
            # Training loop for this episode
            while not done and steps < max_steps:
                # Convert position to state index for Q-table lookup
                state = self.get_state(position)
                
                # Epsilon-greedy action selection
                if random.random() < self.epsilon:
                    # Exploration: try a random valid action to discover new strategies
                    valid_actions = self.get_valid_actions(position)
                    action = random.choice(valid_actions)
                else:
                    # Exploitation: use the best action according to current Q-values
                    action = np.argmax(self.q_table[state])
                
                # Execute the chosen action in the environment
                next_position, reward, done = self.take_action(position, action)
                next_state = self.get_state(next_position)
                
                # Update Q-table using the Q-learning update rule
                # This is where the learning actually happens
                current_q = self.q_table[state][action]
                max_next_q = np.max(self.q_table[next_state])
                
                # Temporal Difference (TD) Error: difference between expected and actual
                # TD Error = r + γ * max(Q(s',a')) - Q(s,a)
                td_error = reward + self.gamma * max_next_q - current_q
                
                # Update Q-value: Q(s,a) ← Q(s,a) + α * TD_error
                self.q_table[state][action] = current_q + self.alpha * td_error
                
                # Accumulate statistics for this episode
                total_reward += reward
                episode_loss += abs(td_error)
                position = next_position
                steps += 1
            
            # Store episode statistics for analysis
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            self.training_losses.append(episode_loss / max(steps, 1))
            
            if done:
                success_count += 1
            
            # Gradually decay epsilon to shift from exploration to exploitation
            # Early training: high epsilon (more exploration)
            # Later training: low epsilon (more exploitation of learned policy)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Print training progress periodically
            if verbose and (episode + 1) % 100 == 0:
                # Calculate recent performance metrics using sliding window
                recent_episodes = self.episode_lengths[-100:]
                recent_success_rate = sum([1 for length in recent_episodes 
                                          if length < max_steps]) / len(recent_episodes) * 100
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_steps = np.mean(recent_episodes)
                
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"Success Rate: {recent_success_rate:.1f}% | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Steps: {avg_steps:.1f}")
        
        # Final statistics
        overall_success_rate = (success_count / num_episodes) * 100
        print(f"\nTraining Complete!")
        print(f"Overall Success Rate: {overall_success_rate:.2f}%")
        print(f"Final Epsilon: {self.epsilon:.4f}")
    
    def get_policy_path(self, start: Optional[Tuple[int, int]] = None,
                       goal: Optional[Tuple[int, int]] = None,
                       max_steps: int = 500) -> Tuple[List[Tuple[int, int]], float]:
        """
        Generate a path using the learned Q-table (greedy policy with loop detection).
        
        After training, this method uses the learned Q-values to navigate from
        start to goal by always choosing the action with the highest Q-value.
        
        Loop Detection:
        The method tracks recent positions to detect if the agent gets stuck
        in a loop. If detected, it tries alternative high-Q-value actions to escape.
        
        Args:
            start: Starting position (uses environment's start if None)
            goal: Goal position (uses environment's goal if None)
            max_steps: Maximum steps before giving up
            
        Returns:
            Tuple of (path, total_cost)
            - path: List of (row, col) positions from start to goal
            - total_cost: Cumulative cost of the path
        """
        position = start or self.env.start
        goal = goal or self.env.goal
        
        path = [position]
        total_cost = 0.0
        recent_positions = []  # Track recent 5 positions to detect loops
        stuck_counter = 0
        
        for step in range(max_steps):
            if position == goal:
                break
            
            state = self.get_state(position)
            
            # Get valid actions for current position
            valid_actions = self.get_valid_actions(position)
            
            if not valid_actions:
                # No valid moves, agent is stuck
                break
            
            # Detect if we're in a loop (position appears in recent history)
            in_recent_loop = recent_positions.count(position) >= 2
            
            # Select action: if stuck in loop, try second-best or random action
            if in_recent_loop and stuck_counter < 10:
                # Try to escape loop by taking a different action
                q_values = self.q_table[state][valid_actions]
                
                # Sort actions by Q-value and try second/third best
                sorted_indices = np.argsort(q_values)[::-1]
                
                # Pick a random action from top 3 (or all if less than 3)
                top_k = min(3, len(sorted_indices))
                random_idx = random.randint(0, top_k - 1)
                best_valid_action_idx = valid_actions[sorted_indices[random_idx]]
                stuck_counter += 1
            else:
                # Normal greedy selection
                q_values = self.q_table[state][valid_actions]
                best_valid_action_idx = valid_actions[np.argmax(q_values)]
                stuck_counter = 0
            
            next_position, cost, done = self.take_action(position, best_valid_action_idx)
            
            # Don't add if we're staying in the same position (hit wall)
            if next_position != position:
                path.append(next_position)
                recent_positions.append(next_position)
                
                # Keep only recent 8 positions for loop detection
                if len(recent_positions) > 8:
                    recent_positions.pop(0)
            else:
                # Hit a wall, try different action
                stuck_counter += 1
                if stuck_counter > 5:
                    # Truly stuck, give up
                    break
            
            total_cost += abs(cost)
            position = next_position
            
            if done:
                break
        
        return path, total_cost
    
    def save_model(self, filepath: str):
        """Save Q-table and parameters to file."""
        model_data = {
            'q_table': self.q_table,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_losses': self.training_losses
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load Q-table and parameters from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = model_data['q_table']
        self.alpha = model_data['alpha']
        self.gamma = model_data['gamma']
        self.epsilon = model_data.get('epsilon', self.epsilon_min)
        self.episode_rewards = model_data.get('episode_rewards', [])
        self.episode_lengths = model_data.get('episode_lengths', [])
        self.training_losses = model_data.get('training_losses', [])
        
        print(f"Model loaded from {filepath}")
        print(f"Trained episodes: {len(self.episode_rewards)}")


# Example usage
if __name__ == "__main__":
    # Create environment
    env = CampusEnvironment(width=10, height=10, obstacle_prob=0.15)
    print("Campus Environment:")
    print(env)
    
    # Create Q-Learning agent
    agent = QLearningAgent(
        environment=env,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    print("\nTraining Q-Learning agent...")
    agent.train(num_episodes=500, max_steps=100, verbose=True)
    
    # Test learned policy
    print("\n--- Testing Learned Policy ---")
    env.add_dynamic_obstacles(count=5)
    path, cost = agent.get_policy_path()
    
    print(f"Path length: {len(path)}")
    print(f"Total cost: {cost:.2f}")
    print(f"Path: {' -> '.join([str(p) for p in path[:10]])}...")
