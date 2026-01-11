"""
Training Script for Campus Navigation Agent

This script trains our Q-Learning agent to navigate a campus grid and
compares its performance against A* search.

Q-Learning allows the agent to learn through experience - it explores
the environment, receives rewards for good actions and penalties for bad ones,
then updates its Q-table with what it learned.

We're using a grid because it makes the problem manageable - instead of
infinite possibilities, we have a finite number of states.

Run with: python train_agent.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

from campus_environment import CampusEnvironment
from qlearning import QLearningAgent
from astar import AStarPlanner



def plot_training_curves(agent, save_path):
    """
    Make graphs showing how the training went.
    
    Creates 4 plots:
    1. Episode rewards - did the agent get better rewards over time?
    2. Steps to goal - is it finding shorter paths as it learns?
    3. Success rate - what percentage actually made it to the goal?
    4. Distribution - histogram of how many steps episodes took
    
    These help us see if the agent is actually learning anything.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # first plot - rewards over time
    axes[0, 0].plot(agent.episode_rewards, alpha=0.6, color='blue')
    axes[0, 0].set_title("Episode Reward Over Time", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True, alpha=0.3)

    # second plot - how many steps it took
    axes[0, 1].plot(agent.episode_lengths, alpha=0.6, color='green')
    axes[0, 1].set_title("Steps to Goal", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps")
    axes[0, 1].grid(True, alpha=0.3)

    # third plot - success rate (using a 100-episode window)
    window = 100
    success_rates = []
    for i in range(len(agent.episode_lengths)):
        recent = agent.episode_lengths[max(0, i - window):i + 1]
        success_rates.append(
            sum(1 for x in recent if x < 200) / len(recent) * 100
        )

    axes[1, 0].plot(success_rates, color='purple')
    axes[1, 0].set_title("Success Rate (%)", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Success Rate (%)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].grid(True, alpha=0.3)

    # 4 Distribution
    axes[1, 1].hist(agent.episode_lengths, bins=40, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title("Distribution of Episode Lengths", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Steps")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"SUCCESS: Training curves saved to: {save_path}")
    plt.close()


def train_and_evaluate(grid_size=15, obstacle_density=0.2, num_episodes=10000):
    """
    Main function that handles the whole training process.
    
    Steps:
    1. Set up the environment
    2. Create the Q-Learning agent
    3. Train it for a bunch of episodes
    4. Save the model so we can use it later
    5. Make some graphs to see how it did
    6. Compare it with A* to see if it's any good
    
    Parameters:
    - grid_size: how big the grid is (we use 15x15)
    - obstacle_density: how many obstacles to put (0.2 = 20%)
    - num_episodes: how many times to train (10000 default)
    """

    print("=" * 60)
    print("TRAINING Q-LEARNING AGENT FOR CAMPUS NAVIGATION")
    print("=" * 60)

    # make the environment (our grid world)
    env = CampusEnvironment(width=grid_size, height=grid_size, obstacle_prob=obstacle_density)

    # Check if a pre-trained model exists for this configuration
    model_filename = f"../models/qlearning_agent_{grid_size}x{grid_size}_{obstacle_density:.2f}.pkl"
    config_filename = f"../models/environment_config_{grid_size}x{grid_size}_{obstacle_density:.2f}.pkl"
    
    if os.path.exists(model_filename) and os.path.exists(config_filename):
        print("\nFOUND EXISTING MODEL - Continuing training from previous Q-table...")
        print("This lets us keep training instead of starting from scratch.")
        
        # load the saved environment
        with open(config_filename, "rb") as f:
            env_config = pickle.load(f)
        
        # Recreate the exact same environment
        env = CampusEnvironment(width=env_config["width"], height=env_config["height"])
        env.start = env_config["start"]
        env.goal = env_config["goal"]
        env.static_grid = env_config["static_grid"]
        env.grid = env_config["static_grid"].copy()
        
        # Load existing model data
        with open(model_filename, "rb") as f:
            saved_data = pickle.load(f)
        
        # Initialize agent with existing Q-table
        agent = QLearningAgent(
            environment=env,
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon=saved_data.get("epsilon", 0.1),  # Resume with previous epsilon
            epsilon_decay=0.9995,
            epsilon_min=0.01
        )
        agent.q_table = saved_data["q_table"]  # Continue from existing knowledge
        
        print(f"Loaded Q-table with shape: {agent.q_table.shape}")
        print(f"Resuming with epsilon: {agent.epsilon:.4f}")
    else:
        print("\nNO EXISTING MODEL - Training from scratch...")
        
        # create our agent with these parameters (took some trial and error to get these right)
        agent = QLearningAgent(
            environment=env,
            learning_rate=0.1,        # how fast it learns (0.1 works well)
            discount_factor=0.99,     # how much it cares about future rewards
            epsilon=1.0,              # start fully random so it explores everything
            epsilon_decay=0.9995,     # slowly make it less random
            epsilon_min=0.01          # but keep it a little random forever
        )

    # training setup
    print(f"Training for {num_episodes} episodes...")
    print(f"Grid size: {env.width}x{env.height}")
    print(f"Obstacle density: {obstacle_density:.2f}")
    print(f"Start: {env.start}, Goal: {env.goal}")

    # run the training! (300 max steps per episode should be enough for 15x15 grid)
    agent.train(
        num_episodes=num_episodes,
        max_steps=300,
        verbose=True
    )

    # save the trained model for later
    os.makedirs("../models", exist_ok=True)
    model_filename = f"../models/qlearning_agent_{grid_size}x{grid_size}_{obstacle_density:.2f}.pkl"
    agent.save_model(model_filename)
    print(f"Model saved to: {model_filename}")

    # also save the environment so we can recreate it exactly
    config_filename = f"../models/environment_config_{grid_size}x{grid_size}_{obstacle_density:.2f}.pkl"
    with open(config_filename, "wb") as f:
        pickle.dump({
            "width": env.width,
            "height": env.height,
            "start": env.start,
            "goal": env.goal,
            "static_grid": env.static_grid
        }, f)

    # make graphs
    os.makedirs("../data", exist_ok=True)
    plot_training_curves(agent, "../data/training_curves.png")

    # compare with A* to see how good our agent actually is
    planner = AStarPlanner(env)
    path, cost, stats = planner.search()

    print("\nTraining completed successfully.")
    print(f"Final epsilon: {agent.epsilon:.4f}")  # should be around 0.01
    
    # Print summary statistics from training
    if len(agent.episode_lengths) > 0:
        avg_last_100 = np.mean(agent.episode_lengths[-100:])
        min_steps = np.min(agent.episode_lengths)
        max_steps = np.max(agent.episode_lengths)
        
        print(f"Average Steps (last 100 episodes): {avg_last_100:.2f}")
        print(f"Minimum Steps Achieved: {min_steps}")
        print(f"Maximum Steps: {max_steps}")
    
    # see how Q-Learning stacks up against A*
    if path:
        print(f"\nSUCCESS: A* found path with {len(path)} steps")
        print(f"   Nodes expanded: {stats['nodes_expanded']}")
        
        # test our trained agent
        q_path, q_cost = agent.get_policy_path(env.start)
        if q_path and len(q_path) > 0:
            reached_goal = q_path[-1] == env.goal
            status = "REACHED GOAL!" if reached_goal else "did not reach goal"
            print(f"SUCCESS: Q-Learning path: {len(q_path)} steps ({status})")
            if reached_goal:
                diff = len(q_path) - len(path)
                efficiency = (len(path) / len(q_path) * 100)
                print(f"   Difference: {diff:+d} steps")
                print(f"   Efficiency: {efficiency:.1f}%")
        else:
            print("ERROR: Q-Learning failed to find path in test")
    
    # Calculate overall training success rate
    success_count = sum(1 for length in agent.episode_lengths if length < 300)
    success_rate = (success_count / len(agent.episode_lengths)) * 100
    print(f"\nTraining Success Rate: {success_rate:.1f}%")
    print(f"   Episodes that reached goal: {success_count}/{len(agent.episode_lengths)}")
    
    print("\n" + "=" * 60)
    print("ALL DONE! You can now run the Streamlit app:")
    print("   streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    # Parse command-line arguments for flexible training
    parser = argparse.ArgumentParser(description="Train Q-Learning agent for campus navigation")
    parser.add_argument('--grid-size', type=int, default=15, help='Grid size (width and height)')
    parser.add_argument('--obstacle-density', type=float, default=0.2, help='Obstacle probability (0.0-1.0)')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.grid_size < 5 or args.grid_size > 50:
        print("ERROR: Grid size must be between 5 and 50")
        sys.exit(1)
    if args.obstacle_density < 0.0 or args.obstacle_density > 0.8:
        print("ERROR: Obstacle density must be between 0.0 and 0.8")
        sys.exit(1)
    
    train_and_evaluate(
        grid_size=args.grid_size,
        obstacle_density=args.obstacle_density,
        num_episodes=args.episodes
    )