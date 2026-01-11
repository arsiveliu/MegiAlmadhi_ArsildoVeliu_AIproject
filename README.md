# Campus Navigation AI Agent

This project shows how an AI agent can learn to navigate a campus environment using reinforcement learning (Q-Learning) and compare it with the classic A* pathfinding algorithm. We built this for our CEN 352 Artificial Intelligence course project.

## Dataset

The AU dataset files are too large for GitHub. Please download them from the original source:

**Download Link:** https://dataverse.orc.gmu.edu/file.xhtml?persistentId=doi:10.13021/ORC2020/JUIW5F/CUWBXZ&version=10.0

After downloading, extract the files and place them in the `data/AU/` folder in this project directory.

## Requirements
- Python 3.9+
- numpy
- matplotlib
- streamlit

## Quick Start

### 1. Install Dependencies
```bash
pip install -r src/requirements.txt
```

### 2. Train the Agent
```bash
python src/train_agent.py
```

### 3. Launch the Interactive App
```bash
streamlit run src/app.py
```

**Note:** All commands should be run from the project root directory to ensure correct paths to `models/` and `data/` folders.

## Project Structure
- **train_agent.py**: Training and evaluation script
- **app.py**: Interactive Streamlit visualization
- **campus_environment.py**: Environment definition and grid world
- **qlearning.py**: Q-learning reinforcement learning agent
- **astar.py**: A* search algorithm implementation
- **data/**: Dataset and training outputs
  - `AU/`: Global Navigation Dataset (GND) - Australian campus sequence
  - `CITATION.md`: Dataset citation and acknowledgments
  - `training_curves.png`: Training performance visualization
- **models/**: Saved trained agent models (.pkl files)
- **src/**: Source code files

## Team
- **Megi Almadhi** - Worked on the Q-Learning agent, ran training experiments, and evaluated results
- **Arsıldo Veliu** - Built the environment setup, implemented A*, and created the web interface


## What AI Techniques We Used

**Q-Learning (Reinforcement Learning)**
- The agent learns through trial and error - no pre-programmed map needed
- Uses epsilon-greedy strategy (explore vs exploit)
- Updates Q-values using the Bellman equation

**A* Search Algorithm**
- Classic pathfinding that always finds the optimal route
- Uses Manhattan distance as the heuristic
- Serves as our "gold standard" to compare against

**Why both?** A* shows us what the perfect path looks like, and we can see how close Q-Learning gets just from learning on its own.

## PEAS Framework (Agent Design)

| Component | Description |
|-----------|-------------|
| **Performance** | Get to the goal successfully, take fewer steps, avoid obstacles |
| **Environment** | Grid-based campus with buildings/obstacles, everything is visible to the agent |
| **Actuators** | Agent can move: Up, Down, Left, Right |
| **Sensors** | Agent knows its current position and can see the grid layout |

## Results

| Grid Size | Obstacles | Success Rate | Avg Steps | Notes |
|-----------|-----------|--------------|-----------|-------|
| 15×15 | 20% | ~95% | ~18 | Q-Learning gets really close to A* optimal |
| 20×20 | 20% | ~92% | ~28 | Paths are slightly longer but still good |

**What we measured:**
- Success rate: How often the agent reaches the goal
- Average steps: How efficient the paths are
- Convergence: How the rewards improve during training

A* always gives us the shortest possible path, and Q-Learning learns to get pretty close to it!

## Ethical Considerations

**Privacy Concerns:**
If this system were used on a real campus with actual student location data, it could potentially track people's movements and reveal behavioral patterns.

**How to address this:**
- Keep all data anonymous (no personal identifiers)
- Only store aggregated data, not individual paths
- Don't keep data longer than necessary
- Let users know what data is collected and let them opt out

## External Code Used

We wrote all the code ourselves for this project. We only used standard Python libraries (numpy, matplotlib, streamlit) - no external AI frameworks or pre-built navigation code.

---

## Training Guide

### Quick Training Commands

#### Default Configuration (15x15, 0.2 density)
```bash
python src/train_agent.py
```

#### Custom Grid Size and Obstacle Density
```bash
# 20x20 grid with 30% obstacles
python src/train_agent.py --grid-size 20 --obstacle-density 0.3

# 25x25 grid with 25% obstacles, 20k episodes
python src/train_agent.py --grid-size 25 --obstacle-density 0.25 --episodes 20000
```

#### Training Multiple Times for Better Performance
```bash
# First training (10k episodes)
python src/train_agent.py --grid-size 15 --obstacle-density 0.2 --episodes 10000

# Continue training (adds 10k more episodes on existing Q-table)
python src/train_agent.py --grid-size 15 --obstacle-density 0.2 --episodes 10000

# Add even more training (another 10k episodes)
python src/train_agent.py --grid-size 15 --obstacle-density 0.2 --episodes 10000
```

**Result:** Each subsequent training improves the agent by continuing from the previous Q-table!

### How Training Works

#### First Time Training
1. Creates new environment with your specified grid size and obstacle density
2. Initializes Q-table with zeros
3. Agent explores and learns through trial and error
4. Epsilon decays from 1.0 (random) to 0.01 (mostly exploitation)
5. Saves trained model to: `models/qlearning_agent_{size}x{size}_{density}.pkl`

#### Subsequent Training (Same Configuration)
1. Loads existing Q-table from previous training
2. Continues with previous epsilon value (e.g., 0.01)
3. Further refines the policy with more episodes
4. Updates the model file with improved Q-values

### Training Performance by Episodes

- **5,000 episodes**: Basic learning, sometimes doesn't reach the goal
- **10,000 episodes**: Pretty good, usually gets there
- **15,000 episodes**: Works well, gets close to the best paths
- **20,000+ episodes**: Works really well, almost as good as A*

### Some Configs We Tried

#### Testing Different Sizes
We trained a few different grids to see what works:

```bash
# Small grid (faster to train)
python src/train_agent.py --grid-size 10 --obstacle-density 0.2 --episodes 10000

# Medium grid (this is the default)
python src/train_agent.py --grid-size 15 --obstacle-density 0.2 --episodes 15000

# Bigger grid (takes longer but more interesting)
python src/train_agent.py --grid-size 20 --obstacle-density 0.3 --episodes 20000
```

#### Training Multiple Times
You can train the same setup multiple times to make it better:

```bash
# First run
python src/train_agent.py --grid-size 15 --obstacle-density 0.2 --episodes 10000

# Run it again to improve
python src/train_agent.py --grid-size 15 --obstacle-density 0.2 --episodes 10000

# One more time
python src/train_agent.py --grid-size 15 --obstacle-density 0.2 --episodes 10000
```

Each time it gets a bit better because it keeps learning from where it left off!

### Understanding Training Output

During training, you'll see:
```
Episode 1000/10000 | Reward: -45.0 | Steps: 67 | Epsilon: 0.951
Episode 2000/10000 | Reward: 85.0 | Steps: 23 | Epsilon: 0.904
...
```

**What it means:**
- **Reward**: Cumulative reward (higher = better, 100+ = reached goal)
- **Steps**: Number of steps taken (fewer = more efficient)
- **Epsilon**: Exploration rate (decreases over time)

---

## Dataset Information

### Real-World Dataset

We based our project on the **Global Navigation Dataset (GND)** from George Mason University. The dataset includes real navigation data from an Australian campus, which you can find in the `data/AU/` folder:
- Point cloud data from LiDAR scans
- Robot trajectory information
- Pose graphs and timestamps
- Scan context descriptors

**Dataset Citation:**
```bibtex
@misc{gnd2024,
  title={Global Navigation Dataset},
  author={Xiao, Jing and others},
  institution={George Mason University},
  year={2024},
  url={https://cs.gmu.edu/~xiao/Research/GND}
}
```

### Why We Use a Grid

Instead of using the raw point cloud data directly, we simplified the campus into a grid (like a checkerboard). Here's why:
- Makes the learning problem manageable - fewer states to learn
- Q-Learning can actually converge in reasonable time
- Still captures the main navigation challenges (obstacles, finding paths)
- Easy to visualize and understand
- Can be trained on a regular laptop in minutes, not hours

Think of it like creating a simplified map for a board game based on a real place - the grid keeps the essential structure but makes it practical to work with.

## Authors
- Megi Almadhi
- Arsıldo Veliu
