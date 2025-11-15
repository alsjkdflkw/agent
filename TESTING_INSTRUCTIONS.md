# Gymnasium API Agent - Testing Instructions

This document provides comprehensive instructions for testing and using the Gymnasium API agents in the Uno game framework.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding the Gymnasium Agent Interface](#understanding-the-gymnasium-agent-interface)
3. [Running Tests](#running-tests)
4. [Creating Custom Agents](#creating-custom-agents)
5. [Example Agents](#example-agents)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

The required dependencies should already be installed. If not, run:

```bash
pip install gymnasium rlcard numpy joblib
```

### Running the Test Suite

Run all tests:

```bash
python test_gymnasium_agent.py
```

Run specific tests:

```bash
# Test random agents
python test_gymnasium_agent.py --test random

# Test heuristic agent
python test_gymnasium_agent.py --test heuristic

# Test with visualization
python test_gymnasium_agent.py --test render

# Test custom agent
python test_gymnasium_agent.py --test custom
```

Customize number of games:

```bash
python test_gymnasium_agent.py --test heuristic --games 500
```

## Understanding the Gymnasium Agent Interface

### The GymnasiumAgent Base Class

All Gymnasium agents inherit from the `GymnasiumAgent` base class:

```python
from uno.gymnasium_agent import GymnasiumAgent
import gymnasium as gym

class MyAgent(GymnasiumAgent):
    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        super().__init__(action_space, observation_space)
        # Your initialization code here
    
    def select_action(self, observation, legal_actions):
        # Your action selection logic here
        return action  # Must be from legal_actions
    
    def reset(self):
        # Optional: Reset internal state
        pass
    
    def learn(self, observation, action, reward, next_observation, done, info):
        # Optional: Learning logic
        pass
```

### Key Components

#### 1. Action Space

The environment uses a `Discrete(61)` action space:
- Actions 0-59: Various Uno cards (numbers, Skip, Reverse, Draw 2, Wild, Wild Draw 4)
- Action 60: Draw a card

#### 2. Observation Dictionary

The observation is a dictionary containing:

```python
observation = {
    'obs': np.ndarray,           # Shape (4, 4, 15) - encoded game state
    'legal_actions': dict,       # Dictionary of legal actions
    'raw_obs': dict,             # Human-readable game info
    'raw_legal_actions': list,   # List of legal action names
    'action_record': list        # History of actions
}
```

The `raw_obs` dictionary contains:

```python
raw_obs = {
    'hand': ['r-5', 'g-skip', ...],  # Your cards
    'target': 'b-4',                  # Card on top of discard pile
    'played_cards': [...],            # History of played cards
    'num_cards': [5, 7],              # Number of cards per player
    'num_players': 2,                 # Number of players
    'current_player': 0               # Current player ID
}
```

#### 3. Legal Actions

Always a list of valid action indices you can choose from:

```python
legal_actions = [5, 12, 34, 60]  # Example
```

**Important:** Your agent must always return an action from `legal_actions`.

## Running Tests

### Test 1: Random vs Random

Tests two random agents against each other. Expected outcome: roughly 50/50 win rate.

```bash
python test_gymnasium_agent.py --test random --games 100
```

**What it tests:**
- Basic agent functionality
- Environment interaction
- Statistical fairness

### Test 2: Heuristic vs Random

Tests a heuristic agent against a random agent. Expected outcome: heuristic should win more often.

```bash
python test_gymnasium_agent.py --test heuristic --games 100
```

**What it tests:**
- Strategic decision making
- Use of observation information
- Action prioritization

### Test 3: Single Game with Rendering

Runs a single game with full visualization of gameplay.

```bash
python test_gymnasium_agent.py --test render
```

**What it tests:**
- Visual verification of agent behavior
- Step-by-step game progression
- Observation structure

### Test 4: Custom Agent

Demonstrates creating and testing a custom agent.

```bash
python test_gymnasium_agent.py --test custom --games 50
```

**What it tests:**
- Custom agent implementation
- Inheritance from GymnasiumAgent
- Integration with tournament system

## Creating Custom Agents

### Template for a Simple Agent

```python
from uno.gymnasium_agent import GymnasiumAgent
import gymnasium as gym

class MySimpleAgent(GymnasiumAgent):
    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        super().__init__(action_space, observation_space)
    
    def select_action(self, observation, legal_actions):
        # Always avoid drawing if possible
        playable = [a for a in legal_actions if a != 60]
        if playable:
            return playable[0]
        return 60
```

### Template for a Learning Agent

```python
from uno.gymnasium_agent import GymnasiumAgent
import gymnasium as gym
import numpy as np

class MyLearningAgent(GymnasiumAgent):
    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        super().__init__(action_space, observation_space)
        self.q_table = {}
        self.epsilon = 0.1
        self.learning_rate = 0.1
        self.discount_factor = 0.9
    
    def select_action(self, observation, legal_actions):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(legal_actions)
        
        # Choose best action based on Q-values
        state_key = self._encode_state(observation)
        q_values = [self.q_table.get((state_key, a), 0) for a in legal_actions]
        return legal_actions[np.argmax(q_values)]
    
    def learn(self, observation, action, reward, next_observation, done, info):
        # Q-learning update
        state_key = self._encode_state(observation)
        next_state_key = self._encode_state(next_observation)
        
        current_q = self.q_table.get((state_key, action), 0)
        
        if not done:
            next_legal = info.get('legal_actions', [])
            max_next_q = max([self.q_table.get((next_state_key, a), 0) 
                             for a in next_legal], default=0)
        else:
            max_next_q = 0
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[(state_key, action)] = new_q
    
    def _encode_state(self, observation):
        # Simple state encoding
        obs_array = observation['obs']
        return tuple(obs_array.flatten().astype(int).tolist())
    
    def reset(self):
        # Called at the start of each episode
        pass
```

### Testing Your Custom Agent

```python
from uno.env import UnoEnv
from test_gymnasium_agent import run_tournament

# Create environment
env = UnoEnv(render_mode=None)

# Create your agent
my_agent = MySimpleAgent(env.action_space, env.observation_space)

# Create opponent (e.g., random agent)
from uno.gymnasium_agent import RandomGymnasiumAgent
opponent = RandomGymnasiumAgent(env.action_space, env.observation_space)

# Run tournament
stats = run_tournament(env, [my_agent, opponent], num_games=100)
```

## Example Agents

### 1. RandomGymnasiumAgent

Selects actions uniformly at random from legal actions.

**Use case:** Baseline for comparison

```python
from uno.gymnasium_agent import RandomGymnasiumAgent
agent = RandomGymnasiumAgent(env.action_space, env.observation_space)
```

### 2. HeuristicGymnasiumAgent

Uses simple rules to make decisions:
1. Avoid drawing if possible
2. Prioritize Draw 2 cards
3. Prioritize Skip cards
4. Prioritize Reverse cards
5. Play Wild cards strategically
6. Default to first legal card

**Use case:** Demonstrates informed decision-making

```python
from uno.gymnasium_agent import HeuristicGymnasiumAgent
agent = HeuristicGymnasiumAgent(env.action_space, env.observation_space)
```

### 3. Wrapping Existing Agents

Convert existing `BaseAgent` implementations to Gymnasium interface:

```python
from uno.gymnasium_agent import create_gymnasium_agent_wrapper
from uno.agents import RandomAgent

# Wrap existing agent
WrappedRandomAgent = create_gymnasium_agent_wrapper(RandomAgent)
agent = WrappedRandomAgent(env.action_space, env.observation_space)
```

## Advanced Usage

### Training a Learning Agent

```python
from uno.env import UnoEnv
from uno.gymnasium_agent import RandomGymnasiumAgent

# Create environment and agents
env = UnoEnv(render_mode=None)
learning_agent = MyLearningAgent(env.action_space, env.observation_space)
opponent = RandomGymnasiumAgent(env.action_space, env.observation_space)

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    
    while not done:
        player_id = info['player_id']
        legal_actions = info['legal_actions']
        
        # Select agent
        agent = learning_agent if player_id == 0 else opponent
        action = agent.select_action(obs, legal_actions)
        
        # Take action
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Learn (only learning_agent)
        if player_id == 0:
            learning_agent.learn(obs, action, reward, next_obs, done, info)
        
        obs = next_obs
    
    if (episode + 1) % 100 == 0:
        print(f"Completed {episode + 1} episodes")

# Save trained agent
import joblib
joblib.dump(learning_agent, 'my_trained_agent.pkl')
```

### Loading and Testing a Trained Agent

```python
import joblib
from uno.env import UnoEnv
from test_gymnasium_agent import run_tournament
from uno.gymnasium_agent import RandomGymnasiumAgent

# Load trained agent
trained_agent = joblib.load('my_trained_agent.pkl')

# Test against random opponent
env = UnoEnv(render_mode=None)
opponent = RandomGymnasiumAgent(env.action_space, env.observation_space)

stats = run_tournament(env, [trained_agent, opponent], num_games=100)
```

### Running Multi-Agent Tournaments

```python
from uno.env import UnoEnv
from uno.gymnasium_agent import RandomGymnasiumAgent, HeuristicGymnasiumAgent

env = UnoEnv(render_mode=None)

# Create multiple agents
agents = [
    RandomGymnasiumAgent(env.action_space, env.observation_space),
    HeuristicGymnasiumAgent(env.action_space, env.observation_space),
]

# Note: Currently the environment supports 2 players
# For more players, you'd need to modify the environment configuration

from test_gymnasium_agent import run_tournament
stats = run_tournament(env, agents, num_games=100)
```

## Troubleshooting

### Common Issues

#### 1. "Action not in legal actions" Error

**Problem:** Your agent returns an action that's not in `legal_actions`.

**Solution:** Always validate your action:

```python
def select_action(self, observation, legal_actions):
    action = self._compute_action(observation)
    
    # Validate action is legal
    if action not in legal_actions:
        # Fallback to first legal action
        action = legal_actions[0]
    
    return action
```

#### 2. Import Errors

**Problem:** Cannot import gymnasium or rlcard modules.

**Solution:**

```bash
pip install --upgrade gymnasium rlcard numpy joblib
```

#### 3. Observation Dictionary KeyError

**Problem:** Trying to access a key that doesn't exist in observation.

**Solution:** Use `.get()` with defaults:

```python
raw_obs = observation.get('raw_obs', {})
hand = raw_obs.get('hand', [])
```

#### 4. Agent Not Learning

**Problem:** Learning agent's performance doesn't improve.

**Possible causes:**
- Learning rate too high or too low
- Insufficient training episodes
- State space too large for Q-table
- Reward signal unclear

**Solution:** Try:
- Adjust hyperparameters
- Increase training episodes
- Use function approximation (neural network)
- Add reward shaping

### Getting Help

If you encounter issues:

1. Check that dependencies are properly installed
2. Verify your agent always returns a legal action
3. Review the example agents for reference patterns
4. Run the test suite to ensure environment is working correctly

## Performance Benchmarks

Expected performance on a modern laptop (approximate):

- **Single Game:** ~0.1-1 seconds (depending on render mode)
- **100 Game Tournament:** ~10-30 seconds (without rendering)
- **1000 Episode Training:** ~2-5 minutes (Q-learning)

## Next Steps

1. **Implement your own agent** using the templates provided
2. **Test against different opponents** to evaluate performance
3. **Experiment with different strategies** (aggressive, defensive, adaptive)
4. **Try machine learning approaches** (Q-learning, DQN, PPO)
5. **Optimize for the tournament** format used in the competition

Good luck building your Uno agent!
