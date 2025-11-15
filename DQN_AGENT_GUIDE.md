# DQN Learning Agent - Documentation

## Overview

The DQN (Deep Q-Network) agent is a sophisticated reinforcement learning agent that learns from its mistakes and improves over time. It uses:

- **Deep Q-Learning**: Neural network to approximate Q-values
- **Experience Replay**: Stores past experiences to break correlation
- **Target Network**: Stabilizes learning
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation

## Quick Start

### 1. Quick Training Demo (2-3 minutes)

Train the agent for 100 episodes and see it improve:

```bash
python test_dqn_agent.py quick-train
```

Expected output:
- Before training: ~30-50% win rate
- After training: ~45-60% win rate (improvement varies)

### 2. Long-Term Training (30+ minutes)

For serious training, use the dedicated training script:

```bash
# Train for 1000 episodes (recommended minimum)
python train_dqn_agent.py --episodes 1000

# Train for 5000 episodes (better performance)
python train_dqn_agent.py --episodes 5000

# Train overnight (10000+ episodes)
python train_dqn_agent.py --episodes 10000 --eval-interval 200
```

### 3. Test a Trained Agent

```bash
python test_dqn_agent.py test models/dqn_agent_final.pkl 100
```

## Training Options

### Basic Training

```bash
python train_dqn_agent.py --episodes 1000
```

### Advanced Options

```bash
python train_dqn_agent.py \
  --episodes 5000 \           # Number of training episodes
  --eval-interval 100 \       # Evaluate every N episodes
  --eval-games 50 \           # Games per evaluation
  --save-interval 500 \       # Save model every N episodes
  --model-dir my_models       # Directory for saved models
```

### Quiet Mode (Less Output)

```bash
python train_dqn_agent.py --episodes 5000 --quiet
```

## Understanding the Agent

### How It Learns

1. **Observes the game state** - Full view of the Uno game
2. **Selects actions** - Using Q-network with epsilon-greedy exploration
3. **Receives rewards** - +1 for winning, -1 for losing
4. **Stores experiences** - In replay buffer (10,000 transitions)
5. **Learns from batches** - Updates Q-network using sampled experiences
6. **Improves over time** - Epsilon decays, agent becomes more confident

### Key Components

#### Q-Network
- **Input**: 240 features (flattened 4x4x15 observation)
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output**: 61 Q-values (one per action)

#### Hyperparameters
- **Learning Rate**: 0.001
- **Discount Factor**: 0.95 (values future rewards at 95%)
- **Epsilon Start**: 1.0 (100% random exploration initially)
- **Epsilon End**: 0.05 (5% exploration after decay)
- **Epsilon Decay**: 0.995 (decays each episode)
- **Batch Size**: 32 transitions
- **Buffer Capacity**: 10,000 transitions
- **Target Update Frequency**: Every 100 steps

## Training Process

### Training Stages

**Episodes 1-200: Heavy Exploration**
- Agent explores randomly (epsilon ~0.8-1.0)
- Building experience in replay buffer
- Win rate: ~30-40%

**Episodes 200-500: Learning Phase**
- Agent starts exploiting learned patterns
- Epsilon decaying (~0.4-0.8)
- Win rate: ~40-50%

**Episodes 500-1000: Refinement**
- More exploitation, less exploration
- Epsilon low (~0.05-0.4)
- Win rate: ~50-60%

**Episodes 1000+: Mastery**
- Consistent performance
- Minimal exploration
- Win rate: ~55-65% (vs random opponent)

### What to Expect

**Short Training (100-200 episodes):**
- Noticeable improvement from random
- Win rate: 45-55%
- Time: 5-10 minutes

**Medium Training (500-1000 episodes):**
- Good strategic play
- Win rate: 50-60%
- Time: 15-30 minutes

**Long Training (5000+ episodes):**
- Near-optimal play
- Win rate: 55-70%
- Time: 2-3 hours

## Saved Files

After training, you'll find in the `models/` directory:

### Model Files
- `dqn_agent_final.pkl` - Final trained model
- `dqn_agent_ep500_*.pkl` - Checkpoint at episode 500
- `dqn_agent_ep1000_*.pkl` - Checkpoint at episode 1000

### Training Artifacts
- `training_progress.png` - Loss and epsilon plots
- `training_summary.txt` - Training configuration and results

## Using a Trained Agent

### Load and Test

```python
from uno.env import UnoEnv
from uno.dqn_agent import DQNAgent

# Create environment
env = UnoEnv(render_mode=None)

# Create and load agent
agent = DQNAgent(env.action_space, env.observation_space)
agent.load('models/dqn_agent_final.pkl')

# Disable exploration for testing
agent.epsilon = 0.0

# Test it
from test_gymnasium_agent import run_tournament
from uno.gymnasium_agent import RandomGymnasiumAgent

opponent = RandomGymnasiumAgent(env.action_space, env.observation_space)
stats = run_tournament(env, [agent, opponent], num_games=100)
```

### Use in Your Code

```python
from uno.dqn_agent import DQNAgent
from uno.env import UnoEnv

env = UnoEnv(render_mode=None)
agent = DQNAgent(env.action_space, env.observation_space)
agent.load('models/dqn_agent_final.pkl')
agent.epsilon = 0.0  # No exploration

# Play a game
obs, info = env.reset()
done = False

while not done:
    if info['player_id'] == 0:  # Your agent's turn
        action = agent.select_action(obs, info['legal_actions'])
        obs, reward, done, truncated, info = env.step(action)
    else:
        # Opponent's turn
        pass
```

## Advanced Usage

### Custom Training Loop

```python
from uno.env import UnoEnv
from uno.dqn_agent import DQNAgent
from uno.gymnasium_agent import RandomGymnasiumAgent

env = UnoEnv(render_mode=None)
agent = DQNAgent(env.action_space, env.observation_space)
opponent = RandomGymnasiumAgent(env.action_space, env.observation_space)

agents = [agent, opponent]

for episode in range(1000):
    obs, info = env.reset()
    done = False
    
    while not done:
        current_player = info['player_id']
        legal_actions = info['legal_actions']
        
        agent_obj = agents[current_player]
        action = agent_obj.select_action(obs, legal_actions)
        
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Only the DQN agent learns
        if current_player == 0:
            agent.learn(obs, action, reward, next_obs, done, info)
        
        obs = next_obs
    
    # Print progress
    if (episode + 1) % 100 == 0:
        stats = agent.get_stats()
        print(f"Episode {episode+1}: Epsilon={stats['epsilon']:.3f}, Loss={stats['avg_loss']:.4f}")

# Save the trained agent
agent.save('my_trained_agent.pkl')
```

### Modify Hyperparameters

```python
agent = DQNAgent(
    action_space=env.action_space,
    observation_space=env.observation_space,
    learning_rate=0.0005,        # Lower = more stable
    discount_factor=0.99,         # Higher = values future more
    epsilon_start=1.0,
    epsilon_end=0.01,             # Lower = less exploration
    epsilon_decay=0.999,          # Slower decay
    buffer_capacity=20000,        # More memory
    batch_size=64,                # Larger batches
    target_update_freq=200,       # Less frequent updates
    hidden_size=256               # Larger network
)
```

## Performance Tips

### For Faster Training
- Use smaller buffer: `buffer_capacity=5000`
- Use smaller batches: `batch_size=16`
- Reduce evaluations: `--eval-interval 200`

### For Better Performance
- Train longer: `--episodes 10000`
- Larger network: `hidden_size=256`
- More experience: `buffer_capacity=20000`
- Tune epsilon decay: experiment with different values

### For Stability
- Lower learning rate: `learning_rate=0.0005`
- More frequent target updates: `target_update_freq=50`
- Larger batches: `batch_size=64`

## Troubleshooting

### Agent Not Improving

**Problem**: Win rate stays around 30-40%

**Solutions**:
1. Train for more episodes (try 2000+)
2. Check that epsilon is decaying (`agent.epsilon` should decrease)
3. Ensure replay buffer has enough samples (needs ~100+ episodes)

### Training is Slow

**Problem**: Training takes too long

**Solutions**:
1. Reduce `eval_interval` (e.g., 200 instead of 100)
2. Reduce `eval_games` (e.g., 10 instead of 20)
3. Use `--quiet` flag to reduce output

### Loss is Unstable

**Problem**: Training loss jumps around

**Solutions**:
1. Lower learning rate: `learning_rate=0.0005`
2. Increase batch size: `batch_size=64`
3. This is partially normal for Q-learning

## Comparison with Other Agents

**vs Random Agent:**
- DQN (untrained): ~50% win rate
- DQN (trained 1000 ep): ~55-60% win rate
- DQN (trained 5000 ep): ~60-70% win rate

**vs Heuristic Agent:**
- DQN (untrained): ~40-45% win rate
- DQN (trained 1000 ep): ~45-55% win rate
- DQN (trained 5000 ep): ~50-60% win rate

## Technical Details

### State Representation
The agent uses the flattened observation tensor:
- Shape: (4, 4, 15) → flattened to 240 features
- Plane 0: Available cards
- Plane 1: Your hand
- Plane 2: Played cards
- Plane 3: Game state info

### Action Space
- 61 total actions
- Actions 0-59: Play specific cards
- Action 60: Draw a card

### Reward Structure
- +1: Win the game
- -1: Lose the game
- 0: Game continues

### Learning Algorithm
1. Collect transition (s, a, r, s', done)
2. Store in replay buffer
3. Sample random batch
4. Compute target: r + γ * max Q(s', a')
5. Update Q-network to minimize (Q(s,a) - target)²
6. Periodically copy Q-network to target network

## Examples

### Example 1: Quick Test
```bash
python test_dqn_agent.py quick-train 200
```

### Example 2: Overnight Training
```bash
nohup python train_dqn_agent.py --episodes 10000 --quiet > training.log 2>&1 &
```

### Example 3: Test Multiple Checkpoints
```bash
for model in models/dqn_agent_ep*.pkl; do
    echo "Testing $model"
    python test_dqn_agent.py test "$model" 50
done
```

## FAQ

**Q: How long should I train?**
A: Minimum 1000 episodes. For best results, 5000+.

**Q: Can I pause and resume training?**
A: Not directly, but you can load a checkpoint and continue training in custom code.

**Q: Why does performance vary?**
A: Uno has randomness. Evaluate over 50+ games for reliable metrics.

**Q: Can I use this against human players?**
A: Yes! Set `render_mode="human"` and use the trained agent.

**Q: How do I know when training is done?**
A: When win rate plateaus and loss stabilizes (typically 2000-5000 episodes).

## Next Steps

1. **Start Simple**: Run `python test_dqn_agent.py quick-train`
2. **Train Longer**: Try `python train_dqn_agent.py --episodes 1000`
3. **Experiment**: Modify hyperparameters and compare results
4. **Deploy**: Use the trained agent in tournaments or competitions

## Support

For issues or questions:
- Check TESTING_INSTRUCTIONS.md for general agent info
- Review the code in `uno/dqn_agent.py` for implementation details
- Test with `test_dqn_agent.py` for debugging
