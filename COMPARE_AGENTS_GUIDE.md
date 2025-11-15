# Comparing Agents - Documentation

## Overview

The `compare_agents.py` script allows you to compare the DQN agent against the Q-Learning agent from `main.py` to see which performs better.

## Quick Start

### Compare Untrained Agents

```bash
python compare_agents.py --quick --games 50
```

This compares:
- Untrained DQN agent
- Default Q-Learning agent (no training)

Expected: Both agents perform similarly (~45-55% win rate each)

### Compare with Trained DQN

```bash
# First, train a DQN agent
python train_dqn_agent.py --episodes 1000

# Then compare it
python compare_agents.py --dqn-model models/dqn_agent_final.pkl --games 100
```

This compares:
- Trained DQN agent (1000 episodes)
- Q-Learning agent (100 episodes training by default)

### Train Both Agents Before Comparing

```bash
# Train DQN agent (longer training)
python train_dqn_agent.py --episodes 5000

# Compare with more Q-Learning training
python compare_agents.py \
  --dqn-model models/dqn_agent_final.pkl \
  --qlearning-episodes 500 \
  --games 200
```

## Command Options

### Basic Options

```bash
python compare_agents.py [options]
```

**Options:**
- `--dqn-model PATH` - Path to trained DQN model (default: untrained)
- `--qlearning-episodes N` - Episodes to train Q-Learning agent (default: 100)
- `--games N` - Number of comparison games (default: 100)
- `--quick` - Quick comparison with untrained agents (50 games)
- `--retrain-qlearning` - Force retrain Q-Learning even if saved

### Examples

**Quick test (untrained agents):**
```bash
python compare_agents.py --quick
```

**Compare with untrained DQN vs trained Q-Learning:**
```bash
python compare_agents.py --qlearning-episodes 200 --games 100
```

**Compare both trained agents:**
```bash
python compare_agents.py \
  --dqn-model models/dqn_agent_final.pkl \
  --qlearning-episodes 500 \
  --games 200
```

**Force retrain Q-Learning agent:**
```bash
python compare_agents.py \
  --dqn-model models/dqn_agent_final.pkl \
  --retrain-qlearning \
  --qlearning-episodes 1000
```

## Understanding Results

### Output Format

```
COMPARISON RESULTS
================================================================================
DQN Agent:
  Wins: 55/100 (55.0%)
  Avg Payoff: 0.100
Q-Learning Agent:
  Wins: 45/100 (45.0%)
  Avg Payoff: -0.100

Average game length: 48.2 turns

================================================================================
🏆 DQN Agent WINS by 10.0%!
================================================================================
```

### Interpreting Results

**Win Rate:**
- 50%: Evenly matched agents
- 55-60%: Slight advantage
- 60-70%: Clear advantage
- 70%+: Strong dominance

**Average Payoff:**
- Ranges from -1.0 to +1.0
- Positive = more wins than losses
- Negative = more losses than wins

**Game Length:**
- Shorter games: More aggressive/effective play
- Longer games: More defensive/careful play
- Typical range: 40-60 turns

## Expected Performance

### Untrained Agents (0 episodes)
- **DQN vs Q-Learning**: ~50/50 (both random)
- Both agents explore randomly

### Short Training (100 episodes)
- **DQN**: ~45-55% win rate
- **Q-Learning**: ~45-55% win rate
- Performance similar, slight variance

### Medium Training (1000 episodes)
- **DQN**: ~55-60% win rate
- **Q-Learning**: ~50-55% win rate
- DQN starts to show advantage with experience replay

### Long Training (5000+ episodes)
- **DQN**: ~60-70% win rate
- **Q-Learning**: ~50-60% win rate
- DQN's neural network and experience replay provide edge

## Q-Learning Agent Details

The Q-Learning agent from `main.py` uses:
- **Q-table**: Dictionary storing state-action values
- **Epsilon-greedy**: Exploration vs exploitation
- **No experience replay**: Learns from immediate transitions
- **State encoding**: Flattened observation tensor

**Advantages:**
- Simple and interpretable
- Fast inference
- Works well for smaller state spaces

**Disadvantages:**
- Large state space (Uno has many states)
- No experience replay
- Can overfit to recent experiences

## DQN Agent Details

The DQN agent uses:
- **Neural network**: Function approximation for Q-values
- **Experience replay**: Learns from stored transitions
- **Target network**: Stabilizes learning
- **Gradient descent**: Efficient parameter updates

**Advantages:**
- Handles large state spaces well
- Experience replay breaks correlation
- Neural network generalizes better
- More stable learning

**Disadvantages:**
- More complex implementation
- Slower training per episode
- Requires more episodes to converge

## Training Recommendations

### For Q-Learning Agent
- **Minimum**: 100 episodes
- **Recommended**: 500-1000 episodes
- **Maximum practical**: 2000 episodes (diminishing returns)

### For DQN Agent
- **Minimum**: 500 episodes
- **Recommended**: 1000-5000 episodes
- **Best results**: 10000+ episodes

## Troubleshooting

### Q-Learning Agent Not Found
**Problem:** `q_learning_agent.pkl` doesn't exist

**Solution:**
```bash
# The script will automatically train one, or:
python main.py  # Train using original script
```

### DQN Model Not Found
**Problem:** Specified DQN model doesn't exist

**Solution:**
```bash
# Train a DQN agent first
python train_dqn_agent.py --episodes 1000
```

### Comparison Takes Too Long
**Problem:** Comparison is slow

**Solution:**
- Reduce number of games: `--games 50`
- Use quick mode: `--quick`
- Use untrained Q-Learning: `--qlearning-episodes 0`

### Results Vary a Lot
**Problem:** Win rates fluctuate between runs

**Solution:**
- Increase number of games: `--games 200`
- Run multiple comparisons and average
- Uno has inherent randomness

## Advanced Usage

### Compare Against Your Own Agent

You can modify `compare_agents.py` to compare against any agent:

```python
from uno.gymnasium_agent import YourCustomAgent

# In the compare_agents function, replace one agent:
your_agent = YourCustomAgent(env.action_space, env.observation_space)
agents = [dqn_agent, your_agent]
```

### Batch Comparisons

Test multiple DQN models:

```bash
for model in models/dqn_agent_ep*.pkl; do
    echo "Testing $model"
    python compare_agents.py --dqn-model "$model" --games 50
done
```

### Save Results

```bash
python compare_agents.py \
  --dqn-model models/dqn_agent_final.pkl \
  --games 200 > comparison_results.txt
```

## FAQ

**Q: Which agent is better?**
A: DQN typically performs better after sufficient training (1000+ episodes).

**Q: How long does comparison take?**
A: 50 games: ~30 seconds, 100 games: ~1 minute, 200 games: ~2 minutes

**Q: Can I compare against human players?**
A: Not directly, but you can use `render_mode="human"` in the code.

**Q: Why does Q-Learning sometimes win?**
A: With limited training, Q-Learning can be competitive. Uno also has randomness.

**Q: Should I retrain Q-Learning each time?**
A: No, it's saved to `q_learning_agent.pkl` and reused. Use `--retrain-qlearning` to force retrain.

## Summary

**Quick comparison:**
```bash
python compare_agents.py --quick
```

**Full comparison:**
```bash
# Train DQN
python train_dqn_agent.py --episodes 5000

# Compare
python compare_agents.py \
  --dqn-model models/dqn_agent_final.pkl \
  --qlearning-episodes 1000 \
  --games 200
```

This helps you evaluate if the DQN agent's complexity is worth it compared to simpler Q-Learning!
