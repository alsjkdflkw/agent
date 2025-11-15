# Quick Start - DQN Learning Agent

## What is it?

A Deep Q-Network agent that **learns from its mistakes** and improves over time through reinforcement learning.

## Fastest Way to Try It

```bash
# 2-3 minute demo - see the agent learn!
python test_dqn_agent.py quick-train 100
```

This will:
1. Test the agent BEFORE training
2. Train for 100 episodes
3. Test the agent AFTER training
4. Show the improvement

Expected result: Agent improves from ~40-50% to ~50-60% win rate.

## For Serious Training

```bash
# Recommended: 1000 episodes (15-30 minutes)
python train_dqn_agent.py --episodes 1000

# Better performance: 5000 episodes (2-3 hours)
python train_dqn_agent.py --episodes 5000

# Best results: 10000+ episodes (overnight)
nohup python train_dqn_agent.py --episodes 10000 > training.log 2>&1 &
```

## After Training

Your trained models will be in the `models/` directory:

```bash
# Test your trained agent
python test_dqn_agent.py test models/dqn_agent_final.pkl 100

# Use in code
from uno.dqn_agent import DQNAgent
agent = DQNAgent(env.action_space, env.observation_space)
agent.load('models/dqn_agent_final.pkl')
agent.epsilon = 0.0  # No exploration for testing
```

## Training Progress

You'll see:
- **Episodes 1-200**: Heavy exploration (~30-40% win rate)
- **Episodes 200-500**: Learning phase (~40-50% win rate)  
- **Episodes 500-1000**: Refinement (~50-60% win rate)
- **Episodes 1000+**: Mastery (~55-70% win rate)

## Commands Reference

```bash
# Quick demo
python test_dqn_agent.py quick-train [episodes]

# Test model
python test_dqn_agent.py test <model_path> [games]

# Compare agents
python test_dqn_agent.py compare

# Long training
python train_dqn_agent.py --episodes 1000

# Training options
python train_dqn_agent.py --help
```

## Files Created

- `uno/dqn_agent.py` - The learning agent
- `train_dqn_agent.py` - Training script
- `test_dqn_agent.py` - Testing utility
- `DQN_AGENT_GUIDE.md` - Full documentation

## Need Help?

Read the complete guide: `DQN_AGENT_GUIDE.md`

It covers:
- How the agent learns
- Training stages and expectations
- Hyperparameter tuning
- Performance optimization
- Troubleshooting

## Example Output

```
Before Training: 40.0% wins
After Training:  55.0% wins
Improvement:     +15.0%
Episodes:        156
Steps:           6234
```

## Pro Tips

1. **Start with quick demo** to see if it works
2. **Train for at least 1000 episodes** for meaningful results
3. **Use saved models** - no need to retrain every time
4. **Check training progress plots** in `models/training_progress.png`
5. **Be patient** - learning takes time but results are worth it!

---

**Start now:** `python test_dqn_agent.py quick-train 100`
