# Uno Agent Training

## Training Options

### Fast Training (Single Agent)
```bash
python fast_train.py --episodes 5000
```

### Evolutionary Training (Population)
```bash
python evolutionary_train.py --population 4 --generations 1000
```

## Testing Your Agent

### Test vs Q-Learning (from main.py)
```bash
# First train Q-Learning agent (if not already done)
python main.py

# Then test your DQN agent against it
python test_agent.py --games 100
```

### Test vs Random Agent
```bash
python test_agent.py --vs-random --games 100
```

### Test Fast-Trained Model
```bash
python test_agent.py --fast-model --games 100
```

**Note:** Testing is READ-ONLY and never modifies your saved models.

## Features

- **Ctrl+C to save and quit** - Training saves progress automatically
- **Auto-resume** - Continues from last checkpoint
- **Backup system** - Saves history in backups/ folder
- **Minimal output** - Optimized for speed
- **READ-ONLY testing** - test_agent.py never overwrites your brains

## Files

- `fast_train.py` - Fast single agent training
- `evolutionary_train.py` - Evolutionary multi-agent training
- `test_agent.py` - Test trained agents (read-only)
- `uno/dqn_agent.py` - DQN agent implementation
- `uno/gymnasium_agent.py` - Agent interface

## Saved Models

- `models/agent.pkl` - Fast training checkpoint
- `evolution/best_agent.pkl` - Best evolved agent
- `evolution/backups/` - Generation backups
- `q_learning_agent.pkl` - Q-Learning from main.py
