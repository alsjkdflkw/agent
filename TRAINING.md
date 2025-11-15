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

## Features

- **Ctrl+C to save and quit** - Training saves progress automatically
- **Auto-resume** - Continues from last checkpoint
- **Backup system** - Saves history in backups/ folder
- **Minimal output** - Optimized for speed

## Files

- `fast_train.py` - Fast single agent training
- `evolutionary_train.py` - Evolutionary multi-agent training
- `uno/dqn_agent.py` - DQN agent implementation
- `uno/gymnasium_agent.py` - Agent interface

## Saved Models

- `models/agent.pkl` - Fast training checkpoint
- `evolution/best_agent.pkl` - Best evolved agent
- `evolution/backups/` - Generation backups
