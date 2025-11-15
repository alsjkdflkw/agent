# Quick Start - Gymnasium API Agent

## Running the Example

The fastest way to see the Gymnasium API in action:

```bash
python example_gymnasium_agent.py
```

This will:
- Create a simple agent
- Play a single game
- Run a 100-game tournament
- Save and load the agent

## Running Tests

Run all tests:
```bash
python test_gymnasium_agent.py
```

Run specific tests:
```bash
python test_gymnasium_agent.py --test random --games 100
python test_gymnasium_agent.py --test heuristic --games 100
python test_gymnasium_agent.py --test custom --games 50
```

## Creating Your Own Agent

```python
from uno.gymnasium_agent import GymnasiumAgent
from uno.env import UnoEnv

class MyAgent(GymnasiumAgent):
    def select_action(self, observation, legal_actions):
        # Your logic here
        # Must return an action from legal_actions
        return legal_actions[0]  # Example: always play first legal card

# Create environment and test
env = UnoEnv(render_mode=None)
my_agent = MyAgent(env.action_space, env.observation_space)

# Test your agent (use test_gymnasium_agent.py functions)
from uno.gymnasium_agent import RandomGymnasiumAgent
from test_gymnasium_agent import run_tournament

opponent = RandomGymnasiumAgent(env.action_space, env.observation_space)
stats = run_tournament(env, [my_agent, opponent], num_games=100)
```

## Key Points

1. **Always return a legal action**: `action in legal_actions`
2. **Access game state**: `observation['raw_obs']['hand']`, `observation['raw_obs']['target']`
3. **Action 60 = Draw card**: All other actions are playing cards
4. **Test frequently**: Run tournaments to evaluate your agent

## Full Documentation

See [TESTING_INSTRUCTIONS.md](TESTING_INSTRUCTIONS.md) for complete documentation including:
- Detailed interface explanation
- Advanced usage patterns
- Learning agent templates
- Troubleshooting guide

## File Overview

- `uno/gymnasium_agent.py` - Agent base classes and implementations
- `test_gymnasium_agent.py` - Test suite with CLI
- `example_gymnasium_agent.py` - Simple usage example
- `TESTING_INSTRUCTIONS.md` - Complete documentation
