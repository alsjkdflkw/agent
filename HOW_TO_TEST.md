# How to Test the Gymnasium API Implementation

## Quick Test (2 minutes)

Run the example script to see a complete working demonstration:

```bash
python example_gymnasium_agent.py
```

This will:
- Create a SimpleAgent and RandomAgent
- Play a single game
- Run a 100-game tournament
- Show win statistics
- Demonstrate save/load functionality

**Expected output**: SimpleAgent should win approximately 45-55% of games (variance is normal).

---

## Full Test Suite (3-5 minutes)

Run all automated tests:

```bash
python test_gymnasium_agent.py
```

This runs 4 test scenarios:
1. **Random vs Random** - Verifies environment works correctly (should be ~50/50)
2. **Heuristic vs Random** - Tests strategic decision making
3. **Custom Agent** - Verifies interface extensibility
4. **Single Game with Rendering** - Visual verification (shows game state)

**Expected result**: All tests should pass with ✓ marks.

---

## Specific Tests

### Test Only Random Agents
```bash
python test_gymnasium_agent.py --test random --games 100
```
Expected: Win rates between 40-60% (balanced)

### Test Heuristic Agent
```bash
python test_gymnasium_agent.py --test heuristic --games 100
```
Expected: Heuristic agent uses strategy (may not always win due to game variance)

### Test with Visualization
```bash
python test_gymnasium_agent.py --test render
```
Expected: Shows detailed game state for one game (card displays, actions, etc.)

### Test Custom Agent
```bash
python test_gymnasium_agent.py --test custom --games 50
```
Expected: Custom agent implementation works correctly

---

## Create and Test Your Own Agent

### Step 1: Create Your Agent

Create a file `my_custom_agent.py`:

```python
from uno.gymnasium_agent import GymnasiumAgent
from uno.env import UnoEnv

class MyAgent(GymnasiumAgent):
    def select_action(self, observation, legal_actions):
        # Your logic here
        # Example: Prefer high-value actions
        playable = [a for a in legal_actions if a != 60]
        return max(playable) if playable else 60

# Test it
if __name__ == "__main__":
    from uno.gymnasium_agent import RandomGymnasiumAgent
    from test_gymnasium_agent import run_tournament
    
    env = UnoEnv(render_mode=None)
    my_agent = MyAgent(env.action_space, env.observation_space)
    opponent = RandomGymnasiumAgent(env.action_space, env.observation_space)
    
    stats = run_tournament(env, [my_agent, opponent], num_games=100)
```

### Step 2: Run Your Test

```bash
python my_custom_agent.py
```

---

## Verify Installation

Check that all dependencies are installed:

```bash
python -c "import gymnasium; import rlcard; import numpy; print('✓ All dependencies installed')"
```

Expected output: `✓ All dependencies installed`

---

## Check What Was Added

List all new files:

```bash
ls -lh GYMNASIUM*.md TESTING*.md example_gymnasium_agent.py test_gymnasium_agent.py uno/gymnasium_agent.py
```

Expected: Should show all 5 main files (3 Python files + 3 documentation files)

---

## Troubleshooting

### If example fails
```bash
# Check dependencies
pip install gymnasium rlcard numpy joblib

# Try again
python example_gymnasium_agent.py
```

### If tests fail
```bash
# Run with more verbose output
python test_gymnasium_agent.py --test random --games 10
```

### If imports fail
```bash
# Make sure you're in the repository root
cd /path/to/agent
python -c "from uno.gymnasium_agent import GymnasiumAgent; print('✓ Imports work')"
```

---

## Performance Benchmarks

On a typical system:
- **Single game**: < 1 second
- **100 games**: ~10-30 seconds
- **Example script**: ~30 seconds
- **Full test suite**: ~2-3 minutes

---

## Success Criteria

✅ Example script runs without errors
✅ Test suite passes all 4 tests
✅ Can create custom agents
✅ Tournament statistics are reasonable (30-70% win rates)
✅ Save/load functionality works

---

## Next Steps After Testing

1. Read [GYMNASIUM_QUICKSTART.md](GYMNASIUM_QUICKSTART.md) for quick reference
2. Read [TESTING_INSTRUCTIONS.md](TESTING_INSTRUCTIONS.md) for full documentation
3. Review [example_gymnasium_agent.py](example_gymnasium_agent.py) for code patterns
4. Create your own agent using the templates
5. Test and iterate on your agent's strategy

---

## Getting Help

- **Quick issues**: Check [TESTING_INSTRUCTIONS.md](TESTING_INSTRUCTIONS.md) troubleshooting section
- **Code examples**: Review [example_gymnasium_agent.py](example_gymnasium_agent.py)
- **API reference**: See docstrings in [uno/gymnasium_agent.py](uno/gymnasium_agent.py)

---

**Summary**: Run `python example_gymnasium_agent.py` first to verify everything works!
