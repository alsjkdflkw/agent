"""
Optimized DQN Training Script - Minimal Output, Maximum Speed

Fast training with automatic save/quit and minimal printing.
"""

import os
import signal
import sys
from datetime import datetime
from uno.env import UnoEnv
from uno.dqn_agent import DQNAgent
from uno.gymnasium_agent import RandomGymnasiumAgent


class FastTrainer:
    """Fast trainer with minimal output."""
    
    def __init__(self, save_dir='models'):
        self.save_dir = save_dir
        self.should_quit = False
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f'{save_dir}/backups', exist_ok=True)
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C to save and quit."""
        print("\n[!] Saving...")
        self.should_quit = True
    
    def train(self, episodes=1000, eval_freq=100, save_freq=500):
        """Train with minimal output."""
        env = UnoEnv(render_mode=None)
        
        # Check for existing model
        model_path = os.path.join(self.save_dir, 'agent.pkl')
        agent = DQNAgent(env.action_space, env.observation_space)
        
        start_episode = 0
        if os.path.exists(model_path):
            agent.load(model_path)
            start_episode = agent.episodes
            print(f"[*] Resumed from episode {start_episode}")
        
        opponent = RandomGymnasiumAgent(env.action_space, env.observation_space)
        agents = [agent, opponent]
        
        print(f"[*] Training to {episodes} episodes (Ctrl+C to save & quit)")
        
        for episode in range(start_episode, episodes):
            obs, info = env.reset()
            done = False
            
            while not done:
                current_player = info['player_id']
                legal_actions = info['legal_actions']
                
                agent_obj = agents[current_player]
                action = agent_obj.select_action(obs, legal_actions)
                
                next_obs, reward, done, truncated, info = env.step(action)
                
                if current_player == 0:
                    agent.learn(obs, action, reward, next_obs, done, info)
                
                obs = next_obs
            
            # Minimal progress output
            if (episode + 1) % eval_freq == 0:
                stats = agent.get_stats()
                print(f"Ep {episode+1:5d} | Eps {stats['epsilon']:.3f} | "
                      f"Loss {stats['avg_loss']:.4f}")
            
            # Periodic save
            if (episode + 1) % save_freq == 0:
                # Backup before overwrite
                backup_path = os.path.join(self.save_dir, 'backups', 
                                          f'agent_ep{episode+1}.pkl')
                agent.save(backup_path)
                agent.save(model_path)
            
            if self.should_quit:
                break
        
        # Final save
        backup_path = os.path.join(self.save_dir, 'backups',
                                  f'agent_final_ep{agent.episodes}.pkl')
        agent.save(backup_path)
        agent.save(model_path)
        
        print(f"\n[✓] Saved to {model_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast DQN Training')
    parser.add_argument('--episodes', type=int, default=5000,
                       help='Training episodes (default: 5000)')
    parser.add_argument('--eval-freq', type=int, default=100,
                       help='Print frequency (default: 100)')
    parser.add_argument('--save-freq', type=int, default=500,
                       help='Save frequency (default: 500)')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Save directory (default: models)')
    
    args = parser.parse_args()
    
    trainer = FastTrainer(save_dir=args.save_dir)
    trainer.train(
        episodes=args.episodes,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq
    )


if __name__ == '__main__':
    main()
