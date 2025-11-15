"""
Evolutionary Training System for DQN Agents

Trains multiple DQN agents against each other, keeping the best performers
and evolving them over generations.
"""

import os
import pickle
import signal
import sys
from datetime import datetime
from uno.env import UnoEnv
from uno.dqn_agent import DQNAgent
import numpy as np


class EvolutionaryTrainer:
    """Manages evolutionary training of DQN agents."""
    
    def __init__(self, population_size=4, generations=1000, 
                 episodes_per_gen=50, save_dir='evolution'):
        self.population_size = population_size
        self.generations = generations
        self.episodes_per_gen = episodes_per_gen
        self.save_dir = save_dir
        self.generation = 0
        self.best_fitness_history = []
        self.should_quit = False
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f'{save_dir}/backups', exist_ok=True)
        
        # Setup signal handler for graceful quit
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C to save and quit gracefully."""
        print("\n\n[!] Quit signal received. Saving current state...")
        self.should_quit = True
    
    def _create_agent(self, env):
        """Create a new DQN agent."""
        return DQNAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            learning_rate=0.001,
            discount_factor=0.95,
            epsilon_start=0.3,  # Lower initial exploration for evolved agents
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_capacity=10000,
            batch_size=32,
            target_update_freq=100,
            hidden_size=128
        )
    
    def _evaluate_fitness(self, agent1, agent2, env, num_games=10):
        """Evaluate fitness of agent1 against agent2."""
        wins = 0
        
        for _ in range(num_games):
            obs, info = env.reset()
            done = False
            
            while not done:
                current_player = info['player_id']
                legal_actions = info['legal_actions']
                
                if current_player == 0:
                    action = agent1.select_action(obs, legal_actions)
                else:
                    action = agent2.select_action(obs, legal_actions)
                
                next_obs, reward, done, truncated, info = env.step(action)
                obs = next_obs
            
            # Check winner
            payoffs = env.env.get_payoffs()
            if payoffs[0] > payoffs[1]:
                wins += 1
        
        return wins / num_games
    
    def _train_against(self, agent1, agent2, env, episodes):
        """Train agent1 against agent2 for specified episodes."""
        agents = [agent1, agent2]
        
        for episode in range(episodes):
            obs, info = env.reset()
            done = False
            
            while not done:
                current_player = info['player_id']
                legal_actions = info['legal_actions']
                
                agent = agents[current_player]
                action = agent.select_action(obs, legal_actions)
                
                next_obs, reward, done, truncated, info = env.step(action)
                
                # Only agent1 learns
                if current_player == 0:
                    agent1.learn(obs, action, reward, next_obs, done, info)
                
                obs = next_obs
    
    def _tournament_fitness(self, agents, env):
        """Run tournament to determine fitness of all agents."""
        fitness_scores = np.zeros(len(agents))
        
        # Each agent plays against all others
        for i in range(len(agents)):
            for j in range(len(agents)):
                if i != j:
                    # Disable exploration for evaluation
                    orig_epsilon_i = agents[i].epsilon
                    orig_epsilon_j = agents[j].epsilon
                    agents[i].epsilon = 0.0
                    agents[j].epsilon = 0.0
                    
                    fitness = self._evaluate_fitness(agents[i], agents[j], env, num_games=5)
                    fitness_scores[i] += fitness
                    
                    # Restore epsilon
                    agents[i].epsilon = orig_epsilon_i
                    agents[j].epsilon = orig_epsilon_j
        
        return fitness_scores
    
    def _save_agent(self, agent, filename):
        """Save agent to file."""
        filepath = os.path.join(self.save_dir, filename)
        agent.save(filepath)
    
    def _load_agent(self, env, filename):
        """Load agent from file."""
        filepath = os.path.join(self.save_dir, filename)
        agent = self._create_agent(env)
        if os.path.exists(filepath):
            agent.load(filepath)
        return agent
    
    def _save_checkpoint(self, agents, fitness_scores):
        """Save current generation state."""
        checkpoint = {
            'generation': self.generation,
            'fitness_scores': fitness_scores,
            'best_fitness_history': self.best_fitness_history,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save checkpoint info
        checkpoint_path = os.path.join(self.save_dir, 'checkpoint.pkl')
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Save all agents
        for i, agent in enumerate(agents):
            self._save_agent(agent, f'agent_{i}.pkl')
    
    def _load_checkpoint(self, env):
        """Load checkpoint if exists."""
        checkpoint_path = os.path.join(self.save_dir, 'checkpoint.pkl')
        
        if not os.path.exists(checkpoint_path):
            return None
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Load agents
        agents = []
        for i in range(self.population_size):
            agent = self._load_agent(env, f'agent_{i}.pkl')
            agents.append(agent)
        
        self.generation = checkpoint['generation']
        self.best_fitness_history = checkpoint['best_fitness_history']
        
        return agents
    
    def run(self, resume=True):
        """Run evolutionary training."""
        env = UnoEnv(render_mode=None)
        
        # Try to load checkpoint
        if resume:
            agents = self._load_checkpoint(env)
            if agents:
                print(f"[*] Resumed from generation {self.generation}")
            else:
                agents = None
        else:
            agents = None
        
        # Initialize population if needed
        if agents is None:
            print("[*] Initializing population...")
            agents = [self._create_agent(env) for _ in range(self.population_size)]
            self.generation = 0
        
        print(f"[*] Population: {self.population_size} agents")
        print(f"[*] Target generations: {self.generations}")
        print(f"[*] Episodes per generation: {self.episodes_per_gen}")
        print(f"[*] Press Ctrl+C to save and quit\n")
        
        # Evolution loop
        while self.generation < self.generations and not self.should_quit:
            self.generation += 1
            
            # Tournament to evaluate fitness
            fitness_scores = self._tournament_fitness(agents, env)
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            self.best_fitness_history.append(best_fitness)
            
            # Print progress (minimal)
            if self.generation % 10 == 0 or self.generation == 1:
                avg_fitness = np.mean(fitness_scores)
                print(f"Gen {self.generation:4d} | Best: {best_fitness:.3f} | "
                      f"Avg: {avg_fitness:.3f} | Eps: {agents[best_idx].epsilon:.3f}")
            
            # Backup best agent before modifications
            best_agent_backup = os.path.join(self.save_dir, 'backups', 
                                            f'best_gen{self.generation}.pkl')
            agents[best_idx].save(best_agent_backup)
            
            # Train top performers against each other
            sorted_indices = np.argsort(fitness_scores)[::-1]
            top_half = sorted_indices[:self.population_size // 2]
            
            for idx in top_half:
                # Train against best agent
                self._train_against(agents[idx], agents[best_idx], env, 
                                  self.episodes_per_gen // 2)
                
                # Train against random top performer
                opponent_idx = np.random.choice(top_half)
                if opponent_idx != idx:
                    self._train_against(agents[idx], agents[opponent_idx], env,
                                      self.episodes_per_gen // 2)
            
            # Replace bottom half with copies of top performers
            bottom_half = sorted_indices[self.population_size // 2:]
            for i, bottom_idx in enumerate(bottom_half):
                # Copy from top half
                top_idx = top_half[i % len(top_half)]
                
                # Save and reload to copy
                temp_path = os.path.join(self.save_dir, 'temp_copy.pkl')
                agents[top_idx].save(temp_path)
                agents[bottom_idx] = self._create_agent(env)
                agents[bottom_idx].load(temp_path)
                os.remove(temp_path)
                
                # Add some exploration for diversity
                agents[bottom_idx].epsilon = 0.5
            
            # Save checkpoint every 50 generations
            if self.generation % 50 == 0:
                self._save_checkpoint(agents, fitness_scores)
                print(f"[*] Checkpoint saved at generation {self.generation}")
            
            if self.should_quit:
                break
        
        # Final save
        print("\n[*] Saving final state...")
        self._save_checkpoint(agents, fitness_scores)
        
        # Save best agent separately
        best_idx = np.argmax(fitness_scores)
        best_final_path = os.path.join(self.save_dir, 'best_agent.pkl')
        agents[best_idx].save(best_final_path)
        
        print(f"\n[✓] Training complete!")
        print(f"[✓] Generations: {self.generation}")
        print(f"[✓] Best fitness: {self.best_fitness_history[-1]:.3f}")
        print(f"[✓] Best agent saved to: {best_final_path}")
        print(f"[✓] All backups in: {self.save_dir}/backups/")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evolutionary DQN Training')
    parser.add_argument('--population', type=int, default=4,
                       help='Population size (default: 4)')
    parser.add_argument('--generations', type=int, default=1000,
                       help='Number of generations (default: 1000)')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Episodes per generation (default: 50)')
    parser.add_argument('--save-dir', type=str, default='evolution',
                       help='Save directory (default: evolution)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh (don\'t resume)')
    
    args = parser.parse_args()
    
    trainer = EvolutionaryTrainer(
        population_size=args.population,
        generations=args.generations,
        episodes_per_gen=args.episodes,
        save_dir=args.save_dir
    )
    
    trainer.run(resume=not args.no_resume)


if __name__ == '__main__':
    main()
