import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
from nqueens_genetic import NQueensGenetic
from nqueens_solver_dfs import OptimizedNQueensSolver
from nqueens_hill_climbing import OptimizedNQueensHillClimbing, solve_nqueens_hill_climbing_optimized

class NQueensComparison:
    def __init__(self, board_sizes: List[int]):
        """
        Initialize the comparison with different board sizes to test.
        
        Args:
            board_sizes: List of board sizes to test
        """
        self.board_sizes = board_sizes
        self.results: Dict[str, Dict[int, Dict[str, Any]]] = {}
        
    def run_genetic_algorithm(self, n: int) -> Dict[str, Any]:
        """Run genetic algorithm solver and return timing results."""
        solver = NQueensGenetic(n, population_size=100, mutation_rate=0.1, 
                              elite_size=20, max_generations=500)
        
        start_time = time.time()
        success = solver.solve_step_by_step()
        end_time = time.time()
        
        return {
            'time': end_time - start_time,
            'success': success,
            'generations': solver.generations_run
        }
    
    def run_optimized_solver(self, n: int) -> Dict[str, Any]:
        """Run optimized backtracking solver and return timing results."""
        solver = OptimizedNQueensSolver(n)
        
        start_time = time.time()
        success = solver.solve_step_by_step(find_all=False)
        end_time = time.time()
        
        return {
            'time': end_time - start_time,
            'success': success,
            'steps': solver.total_steps
        }
    
    def run_hill_climbing(self, n: int) -> Dict[str, Any]:
        """Run hill climbing solver and return timing results."""
        solver = OptimizedNQueensHillClimbing(n, max_restarts=50, max_sideways=100)
        
        start_time = time.time()
        success = solver.solve_step_by_step(use_smart_restart=True, allow_sideways=True)
        end_time = time.time()
        
        return {
            'time': end_time - start_time,
            'success': success,
            'steps': solver.total_steps,
            'restarts': solver.restart_count
        }
    
    def run_simulated_annealing(self, n: int) -> Dict[str, Any]:
        """Run simulated annealing solver and return timing results."""
        solver = OptimizedNQueensHillClimbing(n, max_restarts=3)  # Only one run needed for SA
        
        start_time = time.time()
        success = solver.solve_with_simulated_annealing(
            initial_temp=100.0,
            cooling_rate=0.95
        )
        end_time = time.time()
        
        return {
            'time': end_time - start_time,
            'success': success,
            'steps': solver.total_steps
        }
    
    def run_comparison(self, num_trials: int = 5) -> None:
        """
        Run comparison tests for all methods and board sizes.
        
        Args:
            num_trials: Number of trials to run for each combination
        """
        methods = {
            'Genetic Algorithm': self.run_genetic_algorithm,
            'Optimized Solver': self.run_optimized_solver,
            'Hill Climbing': self.run_hill_climbing,
            'Simulated Annealing': self.run_simulated_annealing
        }
        
        for method_name, method_func in methods.items():
            self.results[method_name] = {}
            print(f"\nRunning {method_name} tests...")
            
            for n in self.board_sizes:
                print(f"Testing board size {n}...")
                trials_data = []
                
                for trial in range(num_trials):
                    try:
                        result = method_func(n)
                        if result['success']:
                            trials_data.append(result)
                            print(f"  Trial {trial + 1}: {result['time']:.4f} seconds")
                        else:
                            print(f"  Trial {trial + 1}: Failed to find solution")
                    except Exception as e:
                        print(f"  Trial {trial + 1}: Error - {str(e)}")
                
                if trials_data:
                    avg_time = sum(r['time'] for r in trials_data) / len(trials_data)
                    self.results[method_name][n] = {
                        'avg_time': avg_time,
                        'min_time': min(r['time'] for r in trials_data),
                        'max_time': max(r['time'] for r in trials_data),
                        'success_rate': len(trials_data) / num_trials
                    }
    
    def plot_results(self, save_path: str = None) -> None:
        """
        Create plots comparing the performance of different methods.
        
        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(15, 10))
        
        # Plot average solving times
        plt.subplot(2, 1, 1)
        markers = ['o', 's', '^', 'D']  # Different markers for each method
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Different colors for each method
        
        for (method_name, results), marker, color in zip(self.results.items(), markers, colors):
            sizes = sorted(results.keys())
            times = [results[n]['avg_time'] for n in sizes]
            
            # Plot average time with error bars
            min_times = [results[n]['min_time'] for n in sizes]
            max_times = [results[n]['max_time'] for n in sizes]
            plt.errorbar(sizes, times, 
                        yerr=[np.array(times) - np.array(min_times), 
                              np.array(max_times) - np.array(times)],
                        marker=marker, color=color, label=method_name,
                        capsize=5, capthick=1, elinewidth=1)
        
        plt.title('N-Queens Solver Comparison (with min/max ranges)')
        plt.xlabel('Board Size (n)')
        plt.ylabel('Average Time (seconds)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.yscale('log')  # Use log scale for better visualization
        
        # Plot success rates
        plt.subplot(2, 1, 2)
        for (method_name, results), marker, color in zip(self.results.items(), markers, colors):
            sizes = sorted(results.keys())
            success_rates = [results[n]['success_rate'] * 100 for n in sizes]
            plt.plot(sizes, success_rates, marker=marker, color=color, 
                    label=method_name, linewidth=2, markersize=8)
        
        plt.title('Solution Success Rate')
        plt.xlabel('Board Size (n)')
        plt.ylabel('Success Rate (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary(self) -> None:
        """Print a summary of the comparison results."""
        print("\nPerformance Summary:")
        print("=" * 80)
        print(f"{'Board Size':<12} {'Method':<20} {'Avg Time (s)':<15} {'Success Rate':<12}")
        print("-" * 80)
        
        for n in sorted(self.board_sizes):
            for method_name, results in self.results.items():
                if n in results:
                    result = results[n]
                    print(f"{n:<12} {method_name:<20} {result['avg_time']:<15.4f} {result['success_rate']*100:<11.1f}%")
        print("=" * 80)

def main():
    # Test with various board sizes
    board_sizes = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    
    # Create comparison object
    comparison = NQueensComparison(board_sizes)
    
    # Run comparison
    print("Starting N-Queens solver comparison...")
    comparison.run_comparison(num_trials=3)
    
    # Print results
    comparison.print_summary()
    
    # Plot and save results
    comparison.plot_results(save_path='nqueens_comparison.png')

if __name__ == "__main__":
    main() 