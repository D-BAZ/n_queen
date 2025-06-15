import time
import psutil
import os
import gc
from typing import Dict, List, Tuple, Optional
import tracemalloc
import sys

class PerformanceTester:
    """
    Performance testing suite for N-Queens solver implementations.
    Tests different algorithms with various board sizes and records metrics.
    """
    
    def __init__(self):
        self.results = {}
        self.test_sizes = [10, 30, 50, 100, 200]
        self.solvers = {}
        
        # Import available solvers
        self._import_solvers()
    
    def _import_solvers(self):
        """Import all available solver implementations"""
        try:
            from nqueens_solver import NQueensSolver
            self.solvers['exhaustive'] = NQueensSolver
            print("✓ Exhaustive search solver imported")
        except ImportError:
            print("✗ Could not import exhaustive search solver (nqueens_solver.py)")
        
        try:
            from nqueens_hill_climbing import NQueensHillClimbing
            self.solvers['hill_climbing'] = NQueensHillClimbing
            print("✓ Hill climbing solver imported")
        except ImportError:
            print("✗ Could not import hill climbing solver (nqueens_hill_climbing.py)")
        
        try:
            from nqueens_simulated_annealing import NQueensSimulatedAnnealing
            self.solvers['simulated_annealing'] = NQueensSimulatedAnnealing
            print("✓ Simulated annealing solver imported")
        except ImportError:
            print("✗ Could not import simulated annealing solver (nqueens_simulated_annealing.py)")
        
        try:
            from nqueens_genetic import NQueensGenetic
            self.solvers['genetic'] = NQueensGenetic
            print("✓ Genetic algorithm solver imported")
        except ImportError:
            print("✗ Could not import genetic algorithm solver (nqueens_genetic.py)")
    
    def measure_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    def test_solver(self, solver_name: str, n: int, warning_time: int = 300) -> Dict:
        """
        Test a specific solver with given board size - keeps trying until solution found
        
        Args:
            solver_name: Name of the solver to test
            n: Board size (n x n)
            warning_time: Time in seconds after which to show warning (but continue running)
            
        Returns:
            Dictionary containing test results
        """
        if solver_name not in self.solvers:
            return {
                'success': False,
                'error': f'Solver {solver_name} not available',
                'execution_time': 0,
                'memory_used': 0,
                'peak_memory': 0
            }
        
        print(f"  Testing {solver_name} with N={n}...")
        print(f"    Will keep trying until solution found or user interrupts (Ctrl+C)")
        
        # Start memory tracking
        tracemalloc.start()
        initial_memory = self.measure_memory_usage()
        
        # Force garbage collection before test
        gc.collect()
        
        start_time = time.time()
        attempt = 0
        success = False
        error_msg = None
        solver_stats = {}
        warning_shown = False
        
        # Keep trying until success or user interrupts
        while not success:
            attempt += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Show warning if taking too long, but continue running
            if elapsed_time > warning_time and not warning_shown:
                print(f"    ⚠️  Warning: Taking longer than {warning_time}s (currently {elapsed_time:.1f}s)")
                print(f"    Still running... Press Ctrl+C to interrupt if needed")
                warning_shown = True
            
            # Show progress for long-running attempts
            if attempt > 1:
                print(f"    Attempt #{attempt} (Total time: {elapsed_time:.1f}s)")
            
            try:
                # Create solver instance with appropriate parameters
                if solver_name == 'exhaustive':
                    solver = self.solvers[solver_name](n)
                    success = solver.solve_step_by_step()
                    solver_stats = solver.get_statistics()
                        
                elif solver_name == 'hill_climbing':
                    solver = self.solvers[solver_name](n, max_restarts=min(1000, n * 10))
                    success = solver.solve_step_by_step()
                    solver_stats = solver.get_statistics()
                    
                elif solver_name == 'simulated_annealing':
                    # Adjust parameters based on board size
                    initial_temp = max(100.0, n * 2)
                    cooling_rate = 0.95
                    solver = self.solvers[solver_name](n, initial_temp=initial_temp, 
                                                     cooling_rate=cooling_rate)
                    success = solver.solve()
                    solver_stats = solver.get_statistics()
                    
                elif solver_name == 'genetic':
                    # Scale parameters with board size
                    pop_size = min(200, max(50, n * 4))
                    mutation_rate = 0.1
                    elite_size = max(5, pop_size // 10)
                    max_gens = min(2000, max(500, n * 20))
                    
                    solver = self.solvers[solver_name](n, population_size=pop_size,
                                                     mutation_rate=mutation_rate,
                                                     elite_size=elite_size,
                                                     max_generations=max_gens)
                    success = solver.solve_step_by_step()
                    solver_stats = solver.get_statistics()
                
                # If not successful and it's a stochastic algorithm, try again
                if not success and solver_name in ['hill_climbing', 'simulated_annealing', 'genetic']:
                    print(f"    Attempt #{attempt} failed, trying again...")
                    continue
                elif not success and solver_name == 'exhaustive':
                    # For exhaustive search, if it fails, there's no solution or it's an error
                    error_msg = "No solution found (exhaustive search complete)"
                    break
                    
            except KeyboardInterrupt:
                print(f"\n    Interrupted by user after {elapsed_time:.1f}s")
                raise
            except Exception as e:
                error_msg = str(e)
                print(f"    Error in attempt #{attempt}: {error_msg}")
                # For critical errors, don't retry
                if "not available" in str(e).lower() or "import" in str(e).lower():
                    break
                # For other errors, try again after a short delay
                time.sleep(1)
                continue
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Get memory usage
        current_memory = self.measure_memory_usage()
        memory_used = max(0, current_memory - initial_memory)
        
        # Get peak memory from tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        peak_memory_mb = peak / 1024 / 1024  # Convert to MB
        tracemalloc.stop()
        
        # Force garbage collection after test
        gc.collect()
        
        result = {
            'success': success,
            'execution_time': execution_time,
            'memory_used': memory_used,
            'peak_memory': peak_memory_mb,
            'solver_stats': solver_stats,
            'board_size': n,
            'solver_name': solver_name,
            'attempts': attempt
        }
        
        if error_msg:
            result['error'] = error_msg
        
        # Print immediate results
        if success:
            print(f"    ✓ Success in {execution_time:.3f}s after {attempt} attempt(s)")
            print(f"    Memory: {memory_used:.1f}MB (Peak: {peak_memory_mb:.1f}MB)")
            if solver_stats and 'solutions_found' in solver_stats:
                print(f"    Solutions found: {solver_stats['solutions_found']}")
        else:
            print(f"    ✗ Failed after {execution_time:.3f}s and {attempt} attempt(s)")
            if error_msg:
                print(f"    Error: {error_msg}")
        
        return result
    
    def run_comprehensive_test(self):
        """Run comprehensive performance tests for all solvers and board sizes"""
        print("=" * 80)
        print("N-QUEENS SOLVER PERFORMANCE TESTING SUITE")
        print("=" * 80)
        print(f"Testing board sizes: {self.test_sizes}")
        print(f"Available solvers: {list(self.solvers.keys())}")
        print()
        
        # Initialize results structure
        for solver_name in self.solvers.keys():
            self.results[solver_name] = {}
        
        # Run tests for each solver and board size
        for n in self.test_sizes:
            print(f"Testing N = {n}")
            print("-" * 40)
            
            for solver_name in self.solvers.keys():
                # Set warning time based on solver type and board size
                if solver_name == 'exhaustive':
                    warning_time = max(300, n * 10)  # Warning after 5+ minutes for exhaustive
                elif solver_name == 'genetic':
                    warning_time = max(120, n * 2)  # Warning after 2+ minutes for genetic
                else:
                    warning_time = max(60, n)  # Warning after 1+ minute for others
                
                try:
                    result = self.test_solver(solver_name, n, warning_time)
                    self.results[solver_name][n] = result
                except KeyboardInterrupt:
                    print(f"\n  User interrupted testing of {solver_name} with N={n}")
                    # Store partial result
                    self.results[solver_name][n] = {
                        'success': False,
                        'error': 'Interrupted by user',
                        'execution_time': 0,
                        'memory_used': 0,
                        'peak_memory': 0,
                        'attempts': 0
                    }
                    # Ask user if they want to continue with next solver/size
                    try:
                        response = input("  Continue with next test? (y/n): ").lower().strip()
                        if response.startswith('n'):
                            print("  Stopping all tests...")
                            return
                    except KeyboardInterrupt:
                        print("\n  Stopping all tests...")
                        return
                
                # Small delay between tests
                time.sleep(0.5)
            
            print()
    
    def generate_summary_report(self):
        """Generate and print a comprehensive summary report"""
        print("=" * 80)
        print("PERFORMANCE TEST SUMMARY REPORT")
        print("=" * 80)
        
        # Create summary table
        print("\nEXECUTION TIME SUMMARY (seconds)")
        print("-" * 80)
        header = f"{'Solver':<20} {'N=10':<10} {'N=30':<10} {'N=50':<10} {'N=100':<12} {'N=200':<12}"
        print(header)
        print("-" * 80)
        
        for solver_name in self.solvers.keys():
            row = f"{solver_name:<20}"
            for n in self.test_sizes:
                if n in self.results[solver_name]:
                    result = self.results[solver_name][n]
                    if result['success']:
                        time_str = f"{result['execution_time']:.3f}s"
                    else:
                        time_str = "FAILED"
                    row += f" {time_str:<9}"
                else:
                    row += f" {'N/A':<9}"
            print(row)
        
        print("\nMEMORY USAGE SUMMARY (MB)")
        print("-" * 80)
        print(header.replace("seconds)", "MB)"))
        print("-" * 80)
        
        for solver_name in self.solvers.keys():
            row = f"{solver_name:<20}"
            for n in self.test_sizes:
                if n in self.results[solver_name]:
                    result = self.results[solver_name][n]
                    if result['success']:
                        mem_str = f"{result['peak_memory']:.1f}MB"
                    else:
                        mem_str = "FAILED"
                    row += f" {mem_str:<9}"
                else:
                    row += f" {'N/A':<9}"
            print(row)
        
        # Detailed results for successful solves
        print("\nDETAILED RESULTS")
        print("-" * 80)
        
        for solver_name in self.solvers.keys():
            print(f"\n{solver_name.upper()} SOLVER:")
            print("-" * 40)
            
            for n in self.test_sizes:
                if n in self.results[solver_name]:
                    result = self.results[solver_name][n]
                    print(f"  N = {n}:")
                    
                    if result['success']:
                        print(f"    Status: SUCCESS")
                        print(f"    Execution Time: {result['execution_time']:.4f} seconds")
                        print(f"    Memory Used: {result['memory_used']:.2f} MB")
                        print(f"    Peak Memory: {result['peak_memory']:.2f} MB")
                        
                        # Print solver-specific statistics
                        if 'solver_stats' in result and result['solver_stats']:
                            stats = result['solver_stats']
                            if 'total_steps' in stats:
                                print(f"    Total Steps: {stats['total_steps']:,}")
                            if 'solutions_found' in stats:
                                print(f"    Solutions Found: {stats['solutions_found']}")
                            if 'backtrack_count' in stats:
                                print(f"    Backtracks: {stats['backtrack_count']:,}")
                            if 'restart_count' in stats:
                                print(f"    Restarts: {stats['restart_count']:,}")
                            if 'generations' in stats:
                                print(f"    Generations: {stats['generations']:,}")
                            if 'acceptance_rate' in stats:
                                print(f"    Acceptance Rate: {stats['acceptance_rate']:.1%}")
                        
                        # Show number of attempts for stochastic algorithms
                        if 'attempts' in result and result['attempts'] > 1:
                            print(f"    Attempts Made: {result['attempts']}")
                            
                    else:
                        print(f"    Status: FAILED")
                        if 'error' in result:
                            print(f"    Error: {result['error']}")
                        print(f"    Time Elapsed: {result['execution_time']:.4f} seconds")
                        if 'attempts' in result:
                            print(f"    Attempts Made: {result['attempts']}")
                    print()
    
    def save_results_to_file(self, filename: str = "nqueens_performance_results.txt"):
        """Save detailed results to a text file"""
        try:
            with open(filename, 'w') as f:
                f.write("N-QUEENS SOLVER PERFORMANCE TEST RESULTS\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("Test Configuration:\n")
                f.write(f"Board Sizes: {self.test_sizes}\n")
                f.write(f"Solvers Tested: {list(self.solvers.keys())}\n")
                f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Write detailed results
                for solver_name in self.solvers.keys():
                    f.write(f"{solver_name.upper()} SOLVER RESULTS:\n")
                    f.write("-" * 30 + "\n")
                    
                    for n in self.test_sizes:
                        if n in self.results[solver_name]:
                            result = self.results[solver_name][n]
                            f.write(f"N = {n}:\n")
                            f.write(f"  Success: {result['success']}\n")
                            f.write(f"  Execution Time: {result['execution_time']:.4f}s\n")
                            f.write(f"  Memory Used: {result['memory_used']:.2f}MB\n")
                            f.write(f"  Peak Memory: {result['peak_memory']:.2f}MB\n")
                            
                            if 'error' in result:
                                f.write(f"  Error: {result['error']}\n")
                            
                            if result['success'] and 'solver_stats' in result:
                                stats = result['solver_stats']
                                f.write(f"  Solver Statistics: {stats}\n")
                            
                            if 'attempts' in result:
                                f.write(f"  Attempts Made: {result['attempts']}\n")
                            
                            f.write("\n")
                    f.write("\n")
            
            print(f"Detailed results saved to: {filename}")
            
        except Exception as e:
            print(f"Error saving results to file: {e}")

def main():
    """Main function to run the performance testing suite"""
    print("N-Queens Solver Performance Testing Suite - UNLIMITED RETRIES")
    print("This will test all available solvers with board sizes: 10, 30, 50, 100, 200")
    print("⚠️  WARNING: Will keep trying until solution found or you press Ctrl+C!")
    print("   - Exhaustive search may take extremely long for large boards")
    print("   - Stochastic algorithms will retry until they find a solution")
    print("   - Press Ctrl+C at any time to interrupt a test")
    print()
    
    # Check available memory
    available_memory = psutil.virtual_memory().available / (1024**3)  # GB
    print(f"Available system memory: {available_memory:.1f} GB")
    
    if available_memory < 2:
        print("Warning: Low available memory. Large tests may fail.")
    
    print()
    
    # Create and run the performance tester
    tester = PerformanceTester()
    
    if not tester.solvers:
        print("Error: No solver implementations found!")
        print("Make sure the following files exist in the same directory:")
        print("- nqueens_solver.py (exhaustive search)")
        print("- nqueens_hill_climbing.py (hill climbing)")
        print("- nqueens_simulated_annealing.py (simulated annealing)")
        print("- nqueens_genetic.py (genetic algorithm)")
        return
    
    try:
        # Run comprehensive tests
        tester.run_comprehensive_test()
        
        # Generate and display summary report
        tester.generate_summary_report()
        
        # Save results to file
        tester.save_results_to_file()
        
        print("\nPerformance testing completed successfully!")
        print("Check the generated report file for detailed results.")
        
    except KeyboardInterrupt:
        print("\nTesting interrupted by user.")
        print("Generating report for completed tests...")
        try:
            tester.generate_summary_report()
            tester.save_results_to_file()
        except:
            print("Could not generate final report.")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()