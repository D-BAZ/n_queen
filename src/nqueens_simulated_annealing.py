import random
import math
import time
from collections import defaultdict

class OptimizedNQueensSimulatedAnnealing:
    def __init__(self, n, initial_temp=None, cooling_rate=0.99, min_temp=0.001, 
                 max_iterations=None, adaptive_cooling=True):
        """
        Optimized Simulated Annealing solver for N-Queens problem.
        
        Args:
            n: Size of the chessboard (n x n)
            initial_temp: Starting temperature (auto-calculated if None)
            cooling_rate: Rate at which temperature decreases
            min_temp: Minimum temperature threshold
            max_iterations: Max iterations per temperature (auto-calculated if None)
            adaptive_cooling: Use adaptive cooling schedule
        """
        self.n = n
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.adaptive_cooling = adaptive_cooling
        
        # Auto-calculate parameters based on problem size
        self.initial_temp = initial_temp or max(10.0, n * 2.0)
        self.max_iterations = max_iterations or max(100, n * 10)
        
        # Conflict tracking arrays for O(1) conflict updates
        self.row_conflicts = [0] * n
        self.diag1_conflicts = [0] * (2 * n - 1)  # / diagonal
        self.diag2_conflicts = [0] * (2 * n - 1)  # \ diagonal
        
        # Statistics
        self.stats = {
            'total_steps': 0,
            'solve_time': 0,
            'temperature_changes': 0,
            'accepted_moves': 0,
            'rejected_moves': 0,
            'solutions_found': 0,
            'best_conflicts': float('inf'),
            'plateau_count': 0
        }
        
        # Current state
        self.board = None
        self.current_conflicts = 0
        self.best_board = None
        self.best_conflicts = float('inf')
        
        # Step tracking (memory efficient)
        self.steps = []
        self.track_steps = False
        self.max_steps_tracked = 1000
        
    def _get_diag1_index(self, row, col):
        """Get index for / diagonal (row - col + n - 1)"""
        return row - col + self.n - 1
    
    def _get_diag2_index(self, row, col):
        """Get index for \ diagonal (row + col)"""
        return row + col
    
    def _update_conflicts_fast(self, col, old_row, new_row, delta):
        """Fast conflict update using tracking arrays"""
        # Update row conflicts
        self.row_conflicts[old_row] += delta
        self.row_conflicts[new_row] -= delta
        
        # Update diagonal conflicts
        self.diag1_conflicts[self._get_diag1_index(old_row, col)] += delta
        self.diag1_conflicts[self._get_diag1_index(new_row, col)] -= delta
        
        self.diag2_conflicts[self._get_diag2_index(old_row, col)] += delta
        self.diag2_conflicts[self._get_diag2_index(new_row, col)] -= delta
    
    def _initialize_conflict_arrays(self, board):
        """Initialize conflict tracking arrays"""
        # Reset arrays
        self.row_conflicts = [0] * self.n
        self.diag1_conflicts = [0] * (2 * self.n - 1)
        self.diag2_conflicts = [0] * (2 * self.n - 1)
        
        # Count conflicts
        for col in range(self.n):
            row = board[col]
            self.row_conflicts[row] += 1
            self.diag1_conflicts[self._get_diag1_index(row, col)] += 1
            self.diag2_conflicts[self._get_diag2_index(row, col)] += 1
    
    def count_conflicts_fast(self):
        """Fast conflict counting using tracking arrays"""
        conflicts = 0
        
        # Row conflicts
        for count in self.row_conflicts:
            if count > 1:
                conflicts += count * (count - 1) // 2
        
        # Diagonal conflicts
        for count in self.diag1_conflicts:
            if count > 1:
                conflicts += count * (count - 1) // 2
                
        for count in self.diag2_conflicts:
            if count > 1:
                conflicts += count * (count - 1) // 2
        
        return conflicts
    
    def get_conflict_delta(self, col, old_row, new_row):
        """Calculate change in conflicts for a move without actually making it"""
        delta = 0
        
        # Row conflict change
        delta -= max(0, self.row_conflicts[old_row] - 1)
        delta -= max(0, self.row_conflicts[new_row])
        delta += max(0, self.row_conflicts[old_row] - 2)
        delta += max(0, self.row_conflicts[new_row] - 1)
        
        # Diagonal conflict changes
        old_diag1 = self._get_diag1_index(old_row, col)
        new_diag1 = self._get_diag1_index(new_row, col)
        old_diag2 = self._get_diag2_index(old_row, col)
        new_diag2 = self._get_diag2_index(new_row, col)
        
        # / diagonal
        delta -= max(0, self.diag1_conflicts[old_diag1] - 1)
        delta -= max(0, self.diag1_conflicts[new_diag1])
        delta += max(0, self.diag1_conflicts[old_diag1] - 2)
        delta += max(0, self.diag1_conflicts[new_diag1] - 1)
        
        # \ diagonal  
        delta -= max(0, self.diag2_conflicts[old_diag2] - 1)
        delta -= max(0, self.diag2_conflicts[new_diag2])
        delta += max(0, self.diag2_conflicts[old_diag2] - 2)
        delta += max(0, self.diag2_conflicts[new_diag2] - 1)
        
        return delta
    
    def get_smart_neighbor(self):
        """Generate neighbor by focusing on conflicted queens"""
        # Find columns with conflicts
        conflicted_cols = []
        for col in range(self.n):
            row = self.board[col]
            if (self.row_conflicts[row] > 1 or 
                self.diag1_conflicts[self._get_diag1_index(row, col)] > 1 or
                self.diag2_conflicts[self._get_diag2_index(row, col)] > 1):
                conflicted_cols.append(col)
        
        # Choose column (prefer conflicted ones)
        if conflicted_cols and random.random() < 0.8:
            col = random.choice(conflicted_cols)
        else:
            col = random.randint(0, self.n - 1)
        
        old_row = self.board[col]
        
        # Find best new row for this column
        best_rows = []
        best_delta = float('inf')
        
        for new_row in range(self.n):
            if new_row == old_row:
                continue
                
            delta = self.get_conflict_delta(col, old_row, new_row)
            if delta < best_delta:
                best_delta = delta
                best_rows = [new_row]
            elif delta == best_delta:
                best_rows.append(new_row)
        
        # Choose randomly among best moves (or occasionally a random move)
        if best_rows and random.random() < 0.9:
            new_row = random.choice(best_rows)
        else:
            new_row = random.randint(0, self.n - 1)
            while new_row == old_row:
                new_row = random.randint(0, self.n - 1)
            best_delta = self.get_conflict_delta(col, old_row, new_row)
        
        return col, old_row, new_row, best_delta
    
    def acceptance_probability(self, delta_cost, temperature):
        """Calculate acceptance probability for a move"""
        if delta_cost <= 0:
            return 1.0
        if temperature <= 0:
            return 0.0
        return math.exp(-delta_cost / temperature)
    
    def adaptive_cooling_schedule(self, temperature, accepted_ratio, plateau_count):
        """Adaptive cooling based on acceptance rate and plateau detection"""
        if not self.adaptive_cooling:
            return temperature * self.cooling_rate
        
        # Slow cooling if accepting too few moves
        if accepted_ratio < 0.1:
            return temperature * 0.99
        # Fast cooling if accepting too many moves
        elif accepted_ratio > 0.9:
            return temperature * 0.8
        # Reheat if stuck in plateau
        elif plateau_count > 50:
            return temperature * 1.2
        else:
            return temperature * self.cooling_rate
    
    def solve(self, track_steps=False):
        """Solve using optimized simulated annealing"""
        self.track_steps = track_steps
        if track_steps:
            self.steps = []
            
        start_time = time.time()
        
        # Initialize with random solution
        self.board = [random.randint(0, self.n - 1) for _ in range(self.n)]
        self._initialize_conflict_arrays(self.board)
        self.current_conflicts = self.count_conflicts_fast()
        
        # Track best solution
        self.best_board = self.board[:]
        self.best_conflicts = self.current_conflicts
        
        # Reset statistics
        for key in self.stats:
            self.stats[key] = 0
        self.stats['best_conflicts'] = self.current_conflicts
        
        temperature = self.initial_temp
        plateau_count = 0
        no_improvement_steps = 0
        
        if track_steps:
            self._record_step(temperature, "Initial random state", None, None)
        
        while temperature > self.min_temp and self.current_conflicts > 0:
            self.stats['temperature_changes'] += 1
            accepted_this_temp = 0
            
            for iteration in range(self.max_iterations):
                self.stats['total_steps'] += 1
                
                if self.current_conflicts == 0:
                    self.stats['solutions_found'] = 1
                    break
                
                # Generate smart neighbor
                col, old_row, new_row, delta = self.get_smart_neighbor()
                
                # Calculate acceptance probability
                prob = self.acceptance_probability(delta, temperature)
                
                # Accept or reject move
                if random.random() < prob:
                    # Make the move
                    self.board[col] = new_row
                    self._update_conflicts_fast(col, old_row, new_row, -1)
                    self.current_conflicts += delta
                    
                    self.stats['accepted_moves'] += 1
                    accepted_this_temp += 1
                    
                    # Update best solution
                    if self.current_conflicts < self.best_conflicts:
                        self.best_board = self.board[:]
                        self.best_conflicts = self.current_conflicts
                        no_improvement_steps = 0
                        plateau_count = 0
                    else:
                        no_improvement_steps += 1
                    
                    if track_steps:
                        msg = f"Moved Q{col} {old_row}→{new_row} (Δ={delta})"
                        self._record_step(temperature, msg, True, prob)
                else:
                    self.stats['rejected_moves'] += 1
                    no_improvement_steps += 1
                    
                    if track_steps:
                        msg = f"Rejected Q{col} {old_row}→{new_row} (p={prob:.3f})"
                        self._record_step(temperature, msg, False, prob)
                
                # Early termination if no improvement for too long
                if no_improvement_steps > self.n * 100:
                    break
            
            # Update plateau counter
            if accepted_this_temp == 0:
                plateau_count += 1
            else:
                plateau_count = 0
            
            # Adaptive cooling
            acceptance_rate = accepted_this_temp / self.max_iterations
            old_temp = temperature
            temperature = self.adaptive_cooling_schedule(temperature, acceptance_rate, plateau_count)
            
            if track_steps and len(self.steps) < self.max_steps_tracked:
                self._record_step(temperature, f"Temp: {old_temp:.3f}→{temperature:.3f}", None, None)
        
        # Restore best solution found
        if self.best_conflicts < self.current_conflicts:
            self.board = self.best_board[:]
            self.current_conflicts = self.best_conflicts
        
        self.stats['solve_time'] = time.time() - start_time
        self.stats['final_conflicts'] = self.current_conflicts
        
        return self.current_conflicts == 0
    
    def _record_step(self, temperature, message, accepted, prob):
        """Record step for visualization (memory efficient)"""
        if len(self.steps) >= self.max_steps_tracked:
            return
            
        self.steps.append({
            'step': len(self.steps),
            'board': self.board[:],
            'conflicts': self.current_conflicts,
            'temperature': temperature,
            'message': message,
            'accepted': accepted,
            'probability': prob
        })
    
    def get_statistics(self):
        """Get comprehensive solver statistics"""
        total_moves = self.stats['accepted_moves'] + self.stats['rejected_moves']
        return {
            **self.stats,
            'acceptance_rate': self.stats['accepted_moves'] / max(1, total_moves),
            'avg_steps_per_temp': self.stats['total_steps'] / max(1, self.stats['temperature_changes']),
            'success_rate': 1.0 if self.stats['solutions_found'] > 0 else 0.0
        }
    
    def print_solution(self, show_conflicts=False):
        """Print the solution with optional conflict visualization"""
        if not self.board:
            print("No solution available.")
            return
        
        print(f"\nSolution (conflicts: {self.current_conflicts}):")
        print("Visual representation:")
        
        for row in range(self.n):
            line = ""
            for col in range(self.n):
                if self.board[col] == row:
                    if show_conflicts and self.current_conflicts > 0:
                        # Mark conflicted queens
                        queen_row = self.board[col]
                        has_conflict = (self.row_conflicts[queen_row] > 1 or
                                      self.diag1_conflicts[self._get_diag1_index(queen_row, col)] > 1 or
                                      self.diag2_conflicts[self._get_diag2_index(queen_row, col)] > 1)
                        line += "X " if has_conflict else "Q "
                    else:
                        line += "Q "
                else:
                    line += ". "
            print(line)
        
        if show_conflicts and self.current_conflicts > 0:
            print("(X = conflicted queen, Q = safe queen)")

def benchmark_solver(sizes=[4, 8, 12, 16], trials=5):
    """Benchmark the optimized solver"""
    print("=== Optimized N-Queens Simulated Annealing Benchmark ===\n")
    
    for n in sizes:
        print(f"Testing n={n} ({trials} trials):")
        successes = 0
        total_time = 0
        total_steps = 0
        
        for trial in range(trials):
            solver = OptimizedNQueensSimulatedAnnealing(n)
            success = solver.solve()
            stats = solver.get_statistics()
            
            if success:
                successes += 1
            total_time += stats['solve_time']
            total_steps += stats['total_steps']
            
            print(f"  Trial {trial+1}: {'✓' if success else '✗'} "
                  f"({stats['solve_time']:.3f}s, {stats['total_steps']:,} steps)")
        
        success_rate = successes / trials
        avg_time = total_time / trials
        avg_steps = total_steps / trials
        
        print(f"  Summary: {success_rate:.1%} success, {avg_time:.3f}s avg, {avg_steps:,.0f} steps avg\n")
        
        # Show one solution if successful
        if successes > 0:
            solver = OptimizedNQueensSimulatedAnnealing(n)
            if solver.solve() and n <= 12:
                solver.print_solution()
                print()

if __name__ == "__main__":
    # Quick test
    print("=== Quick Test ===")
    solver = OptimizedNQueensSimulatedAnnealing(8)
    success = solver.solve()
    stats = solver.get_statistics()
    
    print(f"8-Queens: {'✓ Solved' if success else '✗ Failed'}")
    print(f"Time: {stats['solve_time']:.3f}s")
    print(f"Steps: {stats['total_steps']:,}")
    print(f"Acceptance rate: {stats['acceptance_rate']:.1%}")
    
    if success:
        solver.print_solution()
    
    print("\n" + "="*60)
    
    # Full benchmark
    benchmark_solver([4, 8, 12, 16], trials=3)