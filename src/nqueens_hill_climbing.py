import time
import random
from typing import List, Tuple, Optional, Set
from collections import defaultdict
import heapq

class OptimizedNQueensHillClimbing:
    """
    Optimized N-Queens solver using Hill Climbing with advanced heuristics.
    Features: Sideways moves, min-conflicts heuristic, adaptive restarts, and efficient conflict calculation.
    """
    
    def __init__(self, n: int, max_restarts: int = 100, max_sideways: int = 100):
        self.n = n
        self.max_restarts = max_restarts
        self.max_sideways = max_sideways  # Allow sideways moves to escape plateaus
        self.board = list(range(n))
        self.solutions = []
        self.steps = []
        self.total_steps = 0
        self.restart_count = 0
        self.conflicts_history = []
        self.sideways_moves = 0
        
        # Optimization: Precompute conflict tables for O(1) updates
        self.row_conflicts = [0] * n
        self.col_conflicts = [0] * n
        self.diag1_conflicts = [0] * (2 * n - 1)  # row - col + (n-1)
        self.diag2_conflicts = [0] * (2 * n - 1)  # row + col
        
        # Track individual queen conflicts for faster computation
        self.queen_conflicts = [0] * n
        
    def _get_diag1_index(self, row: int, col: int) -> int:
        """Get diagonal 1 index (row - col + (n-1))"""
        return row - col + (self.n - 1)
    
    def _get_diag2_index(self, row: int, col: int) -> int:
        """Get diagonal 2 index (row + col)"""
        return row + col
    
    def initialize_conflict_tables(self, board: List[int]) -> None:
        """Initialize conflict counting tables."""
        # Reset all conflict counters
        self.row_conflicts = [0] * self.n
        self.col_conflicts = [0] * self.n
        self.diag1_conflicts = [0] * (2 * self.n - 1)
        self.diag2_conflicts = [0] * (2 * self.n - 1)
        self.queen_conflicts = [0] * self.n
        
        # Count conflicts for each position
        for row in range(self.n):
            col = board[row]
            self.col_conflicts[col] += 1
            self.diag1_conflicts[self._get_diag1_index(row, col)] += 1
            self.diag2_conflicts[self._get_diag2_index(row, col)] += 1
        
        # Calculate individual queen conflicts
        for row in range(self.n):
            self.queen_conflicts[row] = self._calculate_queen_conflicts(row, board[row])
    
    def _calculate_queen_conflicts(self, row: int, col: int) -> int:
        """Calculate conflicts for a specific queen position."""
        conflicts = 0
        
        # Column conflicts (subtract 1 because we count the queen itself)
        conflicts += max(0, self.col_conflicts[col] - 1)
        
        # Diagonal conflicts
        conflicts += max(0, self.diag1_conflicts[self._get_diag1_index(row, col)] - 1)
        conflicts += max(0, self.diag2_conflicts[self._get_diag2_index(row, col)] - 1)
        
        return conflicts
    
    def calculate_total_conflicts(self, board: List[int]) -> int:
        """Calculate total conflicts efficiently using conflict tables."""
        return sum(self.queen_conflicts) // 2  # Each conflict is counted twice
    
    def move_queen(self, row: int, old_col: int, new_col: int) -> None:
        """Update conflict tables when moving a queen."""
        # Remove old position
        self.col_conflicts[old_col] -= 1
        self.diag1_conflicts[self._get_diag1_index(row, old_col)] -= 1
        self.diag2_conflicts[self._get_diag2_index(row, old_col)] -= 1
        
        # Add new position
        self.col_conflicts[new_col] += 1
        self.diag1_conflicts[self._get_diag1_index(row, new_col)] += 1
        self.diag2_conflicts[self._get_diag2_index(row, new_col)] += 1
        
        # Recalculate conflicts for all affected queens
        affected_rows = set()
        
        # All queens in the old and new columns
        for r in range(self.n):
            if self.board[r] == old_col or self.board[r] == new_col:
                affected_rows.add(r)
        
        # All queens on the old and new diagonals
        for r in range(self.n):
            if (self._get_diag1_index(r, self.board[r]) == self._get_diag1_index(row, old_col) or
                self._get_diag1_index(r, self.board[r]) == self._get_diag1_index(row, new_col) or
                self._get_diag2_index(r, self.board[r]) == self._get_diag2_index(row, old_col) or
                self._get_diag2_index(r, self.board[r]) == self._get_diag2_index(row, new_col)):
                affected_rows.add(r)
        
        # Update conflicts for affected queens
        for r in affected_rows:
            self.queen_conflicts[r] = self._calculate_queen_conflicts(r, self.board[r])
    
    def get_min_conflicts_move(self, board: List[int]) -> Tuple[Optional[Tuple[int, int, int]], int]:
        """
        Find the move that minimizes conflicts using min-conflicts heuristic.
        Returns ((row, old_col, new_col), new_total_conflicts) or (None, current_conflicts).
        """
        current_conflicts = self.calculate_total_conflicts(board)
        
        # Find the queen with maximum conflicts
        max_conflicts_queen = max(range(self.n), key=lambda r: self.queen_conflicts[r])
        
        # If no conflicts, we're done
        if self.queen_conflicts[max_conflicts_queen] == 0:
            return None, current_conflicts
        
        best_move = None
        best_conflicts = current_conflicts
        
        # Try moving the most conflicted queen to each column
        row = max_conflicts_queen
        old_col = board[row]
        
        for new_col in range(self.n):
            if new_col == old_col:
                continue
            
            # Temporarily make the move to calculate new conflicts
            self.move_queen(row, old_col, new_col)
            board[row] = new_col
            
            new_total_conflicts = self.calculate_total_conflicts(board)
            
            if new_total_conflicts < best_conflicts:
                best_conflicts = new_total_conflicts
                best_move = (row, old_col, new_col)
            
            # Undo the move
            self.move_queen(row, new_col, old_col)
            board[row] = old_col
        
        return best_move, best_conflicts
    
    def get_best_neighbor_optimized(self, board: List[int]) -> Tuple[Optional[Tuple[int, int, int]], int]:
        """
        Find the best neighboring state with optimized conflict calculation.
        Uses min-conflicts heuristic for better performance.
        """
        return self.get_min_conflicts_move(board)
    
    def random_restart_smart(self) -> List[int]:
        """
        Generate a smarter initial configuration using least-conflicts placement.
        """
        board = [-1] * self.n
        
        for row in range(self.n):
            min_conflicts = float('inf')
            best_cols = []
            
            for col in range(self.n):
                conflicts = 0
                # Count conflicts with already placed queens
                for placed_row in range(row):
                    placed_col = board[placed_row]
                    if (placed_col == col or 
                        abs(placed_col - col) == abs(placed_row - row)):
                        conflicts += 1
                
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_cols = [col]
                elif conflicts == min_conflicts:
                    best_cols.append(col)
            
            # Randomly choose among best columns
            board[row] = random.choice(best_cols)
        
        return board
    
    def solve_step_by_step(self, use_smart_restart: bool = True, 
                          allow_sideways: bool = True) -> bool:
        """
        Solve N-Queens using optimized Hill Climbing.
        
        Args:
            use_smart_restart: Use intelligent restart strategy
            allow_sideways: Allow sideways moves to escape plateaus
        """
        self.solutions = []
        self.steps = []
        self.total_steps = 0
        self.restart_count = 0
        self.conflicts_history = []
        self.sideways_moves = 0
        
        start_time = time.time()
        
        # Initial configuration
        if use_smart_restart:
            self.board = self.random_restart_smart()
        else:
            self.board = list(range(self.n))
            random.shuffle(self.board)
        
        self.initialize_conflict_tables(self.board)
        initial_conflicts = self.calculate_total_conflicts(self.board)
        
        self.record_step('initial_state', self.board.copy(), initial_conflicts,
                        f"Initial state with {initial_conflicts} conflicts")
        
        current_board = self.board.copy()
        
        for restart in range(self.max_restarts):
            if restart > 0:
                # Smart restart
                if use_smart_restart:
                    current_board = self.random_restart_smart()
                else:
                    current_board = list(range(self.n))
                    random.shuffle(current_board)
                
                self.board = current_board.copy()
                self.initialize_conflict_tables(self.board)
                conflicts = self.calculate_total_conflicts(current_board)
                self.restart_count += 1
                
                self.record_step('restart', current_board.copy(), conflicts,
                               f"Restart #{restart} with {conflicts} conflicts")
            
            sideways_count = 0
            
            # Hill climbing from current state
            while True:
                self.total_steps += 1
                current_conflicts = self.calculate_total_conflicts(current_board)
                self.conflicts_history.append(current_conflicts)
                
                # Check if solution found
                if current_conflicts == 0:
                    solution = [(i, current_board[i]) for i in range(self.n)]
                    self.solutions.append(solution)
                    
                    self.record_step('solution_found', current_board.copy(), 0,
                                   "Solution found! No conflicts remaining.")
                    
                    end_time = time.time()
                    self.solve_time = end_time - start_time
                    return True
                
                # Find best move
                best_move, best_conflicts = self.get_best_neighbor_optimized(current_board)
                
                if best_move is None:
                    # No improving move found
                    self.record_step('local_optimum', current_board.copy(), current_conflicts,
                                   f"Local optimum reached with {current_conflicts} conflicts")
                    break
                
                row, old_col, new_col = best_move
                
                # Check if this is a sideways move
                if best_conflicts == current_conflicts:
                    if not allow_sideways or sideways_count >= self.max_sideways:
                        self.record_step('plateau', current_board.copy(), current_conflicts,
                                       f"Plateau reached with {current_conflicts} conflicts")
                        break
                    sideways_count += 1
                    self.sideways_moves += 1
                elif best_conflicts < current_conflicts:
                    sideways_count = 0  # Reset sideways counter on improvement
                else:
                    # Worse move - shouldn't happen with proper hill climbing
                    break
                
                # Make the move
                self.move_queen(row, old_col, new_col)
                current_board[row] = new_col
                
                move_type = 'sideways_move' if best_conflicts == current_conflicts else 'improvement'
                self.record_step(move_type, current_board.copy(), best_conflicts,
                               f"Moved queen from ({row},{old_col}) to ({row},{new_col}), "
                               f"conflicts: {best_conflicts}")
        
        # No solution found
        end_time = time.time()
        self.solve_time = end_time - start_time
        
        final_conflicts = self.calculate_total_conflicts(current_board)
        self.record_step('no_solution', current_board.copy(), final_conflicts,
                        f"No solution found after {self.max_restarts} restarts")
        
        return False
    

    
    def record_step(self, step_type: str, board_state: List[int], conflicts: int, message: str) -> None:
        """Record a step with essential information."""
        self.steps.append({
            'type': step_type,
            'board_state': board_state,
            'conflicts': conflicts,
            'message': message,
            'step_number': len(self.steps) + 1
        })
    
    def solve_with_simulated_annealing(self, initial_temp: float = 100.0, 
                                     cooling_rate: float = 0.95) -> bool:
        """
        Solve using Simulated Annealing for comparison.
        """
        self.solutions = []
        self.steps = []
        self.total_steps = 0
        
        start_time = time.time()
        
        current_board = self.random_restart_smart()
        self.board = current_board.copy()
        self.initialize_conflict_tables(self.board)
        
        current_conflicts = self.calculate_total_conflicts(current_board)
        temperature = initial_temp
        
        self.record_step('initial_state', current_board.copy(), current_conflicts,
                        f"SA initial state: {current_conflicts} conflicts, temp: {temperature:.2f}")
        
        while temperature > 0.01:
            self.total_steps += 1
            
            if current_conflicts == 0:
                solution = [(i, current_board[i]) for i in range(self.n)]
                self.solutions.append(solution)
                
                end_time = time.time()
                self.solve_time = end_time - start_time
                return True
            
            # Random move
            row = random.randint(0, self.n - 1)
            old_col = current_board[row]
            new_col = random.randint(0, self.n - 1)
            
            if new_col == old_col:
                continue
            
            # Calculate new conflicts
            self.move_queen(row, old_col, new_col)
            current_board[row] = new_col
            new_conflicts = self.calculate_total_conflicts(current_board)
            
            # Accept or reject move
            delta = new_conflicts - current_conflicts
            if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
                # Accept move
                current_conflicts = new_conflicts
                if len(self.steps) < 1000:  # Limit step recording
                    self.record_step('sa_accept', current_board.copy(), current_conflicts,
                                   f"SA accepted move: conflicts={current_conflicts}, temp={temperature:.2f}")
            else:
                # Reject move
                self.move_queen(row, new_col, old_col)
                current_board[row] = old_col
                if len(self.steps) < 1000:
                    self.record_step('sa_reject', current_board.copy(), current_conflicts,
                                   f"SA rejected move: temp={temperature:.2f}")
            
            temperature *= cooling_rate
        
        end_time = time.time()
        self.solve_time = end_time - start_time
        return False
    
    def get_statistics(self) -> dict:
        """Get comprehensive solving statistics."""
        min_conflicts = min(self.conflicts_history) if self.conflicts_history else 0
        avg_conflicts = sum(self.conflicts_history) / len(self.conflicts_history) if self.conflicts_history else 0
        
        return {
            'total_steps': self.total_steps,
            'restart_count': self.restart_count,
            'sideways_moves': self.sideways_moves,
            'solutions_found': len(self.solutions),
            'solve_time': getattr(self, 'solve_time', 0),
            'steps_recorded': len(self.steps),
            'min_conflicts': min_conflicts,
            'avg_conflicts': avg_conflicts,
            'max_restarts': self.max_restarts,
            'steps_per_second': self.total_steps / max(getattr(self, 'solve_time', 1), 0.001)
        }
    
    def get_current_solution(self) -> List[Tuple[int, int]]:
        """Get the current solution."""
        return self.solutions[0] if self.solutions else []
    
    def get_step(self, step_index: int) -> Optional[dict]:
        """Get a specific step by index."""
        if 0 <= step_index < len(self.steps):
            return self.steps[step_index]
        return None

def solve_nqueens_hill_climbing_optimized(n: int, method: str = 'hill_climbing', 
                                        max_restarts: int = 100, 
                                        max_sideways: int = 100) -> dict:
    """
    Solve N-Queens using optimized Hill Climbing variants.
    
    Args:
        n: Board size
        method: 'hill_climbing', 'simulated_annealing'
        max_restarts: Maximum random restarts
        max_sideways: Maximum sideways moves
    """
    solver = OptimizedNQueensHillClimbing(n, max_restarts, max_sideways)
    
    print(f"Solving {n}-Queens using optimized {method.replace('_', ' ').title()}...")
    
    start_time = time.time()
    
    if method == 'simulated_annealing':
        success = solver.solve_with_simulated_annealing()
    else:
        success = solver.solve_step_by_step(use_smart_restart=True, allow_sideways=True)
    
    stats = solver.get_statistics()
    
    result = {
        'success': success,
        'solution': solver.get_current_solution() if success else [],
        'statistics': stats,
        'solver': solver
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"N-Queens Problem (n={n}) - {method.replace('_', ' ').title()} Results")
    print(f"{'='*60}")
    print(f"Solution found: {'Yes' if success else 'No'}")
    print(f"Total steps: {stats['total_steps']:,}")
    print(f"Restarts used: {stats['restart_count']:,}")
    if method == 'hill_climbing':
        print(f"Sideways moves: {stats['sideways_moves']:,}")
    print(f"Time taken: {stats['solve_time']:.6f} seconds")
    print(f"Steps per second: {stats['steps_per_second']:,.0f}")
    print(f"Min conflicts reached: {stats['min_conflicts']}")
    print(f"Average conflicts: {stats['avg_conflicts']:.2f}")
    
    if success:
        print(f"Solution: {result['solution']}")
    
    return result

def benchmark_hill_climbing_methods(n: int) -> None:
    """Benchmark different hill climbing approaches."""
    print(f"\nBenchmarking Hill Climbing methods for {n}-Queens:")
    print("=" * 60)
    
    methods = ['hill_climbing', 'simulated_annealing']
    results = {}
    
    for method in methods:
        try:
            result = solve_nqueens_hill_climbing_optimized(n, method=method, max_restarts=50)
            results[method] = result
            print(f"{method.replace('_', ' ').title()}: "
                  f"{'Success' if result['success'] else 'Failed'} in "
                  f"{result['statistics']['solve_time']:.4f}s, "
                  f"{result['statistics']['total_steps']:,} steps")
        except Exception as e:
            print(f"{method.replace('_', ' ').title()}: Failed - {e}")
    
    return results

# Example usage and testing
if __name__ == "__main__":
    # Test with various board sizes
    test_sizes = [4, 8, 16, 32]
    
    for n in test_sizes:
        print(f"\n{'='*70}")
        print(f"Testing Optimized Hill Climbing for {n}-Queens")
        print(f"{'='*70}")
        
        result = solve_nqueens_hill_climbing_optimized(n, max_restarts=50)
        
        if result['success']:
            solver = result['solver']
            print(f"\nFirst few solution steps:")
            solution_steps = [s for s in solver.steps if s['type'] in ['solution_found', 'improvement']]
            for i, step in enumerate(solution_steps[:3]):
                print(f"  {step['message']}")
    
    # Benchmark comparison
    print(f"\n{'='*70}")
    benchmark_hill_climbing_methods(8)