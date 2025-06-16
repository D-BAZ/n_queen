import time
from typing import List, Tuple, Optional, Set

class OptimizedNQueensSolver:
    """
    Optimized N-Queens solver using constraint propagation and efficient conflict detection.
    Provides step-by-step solution tracking with improved performance.
    """
    
    def __init__(self, n: int):
        self.n = n
        self.board = [-1] * n  # board[i] = column position of queen in row i
        self.solutions = []
        
        # Optimization: Use sets for O(1) conflict checking
        self.occupied_cols: Set[int] = set()
        self.occupied_diag1: Set[int] = set()  # row - col
        self.occupied_diag2: Set[int] = set()  # row + col
        
        # Step tracking (optimized to reduce memory usage)
        self.steps = []
        self.total_steps = 0
        self.backtrack_count = 0
        self.max_steps = 100000  # Limit steps to prevent memory issues
        self.track_steps = True
        
    def is_safe_optimized(self, row: int, col: int) -> bool:
        """
        Optimized safety check using O(1) set lookups instead of O(n) iteration.
        """
        return (col not in self.occupied_cols and
                (row - col) not in self.occupied_diag1 and
                (row + col) not in self.occupied_diag2)
    

    def place_queen(self, row: int, col: int) -> None:
        """Place a queen and update constraint sets."""
        self.board[row] = col
        self.occupied_cols.add(col)
        self.occupied_diag1.add(row - col)
        self.occupied_diag2.add(row + col)
    
    def remove_queen(self, row: int) -> None:
        """Remove a queen and update constraint sets."""
        if self.board[row] != -1:
            col = self.board[row]
            self.occupied_cols.remove(col)
            self.occupied_diag1.remove(row - col)
            self.occupied_diag2.remove(row + col)
            self.board[row] = -1
    
    def record_step(self, step_type: str, row: int = -1, col: int = -1, message: str = "") -> None:
        """Record a step with memory optimization."""
        if not self.track_steps or len(self.steps) >= self.max_steps:
            return
            
        # Only store essential information to reduce memory usage
        step = {
            'type': step_type,
            'row': row,
            'col': col,
            'message': message,
            'board_state': self.board.copy()  # Always store board state for visualization
        }
            
        self.steps.append(step)
    
    def solve_step_by_step(self, find_all: bool = False) -> bool:
        """
        Solve N-Queens with optimizations.
        
        Args:
            find_all: If True, find all solutions. If False, stop at first solution.
        """
        self.solutions = []
        self.steps = []
        self.total_steps = 0
        self.backtrack_count = 0
        
        # Reset constraint sets
        self.occupied_cols.clear()
        self.occupied_diag1.clear()
        self.occupied_diag2.clear()
        
        start_time = time.time()
        
        # Use iterative approach for better performance on larger boards
        if self.n <= 8:
            result = self._solve_recursive_optimized(0, find_all)
        else:
            # For larger boards, disable step tracking to save memory
            original_track = self.track_steps
            self.track_steps = False
            result = self._solve_iterative(find_all)
            self.track_steps = original_track
            
        end_time = time.time()
        self.solve_time = end_time - start_time
        return result
    
    def _solve_recursive_optimized(self, row: int, find_all: bool) -> bool:
        """
        Optimized recursive backtracking with constraint propagation.
        """
        if row == self.n:
            # Found a solution
            solution = [(i, self.board[i]) for i in range(self.n)]
            self.solutions.append(solution.copy())
            self.record_step('solution_found', message=f"Solution {len(self.solutions)} found!")
            
            if not find_all:
                return True
            return False  # Continue searching for more solutions
        
        # Early pruning: if remaining rows > available columns, impossible
        available_cols = self.n - len(self.occupied_cols)
        remaining_rows = self.n - row
        if available_cols < remaining_rows:
            return False
        
        # Try columns in order (could be optimized with heuristics)
        for col in range(self.n):
            self.total_steps += 1
            
            if self.is_safe_optimized(row, col):
                # Place queen
                self.place_queen(row, col)
                self.record_step('place_queen', row, col, f"Queen placed at ({row}, {col})")
                
                # Recursively solve for next row
                if self._solve_recursive_optimized(row + 1, find_all):
                    if not find_all:
                        return True
                
                # Backtrack
                self.remove_queen(row)
                self.backtrack_count += 1
                self.record_step('backtrack', row, col, f"Backtracking from ({row}, {col})")
            else:
                self.record_step('conflict', row, col, f"Conflict at ({row}, {col})")
        
        return len(self.solutions) > 0 if find_all else False
    
    def _solve_iterative(self, find_all: bool) -> bool:
        """
        Iterative solution for better performance on large boards.
        Uses explicit stack instead of recursion to avoid stack overflow.
        FIXED: Proper bounds checking to prevent index out of range errors.
        """
        # Reset board and constraint sets
        self.board = [-1] * self.n
        self.occupied_cols.clear()
        self.occupied_diag1.clear()
        self.occupied_diag2.clear()
        
        stack = [0]  # Start with row 0
        cols = [0] * self.n  # Track current column for each row
        
        while stack:
            row = stack[-1]
            
            if row == self.n:
                # Found solution
                solution = [(i, self.board[i]) for i in range(self.n)]
                self.solutions.append(solution.copy())
                
                if not find_all:
                    return True
                    
                # Backtrack to continue searching
                stack.pop()
                if stack:
                    # Remove queen from the previous row
                    prev_row = stack[-1]
                    self.remove_queen(prev_row)
                continue
            
            found_valid_col = False
            
            # Try columns starting from current position
            for col in range(cols[row], self.n):
                self.total_steps += 1
                
                if self.is_safe_optimized(row, col):
                    self.place_queen(row, col)
                    cols[row] = col + 1  # Next time, start from next column
                    stack.append(row + 1)
                    
                    # Initialize column counter for next row (with bounds check)
                    if row + 1 < self.n:
                        cols[row + 1] = 0
                    
                    found_valid_col = True
                    break
            
            if not found_valid_col:
                # Backtrack
                stack.pop()
                cols[row] = 0  # Reset column counter
                self.backtrack_count += 1
                
                if stack:
                    # Remove queen from current row before backtracking
                    current_row = stack[-1]
                    if current_row < self.n:
                        self.remove_queen(current_row)
        
        return len(self.solutions) > 0
    
    def solve_with_heuristics(self) -> bool:
        """
        Solve using most constrained variable heuristic for even better performance.
        """
        self.solutions = []
        self.total_steps = 0
        self.backtrack_count = 0
        
        # Reset constraint sets
        self.occupied_cols.clear()
        self.occupied_diag1.clear()
        self.occupied_diag2.clear()
        
        start_time = time.time()
        result = self._solve_with_column_ordering()
        end_time = time.time()
        
        self.solve_time = end_time - start_time
        return result
    
    def _solve_with_column_ordering(self) -> bool:
        """
        Try columns in order of most constrained first.
        """
        def count_conflicts(row: int, col: int) -> int:
            """Count how many future positions this would eliminate."""
            conflicts = 0
            for future_row in range(row + 1, self.n):
                for future_col in range(self.n):
                    if (future_col == col or 
                        abs(future_col - col) == abs(future_row - row)):
                        conflicts += 1
            return conflicts
        
        def solve_recursive_heuristic(row: int) -> bool:
            if row == self.n:
                solution = [(i, self.board[i]) for i in range(self.n)]
                self.solutions.append(solution)
                return True
            
            # Get available columns sorted by number of conflicts
            available_cols = []
            for col in range(self.n):
                if self.is_safe_optimized(row, col):
                    conflicts = count_conflicts(row, col)
                    available_cols.append((conflicts, col))
            
            # Sort by number of conflicts (ascending - try least constraining first)
            available_cols.sort()
            
            for _, col in available_cols:
                self.total_steps += 1
                self.place_queen(row, col)
                
                if solve_recursive_heuristic(row + 1):
                    return True
                
                self.remove_queen(row)
                self.backtrack_count += 1
            
            return False
        
        return solve_recursive_heuristic(0)
    
    def get_statistics(self) -> dict:
        """Get solving statistics."""
        return {
            'total_steps': self.total_steps,
            'backtrack_count': self.backtrack_count,
            'solutions_found': len(self.solutions),
            'solve_time': getattr(self, 'solve_time', 0),
            'steps_recorded': len(self.steps),
            'steps_per_second': self.total_steps / max(getattr(self, 'solve_time', 1), 0.001)
        }
    
    def get_current_solution(self) -> List[Tuple[int, int]]:
        """Get the first solution found."""
        return self.solutions[0] if self.solutions else []
    
    def get_all_solutions(self) -> List[List[Tuple[int, int]]]:
        """Get all solutions found."""
        return self.solutions.copy()
    
    def get_step(self, step_index: int) -> Optional[dict]:
        """Get a specific step by index."""
        if 0 <= step_index < len(self.steps):
            return self.steps[step_index]
        return None
    
    def convert_board_to_positions(self, board_state: List[int]) -> List[Tuple[int, int]]:
        """Convert board state (list of row indices) to list of (row, col) positions."""
        return [(i, col) for i, col in enumerate(board_state) if col != -1]

def solve_nqueens_optimized(n: int, method: str = 'optimized', find_all: bool = False) -> dict:
    """
    Solve N-Queens problem with various optimization methods.
    
    Args:
        n: Board size
        method: 'optimized', 'heuristic', or 'iterative'
        find_all: Whether to find all solutions
    
    Returns:
        Dictionary with results and statistics
    """
    solver = OptimizedNQueensSolver(n)
    
    print(f"Solving {n}-Queens problem using {method} method...")
    if find_all:
        print("Finding all solutions...")
    
    if method == 'heuristic':
        success = solver.solve_with_heuristics()
    else:
        success = solver.solve_step_by_step(find_all=find_all)
    
    stats = solver.get_statistics()
    
    result = {
        'success': success,
        'solution': solver.get_current_solution() if success else [],
        'all_solutions': solver.get_all_solutions() if find_all else [],
        'statistics': stats,
        'solver': solver
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"N-Queens Problem (n={n}) - {method.capitalize()} Method Results")
    print(f"{'='*60}")
    print(f"Solution found: {'Yes' if success else 'No'}")
    if find_all:
        print(f"Total solutions: {len(solver.solutions)}")
    print(f"Total steps: {stats['total_steps']:,}")
    print(f"Backtracks: {stats['backtrack_count']:,}")
    print(f"Time taken: {stats['solve_time']:.6f} seconds")
    print(f"Steps per second: {stats['steps_per_second']:,.0f}")
    print(f"Steps recorded: {stats['steps_recorded']:,}")
    
    if success and not find_all:
        print(f"First solution: {result['solution']}")
    
    return result

def benchmark_methods(n: int) -> None:
    """
    Benchmark different solving methods.
    """
    print(f"\nBenchmarking different methods for {n}-Queens:")
    print("=" * 50)
    
    methods = ['optimized', 'heuristic']
    
    for method in methods:
        try:
            result = solve_nqueens_optimized(n, method=method)
            print(f"{method.capitalize()}: {result['statistics']['solve_time']:.6f}s, "
                  f"{result['statistics']['total_steps']:,} steps")
        except Exception as e:
            print(f"{method.capitalize()}: Failed - {e}")

# Example usage and testing
if __name__ == "__main__":
    # Test with different board sizes
    test_sizes = [4, 8, 12]
    
    for n in test_sizes:
        print(f"\n{'='*70}")
        print(f"Testing N-Queens solver with n={n}")
        print(f"{'='*70}")
        
        # Solve with optimized method
        result = solve_nqueens_optimized(n, method='optimized')
        
        # For smaller boards, also find all solutions
        if n <= 8:
            print(f"\nFinding all solutions for {n}-Queens...")
            all_result = solve_nqueens_optimized(n, find_all=True)
            print(f"Total unique solutions: {len(all_result['all_solutions'])}")
    
    # Benchmark comparison for n=8
    print(f"\n{'='*70}")
    benchmark_methods(8)