import time
from typing import List, Tuple, Optional

class NQueensSolver:
    """
    N-Queens solver using exhaustive search (depth-first search).
    Provides step-by-step solution tracking and visualization support.
    """
    
    def __init__(self, n: int):
        self.n = n
        self.board = [-1] * n  # board[i] = column position of queen in row i
        self.solutions = []
        self.steps = []  # Store all steps for visualization
        self.total_steps = 0
        self.backtrack_count = 0
        
    def is_safe(self, row: int, col: int) -> bool:
        """
        Check if placing a queen at (row, col) is safe.
        Returns True if no conflicts, False otherwise.
        """
        for i in range(row):
            # Check column conflict
            if self.board[i] == col:
                return False
            
            # Check diagonal conflicts
            if abs(self.board[i] - col) == abs(i - row):
                return False
        
        return True
    
    def solve_step_by_step(self) -> bool:
        """
        Solve N-Queens using exhaustive search with step tracking.
        Returns True if solution found, False otherwise.
        """
        self.solutions = []
        self.steps = []
        self.total_steps = 0
        self.backtrack_count = 0
        
        start_time = time.time()
        result = self._solve_recursive(0)
        end_time = time.time()
        
        self.solve_time = end_time - start_time
        return result
    
    def _solve_recursive(self, row: int) -> bool:
        """
        Recursive backtracking function with step tracking.
        """
        if row == self.n:
            # Found a solution
            solution = [(i, self.board[i]) for i in range(self.n)]
            self.solutions.append(solution.copy())
            self.steps.append({
                'type': 'solution_found',
                'board_state': self.board.copy(),
                'message': f"Solution found! Queens at: {solution}"
            })
            return True
        
        for col in range(self.n):
            self.total_steps += 1
            
            # Record the attempt
            self.steps.append({
                'type': 'try_position',
                'row': row,
                'col': col,
                'board_state': self.board.copy(),
                'message': f"Trying to place queen at ({row}, {col})"
            })
            
            if self.is_safe(row, col):
                # Place queen
                self.board[row] = col
                self.steps.append({
                    'type': 'place_queen',
                    'row': row,
                    'col': col,
                    'board_state': self.board.copy(),
                    'message': f"Queen placed at ({row}, {col})"
                })
                
                # Recursively solve for next row
                if self._solve_recursive(row + 1):
                    return True
                
                # Backtrack
                self.board[row] = -1
                self.backtrack_count += 1
                self.steps.append({
                    'type': 'backtrack',
                    'row': row,
                    'col': col,
                    'board_state': self.board.copy(),
                    'message': f"Backtracking from ({row}, {col})"
                })
            else:
                # Position not safe
                self.steps.append({
                    'type': 'conflict',
                    'row': row,
                    'col': col,
                    'board_state': self.board.copy(),
                    'message': f"Conflict at ({row}, {col}) - cannot place queen"
                })
        
        return False
    
    def get_statistics(self) -> dict:
        """
        Get solving statistics.
        """
        return {
            'total_steps': self.total_steps,
            'backtrack_count': self.backtrack_count,
            'solutions_found': len(self.solutions),
            'solve_time': getattr(self, 'solve_time', 0),
            'steps_recorded': len(self.steps)
        }
    
    def get_current_solution(self) -> List[Tuple[int, int]]:
        """
        Get the current solution as list of (row, col) tuples.
        """
        if self.solutions:
            return self.solutions[0]
        return []
    
    def get_step(self, step_index: int) -> Optional[dict]:
        """
        Get a specific step by index.
        """
        if 0 <= step_index < len(self.steps):
            return self.steps[step_index]
        return None
    
    def get_board_state_at_step(self, step_index: int) -> List[int]:
        """
        Get board state at a specific step.
        """
        if 0 <= step_index < len(self.steps):
            return self.steps[step_index]['board_state']
        return [-1] * self.n
    
    def convert_board_to_positions(self, board_state: List[int]) -> List[Tuple[int, int]]:
        """
        Convert board array to list of (row, col) positions.
        Only includes positions where queens are placed.
        """
        positions = []
        for row, col in enumerate(board_state):
            if col != -1:
                positions.append((row, col))
        return positions

def solve_nqueens_exhaustive(n: int) -> dict:
    """
    Convenience function to solve N-Queens and return results.
    
    Args:
        n: Board size (n x n)
    
    Returns:
        Dictionary containing solution, statistics, and steps
    """
    solver = NQueensSolver(n)
    
    print(f"Solving {n}-Queens problem using exhaustive search...")
    print("This may take a while for large values of n...")
    
    success = solver.solve_step_by_step()
    stats = solver.get_statistics()
    
    result = {
        'success': success,
        'solution': solver.get_current_solution() if success else [],
        'statistics': stats,
        'solver': solver  # Return solver object for step-by-step access
    }
    
    # Print results
    print(f"\n{'='*50}")
    print(f"N-Queens Problem (n={n}) - Exhaustive Search Results")
    print(f"{'='*50}")
    print(f"Solution found: {'Yes' if success else 'No'}")
    print(f"Total steps: {stats['total_steps']:,}")
    print(f"Backtracks: {stats['backtrack_count']:,}")
    print(f"Time taken: {stats['solve_time']:.4f} seconds")
    print(f"Steps recorded: {stats['steps_recorded']:,}")
    
    if success:
        print(f"Solution: {result['solution']}")
    
    return result

