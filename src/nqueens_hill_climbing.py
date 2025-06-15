import time
import random
from typing import List, Tuple, Optional

class NQueensHillClimbing:
    """
    N-Queens solver using Hill Climbing (Local Greedy Search).
    Uses random restarts to escape local optima.
    """
    
    def __init__(self, n: int, max_restarts: int = 100):
        self.n = n
        self.max_restarts = max_restarts
        self.board = list(range(n))  # Initial random placement
        self.solutions = []
        self.steps = []
        self.total_steps = 0
        self.restart_count = 0
        self.conflicts_history = []
        
    def calculate_conflicts(self, board: List[int]) -> int:
        """
        Calculate total number of conflicts (attacking pairs) on the board.
        """
        conflicts = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Check if queens attack each other
                if board[i] == board[j]:  # Same column
                    conflicts += 1
                elif abs(board[i] - board[j]) == abs(i - j):  # Same diagonal
                    conflicts += 1
        return conflicts
    
    def get_neighbors(self, board: List[int]) -> List[Tuple[List[int], int, int, int]]:
        """
        Get all neighboring states by moving each queen to different columns.
        Returns list of (new_board, row, old_col, new_col) tuples.
        """
        neighbors = []
        for row in range(self.n):
            for col in range(self.n):
                if col != board[row]:  # Don't include current position
                    new_board = board.copy()
                    new_board[row] = col
                    neighbors.append((new_board, row, board[row], col))
        return neighbors
    
    def get_best_neighbor(self, board: List[int]) -> Tuple[Optional[List[int]], int, int, int, int]:
        """
        Find the best neighboring state (lowest conflicts).
        Returns (best_board, conflicts, row, old_col, new_col) or (None, current_conflicts, -1, -1, -1) if no improvement.
        """
        current_conflicts = self.calculate_conflicts(board)
        best_board = None
        best_conflicts = current_conflicts
        best_move = (-1, -1, -1)  # row, old_col, new_col
        
        neighbors = self.get_neighbors(board)
        
        for new_board, row, old_col, new_col in neighbors:
            conflicts = self.calculate_conflicts(new_board)
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_board = new_board
                best_move = (row, old_col, new_col)
        
        return best_board, best_conflicts, best_move[0], best_move[1], best_move[2]
    
    def random_restart(self) -> List[int]:
        """
        Generate a random initial configuration.
        """
        board = list(range(self.n))
        random.shuffle(board)
        return board
    
    def solve_step_by_step(self) -> bool:
        """
        Solve N-Queens using Hill Climbing with random restarts.
        """
        self.solutions = []
        self.steps = []
        self.total_steps = 0
        self.restart_count = 0
        self.conflicts_history = []
        
        start_time = time.time()
        
        # Initial random configuration
        self.board = self.random_restart()
        initial_conflicts = self.calculate_conflicts(self.board)
        
        self.steps.append({
            'type': 'initial_state',
            'board_state': self.board.copy(),
            'conflicts': initial_conflicts,
            'message': f"Initial random state with {initial_conflicts} conflicts"
        })
        
        current_board = self.board.copy()
        
        for restart in range(self.max_restarts):
            if restart > 0:
                # Random restart
                current_board = self.random_restart()
                conflicts = self.calculate_conflicts(current_board)
                self.restart_count += 1
                
                self.steps.append({
                    'type': 'restart',
                    'board_state': current_board.copy(),
                    'conflicts': conflicts,
                    'restart_number': restart,
                    'message': f"Random restart #{restart} with {conflicts} conflicts"
                })
            
            # Hill climbing from current state
            while True:
                self.total_steps += 1
                current_conflicts = self.calculate_conflicts(current_board)
                self.conflicts_history.append(current_conflicts)
                
                # Check if solution found
                if current_conflicts == 0:
                    solution = [(i, current_board[i]) for i in range(self.n)]
                    self.solutions.append(solution)
                    
                    self.steps.append({
                        'type': 'solution_found',
                        'board_state': current_board.copy(),
                        'conflicts': 0,
                        'message': f"Solution found! No conflicts remaining."
                    })
                    
                    end_time = time.time()
                    self.solve_time = end_time - start_time
                    return True
                
                # Find best neighbor
                best_board, best_conflicts, row, old_col, new_col = self.get_best_neighbor(current_board)
                
                if best_board is None or best_conflicts >= current_conflicts:
                    # Local optimum reached
                    self.steps.append({
                        'type': 'local_optimum',
                        'board_state': current_board.copy(),
                        'conflicts': current_conflicts,
                        'message': f"Local optimum reached with {current_conflicts} conflicts"
                    })
                    break
                
                # Move to best neighbor
                current_board = best_board
                self.steps.append({
                    'type': 'move',
                    'board_state': current_board.copy(),
                    'conflicts': best_conflicts,
                    'row': row,
                    'old_col': old_col,
                    'new_col': new_col,
                    'message': f"Moved queen from ({row},{old_col}) to ({row},{new_col}), conflicts: {best_conflicts}"
                })
        
        # No solution found after all restarts
        end_time = time.time()
        self.solve_time = end_time - start_time
        
        self.steps.append({
            'type': 'no_solution',
            'board_state': current_board.copy(),
            'conflicts': self.calculate_conflicts(current_board),
            'message': f"No solution found after {self.max_restarts} restarts"
        })
        
        return False
    
    def get_statistics(self) -> dict:
        """
        Get solving statistics.
        """
        min_conflicts = min(self.conflicts_history) if self.conflicts_history else 0
        avg_conflicts = sum(self.conflicts_history) / len(self.conflicts_history) if self.conflicts_history else 0
        
        return {
            'total_steps': self.total_steps,
            'restart_count': self.restart_count,
            'solutions_found': len(self.solutions),
            'solve_time': getattr(self, 'solve_time', 0),
            'steps_recorded': len(self.steps),
            'min_conflicts': min_conflicts,
            'avg_conflicts': avg_conflicts,
            'max_restarts': self.max_restarts
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
        return list(range(self.n))
    
    def convert_board_to_positions(self, board_state: List[int]) -> List[Tuple[int, int]]:
        """
        Convert board array to list of (row, col) positions.
        """
        positions = []
        for row, col in enumerate(board_state):
            positions.append((row, col))
        return positions

def solve_nqueens_hill_climbing(n: int, max_restarts: int = 100) -> dict:
    """
    Convenience function to solve N-Queens using Hill Climbing.
    
    Args:
        n: Board size (n x n)
        max_restarts: Maximum number of random restarts
    
    Returns:
        Dictionary containing solution, statistics, and steps
    """
    solver = NQueensHillClimbing(n, max_restarts)
    
    print(f"Solving {n}-Queens problem using Hill Climbing...")
    print(f"Maximum restarts: {max_restarts}")
    
    success = solver.solve_step_by_step()
    stats = solver.get_statistics()
    
    result = {
        'success': success,
        'solution': solver.get_current_solution() if success else [],
        'statistics': stats,
        'solver': solver
    }
    
    # Print results
    print(f"\n{'='*50}")
    print(f"N-Queens Problem (n={n}) - Hill Climbing Results")
    print(f"{'='*50}")
    print(f"Solution found: {'Yes' if success else 'No'}")
    print(f"Total steps: {stats['total_steps']:,}")
    print(f"Restarts used: {stats['restart_count']:,}")
    print(f"Time taken: {stats['solve_time']:.4f} seconds")
    print(f"Steps recorded: {stats['steps_recorded']:,}")
    print(f"Min conflicts reached: {stats['min_conflicts']}")
    print(f"Average conflicts: {stats['avg_conflicts']:.2f}")
    
    if success:
        print(f"Solution: {result['solution']}")
    
    return result

# Example usage and testing
if __name__ == "__main__":
    # Test with various board sizes
    for n in [4, 8, 12]:
        result = solve_nqueens_hill_climbing(n, max_restarts=50)
        print()
        
        # Demonstrate step-by-step access
        if result['success']:
            solver = result['solver']
            print(f"First 5 steps for {n}-Queens:")
            for i in range(min(5, len(solver.steps))):
                step = solver.get_step(i)
                print(f"  Step {i+1}: {step['message']}")
            print()