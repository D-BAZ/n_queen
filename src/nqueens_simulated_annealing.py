import random
import math
import time

class NQueensSimulatedAnnealing:
    def __init__(self, n, initial_temp=100.0, cooling_rate=0.95, min_temp=0.01, max_iterations=10000):
        """
        Initialize the Simulated Annealing solver for N-Queens problem.
        
        Args:
            n: Size of the chessboard (n x n)
            initial_temp: Starting temperature for simulated annealing
            cooling_rate: Rate at which temperature decreases (0 < rate < 1)
            min_temp: Minimum temperature threshold
            max_iterations: Maximum number of iterations per temperature
        """
        self.n = n
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations
        
        # Statistics tracking
        self.total_steps = 0
        self.solve_time = 0
        self.temperature_changes = 0
        self.accepted_moves = 0
        self.rejected_moves = 0
        self.solutions_found = 0
        
        # Step-by-step tracking
        self.steps = []
        self.step_by_step_enabled = False
        
        # Current state
        self.board = None
        self.current_conflicts = 0
        
    def count_conflicts(self, board):
        """Count the number of queen conflicts on the board."""
        conflicts = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Check if queens attack each other
                if (board[i] == board[j] or  # Same row
                    abs(board[i] - board[j]) == abs(i - j)):  # Same diagonal
                    conflicts += 1
        return conflicts
    
    def get_random_neighbor(self, board):
        """Generate a random neighbor by moving one queen to a different row."""
        new_board = board[:]
        col = random.randint(0, self.n - 1)
        old_row = new_board[col]
        
        # Choose a different row
        new_row = random.randint(0, self.n - 1)
        while new_row == old_row:
            new_row = random.randint(0, self.n - 1)
        
        new_board[col] = new_row
        return new_board, col, old_row, new_row
    
    def acceptance_probability(self, current_cost, new_cost, temperature):
        """Calculate acceptance probability for a move."""
        if new_cost < current_cost:
            return 1.0
        if temperature == 0:
            return 0.0
        return math.exp(-(new_cost - current_cost) / temperature)
    
    def solve(self):
        """Solve the N-Queens problem using Simulated Annealing."""
        self.step_by_step_enabled = False
        start_time = time.time()
        
        # Initialize with random solution
        self.board = [random.randint(0, self.n - 1) for _ in range(self.n)]
        self.current_conflicts = self.count_conflicts(self.board)
        
        # Reset statistics
        self.total_steps = 0
        self.temperature_changes = 0
        self.accepted_moves = 0
        self.rejected_moves = 0
        
        temperature = self.initial_temp
        
        while temperature > self.min_temp:
            self.temperature_changes += 1
            
            for iteration in range(self.max_iterations):
                self.total_steps += 1
                
                # Check if solution found
                if self.current_conflicts == 0:
                    self.solutions_found = 1
                    self.solve_time = time.time() - start_time
                    return True
                
                # Generate neighbor
                neighbor_board, col, old_row, new_row = self.get_random_neighbor(self.board)
                neighbor_conflicts = self.count_conflicts(neighbor_board)
                
                # Calculate acceptance probability
                prob = self.acceptance_probability(self.current_conflicts, neighbor_conflicts, temperature)
                
                # Accept or reject the move
                if random.random() < prob:
                    self.board = neighbor_board
                    self.current_conflicts = neighbor_conflicts
                    self.accepted_moves += 1
                else:
                    self.rejected_moves += 1
            
            # Cool down
            temperature *= self.cooling_rate
        
        self.solve_time = time.time() - start_time
        return False
    
    def solve_step_by_step(self):
        """Solve with step-by-step tracking for visualization."""
        self.step_by_step_enabled = True
        self.steps = []
        start_time = time.time()
        
        # Initialize with random solution
        self.board = [random.randint(0, self.n - 1) for _ in range(self.n)]
        self.current_conflicts = self.count_conflicts(self.board)
        
        # Reset statistics
        self.total_steps = 0
        self.temperature_changes = 0
        self.accepted_moves = 0
        self.rejected_moves = 0
        
        # Record initial state
        self.steps.append({
            'board_state': self.board[:],
            'conflicts': self.current_conflicts,
            'temperature': self.initial_temp,
            'message': f'Initial random state with {self.current_conflicts} conflicts',
            'move_accepted': None,
            'acceptance_prob': None
        })
        
        temperature = self.initial_temp
        
        while temperature > self.min_temp:
            self.temperature_changes += 1
            
            for iteration in range(self.max_iterations):
                self.total_steps += 1
                
                # Check if solution found
                if self.current_conflicts == 0:
                    self.solutions_found = 1
                    self.solve_time = time.time() - start_time
                    self.steps.append({
                        'board_state': self.board[:],
                        'conflicts': self.current_conflicts,
                        'temperature': temperature,
                        'message': f'Solution found! 0 conflicts achieved.',
                        'move_accepted': True,
                        'acceptance_prob': 1.0
                    })
                    return True
                
                # Generate neighbor
                neighbor_board, col, old_row, new_row = self.get_random_neighbor(self.board)
                neighbor_conflicts = self.count_conflicts(neighbor_board)
                
                # Calculate acceptance probability
                prob = self.acceptance_probability(self.current_conflicts, neighbor_conflicts, temperature)
                
                # Accept or reject the move
                move_accepted = random.random() < prob
                
                if move_accepted:
                    self.board = neighbor_board
                    self.current_conflicts = neighbor_conflicts
                    self.accepted_moves += 1
                    
                    action_msg = "better" if neighbor_conflicts < self.current_conflicts else "worse (accepted by probability)"
                    message = f'Moved queen in col {col} from row {old_row} to {new_row} - {action_msg} solution'
                else:
                    self.rejected_moves += 1
                    message = f'Rejected move: col {col} row {old_row}→{new_row} (prob={prob:.3f})'
                
                # Record step
                self.steps.append({
                    'board_state': self.board[:],
                    'conflicts': self.current_conflicts,
                    'temperature': temperature,
                    'message': message,
                    'move_accepted': move_accepted,
                    'acceptance_prob': prob,
                    'row': new_row if move_accepted else None,
                    'col': col if move_accepted else None,
                    'old_row': old_row,
                    'new_row': new_row
                })
                
                # Limit steps for visualization (prevent memory issues)
                if len(self.steps) > 5000:
                    break
            
            # Cool down
            old_temp = temperature
            temperature *= self.cooling_rate
            
            # Record temperature change
            if len(self.steps) <= 5000:
                self.steps.append({
                    'board_state': self.board[:],
                    'conflicts': self.current_conflicts,
                    'temperature': temperature,
                    'message': f'Temperature cooled: {old_temp:.3f} → {temperature:.3f}',
                    'move_accepted': None,
                    'acceptance_prob': None
                })
        
        self.solve_time = time.time() - start_time
        
        # Add final step
        if len(self.steps) <= 5000:
            self.steps.append({
                'board_state': self.board[:],
                'conflicts': self.current_conflicts,
                'temperature': temperature,
                'message': f'Annealing completed. Final conflicts: {self.current_conflicts}',
                'move_accepted': None,
                'acceptance_prob': None
            })
        
        return self.current_conflicts == 0
    
    def get_step(self, step_index):
        """Get a specific step for visualization."""
        if 0 <= step_index < len(self.steps):
            return self.steps[step_index]
        return None
    
    def convert_board_to_positions(self, board):
        """Convert board representation to list of (row, col) positions."""
        return [(board[col], col) for col in range(len(board))]
    
    def get_statistics(self):
        """Get solver statistics."""
        return {
            'solve_time': self.solve_time,
            'total_steps': self.total_steps,
            'solutions_found': self.solutions_found,
            'temperature_changes': self.temperature_changes,
            'accepted_moves': self.accepted_moves,
            'rejected_moves': self.rejected_moves,
            'acceptance_rate': self.accepted_moves / max(1, self.accepted_moves + self.rejected_moves),
            'final_conflicts': self.current_conflicts if hasattr(self, 'current_conflicts') else 0
        }
    
    def print_solution(self):
        """Print the current solution."""
        if not self.board:
            print("No solution available.")
            return
            
        print(f"\nSolution found with {self.current_conflicts} conflicts:")
        print("Board representation (column index -> row position):")
        for col, row in enumerate(self.board):
            print(f"Column {col}: Row {row}")
        
        print("\nVisual representation:")
        for row in range(self.n):
            line = ""
            for col in range(self.n):
                if self.board[col] == row:
                    line += "Q "
                else:
                    line += ". "
            print(line)

# Example usage and testing
if __name__ == "__main__":
    def test_solver(n, show_solution=True):
        print(f"\n=== Testing Simulated Annealing N-Queens Solver (n={n}) ===")
        
        solver = NQueensSimulatedAnnealing(n, initial_temp=100.0, cooling_rate=0.95)
        
        print("Attempting to solve...")
        success = solver.solve()
        
        stats = solver.get_statistics()
        print(f"\nResults:")
        print(f"Success: {success}")
        print(f"Time: {stats['solve_time']:.4f} seconds")
        print(f"Total steps: {stats['total_steps']:,}")
        print(f"Temperature changes: {stats['temperature_changes']}")
        print(f"Accepted moves: {stats['accepted_moves']:,}")
        print(f"Rejected moves: {stats['rejected_moves']:,}")
        print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
        print(f"Final conflicts: {stats['final_conflicts']}")
        
        if success and show_solution:
            solver.print_solution()
        
        return success
    
    # Test different board sizes
    test_sizes = [4, 8, 12]
    for size in test_sizes:
        success = test_solver(size, show_solution=(size <= 8))
        if not success:
            print(f"Failed to find solution for n={size}. Try running again or adjusting parameters.")
        print("-" * 60)