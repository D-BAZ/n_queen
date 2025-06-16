import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants - Increased board and window size
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
BOARD_SIZE = 640
BOARD_OFFSET_X = 50
BOARD_OFFSET_Y = 80

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_BROWN = (240, 217, 181)
DARK_BROWN = (181, 136, 99)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
GRAY = (128, 128, 128)
LIGHT_GREEN = (144, 238, 144)
ORANGE = (255, 165, 0)
LIGHT_BLUE = (173, 216, 230)
MAGENTA = (255, 0, 255)

class ChessBoard:
    def __init__(self, n):
        self.n = n
        self.square_size = BOARD_SIZE // n
        self.solver_objects = {}  # Dictionary to store solver objects
        self.highlighted_square = None  # For highlighting current solver position
        self.solver_type = None  # Track which solver is being used
        
    def draw_board(self, screen):
        # Draw the chess board
        for row in range(self.n):
            for col in range(self.n):
                x = BOARD_OFFSET_X + col * self.square_size
                y = BOARD_OFFSET_Y + row * self.square_size
                
                # Determine square color
                if (row + col) % 2 == 0:
                    color = LIGHT_BROWN
                else:
                    color = DARK_BROWN
                
                # Highlight current square if needed
                if self.highlighted_square == (row, col):
                    color = LIGHT_GREEN
                
                pygame.draw.rect(screen, color, (x, y, self.square_size, self.square_size))
                
                # Draw border
                pygame.draw.rect(screen, BLACK, (x, y, self.square_size, self.square_size), 1)
    
    def draw_objects(self, screen, font):
        # Draw solver objects with different colors based on solver type
        if self.solver_type == 'exhaustive':
            queen_color = BLUE
        elif self.solver_type == 'hill_climbing':
            queen_color = PURPLE
        elif self.solver_type == 'simulated_annealing':
            queen_color = ORANGE
        elif self.solver_type == 'genetic':
            queen_color = MAGENTA
        else:
            queen_color = BLACK
        
        for (row, col) in self.solver_objects:
            self._draw_queen(screen, font, row, col, queen_color, "Q")
    
    def _draw_queen(self, screen, font, row, col, color, label):
        x = BOARD_OFFSET_X + col * self.square_size + self.square_size // 2
        y = BOARD_OFFSET_Y + row * self.square_size + self.square_size // 2
        
        # Draw circle
        circle_radius = min(self.square_size // 3, 30)
        pygame.draw.circle(screen, color, (x, y), circle_radius)
        pygame.draw.circle(screen, BLACK, (x, y), circle_radius, 2)
        
        # Draw label if square is large enough
        if self.square_size > 40:
            text = font.render(label, True, WHITE)
            text_rect = text.get_rect(center=(x, y))
            screen.blit(text, text_rect)
    
    def get_square_from_pos(self, pos):
        # Convert mouse position to board square
        x, y = pos
        if (BOARD_OFFSET_X <= x <= BOARD_OFFSET_X + BOARD_SIZE and 
            BOARD_OFFSET_Y <= y <= BOARD_OFFSET_Y + BOARD_SIZE):
            col = (x - BOARD_OFFSET_X) // self.square_size
            row = (y - BOARD_OFFSET_Y) // self.square_size
            if 0 <= row < self.n and 0 <= col < self.n:
                return (row, col)
        return None
    
    def add_object(self, row, col):
        # Method kept for compatibility but disabled
        pass
    
    def set_solver_objects(self, positions, solver_type='exhaustive'):
        # Set objects from solver
        self.solver_objects = {pos: True for pos in positions}
        self.solver_type = solver_type
    
    def highlight_square(self, row, col):
        self.highlighted_square = (row, col)
    
    def clear_highlight(self):
        self.highlighted_square = None
    
    def clear_board(self):
        self.solver_objects.clear()
        self.clear_highlight()
        self.solver_type = None

class TextInputBox:
    def __init__(self, x, y, width, height, font):
        self.rect = pygame.Rect(x, y, width, height)
        self.font = font
        self.text = ""
        self.active = False
        self.cursor_visible = True
        self.cursor_timer = 0
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
            
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                return self.text
            elif event.unicode.isdigit() and len(self.text) < 6:  # Limit to 6 digits
                self.text += event.unicode
                
        return None
    
    def update(self, dt):
        self.cursor_timer += dt
        if self.cursor_timer >= 500:  # Blink every 500ms
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0
    
    def draw(self, screen):
        # Draw background
        color = WHITE if self.active else GRAY
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)
        
        # Draw text
        text_surface = self.font.render(self.text, True, BLACK)
        screen.blit(text_surface, (self.rect.x + 5, self.rect.y + 5))
        
        # Draw cursor
        if self.active and self.cursor_visible:
            cursor_x = self.rect.x + 5 + text_surface.get_width()
            pygame.draw.line(screen, BLACK, 
                           (cursor_x, self.rect.y + 3), 
                           (cursor_x, self.rect.y + self.rect.height - 3), 2)
    
    def get_value(self):
        try:
            return int(self.text) if self.text else 0
        except ValueError:
            return 0
    
    def clear(self):
        self.text = ""

def get_board_size():
    """Get board size from user input"""
    print("Enter the size of the chess board (n for n×n board):")
    while True:
        try:
            n = int(input("Board size (4-12 recommended): "))
            if 4 <= n <= 20:
                return n
            else:
                print("Please enter a number between 4 and 20")
        except ValueError:
            print("Please enter a valid number")

def draw_ui(screen, font, board, solver_active=False, step_info=None, stats=None, input_box=None, solver_type=None):
    """Draw the user interface"""
    # Title
    title_font = pygame.font.Font(None, 36)
    solver_name = ""
    if solver_type == 'exhaustive':
        solver_name = " (Exhaustive Search)"
    elif solver_type == 'hill_climbing':
        solver_name = " (Hill Climbing)"
    elif solver_type == 'simulated_annealing':
        solver_name = " (Simulated Annealing)"
    elif solver_type == 'genetic':
        solver_name = " (Genetic Algorithm)"
    
    title = title_font.render(f"N-Queens Solver {board.n}×{board.n}{solver_name}", True, BLACK)
    screen.blit(title, (BOARD_OFFSET_X, 20))
    
    # Instructions
    instructions = [
        "Controls:",
        "E: Solve using Exhaustive Search",
        "H: Solve using Hill Climbing",
        "S: Solve using Simulated Annealing",
        "G: Solve using Genetic Algorithm",
        "SPACE: Next Step",
        "ENTER: Skip to specified step",
        "R: Reset to start",
        "C: Clear Board",
        "ESC: Quit"
    ]
    
    y_offset = BOARD_OFFSET_Y
    for instruction in instructions:
        text = font.render(instruction, True, BLACK)
        screen.blit(text, (BOARD_OFFSET_X + BOARD_SIZE + 30, y_offset))
        y_offset += 25
    
    # Skip steps input
    y_offset += 20
    skip_label = font.render("Skip to step:", True, BLACK)
    screen.blit(skip_label, (BOARD_OFFSET_X + BOARD_SIZE + 30, y_offset))
    
    if input_box:
        input_box.draw(screen)
    
    # Display solver queen count
    solver_count = len(board.solver_objects)
    y_offset += 60
    
    if solver_type == 'exhaustive':
        count_color = BLUE
    elif solver_type == 'hill_climbing':
        count_color = PURPLE
    elif solver_type == 'simulated_annealing':
        count_color = ORANGE
    elif solver_type == 'genetic':
        count_color = MAGENTA
    else:
        count_color = BLACK
        
    count_text = font.render(f"Queens placed: {solver_count}", True, count_color)
    screen.blit(count_text, (BOARD_OFFSET_X + BOARD_SIZE + 30, y_offset))
    
    # Show conflicts for hill climbing, simulated annealing, and genetic algorithm
    if (solver_type in ['hill_climbing', 'simulated_annealing', 'genetic'] and 
        step_info and 'conflicts' in step_info):
        y_offset += 25
        conflicts_text = font.render(f"Conflicts: {step_info['conflicts']}", True, RED)
        screen.blit(conflicts_text, (BOARD_OFFSET_X + BOARD_SIZE + 30, y_offset))
        
    # Show temperature for simulated annealing
    if (solver_type == 'simulated_annealing' and 
        step_info and 'temperature' in step_info):
        y_offset += 25
        temp_text = font.render(f"Temperature: {step_info['temperature']:.3f}", True, BLUE)
        screen.blit(temp_text, (BOARD_OFFSET_X + BOARD_SIZE + 30, y_offset))
    
    # Show fitness and generation for genetic algorithm
    if (solver_type == 'genetic' and step_info):
        if 'fitness' in step_info:
            y_offset += 25
            fitness_text = font.render(f"Fitness: {step_info['fitness']}", True, GREEN)
            screen.blit(fitness_text, (BOARD_OFFSET_X + BOARD_SIZE + 30, y_offset))
        
        if 'generation' in step_info:
            y_offset += 25
            gen_text = font.render(f"Generation: {step_info['generation']}", True, BLUE)
            screen.blit(gen_text, (BOARD_OFFSET_X + BOARD_SIZE + 30, y_offset))
        
        if 'avg_fitness' in step_info:
            y_offset += 25
            avg_text = font.render(f"Avg Fitness: {step_info['avg_fitness']:.1f}", True, GRAY)
            screen.blit(avg_text, (BOARD_OFFSET_X + BOARD_SIZE + 30, y_offset))
    
    # Show solver status and statistics
    if stats:
        y_offset += 50
        stats_title = font.render("Solver Statistics:", True, BLACK)
        screen.blit(stats_title, (BOARD_OFFSET_X + BOARD_SIZE + 30, y_offset))
        
        stats_info = [
            f"Time: {stats['solve_time']:.4f}s",
            f"Total Steps: {stats['total_steps']:,}",
            f"Solutions: {stats['solutions_found']}"
        ]
        
        # Add solver-specific stats
        if solver_type == 'exhaustive':
            if 'backtrack_count' in stats:
                stats_info.insert(2, f"Backtracks: {stats['backtrack_count']:,}")
        elif solver_type == 'hill_climbing':
            if 'restart_count' in stats:
                stats_info.insert(2, f"Restarts: {stats['restart_count']:,}")
            if 'min_conflicts' in stats:
                stats_info.append(f"Min Conflicts: {stats['min_conflicts']}")
            if 'sideways_moves' in stats:
                stats_info.append(f"Sideways Moves: {stats['sideways_moves']:,}")
        elif solver_type == 'simulated_annealing':
            if 'temperature_changes' in stats:
                stats_info.insert(2, f"Temp Changes: {stats['temperature_changes']:,}")
            if 'accepted_moves' in stats:
                stats_info.append(f"Accepted: {stats['accepted_moves']:,}")
            if 'rejected_moves' in stats:
                stats_info.append(f"Rejected: {stats['rejected_moves']:,}")
            if 'acceptance_rate' in stats:
                stats_info.append(f"Acceptance Rate: {stats['acceptance_rate']:.1%}")
            if 'final_conflicts' in stats:
                stats_info.append(f"Final Conflicts: {stats['final_conflicts']}")
        elif solver_type == 'genetic':
            if 'generations' in stats:
                stats_info.insert(2, f"Generations: {stats['generations']:,}")
            if 'population_size' in stats:
                stats_info.append(f"Population: {stats['population_size']}")
            if 'mutation_rate' in stats:
                stats_info.append(f"Mutation Rate: {stats['mutation_rate']:.1%}")
            if 'elite_size' in stats:
                stats_info.append(f"Elite Size: {stats['elite_size']}")
            if 'best_fitness' in stats:
                stats_info.append(f"Best Fitness: {stats['best_fitness']}")
            if 'convergence_generation' in stats and stats['convergence_generation']:
                stats_info.append(f"Converged at Gen: {stats['convergence_generation']}")
        
        for i, info in enumerate(stats_info):
            text = font.render(info, True, BLACK)
            screen.blit(text, (BOARD_OFFSET_X + BOARD_SIZE + 30, y_offset + 25 + i * 20))
    
    # Show current step info
    if step_info:
        y_offset += 280 if stats else 70
        step_title = font.render("Current Step:", True, BLACK)
        screen.blit(step_title, (BOARD_OFFSET_X + BOARD_SIZE + 30, y_offset))
        
        step_text = font.render(f"Step {step_info['current']}/{step_info['total']}", True, BLACK)
        screen.blit(step_text, (BOARD_OFFSET_X + BOARD_SIZE + 30, y_offset + 25))
        
        # Wrap long messages
        message = step_info['message']
        max_chars = 40
        if len(message) > max_chars:
            # Split into multiple lines
            words = message.split()
            lines = []
            current_line = ""
            
            for word in words:
                if len(current_line + " " + word) <= max_chars:
                    current_line += (" " if current_line else "") + word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Display up to 3 lines
            for i, line in enumerate(lines[:3]):
                msg_text = font.render(line, True, BLACK)
                screen.blit(msg_text, (BOARD_OFFSET_X + BOARD_SIZE + 30, y_offset + 50 + i * 20))
        else:
            msg_text = font.render(message, True, BLACK)
            screen.blit(msg_text, (BOARD_OFFSET_X + BOARD_SIZE + 30, y_offset + 50))

def main():
    # Get initial board size
    n = get_board_size()
    
    # Try to import the solvers
    try:
        # Assuming the solver files are saved as 'nqueens_solver.py' and 'nqueens_hill_climbing.py'
        from nqueens_solver_dfs import OptimizedNQueensSolver as NQueensSolver
        print("N-Queens exhaustive solver imported successfully!")
        exhaustive_available = True
    except ImportError:
        print("Warning: Could not import nqueens_solver.py")
        exhaustive_available = False
        NQueensSolver = None
    
    try:
        from nqueens_hill_climbing import OptimizedNQueensHillClimbing as NQueensHillClimbing
        print("N-Queens hill climbing solver imported successfully!")
        hill_climbing_available = True
    except ImportError:
        print("Warning: Could not import nqueens_hill_climbing.py")
        hill_climbing_available = False
        NQueensHillClimbing = None
    
    try:
        # Simulated annealing is now part of hill climbing
        simulated_annealing_available = hill_climbing_available
        if not simulated_annealing_available:
            print("Warning: Simulated annealing requires hill climbing solver")
    except ImportError:
        print("Warning: Could not import nqueens_simulated_annealing.py")
        simulated_annealing_available = False
    
    try:
        from nqueens_genetic import NQueensGenetic
        print("N-Queens genetic algorithm solver imported successfully!")
        genetic_available = True
    except ImportError:
        print("Warning: Could not import nqueens_genetic.py")
        genetic_available = False
        NQueensGenetic = None
    
    # Create the display
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"Interactive N-Queens Solver {n}×{n}")
    
    # Create fonts
    font = pygame.font.Font(None, 20)
    small_font = pygame.font.Font(None, 18)
    
    # Create chess board
    board = ChessBoard(n)
    
    # Create text input box for step skipping
    input_box = TextInputBox(BOARD_OFFSET_X + BOARD_SIZE + 150, BOARD_OFFSET_Y + 275, 100, 30, font)
    
    # Solver variables
    solver = None
    current_step = 0
    solver_result = None
    step_by_step_mode = False
    current_solver_type = None
    
    # Game loop
    clock = pygame.time.Clock()
    running = True
    
    print(f"\nN-Queens Solver ready!")
    print("Press 'E' for exhaustive search, 'H' for hill climbing, 'S' for simulated annealing, or 'G' for genetic algorithm.")
    
    while running:
        dt = clock.tick(60)
        input_box.update(dt)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Handle text input box events
            skip_value = input_box.handle_event(event)
            if skip_value is not None and step_by_step_mode and solver:
                try:
                    target_step = int(skip_value) - 1  # Convert to 0-based index
                    if 0 <= target_step < len(solver.steps):
                        current_step = target_step
                        step = solver.get_step(current_step)
                        
                        # Update board visualization
                        positions = solver.convert_board_to_positions(step['board_state'])
                        board.set_solver_objects(positions, current_solver_type)
                        
                        # Highlight current position if applicable
                        if current_solver_type == 'genetic':
                            # For genetic algorithm, no specific highlighting needed
                            board.clear_highlight()
                        elif 'row' in step and 'col' in step:
                            if current_solver_type == 'hill_climbing':
                                # For hill climbing, highlight the new position
                                if 'new_col' in step:
                                    board.highlight_square(step['row'], step['new_col'])
                                else:
                                    board.highlight_square(step['row'], step['col'])
                            elif current_solver_type == 'simulated_annealing':
                                # For simulated annealing, highlight the moved queen
                                board.highlight_square(step['row'], step['col'])
                            else:
                                board.highlight_square(step['row'], step['col'])
                        else:
                            board.clear_highlight()
                        
                        print(f"Jumped to step {target_step + 1}")
                    else:
                        print(f"Invalid step number. Must be between 1 and {len(solver.steps)}")
                except ValueError:
                    print("Please enter a valid number")
                
                input_box.clear()
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                elif event.key == pygame.K_c:  # Clear board
                    board.clear_board()
                    solver = None
                    solver_result = None
                    step_by_step_mode = False
                    current_step = 0
                    current_solver_type = None
                    input_box.clear()
                
                elif event.key == pygame.K_e and exhaustive_available:  # Solve using exhaustive search
                    print("Starting exhaustive search solver...")
                    board.clear_board()
                    solver = NQueensSolver(n)
                    current_solver_type = 'exhaustive'
                    
                    success = solver.solve_step_by_step(find_all=False)  # Only find first solution for visualization
                    solver_result = {
                        'success': success,
                        'statistics': solver.get_statistics(),
                        'solver': solver
                    }
                    
                    if success:
                        print(f"Solution found! {len(solver.steps)} steps recorded.")
                        print("Press SPACE to step through or enter a step number to jump to it.")
                        step_by_step_mode = True
                        current_step = 0
                        input_box.clear()
                    else:
                        print("No solution exists for this board size.")
                
                elif event.key == pygame.K_h and hill_climbing_available:  # Solve using hill climbing
                    print("Starting hill climbing solver...")
                    board.clear_board()
                    solver = NQueensHillClimbing(n, max_restarts=100, max_sideways=100)
                    current_solver_type = 'hill_climbing'
                    
                    success = solver.solve_step_by_step(use_smart_restart=True, allow_sideways=True)
                    solver_result = {
                        'success': success,
                        'statistics': solver.get_statistics(),
                        'solver': solver
                    }
                    
                    if success:
                        print(f"Solution found! {len(solver.steps)} steps recorded.")
                        print("Press SPACE to step through or enter a step number to jump to it.")
                        step_by_step_mode = True
                        current_step = 0
                        input_box.clear()
                    else:
                        print("No solution found after maximum restarts.")
                
                elif event.key == pygame.K_s and simulated_annealing_available:  # Solve using simulated annealing
                    print("Starting simulated annealing solver...")
                    board.clear_board()
                    # Use hill climbing solver with simulated annealing
                    solver = NQueensHillClimbing(n, max_restarts=3)  # Only one run needed for SA
                    current_solver_type = 'simulated_annealing'
                    
                    success = solver.solve_with_simulated_annealing(
                        initial_temp=100.0,
                        cooling_rate=0.95
                    )
                    solver_result = {
                        'success': success,
                        'statistics': solver.get_statistics(),
                        'solver': solver
                    }
                    
                    if success:
                        print(f"Solution found! {len(solver.steps)} steps recorded.")
                        print("Press SPACE to step through or enter a step number to jump to it.")
                        step_by_step_mode = True
                        current_step = 0
                        input_box.clear()
                    else:
                        print("No solution found. Try running again or adjusting parameters.")
                
                elif event.key == pygame.K_g and genetic_available:  # Solve using genetic algorithm
                    print("Starting genetic algorithm solver...")
                    board.clear_board()
                    # Adjust parameters based on board size for better performance
                    pop_size = min(100, max(50, n * 8))
                    mutation_rate = 0.1
                    elite_size = max(5, pop_size // 10)
                    max_gens = min(1000, max(200, n * 50))
                    
                    solver = NQueensGenetic(n, population_size=pop_size, mutation_rate=mutation_rate, 
                                          elite_size=elite_size, max_generations=max_gens)
                    current_solver_type = 'genetic'
                    
                    success = solver.solve_step_by_step()
                    solver_result = {
                        'success': success,
                        'statistics': solver.get_statistics(),
                        'solver': solver
                    }
                    
                    if success:
                        print(f"Solution found! {len(solver.steps)} steps recorded.")
                        print("Press SPACE to step through or enter a step number to jump to it.")
                        step_by_step_mode = True
                        current_step = 0
                        input_box.clear()
                    else:
                        print("No solution found. Try running again or adjusting parameters.")
                
                elif event.key == pygame.K_SPACE and step_by_step_mode and solver:  # Next step
                    if current_step < len(solver.steps) - 1:
                        current_step += 1
                        step = solver.get_step(current_step)
                        
                        # Update board visualization
                        positions = solver.convert_board_to_positions(step['board_state'])
                        board.set_solver_objects(positions, current_solver_type)
                        
                        # Highlight current position if applicable
                        if current_solver_type == 'genetic':
                            # For genetic algorithm, no specific highlighting needed
                            board.clear_highlight()
                        elif 'row' in step and 'col' in step:
                            if current_solver_type == 'hill_climbing':
                                # For hill climbing, highlight the new position
                                if 'new_col' in step:
                                    board.highlight_square(step['row'], step['new_col'])
                                else:
                                    board.highlight_square(step['row'], step['col'])
                            elif current_solver_type == 'simulated_annealing':
                                # For simulated annealing, highlight the moved queen
                                board.highlight_square(step['row'], step['col'])
                            else:
                                board.highlight_square(step['row'], step['col'])
                        else:
                            board.clear_highlight()
                    else:
                        print("Reached the end of the solution steps.")
                
                elif event.key == pygame.K_r and solver:  # Reset to start
                    current_step = 0
                    board.clear_highlight()
                    board.solver_objects.clear()
                    input_box.clear()
                    print("Reset to beginning of solution.")
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Mouse clicks are now only used for the text input box
                pass
        
        # Clear screen
        screen.fill(WHITE)
        
        # Draw board and objects
        board.draw_board(screen)
        board.draw_objects(screen, small_font)
        
        # Prepare step info for UI
        step_info = None
        if step_by_step_mode and solver and current_step < len(solver.steps):
            step = solver.get_step(current_step)
            step_info = {
                'current': current_step + 1,
                'total': len(solver.steps),
                'message': step['message'] if step else ""
            }
            
            # Add conflicts info for hill climbing, simulated annealing, and genetic algorithm
            if current_solver_type in ['hill_climbing', 'simulated_annealing', 'genetic'] and 'conflicts' in step:
                step_info['conflicts'] = step['conflicts']
                
            # Add temperature info for simulated annealing
            if current_solver_type == 'simulated_annealing' and 'temperature' in step:
                step_info['temperature'] = step['temperature']
            
            # Add genetic algorithm specific info
            if current_solver_type == 'genetic':
                if 'fitness' in step:
                    step_info['fitness'] = step['fitness']
                if 'generation' in step:
                    step_info['generation'] = step['generation']
                if 'avg_fitness' in step:
                    step_info['avg_fitness'] = step['avg_fitness']
        
        # Draw UI
        stats = solver_result['statistics'] if solver_result else None
        draw_ui(screen, font, board, step_by_step_mode, step_info, stats, input_box, current_solver_type)
        
        # Update display
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()