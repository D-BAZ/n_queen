import random
import time
from typing import List, Tuple, Dict, Any

class NQueensGenetic:
    def __init__(self, n: int, population_size: int = 50, mutation_rate: float = 0.1, 
                 elite_size: int = 10, max_generations: int = 1000):
        """
        Initialize the Genetic Algorithm solver for N-Queens problem.
        
        Args:
            n: Board size (n x n)
            population_size: Number of individuals in each generation
            mutation_rate: Probability of mutation for each gene
            elite_size: Number of best individuals to keep unchanged
            max_generations: Maximum number of generations to run
        """
        self.n = n
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.max_generations = max_generations
        
        # Statistics tracking
        self.start_time = None
        self.end_time = None
        self.generations_run = 0
        self.best_fitness_history = []
        self.average_fitness_history = []
        self.solution_found = False
        
        # Step-by-step tracking
        self.steps = []
        self.current_population = []
        self.current_generation = 0
        
    def create_individual(self) -> List[int]:
        """Create a random individual (chromosome) representing queen positions."""
        return [random.randint(0, self.n - 1) for _ in range(self.n)]
    
    def create_initial_population(self) -> List[List[int]]:
        """Create the initial population of random individuals."""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def fitness(self, individual: List[int]) -> int:
        """
        Calculate fitness of an individual.
        Fitness = n*(n-1)/2 - conflicts (higher is better)
        Perfect solution has fitness = n*(n-1)/2
        """
        conflicts = self.count_conflicts(individual)
        max_conflicts = self.n * (self.n - 1) // 2
        return max_conflicts - conflicts
    
    def count_conflicts(self, individual: List[int]) -> int:
        """Count the number of queen conflicts in the given configuration."""
        conflicts = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Queens are in different columns by design
                # Check row conflicts
                if individual[i] == individual[j]:
                    conflicts += 1
                # Check diagonal conflicts
                elif abs(individual[i] - individual[j]) == abs(i - j):
                    conflicts += 1
        return conflicts
    
    def selection(self, population: List[List[int]], fitnesses: List[int]) -> List[int]:
        """Tournament selection to choose a parent."""
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_idx = max(tournament_indices, key=lambda idx: fitnesses[idx])
        return population[best_idx].copy()
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Order crossover (OX) adapted for N-Queens."""
        # For N-Queens, we'll use single-point crossover
        crossover_point = random.randint(1, self.n - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def mutate(self, individual: List[int]) -> List[int]:
        """Mutate an individual by randomly changing some queen positions."""
        mutated = individual.copy()
        for i in range(self.n):
            if random.random() < self.mutation_rate:
                mutated[i] = random.randint(0, self.n - 1)
        return mutated
    
    def solve_step_by_step(self) -> bool:
        """Solve the N-Queens problem using genetic algorithm with step recording."""
        self.start_time = time.time()
        self.steps = []
        self.generations_run = 0
        self.best_fitness_history = []
        self.average_fitness_history = []
        self.solution_found = False
        
        # Initialize population
        population = self.create_initial_population()
        self.current_population = population
        
        for generation in range(self.max_generations):
            self.current_generation = generation
            self.generations_run = generation + 1
            
            # Calculate fitness for all individuals
            fitnesses = [self.fitness(individual) for individual in population]
            best_fitness = max(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            
            self.best_fitness_history.append(best_fitness)
            self.average_fitness_history.append(avg_fitness)
            
            # Find best individual
            best_idx = fitnesses.index(best_fitness)
            best_individual = population[best_idx]
            
            # Record step
            step_info = {
                'generation': generation + 1,
                'board_state': best_individual,
                'fitness': best_fitness,
                'conflicts': self.count_conflicts(best_individual),
                'avg_fitness': avg_fitness,
                'message': f"Generation {generation + 1}: Best fitness {best_fitness}, Conflicts {self.count_conflicts(best_individual)}"
            }
            self.steps.append(step_info)
            
            # Check if solution found
            max_fitness = self.n * (self.n - 1) // 2
            if best_fitness == max_fitness:
                self.solution_found = True
                self.end_time = time.time()
                final_step = {
                    'generation': generation + 1,
                    'board_state': best_individual,
                    'fitness': best_fitness,
                    'conflicts': 0,
                    'avg_fitness': avg_fitness,
                    'message': f"Solution found in generation {generation + 1}!"
                }
                self.steps.append(final_step)
                return True
            
            # Create new generation
            new_population = []
            
            # Elitism: keep best individuals
            elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate rest of population through crossover and mutation
            while len(new_population) < self.population_size:
                parent1 = self.selection(population, fitnesses)
                parent2 = self.selection(population, fitnesses)
                
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            population = new_population[:self.population_size]
            self.current_population = population
        
        self.end_time = time.time()
        return False
    
    def get_step(self, step_index: int) -> Dict[str, Any]:
        """Get information about a specific step."""
        if 0 <= step_index < len(self.steps):
            return self.steps[step_index]
        return {}
    
    def convert_board_to_positions(self, board_state: List[int]) -> List[Tuple[int, int]]:
        """Convert board state (list of row indices) to list of (row, col) positions."""
        return [(board_state[col], col) for col in range(len(board_state))]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get solving statistics."""
        solve_time = (self.end_time - self.start_time) if self.end_time and self.start_time else 0
        
        stats = {
            'solve_time': solve_time,
            'total_steps': len(self.steps),
            'generations': self.generations_run,
            'population_size': self.population_size,
            'mutation_rate': self.mutation_rate,
            'elite_size': self.elite_size,
            'solutions_found': 1 if self.solution_found else 0,
            'best_fitness': max(self.best_fitness_history) if self.best_fitness_history else 0,
            'final_avg_fitness': self.average_fitness_history[-1] if self.average_fitness_history else 0,
            'convergence_generation': None
        }
        
        # Find convergence generation (when best fitness stopped improving significantly)
        if len(self.best_fitness_history) > 10:
            max_fitness = self.n * (self.n - 1) // 2
            for i in range(10, len(self.best_fitness_history)):
                if self.best_fitness_history[i] == max_fitness:
                    stats['convergence_generation'] = i + 1
                    break
        
        return stats
    
    def print_solution(self, individual: List[int]):
        """Print the solution in a readable format."""
        print(f"\nSolution for {self.n}-Queens:")
        print("Board representation (Q = Queen, . = Empty):")
        
        for row in range(self.n):
            line = ""
            for col in range(self.n):
                if individual[col] == row:
                    line += "Q "
                else:
                    line += ". "
            print(line)
        
        print(f"\nQueen positions (column, row): {[(col, individual[col]) for col in range(self.n)]}")
        print(f"Conflicts: {self.count_conflicts(individual)}")


# Example usage and testing
if __name__ == "__main__":
    def test_genetic_algorithm():
        print("Testing N-Queens Genetic Algorithm Solver")
        print("=" * 50)
        
        # Test different board sizes
        test_sizes = [4, 8, 12]
        
        for n in test_sizes:
            print(f"\nTesting {n}-Queens problem:")
            solver = NQueensGenetic(n, population_size=100, mutation_rate=0.1, 
                                  elite_size=20, max_generations=500)
            
            success = solver.solve_step_by_step()
            stats = solver.get_statistics()
            
            print(f"Success: {success}")
            print(f"Time: {stats['solve_time']:.4f} seconds")
            print(f"Generations: {stats['generations']}")
            print(f"Steps recorded: {stats['total_steps']}")
            
            if success:
                final_step = solver.get_step(len(solver.steps) - 1)
                solver.print_solution(final_step['board_state'])
    
    test_genetic_algorithm()