import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random
import time

class TouristRoute:
    def __init__(self, num_points: int = 20):
        """Initialize tourist points with random coordinates and visit times"""
        self.num_points = num_points
        # Generate random coordinates for tourist points
        self.coordinates = np.random.rand(num_points, 2) * 100
        # Generate random visit times between 1 and 3 hours
        self.visit_times = np.random.uniform(1, 3, num_points)
        
    def calculate_total_time(self, route: List[int]) -> float:
        """Calculate total time including travel and visit times"""
        total_time = 0
        for i in range(len(route)):
            # Add visit time for current point
            total_time += self.visit_times[route[i]]
            # Add travel time to next point
            if i < len(route) - 1:
                total_time += self.calculate_distance(route[i], route[i + 1])
        return total_time

    def calculate_distance(self, point1: int, point2: int) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt(np.sum((self.coordinates[point1] - self.coordinates[point2]) ** 2))

class MemeticAlgorithm:
    def __init__(self, 
                 tourist_route: TouristRoute,
                 population_size: int = 100,
                 generations: int = 1000,
                 mutation_rate: float = 0.1,
                 local_search_frequency: int = 10,
                 local_search_subset: float = 0.2,
                 lamarckian: bool = True):
        self.tourist_route = tourist_route
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.local_search_frequency = local_search_frequency
        self.local_search_subset = local_search_subset
        self.lamarckian = lamarckian
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
    def initialize_population(self) -> List[List[int]]:
        """Create initial population of random permutations"""
        population = []
        for _ in range(self.population_size):
            route = list(range(self.tourist_route.num_points))
            random.shuffle(route)
            population.append(route)
        return population

    def fitness(self, route: List[int]) -> float:
        """Calculate fitness (inverse of total time)"""
        total_time = self.tourist_route.calculate_total_time(route)
        return 1.0 / total_time

    def tournament_selection(self, population: List[List[int]], tournament_size: int = 3) -> List[int]:
        """Select individual using tournament selection"""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=self.fitness)

    def order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Implement Order Crossover (OX) for permutation"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        # Initialize offspring with empty values
        offspring = [-1] * size
        
        # Copy segment from parent1
        offspring[start:end] = parent1[start:end]
        
        # Fill remaining positions with elements from parent2
        current_pos = end
        for item in parent2[end:] + parent2[:end]:
            if item not in offspring:
                offspring[current_pos % size] = item
                current_pos += 1
                
        return offspring

    def swap_mutation(self, route: List[int]) -> List[int]:
        """Apply swap mutation with given probability"""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route

    def local_search_2opt(self, route: List[int], max_iterations: int = 100) -> List[int]:
        """Apply 2-opt local search to improve route"""
        best_route = route.copy()
        best_time = self.tourist_route.calculate_total_time(best_route)
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    new_route = best_route.copy()
                    new_route[i:j] = new_route[i:j][::-1]  # Reverse segment
                    new_time = self.tourist_route.calculate_total_time(new_route)
                    
                    if new_time < best_time:
                        best_route = new_route
                        best_time = new_time
                        improved = True
                        break
                if improved:
                    break
                    
        return best_route

    def apply_local_search(self, population: List[List[int]], generation: int) -> List[List[int]]:
        """Apply local search based on strategy (static or dynamic)"""
        if self.local_search_frequency == 0:  # Static strategy
            if generation == self.generations - 1:
                population = [self.local_search_2opt(route) for route in population]
        else:  # Dynamic strategy
            if generation % self.local_search_frequency == 0:
                num_individuals = int(self.population_size * self.local_search_subset)
                indices = random.sample(range(self.population_size), num_individuals)
                for idx in indices:
                    improved_route = self.local_search_2opt(population[idx])
                    if self.lamarckian:
                        population[idx] = improved_route
                    else:  # Baldwinian model
                        if self.fitness(improved_route) > self.fitness(population[idx]):
                            population[idx] = improved_route
        return population

    def evolve(self) -> Tuple[List[int], List[float], List[float]]:
        """Run the memetic algorithm"""
        population = self.initialize_population()
        best_solution = None
        best_fitness = float('-inf')
        
        for generation in range(self.generations):
            # Apply local search
            population = self.apply_local_search(population, generation)
            
            # Create new population
            new_population = []
            
            # Elitism - keep best individual
            elite = max(population, key=self.fitness)
            new_population.append(elite)
            
            # Generate rest of new population
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                offspring = self.order_crossover(parent1, parent2)
                offspring = self.swap_mutation(offspring)
                new_population.append(offspring)
            
            population = new_population
            
            # Track statistics
            current_fitnesses = [self.fitness(route) for route in population]
            best_current = max(current_fitnesses)
            avg_current = sum(current_fitnesses) / len(current_fitnesses)
            
            self.best_fitness_history.append(best_current)
            self.avg_fitness_history.append(avg_current)
            
            if best_current > best_fitness:
                best_fitness = best_current
                best_solution = population[current_fitnesses.index(best_current)]
                
        return best_solution, self.best_fitness_history, self.avg_fitness_history

def compare_algorithms():
    """Compare different algorithm variants"""
    np.random.seed(42)
    random.seed(42)
    
    # Problem instance
    tourist_problem = TouristRoute(num_points=20)
    
    # Algorithm variants
    variants = [
        ("Genetic Algorithm", {"local_search_frequency": 0}),
        ("Static Memetic (Lamarckian)", {"local_search_frequency": 0, "lamarckian": True}),
        ("Dynamic Memetic (Baldwinian)", {"local_search_frequency": 10, "lamarckian": False})
    ]
    
    results = {}
    
    for name, params in variants:
        print(f"\nRunning {name}...")
        start_time = time.time()
        
        algorithm = MemeticAlgorithm(tourist_problem, **params)
        best_route, best_history, avg_history = algorithm.evolve()
        
        execution_time = time.time() - start_time
        final_time = tourist_problem.calculate_total_time(best_route)
        
        results[name] = {
            "best_route": best_route,
            "final_time": final_time,
            "execution_time": execution_time,
            "best_history": best_history,
            "avg_history": avg_history
        }
        
        print(f"Final route time: {final_time:.2f}")
        print(f"Execution time: {execution_time:.2f} seconds")
    
    # Plot convergence comparison
    plt.figure(figsize=(12, 6))
    for name, data in results.items():
        plt.plot(data["best_history"], label=f"{name} (Best)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Convergence Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("convergence_comparison.png")
    plt.close()
    
    return results

if __name__ == "__main__":
    results = compare_algorithms()