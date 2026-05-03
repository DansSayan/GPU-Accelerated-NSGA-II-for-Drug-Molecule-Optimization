import cupy as cp
import cudf
import time
import pandas as pd
import pickle
import argparse

# --------------------------------
# LOAD DATA FUNCTION
# --------------------------------
def load_data(csv_file="processed_molecules.csv"):
    """Load molecular data from CSV file"""
    df = cudf.read_csv(csv_file)
    data = cp.asarray(df[['mol_wt', 'logp', 'h_donors', 'h_acceptors', 'tpsa']].values)
    return data

# --------------------------------
# GLOBAL MUTATION_STD (needed for mutation function)
# --------------------------------
MUTATION_STD = 0.1
# --------------------------------
# OBJECTIVE FUNCTIONS
# --------------------------------
def evaluate(pop):
    mol_wt = pop[:, 0]
    logp = pop[:, 1]
    tpsa = pop[:, 4]

    # Multi-objective:
    # Minimize mol_wt, minimize tpsa, maximize logp
    return cp.stack([
        mol_wt,          # minimize
        -logp,           # maximize logp → minimize negative
        tpsa             # minimize
    ], axis=1)


# --------------------------------
# NON-DOMINATED SORTING
# --------------------------------
def non_dominated_sort(fitness):
    n = fitness.shape[0]
    
    # Expand dims for broadcasting
    f1 = fitness[:, None, :]
    f2 = fitness[None, :, :]
    
    # Check dominance
    less_equal = cp.all(f1 <= f2, axis=2)
    strictly_less = cp.any(f1 < f2, axis=2)
    
    dominance = less_equal & strictly_less
    
    # Count how many dominate each solution
    ranks = cp.sum(dominance, axis=1)
    
    return ranks


# --------------------------------
# SELECTION (now takes pop_size as parameter)
# --------------------------------
def select(pop, fitness, pop_size):
    ranks = non_dominated_sort(fitness)
    idx = cp.argsort(ranks)
    return pop[idx[:pop_size]]


# --------------------------------
# CROSSOVER
# --------------------------------
def crossover(parent1, parent2):
    alpha = cp.random.rand()
    child = alpha * parent1 + (1 - alpha) * parent2
    return child


# --------------------------------
# MUTATION (now takes mutation_std as parameter)
# --------------------------------
def mutate(child, mutation_std):
    noise = cp.random.normal(0, mutation_std, size=child.shape)
    return child + noise


# --------------------------------
# MAIN NSGA-II OPTIMIZATION FUNCTION
# --------------------------------
def run_nsga2_optimization(pop_size=500, generations=50, mutation_std=0.1, seed=None):
    """
    Run NSGA-II optimization algorithm on GPU
    
    Parameters:
    -----------
    pop_size : int
        Population size
    generations : int
        Number of generations to run
    mutation_std : float
        Standard deviation for Gaussian mutation
    seed : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    pareto_df : pd.DataFrame
        Pareto front solutions
    history : dict
        Optimization history with 'best_fitness' key
    """
    global MUTATION_STD
    MUTATION_STD = mutation_std
    
    if seed is not None:
        cp.random.seed(seed)
    
    # Load data
    data = load_data()
    population = data[:pop_size]
    
    history_best_fitness = []
    
    # --------------------------------
    # MAIN NSGA LOOP
    # --------------------------------
    for gen in range(generations):
        fitness = evaluate(population)
        
        # Selection
        selected = select(population, fitness, pop_size)
        
        # Generate offspring
        offspring = []
        for i in range(0, pop_size, 2):
            p1 = selected[i % pop_size]
            p2 = selected[(i+1) % pop_size]
            
            child1 = mutate(crossover(p1, p2), mutation_std)
            child2 = mutate(crossover(p2, p1), mutation_std)
            
            offspring.append(child1)
            offspring.append(child2)
        
        offspring = cp.stack(offspring)
        
        # Combine
        population = cp.concatenate([selected, offspring], axis=0)
        
        # Reduce to pop_size
        fitness = evaluate(population)
        population = select(population, fitness, pop_size)
        
        # Track best fitness
        ranks = non_dominated_sort(fitness)
        best_fitness = cp.min(fitness[ranks == 0]).item() if cp.any(ranks == 0) else cp.min(fitness).item()
        history_best_fitness.append(best_fitness)
        
        print(f"Generation {gen+1}/{generations} completed")
    
    # --------------------------------
    # FINAL PARETO FRONT
    # --------------------------------
    final_fitness = evaluate(population)
    ranks = non_dominated_sort(final_fitness)
    
    pareto_front = population[ranks == 0]
    
    print(f"Pareto optimal solutions: {pareto_front.shape[0]}")
    
    pareto_np = cp.asnumpy(pareto_front)
    cols = ['mol_wt', 'logp', 'h_donors', 'h_acceptors', 'tpsa']
    pareto_df = pd.DataFrame(pareto_np, columns=cols)
    
    history = {'best_fitness': history_best_fitness}
    
    return pareto_df, history


# --------------------------------
# INTERACTIVE CLI MODE
# --------------------------------
if __name__ == "__main__":
    POP_SIZE     = int(input("Population size       : ") or 500)
    GENERATIONS  = int(input("Number of generations : ") or 50)
    MUTATION_STD = float(input("Mutation std          : ") or 0.1)
    seed_input   = input("Random seed (leave blank to skip): ").strip()
    
    seed = int(seed_input) if seed_input else None
    
    pareto_df, history = run_nsga2_optimization(
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        mutation_std=MUTATION_STD,
        seed=seed
    )
    
    pareto_df.to_csv("pareto_front.csv", index=False)
    print("Saved: pareto_front.csv")