import cupy as cp
import cudf
import time
import pandas as pd
import pickle
import argparse
# --------------------------------
# LOAD GPU DATA
# --------------------------------
POP_SIZE     = int(input("Population size       : ") or 500)
GENERATIONS  = int(input("Number of generations : ") or 50)
MUTATION_STD = float(input("Mutation std          : ") or 0.1)
seed_input   = input("Random seed (leave blank to skip): ").strip()
if seed_input:
    cp.random.seed(int(seed_input))

df = cudf.read_csv("processed_molecules.csv")
data = cp.asarray(df[['mol_wt', 'logp', 'h_donors', 'h_acceptors', 'tpsa']].values)

population = data[:POP_SIZE]

history = []
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
# SELECTION
# --------------------------------
def select(pop, fitness):
    ranks = non_dominated_sort(fitness)
    idx = cp.argsort(ranks)
    return pop[idx[:POP_SIZE]]


# --------------------------------
# CROSSOVER
# --------------------------------
def crossover(parent1, parent2):
    alpha = cp.random.rand()
    child = alpha * parent1 + (1 - alpha) * parent2
    return child


# --------------------------------
# MUTATION
# --------------------------------
def mutate(child):
    noise = cp.random.normal(0, MUTATION_STD, size=child.shape)
    return child + noise


# --------------------------------
# MAIN NSGA LOOP
# --------------------------------
for gen in range(GENERATIONS):
    # start = time.time()
    fitness = evaluate(population)

    # Selection
    selected = select(population, fitness)

    # Generate offspring
    offspring = []
    for i in range(0, POP_SIZE, 2):
        p1 = selected[i]
        p2 = selected[i+1]

        child1 = mutate(crossover(p1, p2))
        child2 = mutate(crossover(p2, p1))

        offspring.append(child1)
        offspring.append(child2)

    offspring = cp.stack(offspring)

    # Combine
    population = cp.concatenate([selected, offspring], axis=0)

    # Convert a small sample to CPU (to reduce overhead)
    sample = cp.asnumpy(population[:200])   # only 200 points
    history.append(sample)

    # Reduce to POP_SIZE
    fitness = evaluate(population)
    population = select(population, fitness)

    print(f"Generation {gen+1} completed")
    # print(f"Gen {gen+1} time:", time.time() - start)

with open("generation_history.pkl", "wb") as f:
    pickle.dump(history, f)

# --------------------------------
# FINAL PARETO FRONT
# --------------------------------
final_fitness = evaluate(population)
ranks = non_dominated_sort(final_fitness)

pareto_front = population[ranks == 0]

print("Pareto optimal solutions:", pareto_front.shape)

pareto_np = cp.asnumpy(pareto_front)
cols = ['mol_wt', 'logp', 'h_donors', 'h_acceptors', 'tpsa']
pareto_df = pd.DataFrame(pareto_np, columns=cols)
pareto_df.to_csv("pareto_front.csv", index=False)
print("Saved: pareto_front.csv")