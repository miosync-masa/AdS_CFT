# Prototype: Λ³ Self-Evolving Automaton with CSP-driven Environment
# ---------------------------------------------------------------
# What this does (quick overview):
# - 2D grid of "organisms", each with a simple genome (bitstring length G)
# - Environment is a time-varying Constraint Satisfaction Problem (CSP):
#     * For each (x,y,t), environment specifies a target Hamming weight s(x,y,t).
#     * Fitness = 1 - |ones(genome) - s| / G. (Higher is better / constraint satisfied)
# - Define Λ = K / |V| per cell:
#     * K (exploration pressure) combines mismatch pressure and innate drive to change
#     * |V| (cohesion) is stability from fitness + neighborhood support (coordination)
# - Local rules per cell:
#     * Λ < 0.9  -> replicate (into empty neighbor if any; low mutation rate)
#     * 0.9–1.1  -> evolve (mutate more aggressively to chase constraint)
#     * Λ > 1.1  -> die (cell becomes empty)
# - We visualize:
#     * Average fitness vs. time
#     * Fraction alive vs. time
#     * Final grid heatmap of fitness
#
# Notes:
# - Uses only numpy + matplotlib (no seaborn), one chart per plot, and no explicit colors.
# - Keep it lightweight and self-contained.
#
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ------------------------
# Parameters
# ------------------------
GRID_H, GRID_W = 40, 40
G = 16  # genome length
STEPS = 60
SEED_DENSITY = 0.15  # initial fill probability
NEIGH_RADIUS = 1     # Moore neighborhood radius

# Λ thresholds
L_REPLICATE = 0.90
L_EVOLVE    = 1.10

# Mutation rates
MUT_LOW  = 0.01
MUT_HIGH = 0.10

# Exploration (K) coefficients
K_BASE = 0.05
K_MISMATCH_SCALE = 0.8

# Cohesion (|V|) coefficients
V_FIT_W = 0.7
V_NEI_W = 0.3
EPS = 1e-6

rng = np.random.default_rng(42)

@dataclass
class Cell:
    alive: bool
    genome: np.ndarray  # shape (G,), dtype=bool

def random_genome():
    return rng.integers(0, 2, size=G, dtype=np.int8).astype(bool)

def init_grid():
    grid = [[None for _ in range(GRID_W)] for _ in range(GRID_H)]
    for i in range(GRID_H):
        for j in range(GRID_W):
            if rng.random() < SEED_DENSITY:
                grid[i][j] = Cell(True, random_genome())
            else:
                grid[i][j] = Cell(False, np.zeros(G, dtype=bool))
    return grid

# ------------------------
# Environment CSP (time-varying)
# Target ones count per position/time: s(x,y,t)
# We define a smooth wave across space/time to force adaptation.
# ------------------------
def target_ones(i, j, t):
    # s in [0, G], wave packet that moves over time
    # Compose two sinusoids and clamp
    val = (G/2
           + (G/4) * np.sin(2*np.pi*(i/GRID_H + 0.03*t))
           + (G/6) * np.cos(2*np.pi*(j/GRID_W + 0.05*t)))
    return int(np.clip(round(val), 0, G))

def fitness(genome, s):
    ones = int(genome.sum())
    return 1.0 - abs(ones - s) / G  # in [0,1]

def neighbors(i, j):
    coords = []
    for di in range(-NEIGH_RADIUS, NEIGH_RADIUS+1):
        for dj in range(-NEIGH_RADIUS, NEIGH_RADIUS+1):
            if di == 0 and dj == 0: 
                continue
            ii = i + di
            jj = j + dj
            if 0 <= ii < GRID_H and 0 <= jj < GRID_W:
                coords.append((ii, jj))
    return coords

def neighbor_support(grid, i, j, s):
    # Average fitness of neighbors against the same local target s
    nb = neighbors(i, j)
    if not nb:
        return 0.0
    accum = 0.0
    cnt = 0
    for ii, jj in nb:
        c = grid[ii][jj]
        if c.alive:
            accum += fitness(c.genome, s)
            cnt += 1
    return accum / cnt if cnt > 0 else 0.0

def mutate(genome, rate):
    mask = rng.random(G) < rate
    newg = genome.copy()
    newg[mask] = ~newg[mask]
    return newg

def step(grid, t):
    # Prepare next grid (copy structure, then update)
    next_grid = [[Cell(False, np.zeros(G, dtype=bool)) for _ in range(GRID_W)] for _ in range(GRID_H)]
    # We also track metrics
    total_fit = 0.0
    total_alive = 0

    # Pass 1: compute Λ and decide actions
    actions = [[None for _ in range(GRID_W)] for _ in range(GRID_H)]
    for i in range(GRID_H):
        for j in range(GRID_W):
            cell = grid[i][j]
            s = target_ones(i, j, t)
            if cell.alive:
                f = fitness(cell.genome, s)
                nei = neighbor_support(grid, i, j, s)
                V = V_FIT_W * f + V_NEI_W * nei
                mismatch = 1.0 - f
                K = K_BASE + K_MISMATCH_SCALE * mismatch
                L = K / (V + EPS)
                total_fit += f
                total_alive += 1

                if L < L_REPLICATE:
                    actions[i][j] = ("replicate", L, f, cell.genome)
                elif L <= L_EVOLVE:
                    actions[i][j] = ("evolve", L, f, cell.genome)
                else:
                    actions[i][j] = ("die", L, f, cell.genome)
            else:
                actions[i][j] = ("empty", 0.0, 0.0, cell.genome)

    # Pass 2: apply actions
    # Priority: deaths first, then evolutions (in place), then replications to neighbors
    # Start with copying living cells by default (will overwrite on die)
    for i in range(GRID_H):
        for j in range(GRID_W):
            act, L, f, g = actions[i][j]
            if act == "die":
                next_grid[i][j] = Cell(False, np.zeros(G, dtype=bool))
            elif act in ("replicate", "evolve"):
                # keep the cell alive (apply potential mutation on evolve)
                if act == "evolve":
                    newg = mutate(g, MUT_HIGH)
                else:
                    newg = mutate(g, MUT_LOW)  # slight drift even when stable
                next_grid[i][j] = Cell(True, newg)
            else:
                # empty stays empty unless later filled by replication
                next_grid[i][j] = Cell(False, np.zeros(G, dtype=bool))

    # Handle replications into neighboring empty cells
    # We sweep again to populate adjacent empties (each replicator claims at most one spot)
    claimed = set()
    for i in range(GRID_H):
        for j in range(GRID_W):
            act, L, f, g = actions[i][j]
            if act == "replicate":
                # look for an empty neighbor to colonize
                nb = neighbors(i, j)
                rng.shuffle(nb)
                for ii, jj in nb:
                    if (ii, jj) not in claimed and not next_grid[ii][jj].alive:
                        # offspring with low mutation rate
                        child = mutate(g, MUT_LOW)
                        next_grid[ii][jj] = Cell(True, child)
                        claimed.add((ii, jj))
                        break

    avg_fit = total_fit / total_alive if total_alive > 0 else 0.0
    frac_alive = total_alive / (GRID_H * GRID_W)
    return next_grid, avg_fit, frac_alive

# ------------------------
# Run simulation
# ------------------------
grid = init_grid()
avg_fits = []
alive_fracs = []

for t in range(STEPS):
    grid, a, z = step(grid, t)
    avg_fits.append(a)
    alive_fracs.append(z)

# ------------------------
# Collect final fitness map for visualization
# ------------------------
final_fit = np.zeros((GRID_H, GRID_W))
for i in range(GRID_H):
    for j in range(GRID_W):
        if grid[i][j].alive:
            s = target_ones(i, j, STEPS)
            final_fit[i, j] = fitness(grid[i][j].genome, s)
        else:
            final_fit[i, j] = 0.0

# ------------------------
# Plots (each on its own figure; no explicit colors set)
# ------------------------
plt.figure()
plt.plot(avg_fits)
plt.title("Average Fitness vs. Time")
plt.xlabel("Step")
plt.ylabel("Average Fitness")
plt.show()

plt.figure()
plt.plot(alive_fracs)
plt.title("Fraction Alive vs. Time")
plt.xlabel("Step")
plt.ylabel("Alive Fraction")
plt.show()

plt.figure()
plt.imshow(final_fit, interpolation='nearest')
plt.title("Final Grid Fitness (heatmap)")
plt.colorbar()
plt.show()
