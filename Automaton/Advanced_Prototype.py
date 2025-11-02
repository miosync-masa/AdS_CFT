# Λ³ Self-Evolving Automaton — Advanced Prototype
# ------------------------------------------------
# Adds:
# 1) Multi-constraint CSP environment (two tasks + weights)
# 2) Game-theoretic neighbor interaction (Prisoner's Dilemma-like payoff)
# 3) Genome as instruction list controlling phenotype & replication "machinery"
# 4) Critical transition logging around Λ≈1 and lineage (phylogeny) tracking
#
# Plots:
#  - Avg fitness, alive fraction, avg cooperation over time
#  - Fraction near-critical (|Λ-1|<ε) over time
#  - Final fitness heatmap, final cooperation heatmap
#  - Saves lineage edges CSV and critical log CSV
#
# Notes: matplotlib only, one chart per figure, no explicit colors.
#
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import deque, defaultdict
import csv
import json

# ------------------------
# Parameters
# ------------------------
GRID_H, GRID_W = 30, 30
G = 24      # genome length (bits, grouped as 3-bit opcodes + args)
STEPS = 80
SEED_DENSITY = 0.18
NEIGH_RADIUS = 1

# Λ thresholds
L_REPLICATE = 0.90
L_EVOLVE    = 1.10
EPS_L_CRIT  = 0.03   # |Λ-1|<EPS => near-critical log

# Mutation rates
MUT_LOW  = 0.005
MUT_HIGH = 0.08

# Exploration (K) coefficients
K_BASE = 0.05
K_MISMATCH_SCALE = 0.7
K_GAME_SCALE = 0.2  # game pressure contributes to K (destabilizing when losing)

# Cohesion (|V|) coefficients
V_FIT_W = 0.55
V_NEI_W = 0.25
V_GAME_W = 0.20

# RNG
rng = np.random.default_rng(1234)

# ------------------------
# Genome & phenotype mapping
# ------------------------
# We interpret genome as simple instruction stream:
#   - 3-bit opcode + 5-bit argument per instruction -> 8 bits/instruction
#   - We will parse first 3 instructions from G=24 bits
# Opcodes (3-bit):
#  000: SET_COOP  (arg: 0..31 -> target cooperation level in [0,1])
#  001: BIAS_REPL (arg: 0..31 -> replication bias: prefer empties, fitness, or help neighbors)
#  010: MASK_ONES (arg: 0..31 -> desired ones modulo mapping for CSP-1 weight)
#  011: MASK_BLOCK (arg -> pattern block size for CSP-2 local pattern match length)
#  100: MUT_TUNE  (arg -> local mutation intensity scaler in [0.5,1.5])
#  101: COOP_RULE (arg -> strategy tweak for PD: Tit-for-Tat weight etc.)
#  110: NOOP
#  111: NOOP
#
# Phenotype fields we derive:
#   cooperation in [0,1]
#   repl_bias in {0,1,2} (0: empty-first, 1: fittest-first, 2: help-ally-first)
#   csp1_pref in [0,1] weight amplifier for constraint 1
#   csp2_len in {1..6} preferred block length for constraint 2
#   mut_scale in [0.5, 1.5]
#   rule_tft in [0,1] tendency toward Tit-for-Tat (memory of last neighbor actions)
#
@dataclass
class Phenotype:
    coop: float
    repl_bias: int
    csp1_pref: float
    csp2_len: int
    mut_scale: float
    rule_tft: float

@dataclass
class Cell:
    alive: bool
    genome: np.ndarray  # bool array length G
    id: int             # unique lineage id
    parent: int         # parent id or -1
    last_action: int    # 1=cooperate, 0=defect (for simple memory)

global_id_counter = 0
def next_id(parent=-1):
    global global_id_counter
    global_id_counter += 1
    return global_id_counter, parent

def random_genome():
    return rng.integers(0, 2, size=G, dtype=np.int8).astype(bool)

def parse_genome(genome: np.ndarray) -> Phenotype:
    # split into 3 instructions of 8 bits
    def bits_to_int(bits):
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return val

    fields = {'coop':0.5,'repl_bias':0,'csp1_pref':1.0,'csp2_len':3,'mut_scale':1.0,'rule_tft':0.0}

    for k in range(3):
        chunk = genome[k*8:(k+1)*8]
        if len(chunk) < 8: break
        opcode = bits_to_int(chunk[:3])
        arg = bits_to_int(chunk[3:])
        if opcode == 0: # SET_COOP
            fields['coop'] = min(1.0, arg/31.0)
        elif opcode == 1: # BIAS_REPL
            fields['repl_bias'] = int((arg/31.0)*2.99)  # 0..2
        elif opcode == 2: # MASK_ONES -> csp1_pref
            fields['csp1_pref'] = 0.5 + 0.5*(arg/31.0)
        elif opcode == 3: # MASK_BLOCK -> length 1..6
            fields['csp2_len'] = 1 + int((arg/31.0)*5.99)
        elif opcode == 4: # MUT_TUNE
            fields['mut_scale'] = 0.5 + (arg/31.0)
        elif opcode == 5: # COOP_RULE
            fields['rule_tft'] = arg/31.0
        else:
            pass  # NOOPs

    return Phenotype(fields['coop'], fields['repl_bias'], fields['csp1_pref'],
                     fields['csp2_len'], fields['mut_scale'], fields['rule_tft'])

# ------------------------
# Environment: Multi-constraint CSP
#   C1: target Hamming weight s1(i,j,t)
#   C2: target "blockiness" — fraction of matching runs of length L around index window
# Fitness = w1 * f1 + w2 * f2
# ------------------------
def target_s1(i, j, t):
    base = (G/2
            + (G/4)*np.sin(2*np.pi*(i/GRID_H + 0.02*t))
            + (G/6)*np.cos(2*np.pi*(j/GRID_W + 0.03*t)))
    return int(np.clip(round(base), 0, G))

def csp1_fitness(genome, s, pref):
    ones = int(genome.sum())
    # preference scales penalty
    return 1.0 - pref*abs(ones - s)/G

def csp2_fitness(genome, L, i, j, t):
    # sliding pattern: at time t choose a target local pattern by phase
    # use a short window of length 8, with desired run-length L around the center
    window_len = 8
    start = (i + j + t) % (G - window_len + 1)
    seg = genome[start:start+window_len].astype(int)
    # score how many runs of identical bits reach length L
    score = 0
    run = 1
    for k in range(1, len(seg)):
        if seg[k] == seg[k-1]:
            run += 1
            if run == L:
                score += 1
        else:
            run = 1
    # normalize by a rough upper bound (window_len)
    return min(1.0, score / max(1, window_len))

def fitness_multi(genome, i, j, t, pheno: Phenotype):
    s = target_s1(i, j, t)
    f1 = csp1_fitness(genome, s, pheno.csp1_pref)
    f2 = csp2_fitness(genome, pheno.csp2_len, i, j, t)
    # weights can vary in space-time to force adaptation
    w1 = 0.6 + 0.2*np.sin(2*np.pi*(t/30 + i/GRID_H))
    w2 = 1.0 - w1
    f = max(0.0, min(1.0, w1*f1 + w2*f2))
    return f, s, f1, f2, w1, w2

def neighbors(i, j):
    coords = []
    for di in range(-NEIGH_RADIUS, NEIGH_RADIUS+1):
        for dj in range(-NEIGH_RADIUS, NEIGH_RADIUS+1):
            if di == 0 and dj == 0:
                continue
            ii, jj = i+di, j+dj
            if 0 <= ii < GRID_H and 0 <= jj < GRID_W:
                coords.append((ii, jj))
    return coords

# ------------------------
# Game-theoretic interaction (Prisoner's Dilemma-like):
# Each step, a cell "acts" (C=1 or D=0). Payoff matrix:
#   R=3 (C,C), S=0 (C,D), T=5 (D,C), P=1 (D,D)
# Effective game score averaged over neighbors; we map to [0,1] then use:
#  - Add to |V| via V_GAME_W * game_score
#  - Add to K via K_GAME_SCALE * (1 - game_score)  (losers destabilize)
# Strategy: probability of C = pheno.coop blended with Tit-for-Tat memory
# ------------------------
R, S, T, P = 3, 0, 5, 1
def play_pd_action(cell: Cell, pheno: Phenotype, avg_nb_last):
    # Tit-for-Tat blend: if neighbors mostly cooperated last round, raise chance to cooperate
    base = pheno.coop
    influence = pheno.rule_tft * (avg_nb_last - 0.5) * 2.0  # -1..+1 scaled
    pC = np.clip(base + influence*0.25, 0.0, 1.0)
    act = 1 if rng.random() < pC else 0
    return act

def pd_payoff(my_act, nb_act):
    if my_act == 1 and nb_act == 1: return R
    if my_act == 1 and nb_act == 0: return S
    if my_act == 0 and nb_act == 1: return T
    return P

def neighbor_game(grid, i, j, pheno: Phenotype):
    nb = neighbors(i, j)
    if not nb:
        return 0.5, 0.5  # neutral
    # compute neighbor average last action
    acts = []
    for ii,jj in nb:
        c = grid[ii][jj]
        if c.alive:
            acts.append(c.last_action)
    avg_nb_last = np.mean(acts) if acts else 0.5
    # choose my action
    me = grid[i][j]
    my_act = play_pd_action(me, pheno, avg_nb_last)
    # play with each neighbor's last action (one-shot memory)
    pay = 0.0
    cnt = 0
    for ii,jj in nb:
        c = grid[ii][jj]
        if c.alive:
            pay += pd_payoff(my_act, c.last_action)
            cnt += 1
    score = pay / (cnt*5.0) if cnt>0 else 0.5  # normalize by T(=5)
    return score, my_act

# ------------------------
# Grid init
# ------------------------
def init_grid():
    grid = [[None for _ in range(GRID_W)] for _ in range(GRID_H)]
    for i in range(GRID_H):
        for j in range(GRID_W):
            if rng.random() < SEED_DENSITY:
                gid, parent = next_id(-1)
                grid[i][j] = Cell(True, random_genome(), gid, parent, last_action=rng.integers(0,2))
            else:
                grid[i][j] = Cell(False, np.zeros(G, dtype=bool), -1, -1, 0)
    return grid

def mutate(genome, rate):
    mask = rng.random(G) < rate
    newg = genome.copy()
    newg[mask] = ~newg[mask]
    return newg

# Lineage & critical logs
lineage_edges = []      # (parent_id, child_id, t)
critical_log = []       # dict per event

def step(grid, t):
    next_grid = [[Cell(False, np.zeros(G, dtype=bool), -1, -1, 0) for _ in range(GRID_W)] for _ in range(GRID_H)]
    total_fit = 0.0
    total_alive = 0
    total_coop = 0.0
    near_crit_count = 0

    # First pass: compute Λ and decide actions
    actions = [[None for _ in range(GRID_W)] for _ in range(GRID_H)]
    phenos = [[None for _ in range(GRID_W)] for _ in range(GRID_H)]
    pd_scores = [[0.5 for _ in range(GRID_W)] for _ in range(GRID_H)]
    pd_actions = [[0 for _ in range(GRID_W)] for _ in range(GRID_H)]

    # Pre-parse phenotypes for alive cells
    for i in range(GRID_H):
        for j in range(GRID_W):
            c = grid[i][j]
            if c.alive:
                phenos[i][j] = parse_genome(c.genome)

    # Play PD & evaluate
    for i in range(GRID_H):
        for j in range(GRID_W):
            c = grid[i][j]
            if not c.alive:
                actions[i][j] = ("empty", 0.0, 0.0, c.genome, 0)
                continue
            ph = phenos[i][j]
            f, s, f1, f2, w1, w2 = fitness_multi(c.genome, i, j, t, ph)

            # Neighbor support and game
            nb = neighbors(i, j)
            nei_acc = 0.0
            nei_cnt = 0
            for ii,jj in nb:
                cc = grid[ii][jj]
                if cc.alive:
                    # neighbor evaluated against same time/space targets for cohesion
                    ff, *_ = fitness_multi(cc.genome, i, j, t, parse_genome(cc.genome))
                    nei_acc += ff
                    nei_cnt += 1
            nei = (nei_acc/nei_cnt) if nei_cnt>0 else 0.0

            gscore, my_act = neighbor_game(grid, i, j, ph)
            pd_scores[i][j] = gscore
            pd_actions[i][j] = my_act

            V = V_FIT_W * f + V_NEI_W * nei + V_GAME_W * gscore
            mismatch = 1.0 - f
            K = K_BASE + K_MISMATCH_SCALE * mismatch + K_GAME_SCALE * (1.0 - gscore)
            L = K / (V + 1e-6)

            if abs(L - 1.0) < EPS_L_CRIT:
                near_crit_count += 1
                critical_log.append({
                    't': t, 'i': i, 'j': j, 'Lambda': float(L),
                    'fitness': float(f), 'nei': float(nei),
                    'gscore': float(gscore)
                })

            total_fit += f
            total_alive += 1
            total_coop += ph.coop

            if L < L_REPLICATE:
                actions[i][j] = ("replicate", L, f, c.genome, my_act)
            elif L <= L_EVOLVE:
                actions[i][j] = ("evolve", L, f, c.genome, my_act)
            else:
                actions[i][j] = ("die", L, f, c.genome, my_act)

    # Second pass: apply actions (deaths/evolve then replication)
    # Evolve in place (mutate more), keep PD action as last_action
    for i in range(GRID_H):
        for j in range(GRID_W):
            act, L, f, g, my_act = actions[i][j]
            c = grid[i][j]
            if act == "die":
                next_grid[i][j] = Cell(False, np.zeros(G, dtype=bool), -1, -1, 0)
            elif act == "evolve":
                ph = parse_genome(g)
                newg = mutate(g, MUT_HIGH * ph.mut_scale)
                next_grid[i][j] = Cell(True, newg, c.id, c.parent, my_act)
            elif act == "replicate":
                ph = parse_genome(g)
                # parent stays with small drift
                newg = mutate(g, MUT_LOW * ph.mut_scale)
                next_grid[i][j] = Cell(True, newg, c.id, c.parent, my_act)
            else:
                next_grid[i][j] = Cell(False, np.zeros(G, dtype=bool), -1, -1, 0)

    # Replication: offspring placement by bias
    def neighbor_slots(i, j):
        nb = neighbors(i, j)
        empties = []
        allies = []
        others = []
        for ii,jj in nb:
            if not next_grid[ii][jj].alive:
                empties.append((ii,jj))
            else:
                # "ally" if similar genome Hamming distance < threshold
                hd = np.sum(next_grid[ii][jj].genome != next_grid[i][j].genome)
                if hd < G*0.25:
                    allies.append((ii,jj))
                else:
                    others.append((ii,jj))
        return empties, allies, others

    claimed = set()
    for i in range(GRID_H):
        for j in range(GRID_W):
            act, L, f, g, my_act = actions[i][j]
            if act != "replicate": 
                continue
            ph = parse_genome(g)
            empties, allies, others = neighbor_slots(i, j)
            choices = []
            if ph.repl_bias == 0:
                choices = empties + allies + others
            elif ph.repl_bias == 1:
                # prefer fittest neighbor site: approximate by fill empties near high neighbor avg fitness
                choices = sorted(empties, key=lambda xy: -fitness_multi(g, xy[0], xy[1], t, ph)[0]) + allies + others
            else:
                # help ally first: try to overwrite an empty near an ally; if none, random empty
                choices = allies + empties + others

            for ii,jj in choices:
                if (ii,jj) not in claimed and not next_grid[ii][jj].alive:
                    # create child
                    child_id, parent = next_id(next_grid[i][j].id)
                    child_genome = mutate(g, MUT_LOW * ph.mut_scale)
                    next_grid[ii][jj] = Cell(True, child_genome, child_id, next_grid[i][j].id, pd_actions[i][j])
                    lineage_edges.append((next_grid[i][j].id, child_id, t))
                    claimed.add((ii,jj))
                    break

    avg_fit = total_fit/total_alive if total_alive>0 else 0.0
    frac_alive = total_alive/(GRID_H*GRID_W)
    avg_coop = total_coop/total_alive if total_alive>0 else 0.0
    frac_crit = near_crit_count/(GRID_H*GRID_W)
    return next_grid, avg_fit, frac_alive, avg_coop, frac_crit

# ------------------------
# Run simulation
# ------------------------
grid = init_grid()
avg_fits = []
alive_fracs = []
avg_coops = []
crit_fracs = []

for t in range(STEPS):
    grid, a, z, c, q = step(grid, t)
    avg_fits.append(a)
    alive_fracs.append(z)
    avg_coops.append(c)
    crit_fracs.append(q)

# ------------------------
# Final maps
# ------------------------
final_fit = np.zeros((GRID_H, GRID_W))
final_coop = np.zeros((GRID_H, GRID_W))
for i in range(GRID_H):
    for j in range(GRID_W):
        if grid[i][j].alive:
            ph = parse_genome(grid[i][j].genome)
            f, *_ = fitness_multi(grid[i][j].genome, i, j, STEPS, ph)
            final_fit[i, j] = f
            final_coop[i, j] = ph.coop
        else:
            final_fit[i, j] = 0.0
            final_coop[i, j] = 0.0

# ------------------------
# Save logs
# ------------------------
crit_path = "/mnt/data/critical_events.csv"
with open(crit_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["t","i","j","Lambda","fitness","nei","gscore"])
    w.writeheader()
    for row in critical_log:
        w.writerow(row)

lineage_path = "/mnt/data/lineage_edges.csv"
with open(lineage_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["parent_id","child_id","t"])
    for p,c,t in lineage_edges:
        w.writerow([p,c,t])

# Also store a simple JSON with last grid metadata (id, parent) for tree reconstruction offline
meta_path = "/mnt/data/final_grid_meta.json"
meta = [[{"id": int(grid[i][j].id), "parent": int(grid[i][j].parent), "alive": bool(grid[i][j].alive)}
         for j in range(GRID_W)] for i in range(GRID_H)]
with open(meta_path, "w") as f:
    json.dump(meta, f)

# ------------------------
# Plots (each on separate figure)
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
plt.plot(avg_coops)
plt.title("Average Cooperation vs. Time")
plt.xlabel("Step")
plt.ylabel("Average Cooperation")
plt.show()

plt.figure()
plt.plot(crit_fracs)
plt.title("Near-Critical Fraction (|Λ-1|<ε) vs. Time")
plt.xlabel("Step")
plt.ylabel("Near-Critical Fraction")
plt.show()

plt.figure()
plt.imshow(final_fit, interpolation='nearest')
plt.title("Final Grid Fitness (heatmap)")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(final_coop, interpolation='nearest')
plt.title("Final Grid Cooperation (heatmap)")
plt.colorbar()
plt.show()

crit_path, lineage_path, meta_path
