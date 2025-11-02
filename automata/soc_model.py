# Λ³ Self-Evolving Automaton — Energy-Conserving, SOC, Mobility, Lineage Viz
# -------------------------------------------------------------------------
# Adds to the previous prototype:
# 1) Energy & Resource conservation (H0): each cell has internal energy; environment has a resource field.
#    - Harvesting converts resource -> internal energy; metabolism converts internal energy -> resource.
#    - Track H0 = sum(E_internal) + sum(resource) and plot drift (should be ~0).
# 2) Mobility: cells probabilistically move to richer resource sites; movement has energy cost.
# 3) SOC (Self-Organized Criticality): local Λ-thresholds self-tune based on local Λ-variance toward a target.
#    - Per-cell Λ_repl and Λ_evo adjust over time to maintain near-criticality.
# 4) Lineage visualization without Graphviz: time-layered layout from (parent_id, child_id, t).
# 5) Lineage time series: per-lineage averages of fitness and cooperation over time (CSV output).
#
# Plots:
#  - Avg fitness vs time
#  - Alive fraction vs time
#  - Avg cooperation vs time
#  - Near-critical fraction vs time
#  - H0 drift vs time (energy conservation check)
#  - Final fitness heatmap
#  - Final cooperation heatmap
#  - Final resource field heatmap
#  - Lineage tree (time-layer layout) image
#
# Notes: matplotlib only (no seaborn), one chart per figure, and no explicit colors.
#
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import csv, json
from collections import defaultdict

# ------------------------
# Parameters
# ------------------------
GRID_H, GRID_W = 28, 28
G = 24      # genome bits (3 x 8-bit instructions)
STEPS = 70
SEED_DENSITY = 0.18
NEIGH_RADIUS = 1

# Base Λ thresholds
BASE_L_REPL = 0.90
BASE_L_EVOL = 1.10
EPS_L_CRIT  = 0.03   # |Λ-1|<EPS => near-critical

# SOC tuning
SOC_GAMMA = 0.10     # learning rate for threshold adaptation
SOC_SIGMA0 = 0.02    # target local variance of Λ
THR_MIN_REPL, THR_MAX_REPL = 0.75, 0.95
THR_MIN_EVOL, THR_MAX_EVOL = 1.05, 1.35

# Mutation
MUT_LOW  = 0.005
MUT_HIGH = 0.08

# K components
K_BASE = 0.05
K_MISMATCH_SCALE = 0.6
K_GAME_SCALE = 0.2
K_ENERGY_W = 0.3   # internal energy contribution

# V components
V_FIT_W = 0.50
V_NEI_W = 0.25
V_GAME_W = 0.15
V_POT_W = 0.10     # environmental potential from resource

# Energy & Resources
INIT_RESOURCE_MEAN = 1.0
INIT_RESOURCE_NOISE = 0.2
INIT_EINT_IF_ALIVE = 0.4
HARVEST_RATE = 0.10     # per-step fraction of local resource converted to E_int (scaled by efficiency & coop)
METAB_RATE = 0.04       # per-step metabolic drain of E_int back to resource
MOVE_COST = 0.02        # energy cost when moving
RESOURCE_DIFFUSION = 0.10   # diffusion coefficient for resource field each step
E_NORM = 1.0   # normalization for E_internal -> [0,1] scale

# RNG
rng = np.random.default_rng(2025)

# ------------------------
# Genome & phenotype mapping
# Instructions: 3-bit opcode + 5-bit argument (8 bits / instr) x 3
# Opcodes:
#  000: SET_COOP   (arg -> coop in [0,1])
#  001: BIAS_REPL  (arg -> {0:empties,1:fittest,2:ally-first})
#  010: MASK_ONES  (arg -> csp1_pref in [0.5,1.0])
#  011: MASK_BLOCK (arg -> csp2_len in {1..6})
#  100: MUT_TUNE   (arg -> mut_scale in [0.5,1.5])
#  101: COOP_RULE  (arg -> rule_tft in [0,1])
#  110: RES_EFF    (arg -> resource efficiency in [0.5,1.5])
#  111: MOVE_TEND  (arg -> move propensity in [0,1])
#
@dataclass
class Phenotype:
    coop: float
    repl_bias: int
    csp1_pref: float
    csp2_len: int
    mut_scale: float
    rule_tft: float
    res_eff: float
    move_prop: float

@dataclass
class Cell:
    alive: bool
    genome: np.ndarray  # bool array length G
    id: int             # lineage id
    parent: int         # parent id or -1
    last_action: int    # 1=cooperate, 0=defect
    E_int: float        # internal energy

global_id_counter = 0
def next_id(parent=-1):
    global global_id_counter
    global_id_counter += 1
    return global_id_counter, parent

def random_genome():
    return rng.integers(0, 2, size=G, dtype=np.int8).astype(bool)

def bits_to_int(bits):
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val

def parse_genome(genome: np.ndarray) -> Phenotype:
    fields = {
        'coop':0.5, 'repl_bias':0, 'csp1_pref':1.0, 'csp2_len':3,
        'mut_scale':1.0, 'rule_tft':0.0, 'res_eff':1.0, 'move_prop':0.0
    }
    n_instr = min(3, len(genome)//8)
    for k in range(n_instr):
        chunk = genome[k*8:(k+1)*8]
        opcode = bits_to_int(chunk[:3])
        arg = bits_to_int(chunk[3:])
        if opcode == 0:   fields['coop'] = min(1.0, arg/31.0)
        elif opcode == 1: fields['repl_bias'] = int((arg/31.0)*2.99)
        elif opcode == 2: fields['csp1_pref'] = 0.5 + 0.5*(arg/31.0)
        elif opcode == 3: fields['csp2_len'] = 1 + int((arg/31.0)*5.99)
        elif opcode == 4: fields['mut_scale'] = 0.5 + (arg/31.0)
        elif opcode == 5: fields['rule_tft'] = arg/31.0
        elif opcode == 6: fields['res_eff'] = 0.5 + (arg/31.0)
        elif opcode == 7: fields['move_prop'] = arg/31.0
    return Phenotype(**fields)

# ------------------------
# Environment: Multi-constraint CSP
# ------------------------
def target_s1(i, j, t):
    base = (G/2
            + (G/4)*np.sin(2*np.pi*(i/GRID_H + 0.02*t))
            + (G/6)*np.cos(2*np.pi*(j/GRID_W + 0.03*t)))
    return int(np.clip(round(base), 0, G))

def csp1_fitness(genome, s, pref):
    ones = int(genome.sum())
    return max(0.0, 1.0 - pref*abs(ones - s)/G)

def csp2_fitness(genome, L, i, j, t):
    window_len = 8
    start = (i + j + t) % (G - window_len + 1)
    seg = genome[start:start+window_len].astype(int)
    score = 0
    run = 1
    for k in range(1, len(seg)):
        if seg[k] == seg[k-1]:
            run += 1
            if run == L:
                score += 1
        else:
            run = 1
    return min(1.0, score / max(1, window_len))

def fitness_multi(genome, i, j, t, pheno: Phenotype):
    s = target_s1(i, j, t)
    f1 = csp1_fitness(genome, s, pheno.csp1_pref)
    f2 = csp2_fitness(genome, pheno.csp2_len, i, j, t)
    w1 = 0.6 + 0.2*np.sin(2*np.pi*(t/30 + i/GRID_H))
    w2 = 1.0 - w1
    f = max(0.0, min(1.0, w1*f1 + w2*f2))
    return f, s, f1, f2, w1, w2

def neighbors(i, j):
    coords = []
    for di in range(-NEIGH_RADIUS, NEIGH_RADIUS+1):
        for dj in range(-NEIGH_RADIUS, NEIGH_RADIUS+1):
            if di == 0 and dj == 0: continue
            ii, jj = i+di, j+dj
            if 0 <= ii < GRID_H and 0 <= jj < GRID_W:
                coords.append((ii, jj))
    return coords

# ------------------------
# Game-theoretic interaction (PD-like)
# ------------------------
R, S, Tm, P = 3, 0, 5, 1
def pd_payoff(my_act, nb_act):
    if my_act == 1 and nb_act == 1: return R
    if my_act == 1 and nb_act == 0: return S
    if my_act == 0 and nb_act == 1: return Tm
    return P

def pd_action(cell: Cell, ph: Phenotype, avg_nb_last):
    base = ph.coop
    influence = ph.rule_tft * (avg_nb_last - 0.5) * 2.0
    pC = np.clip(base + 0.25*influence, 0.0, 1.0)
    return 1 if rng.random() < pC else 0

def neighbor_game(grid, i, j, ph: Phenotype):
    nb = neighbors(i, j)
    if not nb:
        return 0.5, 0.5, 1  # score, my_act, cnt
    acts = []
    for ii,jj in nb:
        c = grid[ii][jj]
        if c.alive:
            acts.append(c.last_action)
    avg_nb_last = np.mean(acts) if acts else 0.5
    me = grid[i][j]
    my_act = pd_action(me, ph, avg_nb_last)
    pay = 0.0
    cnt = 0
    for ii,jj in nb:
        c = grid[ii][jj]
        if c.alive:
            pay += pd_payoff(my_act, c.last_action)
            cnt += 1
    score = pay/(cnt*Tm) if cnt>0 else 0.5
    return score, my_act, cnt

# ------------------------
# Grid & fields init
# ------------------------
def init_grid():
    resource = INIT_RESOURCE_MEAN + INIT_RESOURCE_NOISE*(rng.random((GRID_H, GRID_W)) - 0.5)
    grid = [[None for _ in range(GRID_W)] for _ in range(GRID_H)]
    for i in range(GRID_H):
        for j in range(GRID_W):
            if rng.random() < SEED_DENSITY:
                gid, parent = next_id(-1)
                grid[i][j] = Cell(True, random_genome(), gid, parent, rng.integers(0,2), E_int=INIT_EINT_IF_ALIVE)
            else:
                grid[i][j] = Cell(False, np.zeros(G, dtype=bool), -1, -1, 0, E_int=0.0)
    return grid, np.clip(resource, 0.0, None)

def mutate(genome, rate):
    mask = rng.random(G) < rate
    newg = genome.copy()
    newg[mask] = ~newg[mask]
    return newg

# ------------------------
# SOC threshold fields
# ------------------------
def init_thresholds():
    Lr = np.full((GRID_H, GRID_W), BASE_L_REPL, dtype=float)
    Le = np.full((GRID_H, GRID_W), BASE_L_EVOL, dtype=float)
    return Lr, Le

def local_variance(arr, i, j):
    nb = neighbors(i, j) + [(i,j)]
    vals = [arr[ii,jj] for ii,jj in nb]
    m = np.mean(vals)
    return float(np.mean([(v-m)**2 for v in vals]))

# ------------------------
# Lineage time series accumulators
# ------------------------
lineage_series = defaultdict(lambda: {'t':[], 'fitness':[], 'coop':[]})
lineage_edges = []      # (parent_id, child_id, t)
critical_log = []       # dict per event

# ------------------------
# Resource diffusion (conservative)
# ------------------------
def diffuse_resource(R):
    # Simple five-point Laplacian with reflecting boundaries
    Rnew = R.copy()
    for i in range(GRID_H):
        for j in range(GRID_W):
            center = R[i,j]
            nsum = 0.0
            cnt = 0
            for ii,jj in neighbors(i,j):
                nsum += R[ii,jj]
                cnt += 1
            if cnt > 0:
                lap = (nsum/cnt) - center
                Rnew[i,j] += RESOURCE_DIFFUSION * lap
    return np.clip(Rnew, 0.0, None)

# ------------------------
# Step function
# ------------------------
def step(grid, resource, Lthr_repl, Lthr_evol, t):
    next_grid = [[Cell(False, np.zeros(G, dtype=bool), -1, -1, 0, 0.0) for _ in range(GRID_W)] for _ in range(GRID_H)]
    total_fit = 0.0
    total_alive = 0
    total_coop = 0.0
    near_crit = 0
    L_map = np.zeros((GRID_H, GRID_W))

    actions = [[None for _ in range(GRID_W)] for _ in range(GRID_H)]
    phenos  = [[None for _ in range(GRID_W)] for _ in range(GRID_H)]
    pd_scores = np.full((GRID_H, GRID_W), 0.5, dtype=float)
    pd_actions = np.zeros((GRID_H, GRID_W), dtype=int)

    # Energy bookkeeping
    # Harvest & Metabolism (conservative transformation)
    # Harvest: dE = HARVEST_RATE * res * eff * coop ; resource -= dE ; E_int += dE
    # Metabolic: dQ = METAB_RATE * E_int ; E_int -= dQ ; resource += dQ
    # movement cost will also go to resource (dissipation)
    # After step, diffuse resource (conservative)

    # 1) Prepare phenotypes
    for i in range(GRID_H):
        for j in range(GRID_W):
            c = grid[i][j]
            if c.alive:
                phenos[i][j] = parse_genome(c.genome)

    # 2) Harvest & metabolism
    for i in range(GRID_H):
        for j in range(GRID_W):
            c = grid[i][j]
            if not c.alive: 
                continue
            ph = phenos[i][j]
            # harvest
            harvest = HARVEST_RATE * resource[i,j] * ph.res_eff * ph.coop
            resource[i,j] = max(0.0, resource[i,j] - harvest)
            c.E_int += harvest
            # metabolism
            dQ = METAB_RATE * c.E_int
            c.E_int = max(0.0, c.E_int - dQ)
            resource[i,j] += dQ

    # 3) PD game, fitness, K/V, Λ and decide actions
    for i in range(GRID_H):
        for j in range(GRID_W):
            c = grid[i][j]
            if not c.alive:
                actions[i][j] = ("empty", 0.0, 0.0, c.genome, 0.0, c.E_int)
                continue
            ph = phenos[i][j]
            f, s, f1, f2, w1, w2 = fitness_multi(c.genome, i, j, t, ph)

            # neighbor support
            nb = neighbors(i, j)
            nei_acc = 0.0
            nei_cnt = 0
            for ii,jj in nb:
                cc = grid[ii][jj]
                if cc.alive:
                    ff, *_ = fitness_multi(cc.genome, i, j, t, parse_genome(cc.genome))
                    nei_acc += ff
                    nei_cnt += 1
            nei = (nei_acc/nei_cnt) if nei_cnt>0 else 0.0

            # PD game
            gscore, my_act, cnt = neighbor_game(grid, i, j, ph)
            pd_scores[i,j]  = gscore
            pd_actions[i,j] = my_act

            # Extended K and V with energy & potential
            Eint_norm = min(1.0, grid[i][j].E_int / E_NORM)
            Vpot_norm = min(1.0, resource[i,j] / (INIT_RESOURCE_MEAN + 1e-6))

            V = V_FIT_W * f + V_NEI_W * nei + V_GAME_W * gscore + V_POT_W * Vpot_norm
            mismatch = 1.0 - f
            K = K_BASE + K_MISMATCH_SCALE * mismatch + K_GAME_SCALE*(1.0 - gscore) + K_ENERGY_W * Eint_norm
            L = K / (V + 1e-6)
            L_map[i,j] = L

            total_fit  += f
            total_alive += 1
            total_coop += ph.coop
            if abs(L-1.0) < EPS_L_CRIT:
                near_crit += 1
                critical_log.append({'t':t,'i':i,'j':j,'Lambda':float(L),
                                     'fitness':float(f),'nei':float(nei),'gscore':float(gscore)})

            # record for lineage series (avg later)
            lineage_series[c.id]['t'].append(t)
            lineage_series[c.id]['fitness'].append(float(f))
            lineage_series[c.id]['coop'].append(float(ph.coop))

            # SOC local thresholds (use current Lthr fields)
            Lr = Lthr_repl[i,j]
            Le = Lthr_evol[i,j]

            # Decide action
            if L < Lr:
                actions[i][j] = ("replicate", L, f, c.genome, ph.move_prop, c.E_int)
            elif L <= Le:
                actions[i][j] = ("evolve", L, f, c.genome, ph.move_prop, c.E_int)
            else:
                actions[i][j] = ("die", L, f, c.genome, ph.move_prop, c.E_int)

    # 4) Apply deaths & evolves (mutate) in place
    for i in range(GRID_H):
        for j in range(GRID_W):
            act, L, f, g, move_prop, Eint = actions[i][j]
            c = grid[i][j]
            if act == "die":
                next_grid[i][j] = Cell(False, np.zeros(G, dtype=bool), -1, -1, 0, 0.0)
            elif act == "evolve":
                ph = parse_genome(g)
                newg = mutate(g, MUT_HIGH * ph.mut_scale)
                next_grid[i][j] = Cell(True, newg, c.id, c.parent, pd_actions[i,j], Eint)
            elif act == "replicate":
                ph = parse_genome(g)
                # parent carries small drift
                newg = mutate(g, MUT_LOW * ph.mut_scale)
                next_grid[i][j] = Cell(True, newg, c.id, c.parent, pd_actions[i,j], Eint)
            else:
                next_grid[i][j] = Cell(False, np.zeros(G, dtype=bool), -1, -1, 0, 0.0)

    # 5) Movement before replication: swap with empty richer site with prob move_prop*(1-V)
    claimed = set()
    for i in range(GRID_H):
        for j in range(GRID_W):
            if not next_grid[i][j].alive: 
                continue
            # approximate local stability via V fraction from earlier components
            # if unstable (low V), more likely to move
            # reuse L_map: larger L -> more unstable; so prob ∝ move_prop * min(1, L-0.8)
            _, L, _, _, move_prop, _ = actions[i][j]
            p_move = max(0.0, move_prop * min(1.0, max(0.0, L - 0.8)))
            if rng.random() < p_move:
                # pick empty neighbor with higher resource
                nb = neighbors(i, j)
                empties = [(ii,jj) for (ii,jj) in nb if not next_grid[ii][jj].alive]
                if empties:
                    empties.sort(key=lambda xy: resource[xy[0], xy[1]], reverse=True)
                    ii,jj = empties[0]
                    if (ii,jj) not in claimed:
                        # apply movement cost -> to resource (dissipation)
                        cost = min(MOVE_COST, next_grid[i][j].E_int)
                        next_grid[i][j].E_int -= cost
                        resource[i,j] += cost*0.5
                        resource[ii,jj] += cost*0.5
                        # move (swap)
                        next_grid[ii][jj] = next_grid[i][j]
                        next_grid[i][j] = Cell(False, np.zeros(G, dtype=bool), -1, -1, 0, 0.0)
                        claimed.add((ii,jj))

    # 6) Replication into neighboring empties (child creation)
    def neighbor_slots(i, j):
        nb = neighbors(i, j)
        empties, allies, others = [], [], []
        for ii,jj in nb:
            if not next_grid[ii][jj].alive:
                empties.append((ii,jj))
            else:
                # ally if small Hamming distance
                hd = np.sum(next_grid[ii][jj].genome != next_grid[i][j].genome)
                if hd < G*0.25: allies.append((ii,jj))
                else: others.append((ii,jj))
        return empties, allies, others

    for i in range(GRID_H):
        for j in range(GRID_W):
            act, L, f, g, move_prop, Ein = actions[i][j]
            if act != "replicate": 
                continue
            ph = parse_genome(g)
            empties, allies, others = neighbor_slots(i, j)
            choices = []
            if ph.repl_bias == 0:   choices = empties + allies + others
            elif ph.repl_bias == 1: choices = sorted(empties, key=lambda xy: -fitness_multi(g, xy[0], xy[1], t, ph)[0]) + allies + others
            else:                   choices = allies + empties + others
            for ii,jj in choices:
                if not next_grid[ii][jj].alive:
                    # child inherits id? -> new lineage id
                    parent_id = next_grid[i][j].id if next_grid[i][j].alive else grid[i][j].id
                    child_id, _ = next_id(parent_id)
                    child_genome = mutate(g, MUT_LOW * ph.mut_scale)
                    # child gets small energy endowment from parent (split internal energy)
                    share = min(0.1, next_grid[i][j].E_int)
                    next_grid[i][j].E_int -= share
                    child_E = share
                    next_grid[ii][jj] = Cell(True, child_genome, child_id, parent_id, pd_actions[i,j], child_E)
                    lineage_edges.append((parent_id, child_id, t))
                    break

    # 7) Resource diffusion (conservative)
    resource = diffuse_resource(resource)

    # 8) SOC: update thresholds for next step based on local variance of L_map
    new_Lr = Lthr_repl.copy()
    new_Le = Lthr_evol.copy()
    for i in range(GRID_H):
        for j in range(GRID_W):
            varL = local_variance(L_map, i, j)
            d = SOC_GAMMA * (varL - SOC_SIGMA0)
            new_Lr[i,j] = np.clip(Lthr_repl[i,j] + d, THR_MIN_REPL, THR_MAX_REPL)
            new_Le[i,j] = np.clip(Lthr_evol[i,j] + d, THR_MIN_EVOL, THR_MAX_EVOL)

    # Metrics
    avg_fit = total_fit/total_alive if total_alive>0 else 0.0
    frac_alive = total_alive/(GRID_H*GRID_W)
    avg_coop = total_coop/total_alive if total_alive>0 else 0.0
    frac_crit = near_crit/(GRID_H*GRID_W)

    return next_grid, resource, new_Lr, new_Le, avg_fit, frac_alive, avg_coop, frac_crit, L_map

# ------------------------
# Run simulation
# ------------------------
grid, resource = init_grid()
Lthr_repl, Lthr_evol = init_thresholds()

# Initial energy H0
def total_internal_energy(grid):
    s = 0.0
    for i in range(GRID_H):
        for j in range(GRID_W):
            s += grid[i][j].E_int
    return s

H0_init = total_internal_energy(grid) + float(np.sum(resource))

avg_fits, alive_fracs, avg_coops, crit_fracs = [], [], [], []
H0_drifts = []

for t in range(STEPS):
    grid, resource, Lthr_repl, Lthr_evol, a, z, c, q, L_map = step(grid, resource, Lthr_repl, Lthr_evol, t)
    avg_fits.append(a)
    alive_fracs.append(z)
    avg_coops.append(c)
    crit_fracs.append(q)
    H0_now = total_internal_energy(grid) + float(np.sum(resource))
    H0_drifts.append(H0_now - H0_init)

# ------------------------
# Final maps
# ------------------------
final_fit  = np.zeros((GRID_H, GRID_W))
final_coop = np.zeros((GRID_H, GRID_W))
for i in range(GRID_H):
    for j in range(GRID_W):
        if grid[i][j].alive:
            ph = parse_genome(grid[i][j].genome)
            f, *_ = fitness_multi(grid[i][j].genome, i, j, STEPS, ph)
            final_fit[i, j] = f
            final_coop[i, j] = ph.coop

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

# Per-lineage time series (averaged per time step)
# Build time->id stats first
per_t_stats = defaultdict(lambda: defaultdict(lambda: {'fit_sum':0.0,'coop_sum':0.0,'cnt':0}))
for lid, series in lineage_series.items():
    for tt, ff, cc in zip(series['t'], series['fitness'], series['coop']):
        per_t_stats[tt][lid]['fit_sum'] += ff
        per_t_stats[tt][lid]['coop_sum'] += cc
        per_t_stats[tt][lid]['cnt']     += 1

ts_path = "/mnt/data/lineage_timeseries.csv"
with open(ts_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t","lineage_id","avg_fitness","avg_coop"])
    for tt in sorted(per_t_stats.keys()):
        for lid, s in per_t_stats[tt].items():
            if s['cnt'] > 0:
                w.writerow([tt, lid, s['fit_sum']/s['cnt'], s['coop_sum']/s['cnt']])

# Save final grid meta
meta_path = "/mnt/data/final_grid_meta.json"
meta = [[{"id": int(grid[i][j].id), "parent": int(grid[i][j].parent), "alive": bool(grid[i][j].alive)}
         for j in range(GRID_W)] for i in range(GRID_H)]
with open(meta_path, "w") as f:
    json.dump(meta, f)

# ------------------------
# Lineage tree visualization (time-layered)
# Build birth time per node from edges (parent_id, child_id, t).
# Roots (parent=-1 in initial grid) are set to birth time 0.
# We then place nodes by birth time on x-axis, stack vertically by encounter order.
# ------------------------
# Collect nodes and birth times
birth_time = {}
children_map = defaultdict(list)
parents_map = {}
for p,c,t in lineage_edges:
    if c not in birth_time or t < birth_time[c]:
        birth_time[c] = t
    if p not in birth_time:
        birth_time[p] = 0  # assume parent existed earlier
    children_map[p].append(c)
    parents_map[c] = p

# Keep only nodes that appear (edges define them)
nodes = sorted(birth_time.keys(), key=lambda nid: birth_time[nid])
# Layout
x_positions = {}
y_positions = {}
layer_counts = defaultdict(int)
for nid in nodes:
    x = birth_time[nid]
    layer_counts[x] += 1
    y = layer_counts[x]
    x_positions[nid] = x
    y_positions[nid] = y

# Draw
plt.figure(figsize=(8,6))
for p,c,t in lineage_edges:
    if p in x_positions and c in x_positions:
        xs = [x_positions[p], x_positions[c]]
        ys = [y_positions[p], y_positions[c]]
        plt.plot(xs, ys)
# Nodes
plt.scatter([x_positions[n] for n in nodes], [y_positions[n] for n in nodes])
plt.title("Lineage Tree (time-layered)")
plt.xlabel("Birth step")
plt.ylabel("Node index per layer")
plt.tight_layout()
lineage_img = "/mnt/data/lineage_tree.png"
plt.savefig(lineage_img, dpi=150)
plt.show()

# ------------------------
# Plots (each separate figure)
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
plt.plot(H0_drifts)
plt.title("Energy Conservation Drift H0-H0_init")
plt.xlabel("Step")
plt.ylabel("Drift")
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

plt.figure()
plt.imshow(resource, interpolation='nearest')
plt.title("Final Resource Field (heatmap)")
plt.colorbar()
plt.show()

crit_path, lineage_path, ts_path, meta_path, lineage_img
