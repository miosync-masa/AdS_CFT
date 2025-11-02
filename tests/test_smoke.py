from lambda3_holo.automaton import Automaton

def test_smoke_small_run():
    ua = Automaton(H=16, W=16, Z=8, gate_delay=1, gate_strength=0.1, c_eff_max=0.18, seed=123)
    ua.boundary = ua.coop_field()
    rec = ua.step_once(1)
    assert "entropy_RT_mo" in rec and "lambda_p99_A_out_pre" in rec
