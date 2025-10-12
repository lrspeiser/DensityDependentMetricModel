import numpy as np
from scripts.next_steps_from_run import xi_rar_plateau_numpy, solar_system_table

def test_xi_limits():
    # High-g: larger V / smaller R -> xi close to 1
    Vbar = np.array([350.0])
    R = np.array([2.0])
    xi, _ = xi_rar_plateau_numpy(Vbar, R, a0_m_s2=1.2e-10)
    assert xi[0] >= 1.0 and xi[0] < 1.1

    # Low-g with plateau cap
    xi2, _ = xi_rar_plateau_numpy(Vbar*0.01, R*100.0, a0_m_s2=1.2e-10, D_max=50.0)
    assert xi2[0] <= 50.0 + 1e-8

def test_solar_system_cassini_gated():
    rows = solar_system_table({"a0_m_s2": 1.2e-10, "zeta_env": 0.0})
    at_10 = [r for r in rows if abs(r["AU"]-10.0)<1e-6][0]
    assert abs(at_10["gamma_minus_1"]) == 0.0
    # Gated deviation should be below Cassini bound near 10 AU for zeta_env=0
    assert at_10["dGoverG_gated"] < 2.3e-5

