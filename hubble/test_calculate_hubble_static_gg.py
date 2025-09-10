import math

def test_xi_void_plateau():
    # Local import from the sibling script
    from hubble.calculate_hubble_static_gg import xi_rar_plateau, A0, D_MAX
    # Extremely small g -> plateau
    g_small = 1e-40
    assert xi_rar_plateau(g_small, a0=A0, D_max=D_MAX) == D_MAX


def test_k_backsolve_identity():
    from hubble.calculate_hubble_static_gg import C_KM_S, D_MAX
    # Using Planck_2018 H0 and xi_void = D_MAX, verify the identity
    H0_planck = 67.4  # km/s/Mpc
    xi_void = D_MAX
    k_planck = H0_planck / (C_KM_S * (xi_void - 1.0))
    H0_pred = C_KM_S * k_planck * (xi_void - 1.0)
    assert abs(H0_pred - H0_planck) < 1e-9

