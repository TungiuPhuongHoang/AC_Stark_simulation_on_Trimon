
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, qeye, destroy, tensor, mesolve, Qobj

# ---------- device parameters  ----------
twopi = 2 * np.pi

omega_u_A = twopi * 5.5585e9
omega_u_B = twopi * 6.1470e9
omega_u_C = twopi * 7.0180e9

alpha_A = twopi * 111e6
alpha_B = twopi * 116e6
alpha_C = twopi * 138.6e6

Jzz_AB = twopi * 201.2e6  # A<->B band split
Jzz_BC = twopi * 253.0e6  # B<->C band split
Jzz_CA = twopi * 232.0e6  # C<->A band split

# ---------- decoherence ----------
T1_A = 20.6e-6
T1_B = 51.4e-6
T1_C = 26.2e-6

T2_A = 39.7e-6
T2_B = 64.8e-6
T2_C = 32.3e-6

USE_NOISE = False

RABI_CAP = twopi * 5e6
USE_DRAG = False
BETA_DRAG = 1.0

# Self-Stark (first-order transmon expression)
USE_STARK = True
STARK_COMPENSATE = False
K_STARK = 1

# Cross-Stark (crosstalk): drive on A shifts B (and C), and vice versa
USE_CROSS_STARK = True
K_XST_B_from_A = 0.5  # when driving A, shift B
K_XST_A_from_B = 0.5  # when driving B, shift A
K_XST_C_from_A = 0.0
K_XST_C_from_B = 0.0

IDLE_NS = 0

# Extra detuning on lower-band π pulses only (turnable) [rad/s]
EXTRA_LOWER_DETUNE_A = 0 * np.pi * 0.8e6  # affects A lower π pulse
EXTRA_LOWER_DETUNE_B = 0 * np.pi * 0.8e6  # affects B lower π pulse

T_pi2 = max(2.0 * (np.pi / 2) / RABI_CAP, 100e-9)
T_pi = max(2.0 * (np.pi) / RABI_CAP, 200e-9)

# ---------- Hilbert space (3x3x3) ----------
d = 3
I3 = qeye(d)
a = destroy(d);
ad = a.dag();
n = ad * a

# Tensor operators A,B,C (order = A ⊗ B ⊗ C)
aA, nA = tensor(a, I3, I3), tensor(n, I3, I3)
aB, nB = tensor(I3, a, I3), tensor(I3, n, I3)
aC, nC = tensor(I3, I3, a), tensor(I3, I3, n)
I27 = tensor(I3, I3, I3)

# Quadratures for each mode
X_A, Y_A = (aA + aA.dag()), 1j * (aA.dag() - aA)
X_B, Y_B = (aB + aB.dag()), 1j * (aB.dag() - aB)
X_C, Y_C = (aC + aC.dag()), 1j * (aC.dag() - aC)

# Z_01 projectors per mode (to implement Stark as local Z on 0-1 manifold)
P0 = basis(3, 0) * basis(3, 0).dag()
P1 = basis(3, 1) * basis(3, 1).dag()
Z01_single = P1 - P0
Z01_A = tensor(Z01_single, I3, I3)
Z01_B = tensor(I3, Z01_single, I3)
Z01_C = tensor(I3, I3, Z01_single)

# ---------- Drift in the rotating frame of upper bands ----------
# We remove omega_u_A nA + omega_u_B nB + omega_u_C nC
H0 = (-0.5 * alpha_A) * (nA * (nA - I27)) \
     + (-0.5 * alpha_B) * (nB * (nB - I27)) \
     + (-0.5 * alpha_C) * (nC * (nC - I27)) \
     + (Jzz_AB * nA * nB) + (Jzz_BC * nB * nC) + (Jzz_CA * nC * nA)


# ---------- helpers ----------
def wrap_pi(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi


def amp(psi, a, b, c):
    v = np.array(psi.full()).ravel()
    idx = a * 9 + b * 3 + c  # (A,B,C)
    return v[idx]


def pops_comp_subspace(psi):
    v = np.array(psi.full()).ravel()
    P00 = abs(v[0 * 9 + 0 * 3 + 0]) ** 2  # |0,0,0>
    P01 = abs(v[0 * 9 + 1 * 3 + 0]) ** 2  # |0,1,0>
    P10 = abs(v[1 * 9 + 0 * 3 + 0]) ** 2  # |1,0,0>
    P11 = abs(v[1 * 9 + 1 * 3 + 0]) ** 2  # |1,1,0>
    Pleak = 1.0 - (P00 + P01 + P10 + P11)
    return P00, P01, P10, P11, Pleak


def rel_phase(psi, bra_tuple_i, bra_tuple_j):
    ai = amp(psi, *bra_tuple_i)
    aj = amp(psi, *bra_tuple_j)
    return wrap_pi(float(np.angle(ai) - np.angle(aj)))


def Tphi_from_T1_T2(T1, T2):
    val = 1.0 / T2 - 0.5 / T1
    if val <= 0:
        return 1e18  # effectively no pure dephasing if T2 >= 2*T1
    return 1.0 / val


def collapse_ops_T1_T2():
    if not USE_NOISE:
        return []
    c_ops = []
    if T1_A > 0: c_ops.append(np.sqrt(1.0 / T1_A) * aA)
    if T1_B > 0: c_ops.append(np.sqrt(1.0 / T1_B) * aB)
    if T1_C > 0: c_ops.append(np.sqrt(1.0 / T1_C) * aC)
    Tphi_A = Tphi_from_T1_T2(T1_A, T2_A)
    Tphi_B = Tphi_from_T1_T2(T1_B, T2_B)
    Tphi_C = Tphi_from_T1_T2(T1_C, T2_C)
    if np.isfinite(Tphi_A) and Tphi_A > 0: c_ops.append(np.sqrt(1.0 / Tphi_A) * nA)
    if np.isfinite(Tphi_B) and Tphi_B > 0: c_ops.append(np.sqrt(1.0 / Tphi_B) * nB)
    if np.isfinite(Tphi_C) and Tphi_C > 0: c_ops.append(np.sqrt(1.0 / Tphi_C) * nC)
    return c_ops


# ---------- envelopes ----------
def hann_on_interval(t, T):
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = (t >= 0.0) & (t < T)
    x = t[m] / T
    y[m] = 0.5 * (1 - np.cos(2 * np.pi * x))
    return y


def hann_env_area(tgrid, T, area, cap):
    s = hann_on_interval(tgrid, T)
    S = np.trapezoid(s, tgrid)
    amp = area / max(S, 1e-18)
    amp = min(amp, cap)
    return amp * s


def drag_from_x(tgrid, ox, beta, delta_anh):
    if (not USE_DRAG) or np.isclose(beta, 0.0):
        return np.zeros_like(ox)
    dt = tgrid[1] - tgrid[0]
    dox = np.gradient(ox, dt)
    return -beta * dox / delta_anh


# ---------- conditional detuning (band-select) ----------
def band_detuning_for(mode, condA, condB, condC):
    if mode == 'A':
        return -(condB * Jzz_AB + condC * Jzz_CA)
    if mode == 'B':
        return -(condA * Jzz_AB + condC * Jzz_BC)
    if mode == 'C':
        return -(condA * Jzz_CA + condB * Jzz_BC)
    raise ValueError("mode must be 'A','B','C'")


# ---------- FREE EVOLUTION (idle gap) ----------
def build_idle(T, dt=0.25e-9):
    npts = max(4, int(np.ceil(T / dt)))
    t = np.linspace(0.0, T, npts, endpoint=False)
    return t, [H0]


# ---------- one pulse segment ----------
def build_segment(mode, band, theta, T, dt=0.25e-9, phi=0.0,
                  condA=0, condB=0, condC=0,
                  extra_detuning=0.0):
    """
    mode: 'A','B','C'
    band: 'upper' or 'lower'
    condA/B/C: which partners are intended 'excited' (1) for selecting the lower band
    extra_detuning: add small offset (Stark tuning / calibration mismatch)
    """
    if mode == 'A':
        X, Y, Zop, alpha = X_A, Y_A, Z01_A, alpha_A
    elif mode == 'B':
        X, Y, Zop, alpha = X_B, Y_B, Z01_B, alpha_B
    else:
        X, Y, Zop, alpha = X_C, Y_C, Z01_C, alpha_C

    base = 0.0 if band == 'upper' else band_detuning_for(mode, condA, condB, condC)
    delta = float(base + extra_detuning)

    # time grid and envelopes
    npts = max(64, int(np.ceil(T / dt)))
    t = np.linspace(0.0, T, npts, endpoint=False)
    ox = hann_env_area(t, T, area=theta, cap=RABI_CAP)
    oy = drag_from_x(t, ox, beta=BETA_DRAG, delta_anh=-alpha)  # alpha>0 mag

    # scalar samplers for qutip
    def Ox(tau):
        return float(np.interp(tau, t, ox, left=0.0, right=0.0))

    def Oy(tau):
        return float(np.interp(tau, t, oy, left=0.0, right=0.0))

    # rotating-frame mixing (carrier delta)
    def fx(tau):
        return 0.5 * Ox(tau) * np.cos(delta * tau + phi)  # X

    def fy(tau):
        return 0.5 * Ox(tau) * np.sin(delta * tau + phi)  # Y

    def gx(tau):
        return 0.5 * Oy(tau) * (-np.sin(delta * tau + phi))  # X

    def gy(tau):
        return 0.5 * Oy(tau) * (np.cos(delta * tau + phi))  # Y

    Hlist = [H0, [X, fx], [Y, fy], [X, gx], [Y, gy]]


    # Self Stark (first-order transmon)
    if USE_STARK:
        Delta_eff = delta if abs(delta) > 1e3 else (2.0 * np.pi * 10e6)

        def d_omega_self(tau):
            Om = Ox(tau)
            # <-- add K_STARK here so a sweep actually changes the shift
            val = K_STARK * (Om * Om) * alpha / (2.0 * Delta_eff * (alpha - Delta_eff))
            return -val if STARK_COMPENSATE else val

        Hlist.append([0.5 * Zop, d_omega_self])

    # Cross-Stark (phenomenological)
    if USE_CROSS_STARK:
        # Use same Delta_eff scale for cross (tunable model)
        Delta_xst = abs(delta) if abs(delta) > 1e3 else (2.0 * np.pi * 10e6)

        if mode == 'A':
            if K_XST_B_from_A != 0.0:
                def d_omega_B_from_A(tau):
                    Om = Ox(tau)
                    return (K_XST_B_from_A * (Om * Om) / Delta_xst)

                Hlist.append([0.5 * Z01_B, d_omega_B_from_A])

            if K_XST_C_from_A != 0.0:
                def d_omega_C_from_A(tau):
                    Om = Ox(tau)
                    return (K_XST_C_from_A * (Om * Om) / Delta_xst)

                Hlist.append([0.5 * Z01_C, d_omega_C_from_A])

        elif mode == 'B':
            if K_XST_A_from_B != 0.0:
                def d_omega_A_from_B(tau):
                    Om = Ox(tau)
                    return (K_XST_A_from_B * (Om * Om) / Delta_xst)

                Hlist.append([0.5 * Z01_A, d_omega_A_from_B])

            if K_XST_C_from_B != 0.0:
                def d_omega_C_from_B(tau):
                    Om = Ox(tau)
                    return (K_XST_C_from_B * (Om * Om) / Delta_xst)

                Hlist.append([0.5 * Z01_C, d_omega_C_from_B])

        # (If you also drive C at some point, mirror the pattern for cross to A/B.)

    return t, Hlist


# ---------- run a chain of segments ----------
def run_chain(psi0, segments, title="sequence"):
    times_all, states_all = [], []
    t_offset = 0.0
    psi = psi0

    c_ops = collapse_ops_T1_T2()

    for tloc, Hlist in segments:
        out = mesolve(Hlist, psi, tloc, c_ops=c_ops, e_ops=[])
        psi = out.states[-1]
        tg = np.array(out.times) + t_offset
        times_all.extend(list(tg))
        states_all.extend(list(out.states))
        t_offset = tg[-1]

    # final report
    P00, P01, P10, P11, Pleak = pops_comp_subspace(psi)
    print(f"[final] P00={P00:.5f}  P01={P01:.5f}  P10={P10:.5f}  P11={P11:.5f}  P_leak={Pleak:.5f}")

    # stackplot including leakage
    t_ns = np.array(times_all) * 1e9
    P00_list, P01_list, P10_list, P11_list, Pleak_list = [], [], [], [], []
    for st in states_all:
        p00, p01, p10, p11, pl = pops_comp_subspace(st)
        P00_list.append(p00);
        P01_list.append(p01)
        P10_list.append(p10);
        P11_list.append(p11)
        Pleak_list.append(pl)

    # plt.figure(figsize=(7.8, 4.3))
    # plt.stackplot(t_ns,
    #               np.array(P00_list),
    #               np.array(P01_list),
    #               np.array(P10_list),
    #               np.array(P11_list),
    #               np.array(Pleak_list),
    #               labels=[r'$P_{00}$', r'$P_{01}$', r'$P_{10}$', r'$P_{11}$', r'$P_{\rm leak}$'],
    #               alpha=0.9)
    # plt.xlabel('time [ns]');
    # plt.ylabel('population')
    # plt.ylim(-0.02, 1.02);
    # plt.title(title)
    # plt.legend(loc='upper right');
    # plt.tight_layout();
    # plt.show()

    return np.array(times_all), states_all, psi


# ---------- 3-pulse loops (C frozen in |0>) with idle gaps + per-pulse extra detuning ----------
def build_loop(loop_id, dt=0.25e-9):
    """
    Loop-B:  B upper π/2  ->  (idle) ->  A lower π (condB=1)  ->  (idle) ->  B lower π (condA=1)
    Loop-A:  A upper π/2  ->  (idle) ->  B lower π (condA=1)  ->  (idle) ->  A lower π (condB=1)
    Lower-band π pulses get the extra_detuning knobs.
    """

    segs = []
    idle_T = max(0.0, float(IDLE_NS))  # seconds

    if loop_id.upper() == "B":
        # B upper π/2
        segs.append(build_segment('B', 'upper', theta=np.pi / 2, T=T_pi2, dt=dt, phi=0.0,
                                  condA=0, condB=0, condC=0))
        if idle_T > 0.0:
            segs.append(build_idle(idle_T, dt=dt))

        # A lower π (condB=1) with extra detune for A lower only
        segs.append(build_segment('A', 'lower', theta=np.pi, T=T_pi, dt=dt, phi=0.0,
                                  condA=0, condB=1, condC=0,
                                  extra_detuning=EXTRA_LOWER_DETUNE_A))
        if idle_T > 0.0:
            segs.append(build_idle(idle_T, dt=dt))

        # B lower π (condA=1) with extra detune for B lower only
        segs.append(build_segment('B', 'lower', theta=np.pi, T=T_pi, dt=dt, phi=0.0,
                                  condA=1, condB=0, condC=0,
                                  extra_detuning=EXTRA_LOWER_DETUNE_B))

    else:
        # A upper π/2
        segs.append(build_segment('A', 'upper', theta=np.pi / 2, T=T_pi2, dt=dt, phi=0.0,
                                  condA=0, condB=0, condC=0))
        if idle_T > 0.0:
            segs.append(build_idle(idle_T, dt=dt))

        # B lower π (condA=1) with extra detune for B lower only
        segs.append(build_segment('B', 'lower', theta=np.pi, T=T_pi, dt=dt, phi=0.0,
                                  condA=1, condB=0, condC=0,
                                  extra_detuning=EXTRA_LOWER_DETUNE_B))
        if idle_T > 0.0:
            segs.append(build_idle(idle_T, dt=dt))

        # A lower π (condB=1) with extra detune for A lower only
        segs.append(build_segment('A', 'lower', theta=np.pi, T=T_pi, dt=dt, phi=0.0,
                                  condA=0, condB=1, condC=0,
                                  extra_detuning=EXTRA_LOWER_DETUNE_A))
    return segs


# ---------- main: run both loops and report phases ----------
if __name__ == "__main__":
    psi000 = tensor(basis(3, 0), basis(3, 0), basis(3, 0))

    # Loop B
    print("\n=== LOOP-B: B upper π/2 -> (idle) -> A lower π -> (idle) -> B lower π (C=0) ===")
    segs_B = build_loop("B", dt=0.25e-9)
    tB, sB, psiB = run_chain(psi000, segs_B, title="Loop-B (3 pulses)")
    phi_B = rel_phase(psiB, (1, 0, 0), (0, 0, 0))  # |100> vs |000>
    print(f"[Loop B] relative phase (|100> vs |000>): {phi_B:+.6f} rad")

    # Loop A
    print("\n=== LOOP-A: A upper π/2 -> (idle) -> B lower π -> (idle) -> A lower π (C=0) ===")
    segs_A = build_loop("A", dt=0.25e-9)
    tA, sA, psiA = run_chain(psi000, segs_A, title="Loop-A (3 pulses)")
    phi_A = rel_phase(psiA, (0, 1, 0), (0, 0, 0))  # |010> vs |000>
    print(f"[Loop A] relative phase (|010> vs |000>): {phi_A:+.6f} rad")

    print(f"\n|phi_A - phi_B| = {abs(phi_A - phi_B):.6f} rad")