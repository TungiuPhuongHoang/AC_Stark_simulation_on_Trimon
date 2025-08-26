from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, qeye, destroy, tensor, mesolve

# ==================== constants & device ====================

twopi = 2 * np.pi

# TR225: X00 transition frequencies (MHz)
omega_u_A = twopi * (5181.0e6)
omega_u_B = twopi * (4479.0e6)
omega_u_C = twopi * (6001.0e6)

# TR225: |alpha| from J_ii (A:128, B:116, C:142) MHz
alpha_A = -twopi * 128.0e6
alpha_B = -twopi * 116.0e6
alpha_C = -twopi * 142.0e6

# TR225: reported cross‑Kerr strengths (MHz)
Jzz_AB = twopi * 78.0e6
Jzz_BC = twopi * 104.0e6
Jzz_CA = twopi * 126.0e6

# ==================== control params ====================
PLOT = True

# Envelope cap (rad/s). Use None during calibration sweeps to avoid clipping.
RABI_CAP = twopi * 80e6

# When a line is anchored on target {A, B, C}, the same LO/envelope hits
# all three modes with these relative couplings and extra phases.
ALPHA = 1.0  # ↦ A
BETA = 0.4  # ↦ B
GAMMA = 0.3  # ↦ C

K_LINE = {
    ('A', 'A'): ALPHA, ('A', 'B'): BETA, ('A', 'C'): GAMMA,
    ('B', 'A'): ALPHA, ('B', 'B'): BETA, ('B', 'C'): GAMMA,
    ('C', 'A'): ALPHA, ('C', 'B'): BETA, ('C', 'C'): GAMMA,
}

# ---- DRAG controls ----
# Add a derivative-quadrature component on the driven branch to suppress |1⟩→|2⟩ leakage.
# Standard choice: Ω_Q(t) = β_DRAG * dΩ/dt / |α| applied on the target branch.
USE_DRAG = False
BETA_DRAG = 1.0  # dimensionless; ≈1 is a good starting point
DRAG_TARGET_ONLY = False  # apply DRAG only to the branch equal to `target`

# Virtual-Z controls (only detune VZ supported)
VZ_COMPENSATE = False  # if True, advance software frame by detune only (Δ·T)
FRAME_PHASE = {'A': 0.0, 'B': 0.0, 'C': 0.0}  # running phase per mode (rad)
EPS_ONRES = twopi * 1e6  # ~1 MHz guard band for "near-resonant"

# Duration synthesis
TARGET_PEAK_FRAC = 1.0  # aim for peak = TARGET_PEAK_FRAC * RABI_CAP
DEFAULT_PEAK = twopi * 10e6  # used if RABI_CAP is None
MIN_SAMPLES = 1000  # minimum time points per segment
MIN_T = 8e-9  # floor on segment duration (s)
AUTO_T_SCALE = True  # stretch T (not amplitude) to respect cap
SAFETY = 1.0  # margin on cap
PRINT_PULSE_INFO = False  # verbose per-pulse summary

# ==================== Hilbert space (3 x 3 x 3) ====================

d = 3
I3 = qeye(d)
a = destroy(d)
ad = a.dag()
n = ad * a

# Modes: A x B x C
aA, nA = tensor(a, I3, I3), tensor(n, I3, I3)
aB, nB = tensor(I3, a, I3), tensor(I3, n, I3)
aC, nC = tensor(I3, I3, a), tensor(I3, I3, n)
I27 = tensor(I3, I3, I3)

# In-phase and quadrature (RWA drive)
X_A, Y_A = (aA + aA.dag()), 1j * (aA.dag() - aA)
X_B, Y_B = (aB + aB.dag()), 1j * (aB.dag() - aB)
X_C, Y_C = (aC + aC.dag()), 1j * (aC.dag() - aC)

# ==================== drift (rotating at upper bands) ====================

H0 = (
        (0.5 * alpha_A) * (nA * (nA - I27))
        + (0.5 * alpha_B) * (nB * (nB - I27))
        + (0.5 * alpha_C) * (nC * (nC - I27))
        + (2 * Jzz_AB * nA * nB) + (2 * Jzz_BC * nB * nC) + (2 * Jzz_CA * nC * nA)
)


# ==================== helpers ====================

def wrap_pi(phi: float) -> float:
    return (phi + np.pi) % (2 * np.pi) - np.pi


def amp(psi, a_idx: int, b_idx: int, c_idx: int):
    """Amplitude of |a, b,c⟩ in the 3×3×3 computational basis."""
    v = np.array(psi.full()).ravel()
    return v[a_idx * 9 + b_idx * 3 + c_idx]


def pops_comp_subspace(psi):
    """Populations in {|000⟩,|010⟩,|100⟩,|110⟩}; everything else is labeled leak."""
    v = np.array(psi.full()).ravel()
    P00 = abs(v[0]) ** 2  # |0,0,0>
    P01 = abs(v[3]) ** 2  # |0,1,0>
    P10 = abs(v[9]) ** 2  # |1,0,0>
    P11 = abs(v[12]) ** 2  # |1,1,0>
    Pleak = 1.0 - (P00 + P01 + P10 + P11)
    return P00, P01, P10, P11, Pleak


def rel_phase(psi, ket_i, ket_j) -> float:
    """Relative phase arg(⟨ket_i⟩) − arg(⟨ket_j⟩)."""
    ai = amp(psi, *ket_i)
    aj = amp(psi, *ket_j)
    return wrap_pi(float(np.angle(ai) - np.angle(aj)))


# Hann with fixed area (rotation "area" at the target), optional amplitude cap

def hann_env_area(tgrid, T: float, area: float, cap: float | None = None):
    s = np.zeros_like(tgrid)
    m = (tgrid >= 0.0) & (tgrid < T)
    x = tgrid[m] / T
    s[m] = 0.5 * (1 - np.cos(2 * np.pi * x))
    S = np.trapezoid(s, tgrid)
    A_s = area / max(S, 1e-18)
    if cap is not None:
        A_s = np.clip(A_s, -cap, cap)
    return A_s * s


def hann_peak_for_area(area: float, T: float) -> float:
    # For Hann with "area" = ∫Ω dt, the peak is 2*area/T
    return 2.0 * area / max(T, 1e-18)


# Lower-band detuning offsets (rotating at upper bands).  If a partner is |1>,
# the lower band of a mode is shifted by minus the corresponding Jzz.

def band_detuning_for(mode: str, condA: int, condB: int, condC: int) -> float:
    if mode == 'A':
        return -2 * (condB * Jzz_AB + condC * Jzz_CA)
    if mode == 'B':
        return -2 * (condA * Jzz_AB + condC * Jzz_BC)
    if mode == 'C':
        return -2 * (condA * Jzz_CA + condB * Jzz_BC)
    raise ValueError("mode must be 'A','B','C'")


# Stark per branch during a segment (computed for reference; never used for VZ).
# Transmon 3-level approximations:
#  • near-resonant: δω ≈ −Ω_R^2 / (2|α|)
#  • off-resonant: δω ≈ α Ω_R^2 / [2 Δ (α − Δ)] with α the *physical* (negative) anharmonicity
def compute_stark_phases_for_segment(tgrid, Omega, target, base):
    al = {'A': alpha_A, 'B': alpha_B, 'C': alpha_C}
    ou = {'A': omega_u_A, 'B': omega_u_B, 'C': omega_u_C}

    omega_d = ou[target] + base[target]

    # Pre-compute envelope derivative for DRAG magnitude (used only to refine Stark calc)
    dOmega_dt = np.gradient(Omega, tgrid, edge_order=2)
    alpha_mag = {'A': abs(alpha_A), 'B': abs(alpha_B), 'C': abs(alpha_C)}

    phis_stark = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    for m in ('A', 'B', 'C'):
        k = K_LINE[(target, m)]
        if np.isclose(k, 0.0):
            continue

        # detuning seen by branch m in this segment
        if m == target:
            Delta_m = base[m]
        else:
            Delta_m = (omega_d - ou[m]) + base[m]

        # α_phys is negative for transmon; use magnitude near resonance
        alpha_phys = float(al[m])  # positive storedl  # negative physical

        # Include the DRAG quadrature magnitude (only affects Stark via |Ω_R|^2)
        if USE_DRAG and (not DRAG_TARGET_ONLY or m == target):
            OmR2 = (0.5 * k) ** 2 * (Omega * Omega + (BETA_DRAG ** 2) * (dOmega_dt * dOmega_dt) / (alpha_mag[m] ** 2))
        else:
            OmR2 = (0.5 * k) ** 2 * (Omega * Omega)

        if m == target or abs(Delta_m) < EPS_ONRES:
            d_omega = - OmR2 / (2.0 * alpha_phys)
        else:
            d_omega = (alpha_phys * OmR2) / (2.0 * Delta_m * (alpha_phys - Delta_m))

        phis_stark[m] = float(np.trapezoid(d_omega, tgrid))

    return phis_stark


# ==================== one pulse segment (single LO, fan-out to A/B/C) ====================

def build_segment(
        target: str,
        band: str,
        theta: float,
        T: float | None = None,
        dt: float = 0.25e-9,
        phi: float = 0.0,
        *,
        condA: int = 0,
        condB: int = 0,
        condC: int = 0
):
    global FRAME_PHASE

    # Operators & references
    Xs = {'A': X_A, 'B': X_B, 'C': X_C}
    Ys = {'A': Y_A, 'B': Y_B, 'C': Y_C}
    ou = {'A': omega_u_A, 'B': omega_u_B, 'C': omega_u_C}

    # Base detunings for this segment
    base = {
        m: (0.0 if band == 'upper' else band_detuning_for(m, condA, condB, condC))
        for m in ('A', 'B', 'C')
    }

    # LO frequency anchored on the target
    omega_d = ou[target] + base[target]

    # Envelope area seen by the *target* branch:
    # θ_target = 0.5 * k_tt * ∫Ω dt  ⇒  ∫Ω dt = 2θ / k_tt
    k_tt = float(K_LINE[(target, target)])
    area_for_envelope = theta / max(k_tt, 1e-12)

    # Choose a duration if not provided (auto-T so the Hann peak hits the cap)
    if T is None:
        target_peak = (float(RABI_CAP) * TARGET_PEAK_FRAC) if (RABI_CAP is not None) else float(DEFAULT_PEAK)
        T_eff = 2.0 * area_for_envelope / max(target_peak, 1e-18)  # from Ω_peak = 2*area/T
        T_eff = max(T_eff, MIN_T, MIN_SAMPLES * dt)
    else:
        T_eff = float(T)

    # If the peak exceeds cap, stretch T
    if AUTO_T_SCALE and (RABI_CAP is not None):
        peak_needed = hann_peak_for_area(area_for_envelope, T_eff)
        cap = float(RABI_CAP) * SAFETY
        if peak_needed > cap:
            T_eff *= (peak_needed / cap)

    # Time grid and shared envelope Ω(t)
    npts = max(MIN_SAMPLES, int(np.ceil(T_eff / dt)))
    t = np.linspace(0.0, T_eff, npts, endpoint=False)
    Omega = hann_env_area(t, T_eff, area=area_for_envelope, cap=None)

    # Envelope time-derivative for DRAG
    dOmega_dt = np.gradient(Omega, t, edge_order=2)

    # Optional verbose pulse info
    if PRINT_PULSE_INFO:
        GHz = twopi * 1e9
        MHz = twopi * 1e6
        Om_peak = float(np.max(np.abs(Omega)))
        area_env = float(np.trapezoid(Omega, t))
        theta_implied = 0.5 * k_tt * area_env
        print(f"[pulse] target={target} band={band} cond={condA}{condB}{condC}  "
              f"LO={omega_d / GHz:.6f} GHz  T={T_eff * 1e9:.2f} ns  "
              f"(∫Ωdt={area_env:.3e}, θ_implied={theta_implied:.3f} rad)")
        print(
            f"      DRAG: {'on' if (USE_DRAG and DRAG_TARGET_ONLY) else ('on (all branches)' if USE_DRAG else 'off')}  β={BETA_DRAG:.3f}")
        for m in ('A', 'B', 'C'):
            k = K_LINE[(target, m)]
            if np.isclose(k, 0.0):
                continue
            Delta_m = (base[m]) if (m == target) else ((omega_d - ou[m]) + base[m])
            OmR_peak = 0.5 * abs(k) * Om_peak
            equiv_theta = 0.5 * abs(k) * area_env  # branch rotation
            print(f"   ↳ branch {m}: k={k:+.3f}  Δ={Delta_m / MHz:+.2f} MHz  "
                  f"Ω_R^peak={OmR_peak / (2 * np.pi) / 1e6:.2f} MHz  "
                  f"equiv_theta={np.degrees(equiv_theta):.1f}°")
        print("")

    # Build drive Hamiltonian list
    Hlist = [H0]
    for m in ('A', 'B', 'C'):
        k = K_LINE[(target, m)]
        if np.isclose(k, 0.0):
            continue

        Delta_m = (base[m]) if (m == target) else ((omega_d - ou[m]) + base[m])
        phi_m = phi + FRAME_PHASE[m]
        Xm, Ym = Xs[m], Ys[m]

        alpha_map = {'A': alpha_A, 'B': alpha_B, 'C': alpha_C}
        beta_eff = (BETA_DRAG if USE_DRAG and (not DRAG_TARGET_ONLY or m == target) else 0.0)
        denom = max(abs(float(alpha_map[m])), 1e-18)  # use |α|

        def fx_m(tau, *, k=k, phi_m=phi_m, Delta_m=Delta_m, beta_eff=beta_eff, denom=denom):
            Om = float(np.interp(tau, t, Omega, left=0.0, right=0.0))
            dOm = float(np.interp(tau, t, dOmega_dt, left=0.0, right=0.0))
            # Complex amplitude A = Ω + i * β * dΩ/dt / |α|
            # Expand to X/Y using cos/sin:
            # X-coeff: Ω cos(⋯) − (β dΩ/dt/|α|) sin(⋯)
            return 0.5 * k * (
                    Om * np.cos(Delta_m * tau + phi_m) - (beta_eff * dOm / denom) * np.sin(Delta_m * tau + phi_m))

        def fy_m(tau, *, k=k, phi_m=phi_m, Delta_m=Delta_m, beta_eff=beta_eff, denom=denom):
            Om = float(np.interp(tau, t, Omega, left=0.0, right=0.0))
            dOm = float(np.interp(tau, t, dOmega_dt, left=0.0, right=0.0))
            # Y-coeff: Ω sin(⋯) + (β dΩ/dt/|α|) cos(⋯)
            return 0.5 * k * (
                    Om * np.sin(Delta_m * tau + phi_m) + (beta_eff * dOm / denom) * np.cos(Delta_m * tau + phi_m))

        Hlist += [[Xm, fx_m], [Ym, fy_m]]

    # --- Analytics per segment (for reporting / post-VZ) ---
    phis_stark = compute_stark_phases_for_segment(t, Omega, target, base)
    # Pure detuning phase per branch: φ_det[m] = Δ_m · T_eff
    Delta_per = {}
    for m in ('A', 'B', 'C'):
        if m == target:
            Delta_per[m] = base[m]
        else:
            Delta_per[m] = (omega_d - ou[m]) + base[m]
    phi_det = {m: float(Delta_per[m]) * float(T_eff) for m in ('A', 'B', 'C')}

    # Optionally update the software frame with DETUNE ONLY (never Stark)
    if VZ_COMPENSATE:
        for q in ('A', 'B', 'C'):
            FRAME_PHASE[q] -= phi_det[q]

    # Return segment and a metadata bundle
    return t, Hlist, {'detune': phi_det, 'stark': phis_stark, 'Delta': Delta_per, 'T': T_eff}


# ==================== run a chain ====================

def run_chain(
        psi0,
        segments,
        title: str = "sequence",
        plot=PLOT,
        print_seg: bool = True,
        apply_vz_to_state: bool = False,  # if True, apply DETUNE-VZ (Δ·T) to the state after each pulse
        log_vz: bool = False,
):
    times_all, states_all, t_off, psi = [], [], 0.0, psi0
    seg_end_ns = []

    for k, seg in enumerate(segments, 1):
        if len(seg) == 3:
            tloc, Hlist, info = seg
        else:
            tloc, Hlist = seg
            info = None

        out = mesolve(Hlist, psi, tloc, e_ops=[])
        psi = out.states[-1]

        # Record segment end time in ns for plotting
        end_ns = (t_off + (tloc[-1] if len(tloc) > 0 else 0.0)) * 1e9
        seg_end_ns.append(end_ns)

        # ----- DETUNE-VZ (optional): apply only φ_det to the state -----
        if apply_vz_to_state and info is not None:
            phi_det = info.get('detune', {})
            # Apply exp(-i φ_det n) per mode
            if 'A' in phi_det and abs(phi_det['A']) > 0:
                psi = ((-1j * float(phi_det['A']) * nA).expm()) * psi
            if 'B' in phi_det and abs(phi_det['B']) > 0:
                psi = ((-1j * float(phi_det['B']) * nB).expm()) * psi
            if 'C' in phi_det and abs(phi_det['C']) > 0:
                psi = ((-1j * float(phi_det['C']) * nC).expm()) * psi
            if log_vz:
                print(f"[seg {k:02d}] applied detune-VZ (rad): "
                      f"A:{phi_det['A']:+.3e}  B:{phi_det['B']:+.3e}  C:{phi_det['C']:+.3e}")

        # Store post-VZ state
        out.states[-1] = psi

        # ----- Reporting -----
        if print_seg:
            P00, P01, P10, P11, Pleak = pops_comp_subspace(psi)
            end_ns = (t_off + (tloc[-1] if len(tloc) > 0 else 0.0)) * 1e9
            # Detune phase (this pulse)
            if info is not None and 'detune' in info:
                pd = info['detune']
                print(f"[seg {k:02d} end @ {end_ns:8.1f} ns]  "
                      f"P00={P00:.3f}  P01={P01:.3f}  P10={P10:.3f}  P11={P11:.3f}  leak={Pleak:.3f}\n"
                      f"   detune φ this pulse (rad):  A:{pd['A']:+.6f}  B:{pd['B']:+.6f}  C:{pd['C']:+.6f}")
            else:
                print(f"[seg {k:02d} end @ {end_ns:8.1f} ns]  "
                      f"P00={P00:.3f}  P01={P01:.3f}  P10={P10:.3f}  P11={P11:.3f}  leak={Pleak:.3f}")

        tt = np.array(out.times) + t_off
        times_all.extend(tt.tolist())
        states_all.extend(out.states)
        t_off = tt[-1]

    if plot:
        t_ns = np.array(times_all) * 1e9
        P00s, P01s, P10s, P11s, Pleaks = [], [], [], [], []
        for st in states_all:
            p00, p01, p10, p11, pl = pops_comp_subspace(st)
            P00s.append(p00)
            P01s.append(p01)
            P10s.append(p10)
            P11s.append(p11)
            Pleaks.append(pl)

        plt.figure(figsize=(7.8, 4.3))
        # Line plots instead of stacked area
        plt.plot(t_ns, np.array(P00s), label=r'$P_{00}$')
        plt.plot(t_ns, np.array(P01s), label=r'$P_{01}$')
        plt.plot(t_ns, np.array(P10s), label=r'$P_{10}$')
        plt.plot(t_ns, np.array(P11s), label=r'$P_{11}$')
        plt.plot(t_ns, np.array(Pleaks), label=r'$P_{\rm leak}$')

        plt.xlabel('time [ns]')
        plt.ylabel('population')
        plt.ylim(-0.02, 1.02)
        plt.title(title)

        # Mark the end of each pulse segment
        for idx, x in enumerate(seg_end_ns, start=1):
            plt.axvline(x=x, linestyle='--', linewidth=1.2, alpha=0.6)
            # annotate a segment number at the top
            plt.text(x, 1.02, f'{idx}', ha='center', va='bottom', fontsize=9)

        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    return np.array(times_all), states_all, psi


# ==================== 3-pulse loops (C frozen in |0⟩) ====================

def build_loop(loop_id: str, dt: float = 0.25e-9):
    segs = []
    if loop_id.upper() == "B":
        # 1) B upper pi/2
        segs.append(build_segment('B', 'upper', theta=np.pi / 2, dt=dt,
                                  condA=0, condB=0, condC=0))
        # 2) A lower pi (condition B=1)
        segs.append(build_segment('A', 'lower', theta=np.pi, dt=dt,
                                  condA=0, condB=1, condC=0))
        # 3) B lower pi (condition A=1)
        segs.append(build_segment('B', 'lower', theta=np.pi, dt=dt,
                                  condA=1, condB=0, condC=0))
    else:
        # 1) A upper pi/2
        segs.append(build_segment('A', 'upper', theta=np.pi / 2, dt=dt,
                                  condA=0, condB=0, condC=0))
        # 2) B lower pi (condition A=1)
        segs.append(build_segment('B', 'lower', theta=np.pi, dt=dt,
                                  condA=1, condB=0, condC=0))
        # 3) A lower pi (condition B=1)
        segs.append(build_segment('A', 'lower', theta=np.pi, dt=dt,
                                  condA=0, condB=1, condC=0))
    return segs


# ==================== main ====================

if __name__ == "__main__":
    psi000 = tensor(basis(3, 0), basis(3, 0), basis(3, 0))

    # -------- LOOP B --------
    FRAME_PHASE = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    print("\n=== LOOP-B: B upper π/2 → A lower π → B lower π (C=0) ===")
    segs_B = build_loop("B", dt=0.25e-11)
    tB, sB, psiB = run_chain(psi000, segs_B, title="Loop-B (3 pulses)",
                             apply_vz_to_state=True)  # set True to apply detune-VZ during a run
    phi_B = rel_phase(psiB, (1, 0, 0), (0, 0, 0))
    print(f"[Loop B] relative phase (|100> vs |000>): {phi_B:+.6f} rad")

    # -------- LOOP A --------
    FRAME_PHASE = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    print("\n=== LOOP-A: A upper π/2 → B lower π → A lower π (C=0) ===")
    segs_A = build_loop("A", dt=0.25e-11)
    tA, sA, psiA = run_chain(psi000, segs_A, title="Loop-A (3 pulses)",
                             apply_vz_to_state=True)  # set True to apply detune-VZ during a run
    phi_A = rel_phase(psiA, (0, 1, 0), (0, 0, 0))
    print(f"[Loop A] relative phase (|010> vs |000>): {phi_A:+.6f} rad\n")
    print(f"|phi_A − phi_B| = {abs(wrap_pi(phi_A - phi_B)):.6f} rad")
