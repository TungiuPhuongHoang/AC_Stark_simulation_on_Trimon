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

# TR225 measured conditional transitions (MHz)
A00_MHz, A10_MHz, A01_MHz, A11_MHz = 5181.0, 5026.0, 4930.0, 4763.0
B00_MHz, B10_MHz, B01_MHz, B11_MHz = 4479.0, 4325.0, 4272.0, 4105.0
C00_MHz, C10_MHz, C01_MHz, C11_MHz = 6001.0, 5750.0, 5794.0, 5531.0

# TR225: |alpha| from J_ii (A:128, B:116, C:142) MHz
alpha_A = -twopi * 128.0e6
alpha_B = -twopi * 116.0e6
alpha_C = -twopi * 142.0e6


# TR225: reported cross‑Kerr strengths (MHz)
Jzz_AB = twopi * 78.0e6
Jzz_BC = twopi * 104.0e6
Jzz_CA = twopi * 126.0e6


# ==================== control params ====================

# Envelope cap (rad/s). Use None during calibration sweeps to avoid clipping.
RABI_CAP = twopi * 10e6

# TR225: relative drive fan‑out (from π‑time ratios)
ALPHA = 1.0                  # ↦ A (reference)
BETA  = 192.0/578.0          # ≈ 0.332 ↦ B
GAMMA = 192.0/262.0          # ≈ 0.733 ↦ C

K_LINE = {
    ('A', 'A'): ALPHA, ('A', 'B'): BETA, ('A', 'C'): GAMMA,
    ('B', 'A'): ALPHA, ('B', 'B'): BETA, ('B', 'C'): GAMMA,
    ('C', 'A'): ALPHA, ('C', 'B'): BETA, ('C', 'C'): GAMMA,
}

# Lab AWG scale per target & band (from report). This multiplies the single-line fan-out.
SEG_SCALE = {
    ('A', 'upper'): 1.0,
    ('A', 'lower'): 1.462,   # A01 / A00 amplitude ratio
    ('B', 'upper'): 1.0,
    ('B', 'lower'): 1.0,
    ('C', 'upper'): 1.0,
    ('C', 'lower'): 0.776,   # C10 / C00 amplitude ratio
}

VZ_KIND = 'detune'   # 'total' | 'stark' | 'detune'

# Virtual‑Z controls
VZ_COMPENSATE = True  # set True to subtract accumulated Z phase
FRAME_PHASE = {'A': 0.0, 'B': 0.0, 'C': 0.0}  # running phase per mode (rad)
EPS_ONRES = twopi * 1e6  # ~1 MHz guard band for "near-resonant"

# Duration synthesis
TARGET_PEAK_FRAC = 1.0  # aim for peak = TARGET_PEAK_FRAC * RABI_CAP
DEFAULT_PEAK = twopi * 10e6  # used if RABI_CAP is None
MIN_SAMPLES = 1000  # minimum time points per segment
MIN_T = 8e-9  # floor on segment duration (s)
AUTO_T_SCALE = True  # stretch T (not amplitude) to respect cap
SAFETY = 1.0  # margin on cap


PRINT_PULSE_INFO = True  # verbose per‑pulse summary

# Analysis/reporting: choose whether to wrap phase differences to (−π,π]
WRAP_PHASES = False  # keep False to preserve additivity in component sums


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

# In‑phase and quadrature (RWA drive)
X_A, Y_A = (aA + aA.dag()), 1j * (aA.dag() - aA)
X_B, Y_B = (aB + aB.dag()), 1j * (aB.dag() - aB)
X_C, Y_C = (aC + aC.dag()), 1j * (aC.dag() - aC)


# ==================== drift Hamiltonian ====================
# (A) DUFFING + cross-Kerr (working frame at upper bands)
H0_DUFFING = (
        (0.5 * alpha_A) * (nA * (nA - I27))
        + (0.5 * alpha_B) * (nB * (nB - I27))
        + (0.5 * alpha_C) * (nC * (nC - I27))
        + (Jzz_AB * nA * nB) + (Jzz_BC * nB * nC) + (Jzz_CA * nC * nA)
)

H0 = H0_DUFFING



# ==================== helpers ====================

def rel_phase_in_rot_frame(psi, H0, t_total, ket_i, ket_j):
    U = (+1j * float(t_total) * H0).expm()
    psi_rot = U * psi
    return rel_phase(psi_rot, ket_i, ket_j)


def wrap_pi(phi: float) -> float:
    return (phi + np.pi) % (2 * np.pi) - np.pi


def amp(psi, a_idx: int, b_idx: int, c_idx: int):
    """Amplitude of |a, b,c⟩ in the 3×3×3 computational basis."""
    v = np.array(psi.full()).ravel()
    return v[a_idx * 9 + b_idx * 3 + c_idx]


def pops_comp_subspace(psi):
    """Populations in {|000⟩,|001⟩,|100⟩,|101⟩}; everything else is labeled leak.
       Basis order is |A,B,C⟩ and index = a*9 + b*3 + c."""
    v = np.array(psi.full()).ravel()
    P00 = abs(v[0]) ** 2  # |0,0,0>
    P01 = abs(v[1]) ** 2  # |0,0,1>   (C=1)
    P10 = abs(v[9]) ** 2  # |1,0,0>   (A=1)
    P11 = abs(v[10]) ** 2  # |1,0,1>   (A=1,C=1)
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
    return 2.0 * area / max(T, 1e-18)



# Lower‑band detuning offsets (rotating at upper bands).  If a partner is |1>,
# the lower band of a mode is shifted by minus the corresponding Jzz.
def band_detuning_for(mode: str, condA: int, condB: int, condC: int) -> float:
    if mode == 'A':
        return -(condB * Jzz_AB + condC * Jzz_CA)
    if mode == 'B':
        return -(condA * Jzz_AB + condC * Jzz_BC)
    if mode == 'C':
        return -(condA * Jzz_CA + condB * Jzz_BC)
    raise ValueError("mode must be 'A','B','C'")



# NOTE on signs:
#   • alpha_* are stored NEGATIVE (transmon convention in this file).
#   • Inside the Stark model we use alpha_phys = -alpha_* (>0) so the analytic
#     formulas (near‑resonant and off‑resonant) are evaluated with magnitudes.
#   • Near‑resonant target: δω ≈ −Ω_R^2/(2|α|).
#   • Off‑resonant spectator: δω ≈ α|Ω_R|^2/[2Δ(α−Δ)] with α taken as |α| in code.

def compute_stark_phases_for_segment(tgrid, Omega, target, base, scale: float = 1.0):
    al = {'A': alpha_A, 'B': alpha_B, 'C': alpha_C}
    # Use upper-band references for carrier/reference frequencies
    ou = {'A': omega_u_A, 'B': omega_u_B, 'C': omega_u_C}
    omega_d = ou[target] + base[target]

    phis = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    for m in ('A', 'B', 'C'):
        k = float(scale) * K_LINE[(target, m)]
        if np.isclose(k, 0.0):
            continue

        # detuning seen by branch m in this segment
        if m == target:
            Delta_m = base[m]
        else:
            Delta_m = (omega_d - ou[m]) + base[m]

        # Use magnitude of α near resonance (α is negative)
        alpha_phys = -float(al[m])

        # In this convention
        OmR = 0.5 * k * Omega

        if m == target or abs(Delta_m) < EPS_ONRES:
            d_omega = -(OmR * OmR) / (2.0 * abs(alpha_phys))
        else:
            d_omega = (alpha_phys * OmR * OmR) / (2.0 * Delta_m * (alpha_phys - Delta_m))

        phis[m] = float(np.trapezoid(d_omega, tgrid))

    return phis


# ==================== idle segment (free evolution) ====================

def build_idle(T: float, dt: float = 0.25e-9):
    npts = max(4, int(np.ceil(T / dt)))
    t = np.linspace(0.0, T, npts, endpoint=False)
    return t, [H0]


# ==================== one pulse segment (single LO, fan‑out to A/B/C) ====================

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
        condC: int = 0,
        scale: float | None = None,
):
    global FRAME_PHASE

    # Operators & references
    Xs = {'A': X_A, 'B': X_B, 'C': X_C}
    Ys = {'A': Y_A, 'B': Y_B, 'C': Y_C}
    # Upper-band references
    ou = {'A': omega_u_A, 'B': omega_u_B, 'C': omega_u_C}

    # Base detuning consistent with H0 (ZZ-only lower-band model):
    # use minus-sum-of-Jzz shifts when a spectator is |1>, and zero for upper-band targets.
    base = {
        m: (0.0 if band == 'upper' else band_detuning_for(m, condA, condB, condC))
        for m in ('A', 'B', 'C')
    }

    # LO frequency anchored on the target
    omega_d = ou[target] + base[target]

    # Effective per-branch couplings include the lab scale (per pulse) times fixed fan-out
    scale_eff = float(SEG_SCALE.get((target, band), 1.0) if scale is None else scale)

    # Envelope area seen by the *target* branch (normalize by k_tt).
    # In our RWA convention, the instantaneous Rabi rate is
    #   Ω_R(t) = 0.5 * k_tt_eff * Ω(t),
    # and the Bloch rotation is θ = ∫ Ω_R(t) dt = 0.5 * k_tt_eff * ∫ Ω(t) dt.
    # Hence the envelope area must satisfy ∫Ω dt = 2θ / k_tt_eff.
    k_tt_eff = float(scale_eff) * float(K_LINE[(target, target)])
    area_for_envelope = theta / max(k_tt_eff, 1e-12)

    # Choose a duration if not provided
    if T is None:
        target_peak = (float(RABI_CAP) * TARGET_PEAK_FRAC) if (RABI_CAP is not None) else float(DEFAULT_PEAK)
        # From the Hann lobe relation: Ω_peak = 2 * area / T
        T_eff = 2.0 * area_for_envelope / max(target_peak, 1e-18)
        # Respect only the physical minimum; do NOT inflate T by MIN_SAMPLES*dt
        T_eff = max(T_eff, MIN_T)
    else:
        T_eff = float(T)

    # Ensure we respect the cap by stretching T if necessary
    scaled_note = ""
    if AUTO_T_SCALE and (RABI_CAP is not None):
        peak_needed = hann_peak_for_area(area_for_envelope, T_eff)
        cap = float(RABI_CAP) * SAFETY
        if peak_needed > cap:
            scale = peak_needed / cap
            T_old = T_eff
            T_eff *= scale
            scaled_note = f"  (auto-stretch {T_old * 1e9:.2f}→{T_eff * 1e9:.2f} ns)"

    # Time grid and shared envelope Ω(t)
    npts = max(MIN_SAMPLES, int(np.ceil(T_eff / dt)))
    t = np.linspace(0.0, T_eff, npts, endpoint=False)
    Omega = hann_env_area(t, T_eff, area=area_for_envelope, cap=None)

    # Optional verbose pulse info
    if PRINT_PULSE_INFO:
        GHz = twopi * 1e9
        MHz = twopi * 1e6
        Om_peak = float(np.max(np.abs(Omega)))
        area_env = float(np.trapezoid(Omega, t))
        theta_implied = 0.5 * k_tt_eff * area_env
        print(f"[pulse] target={target} band={band} cond={condA}{condB}{condC}  "
              f"LO={omega_d / GHz:.6f} GHz  T={T_eff * 1e9:.2f} ns{scaled_note}"
              f"  (∫Ωdt={area_env:.3e}, θ_implied={theta_implied:.3f} rad)")
        print(f"      scale={scale_eff:.3f}  k_tt_eff={k_tt_eff:.3f}")
        for m in ('A', 'B', 'C'):
            k = float(scale_eff) * K_LINE[(target, m)]
            if np.isclose(k, 0.0):
                continue
            Delta_m = (base[m]) if (m == target) else ((omega_d - ou[m]) + base[m])
            OmR_peak = 0.5 * abs(k) * Om_peak
            equiv_theta = 0.5 * abs(k) * area_env
            print(f"   ↳ branch {m}: k={k:+.3f}  Δ={Delta_m / MHz:+.2f} MHz  "
                  f"Ω_R^peak={OmR_peak / (2 * np.pi) / 1e6:.2f} MHz  "
                  f"equiv_theta={np.degrees(equiv_theta):.1f}°")
        print("")

    # Build drive Hamiltonian list
    Hlist = [H0]
    for m in ('A', 'B', 'C'):
        k = float(scale_eff) * K_LINE[(target, m)]
        if np.isclose(k, 0.0):
            continue

        Delta_m = (base[m]) if (m == target) else ((omega_d - ou[m]) + base[m])
        phi_m = phi + FRAME_PHASE[m]
        Xm, Ym = Xs[m], Ys[m]

        def fx_m(tau, *, k=k, phi_m=phi_m, Delta_m=Delta_m):
            return 0.5 * k * float(np.interp(tau, t, Omega, left=0.0, right=0.0)) * np.cos(Delta_m * tau + phi_m)

        def fy_m(tau, *, k=k, phi_m=phi_m, Delta_m=Delta_m):
            return 0.5 * k * float(np.interp(tau, t, Omega, left=0.0, right=0.0)) * np.sin(Delta_m * tau + phi_m)

        Hlist += [[Xm, fx_m], [Ym, fy_m]]

    # AC‑Stark phases accumulated during this segment (Stark-only)
    phis_stark = compute_stark_phases_for_segment(t, Omega, target, base, scale=scale_eff)

    # Add pure detuning frame phase for driven branches: φ_det[m] = Δ_m · T_eff
    phi_det = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    for m in ('A', 'B', 'C'):
        k_eff = float(scale_eff) * K_LINE[(target, m)]
        if np.isclose(k_eff, 0.0):
            continue
        Delta_m = (base[m]) if (m == target) else ((omega_d - ou[m]) + base[m])
        phi_det[m] = float(Delta_m) * float(T_eff)

    # Compose totals per branch
    phis_total = {m: float(phis_stark.get(m, 0.0)) + float(phi_det.get(m, 0.0)) for m in ('A', 'B', 'C')}

    # Optionally update software frame phases to cancel total deterministic Z
    if VZ_COMPENSATE:
        bundle = {'total': phis_total, 'stark': phis_stark, 'detune': phi_det}[VZ_KIND]
        for q in ('A', 'B', 'C'):
            FRAME_PHASE[q] -= float(bundle.get(q, 0.0))

    return t, Hlist, {'total': phis_total, 'stark': phis_stark, 'detune': phi_det}


# ==================== run a chain ====================

def run_chain(
        psi0,
        segments,
        title: str = "sequence",
        plot: bool = True,
        print_seg: bool = True,
        apply_vz_to_state: bool = False,
        vz_kind: str = 'total',
        log_vz: bool = False,
):
    times_all, states_all, t_off, psi = [], [], 0.0, psi0
    seg_end_ns = []

    for k, seg in enumerate(segments, 1):
        if len(seg) == 3:
            tloc, Hlist, vz_info = seg
            # Support either a flat dict or a component bundle
            if isinstance(vz_info, dict) and all(k in vz_info for k in ('total', 'stark', 'detune')):
                vz_phis = vz_info.get(vz_kind, vz_info['total']) if apply_vz_to_state else None
            else:
                vz_phis = vz_info if apply_vz_to_state else None
        else:
            tloc, Hlist = seg
            vz_phis = None

        seg_end_time_abs = t_off + (tloc[-1] if len(tloc) > 0 else 0.0)
        seg_end_ns.append(seg_end_time_abs * 1e9)
        out = mesolve(Hlist, psi, tloc, e_ops=[])
        psi = out.states[-1]

        # Optionally apply VZ directly to the state (showing what would be canceled)
        if apply_vz_to_state and vz_phis is not None:
            for m in ('A', 'B', 'C'):
                phi = float(vz_phis.get(m, 0.0))
                if abs(phi) < 1e-15:
                    continue
                if m == 'A':
                    U = (-1j * phi * nA).expm()
                elif m == 'B':
                    U = (-1j * phi * nB).expm()
                else:
                    U = (-1j * phi * nC).expm()
                psi = U * psi
            if log_vz:
                print(f"[seg {k:02d}] applied VZ to state (rad): "
                      f"{{'A': {float(vz_phis['A']):+.3e}, 'B': {float(vz_phis['B']):+.3e}, 'C': {float(vz_phis['C']):+.3e}}}")

        # Store post‑VZ state
        out.states[-1] = psi

        if print_seg:
            P00, P01, P10, P11, Pleak = pops_comp_subspace(psi)
            end_ns = seg_end_time_abs * 1e9
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

        # High-quality population plot with segment markers
        fig = plt.figure(figsize=(9.5, 5.2), dpi=200)
        ax = plt.gca()
        ax.stackplot(
            t_ns,
            np.array(P00s), np.array(P01s), np.array(P10s), np.array(P11s), np.array(Pleaks),
            labels=[r'$P_{000}$', r'$P_{001}$', r'$P_{100}$', r'$P_{101}$', r'$P_{\rm leak}$'],
            alpha=0.85,
        )

        # Axis labels and limits
        ax.set_xlabel('time [ns]')
        ax.set_ylabel('population')
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(t_ns.min() if len(t_ns) else 0.0, t_ns.max() if len(t_ns) else 1.0)
        ax.set_title(title)

        # Segment end markers with numbers (placed just above the dashed line)
        y_top_lim = ax.get_ylim()[1]
        for idx, x in enumerate(seg_end_ns, start=1):
            ax.axvline(x=x, linestyle='--', linewidth=1.5, alpha=0.7)
            # Put the number a few pixels above the top edge at this x-position
            ax.annotate(
                f'{idx}', xy=(x, y_top_lim), xytext=(0, 3),
                textcoords='offset points', ha='center', va='bottom',
                fontsize=9, clip_on=False
            )

        # Legend and layout
        ax.legend(loc='upper right', ncol=1, frameon=True)
        plt.tight_layout()
        plt.show()

    return np.array(times_all), states_all, psi


# ==================== 3‑pulse loops (C frozen in |0⟩) ====================



def build_loop(loop_id: str, dt: float = 0.25e-9):
    """
    Two 3-pulse loops with B frozen in |0⟩:

      Loop 'C':  C upper π/2  →  A lower π (C=1)  →  C lower π (A=1)
      Loop 'A':  A upper π/2  →  C lower π (A=1)  →  A lower π (C=1)
    """
    segs = []
    L = loop_id.upper()
    if L == "C":
        # 1) C upper π/2   (A=0, B=0, C=0)
        segs.append(build_segment('C', 'upper', theta=np.pi / 2, dt=dt,
                                  condA=0, condB=0, condC=0,
                                  scale=SEG_SCALE[('C','upper')]))
        # 2) A lower π  conditioned on C=1
        segs.append(build_segment('A', 'lower', theta=np.pi, dt=dt,
                                  condA=0, condB=0, condC=1,
                                  scale=SEG_SCALE[('A','lower')]))
        # 3) C lower π  conditioned on A=1
        segs.append(build_segment('C', 'lower', theta=np.pi, dt=dt,
                                  condA=1, condB=0, condC=0,
                                  scale=SEG_SCALE[('C','lower')]))
    else:  # "A"
        # 1) A upper π/2   (A=0, B=0, C=0)
        segs.append(build_segment('A', 'upper', theta=np.pi / 2, dt=dt,
                                  condA=0, condB=0, condC=0,
                                  scale=SEG_SCALE[('A','upper')]))
        # 2) C lower π  conditioned on A=1
        segs.append(build_segment('C', 'lower', theta=np.pi, dt=dt,
                                  condA=1, condB=0, condC=0,
                                  scale=SEG_SCALE[('C','lower')]))
        # 3) A lower π  conditioned on C=1
        segs.append(build_segment('A', 'lower', theta=np.pi, dt=dt,
                                  condA=0, condB=0, condC=1,
                                  scale=SEG_SCALE[('A','lower')]))
    return segs


# ==================== phase component analysis helpers ====================

def _sum_components_over_segments(segs):
    zero = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    S, D, Tt = dict(zero), dict(zero), dict(zero)
    for seg in segs:
        if len(seg) != 3:
            continue
        _, _, info = seg
        if isinstance(info, dict) and all(k in info for k in ('total', 'stark', 'detune')):
            for m in ('A', 'B', 'C'):
                S[m] += float(info['stark'].get(m, 0.0))
                D[m] += float(info['detune'].get(m, 0.0))
                Tt[m] += float(info['total'].get(m, 0.0))
    return S, D, Tt


def _measure_loop_phase(loop_id: str, dt: float = 0.25e-9,
                        apply_vz: bool = False, vz_kind: str = 'total'):
    """
    For loop 'C', report phase on A:    arg⟨100⟩ − arg⟨000⟩
    For loop 'A', report phase on C:    arg⟨001⟩ − arg⟨000⟩
    """
    psi000 = tensor(basis(3, 0), basis(3, 0), basis(3, 0))
    FRAME_PHASE.update({'A': 0.0, 'B': 0.0, 'C': 0.0})
    segs = build_loop(loop_id, dt=dt)
    _, _, psi = run_chain(psi000, segs, plot=False, print_seg=False,
                          apply_vz_to_state=apply_vz, vz_kind=vz_kind)
    if loop_id.upper() == 'C':
        # Loop on C → read phase on A
        return rel_phase(psi, (1, 0, 0), (0, 0, 0))
    else:  # 'A'
        # Loop on A → read phase on C
        return rel_phase(psi, (0, 0, 1), (0, 0, 0))


def analyze_phase_components(dt: float = 0.25e-9):
    # Measured (no VZ)
    phi_C_onA = _measure_loop_phase('C', dt=dt, apply_vz=False)
    phi_A_onC = _measure_loop_phase('A', dt=dt, apply_vz=False)
    dphi_meas = abs(phi_A_onC - phi_C_onA) if not WRAP_PHASES else abs(wrap_pi(phi_A_onC - phi_C_onA))

    # Stark-only (cancel detune on the state)
    phi_C_onA_stark = _measure_loop_phase('C', dt=dt, apply_vz=True, vz_kind='detune')
    phi_A_onC_stark = _measure_loop_phase('A', dt=dt, apply_vz=True, vz_kind='detune')
    dphi_stark_only = abs(phi_A_onC_stark - phi_C_onA_stark) if not WRAP_PHASES else abs(wrap_pi(phi_A_onC_stark - phi_C_onA_stark))

    # Detune-only (cancel Stark on the state)
    phi_C_onA_det = _measure_loop_phase('C', dt=dt, apply_vz=True, vz_kind='stark')
    phi_A_onC_det = _measure_loop_phase('A', dt=dt, apply_vz=True, vz_kind='stark')
    dphi_detune_only = abs(phi_A_onC_det - phi_C_onA_det) if not WRAP_PHASES else abs(wrap_pi(phi_A_onC_det - phi_C_onA_det))

    # Logged sums for intuition
    FRAME_PHASE.update({'A': 0.0, 'B': 0.0, 'C': 0.0})
    segs_C = build_loop('C', dt=dt);
    run_chain(tensor(basis(3, 0), basis(3, 0), basis(3, 0)), segs_C, plot=False, print_seg=False)
    SC, DC, TC = _sum_components_over_segments(segs_C)
    FRAME_PHASE.update({'A': 0.0, 'B': 0.0, 'C': 0.0})
    segs_A = build_loop('A', dt=dt);
    run_chain(tensor(basis(3, 0), basis(3, 0), basis(3, 0)), segs_A, plot=False, print_seg=False)
    SA, DA, TA = _sum_components_over_segments(segs_A)

    # Compare “other” branches: (segs_A → C) minus (segs_C → A)
    _stark_diff = float(SA['C']) - float(SC['A'])
    _detune_diff = float(DA['C']) - float(DC['A'])
    _total_diff  = float(TA['C']) - float(TC['A'])
    if WRAP_PHASES:
        dphi_stark_log = abs(wrap_pi(_stark_diff))
        dphi_detune_log = abs(wrap_pi(_detune_diff))
        dphi_total_log  = abs(wrap_pi(_total_diff))
    else:
        dphi_stark_log = abs(_stark_diff)
        dphi_detune_log = abs(_detune_diff)
        dphi_total_log  = abs(_total_diff)

    print("\n=== Phase-difference components (A↔C, B frozen) ===")
    print(f"|Δφ| (measured, no VZ)       = {dphi_meas:.6f} rad")
    print(f"|Δφ| (measured, Stark-only)  = {dphi_stark_only:.6f} rad")
    print(f"|Δφ| (measured, detune-only) = {dphi_detune_only:.6f} rad")
    print("    (logged sums; intuition only)")
    print(f"|Δφ_total| (log) = {dphi_total_log:.6f} rad   "
          f"|Δφ_Stark| (log) = {dphi_stark_log:.6f} rad   "
          f"|Δφ_detune| (log) = {dphi_detune_log:.6f} rad")


# ---- convenience for external sweep scripts ----
def set_rabi_cap_MHz(cap_MHz: float | None):
    """Set RABI_CAP in rad/s given a cap in MHz (None to disable the cap)."""
    global RABI_CAP
    if cap_MHz is None:
        RABI_CAP = None
    else:
        RABI_CAP = twopi * float(cap_MHz) * 1e6

# ==================== main ====================

if __name__ == "__main__":
    psi000 = tensor(basis(3, 0), basis(3, 0), basis(3, 0))

    # Loop-C (anchor C), measure phase on A
    FRAME_PHASE = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    print("\n=== LOOP-C: C upper π/2 → A lower π (C=1) → C lower π (A=1);  B=0 ===")
    segs_C = build_loop("C", dt=0.25e-9)
    tC, sC, psiC = run_chain(psi000, segs_C, title="Loop-C (A–C, B frozen)")
    phi_C_onA = rel_phase(psiC, (1, 0, 0), (0, 0, 0))
    print(f"[Loop C] phase on A (|100⟩ vs |000⟩): {phi_C_onA:+.6f} rad")

    # Loop-A (anchor A), measure phase on C
    FRAME_PHASE = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    print("\n=== LOOP-A: A upper π/2 → C lower π (A=1) → A lower π (C=1);  B=0 ===")
    segs_A = build_loop("A", dt=0.25e-9)
    tA, sA, psiA = run_chain(psi000, segs_A, title="Loop-A (A–C, B frozen)")
    phi_A_onC = rel_phase(psiA, (0, 0, 1), (0, 0, 0))
    print(f"[Loop A] phase on C (|001⟩ vs |000⟩): {phi_A_onC:+.6f} rad")

    _raw_diff = phi_A_onC - phi_C_onA
    _show_diff = abs(_raw_diff) if not WRAP_PHASES else abs(wrap_pi(_raw_diff))
    print(f"\n|φ_(A←Loop A) − φ_(A←Loop C)| = |{phi_A_onC:+.6f} − {phi_C_onA:+.6f}| = {_show_diff:.6f} rad")
    #
    # analyze_phase_components(dt=0.25e-9)