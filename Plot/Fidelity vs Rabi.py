# Fidelity_vs_RABI_cap_AB_ZInvariant.py
# Sweep RABI cap and compute phase-optimized (Z-invariant) state fidelity:
#   Loop-B → target subspace { |000>, |100> }  =>  Fmax = (|A000| + |A100|)^2 / 2
#   Loop-A → target subspace { |000>, |010> }  =>  Fmax = (|A000| + |A010|)^2 / 2
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, Qobj
import core as core  # your simulator (must be importable)

twopi = 2 * np.pi

# --- config ---
DT = 0.25e-11  # simulation time step for building segments
CAPS_MHZ = np.linspace(2.0, 100.0, 40)  # sweep Ω_max/(2π) in MHz


# ---------- utilities ----------
def set_core_quiet() -> None:
    """Make core not print/plot and never apply VZ during sweeps."""
    for attr, val in [
        ("PRINT_PULSE_INFO", False),
        ("VZ_COMPENSATE", False),  # important: we don't need VZ if we use Z-invariant fidelity
    ]:
        if hasattr(core, attr):
            setattr(core, attr, val)


def set_rabi_cap(cap_MHz: float) -> None:
    """Set peak envelope cap in core (rad/s) using MHz input."""
    if hasattr(core, "set_rabi_cap_MHz"):
        core.set_rabi_cap_MHz(cap_MHz)
    else:
        core.RABI_CAP = twopi * float(cap_MHz) * 1e6


def _amp(psi: Qobj, a: int, b: int, c: int) -> complex:
    """Amplitude ⟨a,b,c|psi⟩ with basis order index = a*9 + b*3 + c."""
    v = np.array(psi.full()).ravel()
    return complex(v[a * 9 + b * 3 + c])


def z_invariant_fidelity_loopB(psi: Qobj) -> float:
    """Loop-B ideal: (|000> + |100>)/√2 ⇒ Fmax = (|A000| + |A100|)^2 / 2."""
    A000 = _amp(psi, 0, 0, 0)
    A100 = _amp(psi, 1, 0, 0)
    return float(0.5 * (abs(A000) + abs(A100)) ** 2)


def z_invariant_fidelity_loopA(psi: Qobj) -> float:
    """Loop-A ideal: (|000> + |010>)/√2 ⇒ Fmax = (|A000| + |A010|)^2 / 2."""
    A000 = _amp(psi, 0, 0, 0)
    A010 = _amp(psi, 0, 1, 0)
    return float(0.5 * (abs(A000) + abs(A010)) ** 2)


def run_loop_get_final(loop_id: str, dt: float = DT):
    """Run 3-pulse loop from |000> → return (psi_final, total_duration_ns)."""
    psi000 = tensor(basis(3, 0), basis(3, 0), basis(3, 0))
    # reset software phases if present
    if hasattr(core, "FRAME_PHASE"):
        core.FRAME_PHASE.update({'A': 0.0, 'B': 0.0, 'C': 0.0})
    segs = core.build_loop(loop_id, dt=dt)
    # IMPORTANT: do NOT apply any VZ to the state here; we want the raw evolution.
    t, _, psi_final = core.run_chain(psi000, segs, plot=False, print_seg=False, apply_vz_to_state=False)
    Tsum_ns = (t[-1] - t[0]) * 1e9 if t.size else 0.0
    return psi_final, Tsum_ns


# ---------- sweep ----------
def sweep_fidelity_vs_cap(caps_MHz: np.ndarray, dt: float = DT, show_plot: bool = True):
    set_core_quiet()
    caps_MHz = np.asarray(caps_MHz, dtype=float)

    F_B = np.zeros_like(caps_MHz)
    F_A = np.zeros_like(caps_MHz)
    T_B = np.zeros_like(caps_MHz)
    T_A = np.zeros_like(caps_MHz)

    def _progress(i):
        W = 28
        frac = (i + 1) / len(caps_MHz)
        bar = "█" * int(W * frac) + "·" * (W - int(W * frac))
        print(f"\r[sweep] cap = {caps_MHz[i]:6.2f} MHz  |{bar}| {100 * frac:5.1f}%", end="", flush=True)

    for i, cap in enumerate(caps_MHz):
        set_rabi_cap(cap)

        psiB, T_B[i] = run_loop_get_final("B", dt=dt)
        psiA, T_A[i] = run_loop_get_final("A", dt=dt)

        # Phase-optimized (Z-invariant) fidelities:
        F_B[i] = z_invariant_fidelity_loopB(psiB)
        F_A[i] = z_invariant_fidelity_loopA(psiA)

        _progress(i)
    print("")

    if show_plot:
        plt.figure(figsize=(8.6, 4.8), dpi=160)
        plt.plot(caps_MHz, F_B, marker='o', markersize=4, linewidth=1.1,
                 label='Loop-B: Z-invariant fidelity to { |000>, |100> }')
        plt.plot(caps_MHz, F_A, marker='s', markersize=4, linewidth=1.1,
                 label='Loop-A: Z-invariant fidelity to { |000>, |010> }')
        plt.xlabel(r'peak envelope $\Omega_{\max}/(2\pi)$ [MHz]')
        plt.ylabel('phase-optimized fidelity')
        plt.ylim(0.6, 1.01)
        plt.grid(True, alpha=0.35)
        plt.legend()
        plt.title('Final-state Z-invariant fidelity vs drive power (AB loops)')
        plt.tight_layout()
        plt.show()

    return {
        "caps_MHz": caps_MHz,
        "F_loopB_Zinv": F_B,
        "F_loopA_Zinv": F_A,
        "T_eff_loopB_ns": T_B,
        "T_eff_loopA_ns": T_A,
    }


if __name__ == "__main__":
    print("\n=== Sweep: Z-invariant final-state fidelity vs RABI cap (AB only) ===\n")
    res = sweep_fidelity_vs_cap(CAPS_MHZ, dt=DT, show_plot=True)

    print("\n cap(MHz)    Fz[B]     Fz[A]     T_B(ns)   T_A(ns)")
    for cap, fb, fa, tb, ta in zip(
            res["caps_MHz"], res["F_loopB_Zinv"], res["F_loopA_Zinv"],
            res["T_eff_loopB_ns"], res["T_eff_loopA_ns"]
    ):
        print(f"  {cap:7.2f}  {fb:8.5f}  {fa:8.5f}  {tb:7.1f}  {ta:7.1f}")