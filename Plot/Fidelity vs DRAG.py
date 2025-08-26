# Fidelity_vs_RABI_cap_AB_ZInvariant_DRAGs.py
# Sweep RABI cap and compute phase-optimized (Z-invariant) state fidelity
# for several DRAG coefficients β:
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
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]  # DRAG coefficients to plot
DRAG_TARGET_ONLY = True  # True: DRAG only on the addressed (target) branch


# ---------- utilities ----------
def set_core_quiet() -> None:
    """Make core not print/plot and never apply VZ during sweeps."""
    for attr, val in [
        ("PRINT_PULSE_INFO", False),
        ("VZ_COMPENSATE", False),  # Z-invariant fidelity ⇒ no need for VZ
        ("PLOT", False),
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
    if hasattr(core, "FRAME_PHASE"):
        core.FRAME_PHASE.update({'A': 0.0, 'B': 0.0, 'C': 0.0})
    segs = core.build_loop(loop_id, dt=dt)
    # IMPORTANT: no VZ to state; raw evolution only
    t, _, psi_final = core.run_chain(psi000, segs, plot=False, print_seg=False, apply_vz_to_state=False)
    Tsum_ns = (t[-1] - t[0]) * 1e9 if t.size else 0.0
    return psi_final, Tsum_ns


# ---------- sweep ----------
def sweep_fidelity_vs_cap_and_drag(
        caps_MHz: np.ndarray,
        betas: list[float],
        dt: float = DT,
        target_only: bool = DRAG_TARGET_ONLY,
        show_plot: bool = True,
):
    set_core_quiet()
    caps_MHz = np.asarray(caps_MHz, dtype=float)

    nb = len(betas)
    nc = caps_MHz.size
    F_B = np.zeros((nb, nc))
    F_A = np.zeros((nb, nc))
    T_B = np.zeros((nb, nc))
    T_A = np.zeros((nb, nc))

    # Fix DRAG target-only/all-branches behavior
    if hasattr(core, "DRAG_TARGET_ONLY"):
        core.DRAG_TARGET_ONLY = bool(target_only)

    def _progress(i_beta, i_cap):
        W = 28
        idx = i_beta * nc + i_cap + 1
        total = nb * nc
        frac = idx / total
        bar = "█" * int(W * frac) + "·" * (W - int(W * frac))
        print(f"\r[sweep] β={betas[i_beta]:>4.2f}  cap = {caps_MHz[i_cap]:6.2f} MHz  |{bar}| {100 * frac:5.1f}%",
              end="", flush=True)

    for ib, beta in enumerate(betas):
        # Configure DRAG for this β
        if hasattr(core, "USE_DRAG"):
            core.USE_DRAG = (beta != 0.0)  # disable for β=0 for a true baseline
        if hasattr(core, "BETA_DRAG"):
            core.BETA_DRAG = float(beta)

        for ic, cap in enumerate(caps_MHz):
            set_rabi_cap(cap)

            psiB, T_B[ib, ic] = run_loop_get_final("B", dt=dt)
            psiA, T_A[ib, ic] = run_loop_get_final("A", dt=dt)

            # Phase-optimized (Z-invariant) fidelities:
            F_B[ib, ic] = z_invariant_fidelity_loopB(psiB)
            F_A[ib, ic] = z_invariant_fidelity_loopA(psiA)

            _progress(ib, ic)
    print("")

    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=160, sharex=True, sharey=True)

        # Loop-B
        ax = axes[0]
        for ib, beta in enumerate(betas):
            ax.plot(caps_MHz, F_B[ib], marker='o', markersize=3.5, linewidth=1.0, label=f'β={beta:g}')
        ax.set_title('Loop-B: Z-invariant fidelity to { |000>, |100> }')
        ax.set_xlabel(r'peak envelope $\Omega_{\max}/(2\pi)$ [MHz]')
        ax.set_ylabel('phase-optimized fidelity')
        ax.grid(True, alpha=0.35)
        ax.set_ylim(0.6, 1.01)
        ax.legend(title='DRAG')

        # Loop-A
        ax = axes[1]
        for ib, beta in enumerate(betas):
            ax.plot(caps_MHz, F_A[ib], marker='s', markersize=3.5, linewidth=1.0, label=f'β={beta:g}')
        ax.set_title('Loop-A: Z-invariant fidelity to { |000>, |010> }')
        ax.set_xlabel(r'peak envelope $\Omega_{\max}/(2\pi)$ [MHz]')
        ax.grid(True, alpha=0.35)
        ax.set_ylim(0.6, 1.01)
        ax.legend(title='DRAG')

        plt.tight_layout()
        plt.show()

    return {
        "caps_MHz": caps_MHz,
        "betas": np.array(betas, dtype=float),
        "F_loopB_Zinv": F_B,  # shape (nbeta, ncap)
        "F_loopA_Zinv": F_A,  # shape (nbeta, ncap)
        "T_eff_loopB_ns": T_B,  # shape (nbeta, ncap)
        "T_eff_loopA_ns": T_A,  # shape (nbeta, ncap)
    }


if __name__ == "__main__":
    print("\n=== Sweep: Z-invariant final-state fidelity vs RABI cap (AB) for multiple DRAG β ===\n")
    res = sweep_fidelity_vs_cap_and_drag(CAPS_MHZ, BETAS, dt=DT, target_only=DRAG_TARGET_ONLY, show_plot=True)

    print("\n cap(MHz)  " + "  ".join([f"Fz[B] β={b:g}" for b in BETAS]))
    for i_cap, cap in enumerate(res["caps_MHz"]):
        row = "  ".join([f"{res['F_loopB_Zinv'][i_b, i_cap]:8.5f}" for i_b in range(len(BETAS))])
        print(f"  {cap:7.2f}  {row}")

    print("\n cap(MHz)  " + "  ".join([f"Fz[A] β={b:g}" for b in BETAS]))
    for i_cap, cap in enumerate(res["caps_MHz"]):
        row = "  ".join([f"{res['F_loopA_Zinv'][i_b, i_cap]:8.5f}" for i_b in range(len(BETAS))])
        print(f"  {cap:7.2f}  {row}")