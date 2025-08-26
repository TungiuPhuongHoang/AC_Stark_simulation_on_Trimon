# P_leak_vs_RABI_cap_DRAG.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor
import core as core  # your simulator

twopi = 2 * np.pi

# ---- sweep config ----
CAPS_MHZ = np.linspace(2.0, 100.0, 36)  # Ω_max/(2π) MHz sweep
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]  # DRAG coefficients β
DT = 0.25e-11  # time step to build segments (match your core usage)


def _measure_leak_for_cap(cap_MHz: float, dt: float = DT):
    """
    Set peak envelope cap Ω_cap/(2π)=cap_MHz (area-locked; duration adapts),
    run both loops (B and A), and return final P_leak and total duration for each.
    """
    # Set global cap (rad/s); keep rotation angles fixed (area-locked)
    core.RABI_CAP = twopi * cap_MHz * 1e6
    psi000 = tensor(basis(3, 0), basis(3, 0), basis(3, 0))

    # No virtual-Z compensation during the sweep
    core.VZ_COMPENSATE = False

    # Loop-B: B upper π/2 → A lower π → B lower π
    core.FRAME_PHASE.update({'A': 0.0, 'B': 0.0, 'C': 0.0})
    segs_B = core.build_loop("B", dt=dt)
    tB, _, psiB = core.run_chain(psi000, segs_B, plot=False, print_seg=False)
    Tsum_B_ns = (tB[-1] - tB[0]) * 1e9 if tB.size else 0.0
    _, _, _, _, Pleak_B = core.pops_comp_subspace(psiB)

    # Loop-A: A upper π/2 → B lower π → A lower π
    core.FRAME_PHASE.update({'A': 0.0, 'B': 0.0, 'C': 0.0})
    segs_A = core.build_loop("A", dt=dt)
    tA, _, psiA = core.run_chain(psi000, segs_A, plot=False, print_seg=False)
    Tsum_A_ns = (tA[-1] - tA[0]) * 1e9 if tA.size else 0.0
    _, _, _, _, Pleak_A = core.pops_comp_subspace(psiA)

    return Pleak_B, Pleak_A, Tsum_B_ns, Tsum_A_ns


def sweep_Pleak_vs_power_for_betas(caps_MHz: np.ndarray, betas: list[float], dt: float = DT, show_plot: bool = True):
    """
    For each DRAG β, sweep Ω_cap/(2π) in MHz (area-locked) and measure P_leak for both loops.
    Returns a dict with arrays indexed as [beta_index, cap_index].
    """
    caps_MHz = np.asarray(caps_MHz, dtype=float)
    nB = len(betas)
    nC = caps_MHz.size

    Pleak_B = np.zeros((nB, nC))
    Pleak_A = np.zeros((nB, nC))
    Tsum_B = np.zeros((nB, nC))
    Tsum_A = np.zeros((nB, nC))

    # make core quiet if the flags exist
    if hasattr(core, "PRINT_PULSE_INFO"):
        core.PRINT_PULSE_INFO = False
    if hasattr(core, "PLOT"):
        core.PLOT = False

    for bi, beta in enumerate(betas):
        # configure DRAG for this beta
        if hasattr(core, "USE_DRAG"):
            core.USE_DRAG = True  # keep True for all; β=0 is the no-DRAG line
        if hasattr(core, "BETA_DRAG"):
            core.BETA_DRAG = float(beta)

        # progress per beta
        print(f"\n=== DRAG β = {beta:.2f} ===")
        for ci, cap in enumerate(caps_MHz):
            # simple inline progress
            W = 30
            frac = (ci + 1) / nC
            bar = "█" * int(W * frac) + "·" * (W - int(W * frac))
            print(f"\r  sweep Ω_cap/(2π) = {cap:6.2f} MHz  |{bar}| {100 * frac:5.1f}%", end="", flush=True)

            pB, pA, tB, tA = _measure_leak_for_cap(cap, dt=dt)
            Pleak_B[bi, ci] = pB
            Pleak_A[bi, ci] = pA
            Tsum_B[bi, ci] = tB
            Tsum_A[bi, ci] = tA
        print("")  # newline per beta

    results = {
        "caps_MHz": caps_MHz,
        "betas": np.array(betas, dtype=float),
        "Pleak_loopB": Pleak_B,  # shape (nbetas, ncaps)
        "Pleak_loopA": Pleak_A,  # shape (nbetas, ncaps)
        "T_eff_loopB_ns": Tsum_B,  # shape (nbetas, ncaps)
        "T_eff_loopA_ns": Tsum_A,  # shape (nbetas, ncaps)
    }

    if show_plot:
        # ---- Plot leakage vs cap for each beta (two panels: Loop-B and Loop-A) ----
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=180, sharex=True, sharey=True)
        axB, axA = axes

        for bi, beta in enumerate(betas):
            label = fr'β={beta:g}'
            axB.plot(caps_MHz, Pleak_B[bi], marker='o', markersize=3.5, linewidth=1.0, label=label)
            axA.plot(caps_MHz, Pleak_A[bi], marker='o', markersize=3.5, linewidth=1.0, label=label)

        for ax, ttl in [(axB, 'Loop-B leakage (final $P_{\\rm leak}$)'),
                        (axA, 'Loop-A leakage (final $P_{\\rm leak}$)')]:
            ax.set_xlabel(r'peak envelope $\Omega_{\max}/(2\pi)$ [MHz]')
            ax.set_ylabel(r'$P_{\rm leak}$')
            ax.grid(True, alpha=0.35)
            ax.set_ylim(-0.01, 0.1)  # adjust if your leakage is smaller/larger
            ax.legend(title='DRAG β', fontsize=9)

        fig.suptitle('Final leakage vs drive power for different DRAG coefficients', y=0.98)
        plt.tight_layout()
        plt.show()

        # ---- Optional: total duration vs cap (just for β=0 and β=1 for readability) ----
        pick = [0, min(4, len(betas) - 1)]  # indices for β=0 and (ideally) β=1.0
        fig2, ax2 = plt.subplots(figsize=(8.2, 4.4), dpi=180)
        for i in pick:
            ax2.plot(caps_MHz, Tsum_B[i], linewidth=1.3, label=f'Loop-B, β={betas[i]:g}')
            ax2.plot(caps_MHz, Tsum_A[i], linewidth=1.3, linestyle='--', label=f'Loop-A, β={betas[i]:g}')
        ax2.set_xlabel(r'peak envelope $\Omega_{\max}/(2\pi)$ [MHz]')
        ax2.set_ylabel('total loop duration [ns]')
        ax2.set_title('Total 3-pulse duration vs cap (area-locked)')
        ax2.grid(True, alpha=0.35)
        ax2.legend()
        plt.tight_layout()
        plt.show()

    return results


if __name__ == "__main__":
    print("\n=== Sweep: final leakage vs RABI cap for multiple DRAG β ===")
    res = sweep_Pleak_vs_power_for_betas(CAPS_MHZ, BETAS, dt=DT, show_plot=True)

    # Compact table for β=0 and β=1 as a quick sanity check
    betas = res["betas"]
    caps = res["caps_MHz"]
    PleakB = res["Pleak_loopB"]
    PleakA = res["Pleak_loopA"]


    # find indices for β=0 and β=1 (if present)
    def _find_beta(target):
        j = np.argmin(np.abs(betas - target))
        return j


    j0 = _find_beta(0.0)
    j1 = _find_beta(1.0)

    print("\n(cap MHz)   Pleak_B[β=0]  Pleak_A[β=0]   ||   Pleak_B[β=1]  Pleak_A[β=1]")
    for c, pB0, pA0, pB1, pA1 in zip(caps, PleakB[j0], PleakA[j0], PleakB[j1], PleakA[j1]):
        print(f" {c:8.2f}     {pB0:9.6f}   {pA0:9.6f}      {pB1:9.6f}   {pA1:9.6f}")