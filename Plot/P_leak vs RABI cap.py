# P_leak vs RABI cap.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor
import core as core  # your simulator

twopi = 2 * np.pi


def _measure_leak_for_cap(cap_MHz: float, dt: float = 0.25e-11):
    """
    Set peak envelope cap Ω_cap/(2π)=cap_MHz (area-locked; duration adapts),
    run both loops (B and A), and return final P_leak and total duration for each.
    """
    # Set the global cap (rad/s); keep rotation angles fixed (area-locked)
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


def sweep_Pleak_vs_power(caps_MHz: np.ndarray, dt: float = 0.25e-11, show_plot: bool = True):
    """
    Sweep Ω_cap/(2π) in MHz (area-locked) and measure P_leak for both loops.
    Returns a dict of arrays and optionally makes plots.
    """
    caps_MHz = np.asarray(caps_MHz, dtype=float)
    n = caps_MHz.size
    Pleak_B = np.zeros(n)
    Pleak_A = np.zeros(n)
    Tsum_B = np.zeros(n)
    Tsum_A = np.zeros(n)

    # Simple text progress bar
    def _progress(i):
        W = 30
        frac = (i + 1) / n
        bar = "█" * int(W * frac) + "·" * (W - int(W * frac))
        print(f"\r[sweep] Ω_cap/(2π) = {caps_MHz[i]:6.2f} MHz  |{bar}| {100 * frac:5.1f}%",
              end="", flush=True)

    for i, cap in enumerate(caps_MHz):
        Pleak_B[i], Pleak_A[i], Tsum_B[i], Tsum_A[i] = _measure_leak_for_cap(cap, dt=dt)
        _progress(i)
    print("")  # newline

    Pleak_mean = 0.5 * (Pleak_B + Pleak_A)
    results = {
        "caps_MHz": caps_MHz,
        "Pleak_loopB": Pleak_B,
        "Pleak_loopA": Pleak_A,
        "Pleak_mean": Pleak_mean,
        "T_eff_loopB_ns": Tsum_B,
        "T_eff_loopA_ns": Tsum_A,
    }

    if show_plot:
        # Leakage vs power
        fig = plt.figure(figsize=(8.8, 5.0), dpi=200)
        ax = plt.gca()
        ax.plot(caps_MHz, Pleak_B, marker='o', markersize=4, markeredgewidth=0.0, linewidth=0.8, label='Loop-B')
        ax.plot(caps_MHz, Pleak_A, marker='s', markersize=4, markeredgewidth=0.0, linewidth=0.8, label='Loop-A')
        ax.plot(caps_MHz, Pleak_mean, marker='^', markersize=4, markeredgewidth=0.0, linewidth=0.8, label='Mean',
                alpha=0.9)
        ax.set_xlabel(r'peak envelope $\Omega_{\max}/(2\pi)$ [MHz]')
        ax.set_ylabel(r'$P_{\rm leak}$ (final)')
        ax.set_title('Leakage vs drive power (area-locked via duration)')
        ax.grid(True, alpha=0.35)
        ax.legend()
        plt.tight_layout()
        plt.show()

        # Total (3-pulse) duration vs power — useful to see the area-lock tradeoff
        fig2 = plt.figure(figsize=(8.2, 4.4), dpi=200)
        ax2 = plt.gca()
        ax2.plot(caps_MHz, Tsum_B, linewidth=1.4, label='Loop-B duration')
        ax2.plot(caps_MHz, Tsum_A, linewidth=1.4, label='Loop-A duration')
        ax2.set_xlabel(r'peak envelope $\Omega_{\max}/(2\pi)$ [MHz]')
        ax2.set_ylabel('total loop duration [ns]')
        ax2.set_title('Total 3-pulse duration vs cap (area-locked)')
        ax2.grid(True, alpha=0.35)
        ax2.legend()
        plt.tight_layout()
        plt.show()

    return results


if __name__ == "__main__":
    # Choose the sweep range (Ω_cap/(2π) in MHz).
    # Area-locked: higher cap → shorter pulses → usually more leakage.
    caps_MHz = np.linspace(2.0, 100.0, 50)
    print("\n=== Sweep: P_leak vs drive power ===")
    print("Area-locked: target rotations fixed; duration adapts to respect the chosen cap.\n")
    res = sweep_Pleak_vs_power(caps_MHz, dt=0.25e-11, show_plot=True)

    # Compact table
    print(" cap(MHz)   Pleak[B]   Pleak[A]   Pleak[mean]   T_B(ns)   T_A(ns)")
    for cap, pb, pa, pm, tb, ta in zip(
            res["caps_MHz"], res["Pleak_loopB"], res["Pleak_loopA"], res["Pleak_mean"],
            res["T_eff_loopB_ns"], res["T_eff_loopA_ns"]
    ):
        print(f"  {cap:7.2f}   {pb:8.5f}   {pa:8.5f}   {pm:10.5f}   {tb:7.1f}   {ta:7.1f}")