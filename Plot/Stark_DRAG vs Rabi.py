# sweep_detuneVZ_vs_DRAG.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import core  # uses your current core.py


def wrap_pi(x: float) -> float:
    return (x + np.pi) % (2 * np.pi) - np.pi


def measure_loop_phase_with_detune_VZ(loop_id: str, dt: float = 0.25e-11) -> float:
    """
    Run one loop and return the loop's diagnostic phase with detune-VZ
    enabled per pulse. Any DRAG settings in core.* are honored.
    """
    # fresh frames and settings for a clean run
    core.FRAME_PHASE = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    core.VZ_COMPENSATE = False  # never use build-time frame advances
    segs = core.build_loop(loop_id, dt=dt)
    psi0 = core.tensor(core.basis(3, 0), core.basis(3, 0), core.basis(3, 0))
    _, _, psi = core.run_chain(
        psi0, segs,
        plot=False, print_seg=False,
        apply_vz_to_state=True  # <-- per-pulse detune-VZ applied to the state
    )
    if loop_id.upper() == "B":
        # Loop-B → read phase on A: arg|100> − arg|000>
        return core.wrap_pi(float(np.angle(core.amp(psi, 1, 0, 0)) - np.angle(core.amp(psi, 0, 0, 0))))
    else:
        # Loop-A → read phase on B: arg|010> − arg|000>
        return core.wrap_pi(float(np.angle(core.amp(psi, 0, 1, 0)) - np.angle(core.amp(psi, 0, 0, 0))))


def measure_asymmetry_detVZ(dt: float = 0.25e-11) -> float:
    """Return |Δφ| with detune-VZ only, honoring current core.* (e.g., DRAG on/off)."""
    phi_B = measure_loop_phase_with_detune_VZ('B', dt=dt)  # read on A
    phi_A = measure_loop_phase_with_detune_VZ('A', dt=dt)  # read on B
    return abs(wrap_pi(phi_A - phi_B))


def main():
    # Quiet down per-pulse prints during the sweep
    core.PRINT_PULSE_INFO = False
    core.VZ_COMPENSATE = False  # we only apply VZ dynamically via apply_vz_to_state
    dt = 0.25e-11

    # --- sweep settings ---
    caps_MHz = np.linspace(2, 60, 30)  # Ω_cap/(2π) in MHz
    betas = [0.25, 0.50, 0.75, 1.00, 1.25]  # DRAG coefficients to compare (target-only)

    # Storage
    dphi_detVZ_only = np.zeros_like(caps_MHz, dtype=float)
    dphi_detVZ_plus_drag = {beta: np.zeros_like(caps_MHz, dtype=float) for beta in betas}

    def _progress(i, total, note=""):
        W = 28
        frac = (i + 1) / total
        bar = "█" * int(W * frac) + "·" * (W - int(W * frac))
        print(f"\r[sweep] {note:>10}  |{bar}| {100 * frac:5.1f}%", end="", flush=True)

    # --- baseline: detune-VZ only (no DRAG) ---
    core.USE_DRAG = False
    for i, cap in enumerate(caps_MHz):
        core.RABI_CAP = 2 * np.pi * cap * 1e6
        dphi_detVZ_only[i] = measure_asymmetry_detVZ(dt=dt)
        _progress(i, len(caps_MHz), "no-DRAG")
    print("")

    # --- detune-VZ + DRAG for several β values (target-only by default) ---
    core.DRAG_TARGET_ONLY = False
    core.USE_DRAG = True
    for beta in betas:
        core.BETA_DRAG = float(beta)
        for i, cap in enumerate(caps_MHz):
            core.RABI_CAP = 2 * np.pi * cap * 1e6
            dphi_detVZ_plus_drag[beta][i] = measure_asymmetry_detVZ(dt=dt)
            _progress(i, len(caps_MHz), f"β={beta:>4.2f}")
        print("")

    # reset DRAG to a safe default after the sweep
    core.USE_DRAG = False

    # --- Plot ---
    plt.figure(figsize=(8.2, 4.9), dpi=160)
    # Baseline
    plt.plot(caps_MHz, dphi_detVZ_only, marker='o', linewidth=1.6, label='detune-VZ (no DRAG)')
    # DRAG variants
    for beta in betas:
        plt.plot(
            caps_MHz, dphi_detVZ_plus_drag[beta],
            marker='s', linewidth=1.2, label=fr'detune-VZ + DRAG $\beta={beta:.2f}$'
        )

    plt.xlabel(r'Rabi cap $\Omega_{\max}/(2\pi)$ [MHz]')
    plt.ylabel(r'Loop asymmetry $|\Delta\phi|$ (rad)')
    plt.title('Loop phase asymmetry vs Rabi cap:\n'
              'detune-VZ only vs detune-VZ + DRAG')
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()