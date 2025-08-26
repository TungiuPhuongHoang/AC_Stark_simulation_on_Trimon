# sweep_stark_vs_cap.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import core  # uses your current core.py


def wrap_pi(x: float) -> float:
    return (x + np.pi) % (2 * np.pi) - np.pi


def measure_loop_phase_with_detune_VZ(loop_id: str, dt: float = 0.25e-11) -> float:
    """Run one loop and return the loop's diagnostic phase with detune-VZ enabled per pulse."""
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
    # match the diagnostics in core.py:
    if loop_id.upper() == "B":
        # Loop-B → read phase on A: arg|100> − arg|000>
        return core.wrap_pi(float(np.angle(core.amp(psi, 1, 0, 0)) - np.angle(core.amp(psi, 0, 0, 0))))
    else:
        # Loop-A → read phase on B: arg|010> − arg|000>
        return core.wrap_pi(float(np.angle(core.amp(psi, 0, 1, 0)) - np.angle(core.amp(psi, 0, 0, 0))))


def calc_stark_only_asymmetry(dt: float = 0.25e-11) -> float:
    """
    Compute loop-level Stark-only asymmetry using the segment metadata produced by build_segment.
    - Loop-B: sum Stark on branch A across its segments
    - Loop-A: sum Stark on branch B across its segments
    Return | wrap_pi( φ_A_from_LoopA  −  φ_A_from_LoopB ) | with the matching branches.
    """
    # Build segments (no dynamics needed to read metadata)
    segs_B = core.build_loop("B", dt=dt)
    segs_A = core.build_loop("A", dt=dt)

    # Sum Stark phases for the measured branch in each loop
    S_B_onA = 0.0
    for seg in segs_B:
        if len(seg) == 3:
            info = seg[2]
            S_B_onA += float(info['stark'].get('A', 0.0))

    S_A_onB = 0.0
    for seg in segs_A:
        if len(seg) == 3:
            info = seg[2]
            S_A_onB += float(info['stark'].get('B', 0.0))

    # Loop-asymmetry (match how we compare loop phases in core main)
    dphi_stark = wrap_pi(S_A_onB - S_B_onA)
    return abs(dphi_stark)


def main():
    # Quiet down per-pulse prints during the sweep
    core.PRINT_PULSE_INFO = False
    core.VZ_COMPENSATE = False  # we only apply VZ dynamically via apply_vz_to_state
    dt = 0.25e-11

    # Rabi cap sweep (MHz)
    caps_MHz = np.linspace(2, 50, 30)  # 4,6,...,40 MHz
    dphi_stark_calc = []
    dphi_meas_detVZ = []
    dphi_stark_VZ = []

    for cap in caps_MHz:
        # set cap in rad/s
        core.RABI_CAP = 2 * np.pi * cap * 1e6

        # --- calculated Stark-only asymmetry
        dphi_calc = calc_stark_only_asymmetry(dt=dt)
        dphi_stark_calc.append(dphi_calc)

        # --- measured asymmetry with detune-VZ (per-pulse)
        phi_B = measure_loop_phase_with_detune_VZ('B', dt=dt)  # read on A
        phi_A = measure_loop_phase_with_detune_VZ('A', dt=dt)  # read on B
        dphi_meas = abs(wrap_pi(phi_A - phi_B))
        dphi_meas_detVZ.append(dphi_meas)

        dphi_stark_VZ.append(dphi_meas-dphi_calc)

        print(f"cap={cap:5.1f} MHz  |  Δφ_stark(calc)={dphi_calc: .6f} rad   "
              f"Δφ_meas(detune-VZ)={dphi_meas - dphi_calc: .6f} rad")

    # --- Plot ---
    plt.figure(figsize=(7.6, 4.6), dpi=160)
    plt.plot(caps_MHz, dphi_stark_VZ, marker='o', label='Simulated Δφ with detune & Stark VZ')
    plt.plot(caps_MHz, dphi_meas_detVZ, marker='s', label='Simulated Δφ with detune-VZ')
    plt.xlabel('Rabi cap (MHz)')
    plt.ylabel('Loop asymmetry Δφ (rad)')
    plt.title('Loop phase asymmetry vs Rabi cap\n(detune-VZ only vs total-VZ)')
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()