import numpy as np
import matplotlib.pyplot as plt

import heat as sim
from qutip import basis, tensor

# ------------------------- CONFIG -------------------------
MODE = "detune_heatmap"

DT = 0.25e-9

BASE_USE_DRAG = False
BASE_USE_STARK = False
BASE_USE_CROSS_STARK = False
BASE_IDLE_NS = 0.0  # seconds

DET_A_MHZ = np.linspace(-3.0, +3.0, 50)
DET_B_MHZ = np.linspace(-3.0, +3.0, 50)


# ------------------------- CORE -------------------------
def apply_baseline():
    sim.USE_DRAG = bool(BASE_USE_DRAG)
    sim.USE_STARK = bool(BASE_USE_STARK)
    sim.USE_CROSS_STARK = bool(BASE_USE_CROSS_STARK)
    sim.IDLE_NS = float(BASE_IDLE_NS)  # seconds


def run_both_loops_and_metrics():
    """Run both loops with current sim globals; return Δφ and extra metrics."""
    psi000 = tensor(basis(3, 0), basis(3, 0), basis(3, 0))

    # Loop B
    segs_B = sim.build_loop("B", dt=DT)
    _, _, psiB = sim.run_chain(psi000, segs_B, title="Loop-B (sweep)")
    phi_B = sim.rel_phase(psiB, (1, 0, 0), (0, 0, 0))  # |100> vs |000>

    # Loop A
    segs_A = sim.build_loop("A", dt=DT)
    _, _, psiA = sim.run_chain(psi000, segs_A, title="Loop-A (sweep)")
    phi_A = sim.rel_phase(psiA, (0, 1, 0), (0, 0, 0))  # |010> vs |000>

    dphi = sim.wrap_pi(phi_A - phi_B)
    return float(phi_A), float(phi_B), float(dphi)


# ------------------------- HEATMAP -------------------------
def heatmap_detunes():
    """
    2D sweep over EXTRA_LOWER_DETUNE_A (x-axis, MHz)
    and EXTRA_LOWER_DETUNE_B (y-axis, MHz).
    Plots |Δφ| in radians.
    """
    apply_baseline()

    X = DET_A_MHZ
    Y = DET_B_MHZ
    Z = np.zeros((len(Y), len(X)), dtype=float)

    twopi = 2.0 * np.pi

    best = {"abs_dphi": -1, "detA": None, "detB": None}

    for j, detB_mhz in enumerate(Y):
        # set B detune (rad/s)
        sim.EXTRA_LOWER_DETUNE_B = twopi * detB_mhz * 1e6
        for i, detA_mhz in enumerate(X):
            # set A detune (rad/s)
            sim.EXTRA_LOWER_DETUNE_A = twopi * detA_mhz * 1e6

            phi_A, phi_B, dphi = run_both_loops_and_metrics()
            Z[j, i] = abs(dphi)  # <-- absolute phase difference

    plt.figure(figsize=(6.4, 5.2))
    im = plt.imshow(
        Z, origin="lower", aspect="auto",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        cmap="magma"
    )
    plt.colorbar(im, label="|Δφ| [rad]")
    plt.xlabel("EXTRA_LOWER_DETUNE_A [MHz]")
    plt.ylabel("EXTRA_LOWER_DETUNE_B [MHz]")
    plt.title("Heatmap of |Δφ| vs lower-band detunes (A,B)")

    plt.tight_layout()
    plt.show()

    print(f"Max |Δφ| ≈ {best['abs_dphi']:.4f} rad at "
          f"detA={best['detA']:+.3f} MHz, detB={best['detB']:+.3f} MHz")


# ------------------------- MAIN -------------------------
def main():
    if MODE == "detune_heatmap":
        heatmap_detunes()
    else:
        raise ValueError("This script implements only MODE='detune_heatmap' for now.")


if __name__ == "__main__":
    main()