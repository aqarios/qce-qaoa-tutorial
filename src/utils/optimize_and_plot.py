from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from luna_quantum import Model, ResultView, Solution
from qiskit import QuantumCircuit
from scipy.optimize import minimize

from .sampling import cost_function, sample


def optimize_and_plot(
    get_circ: Callable[[int], QuantumCircuit],
    model: Model,
    ps: Sequence[int],
    shots: int = 1000,
) -> tuple[Solution, ResultView]:
    global current_it
    best = None
    final_solution = None
    result = None
    all_logs = {}

    # Set up the plot
    _, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Energy Expectation $\langle H_C \rangle$")
    ax.set_title("QAOA Optimization Progress for Different Circuit Depths")
    plt.draw()

    # Run QAOA for different numbers of layers
    for p in ps:
        print(f"=== Running optimization for p={p} layers ===")

        current_it = 1

        def progress_callback(*args):
            """Callback function to print optimization progress."""
            global current_it

            if current_it % 5 == 0:
                print(f"Iteration {current_it}: Energy = {all_logs[p][-1]:.3f}")
            current_it += 1

        # Create QAOA circuit with p layers
        circ = get_circ(p)
        all_logs[p] = []

        print(f"Circuit depth: {circ.depth()}, Parameters: {len(circ.parameters)}")

        # Run classical optimization
        initial_params = np.concat(
            [np.linspace(0.5, 0, p), np.linspace(0, 0.5, p)]
        )  # p beta parameters + p gamma parameters

        result = minimize(
            cost_function,  # Function to minimize
            initial_params,  # Initial parameters
            args=(model, circ, all_logs[p], shots),  # Additional arguments
            method="COBYLA",  # Optimization algorithm
            options={"rhobeg": 0.1},  # COBYLA-specific options
            callback=progress_callback,  # Progress monitoring
        )

        # Plot the optimization trajectory
        ax.plot(all_logs[p], label=f"p={p} ({len(all_logs[p])} iterations)")

        final_solution = sample(model, circ, result.x)
        best = final_solution.best()
        assert best is not None

        print("-------------------------")
        print(f"Energy: {result.fun:.3f}")
        print(f"Best objective value: {best.obj_value:.3f}")
        print()

    if final_solution is None or best is None or result is None:
        raise RuntimeError("No Feasible Solution Found.")

    print("\n=== Final Results ===")
    print(f"Best objective value: {best.obj_value:.3f}")
    print(f"Final energy: {result.fun:.3f}")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return final_solution, best
