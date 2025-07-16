from aqmodels import Solution
from qiskit.circuit.gate import np


def get_metrics(
    solution: Solution, optimal: float | None = None, circuit_depth: int | None = None
) -> dict[str, float]:
    """Computes the metrics for a given solution.

    Parameters
    ----------
    solution: Solution
        The solution output from the algorithm excecution.
    optimal: float | None
        The optimal value of the optimization problem. If none provided 'p_opt', 'p_90'
        and 'tts' cannot be computed.
    circuit_depth: float | None
        The circuit depth of the algorithm. If none provided 'tts' cannot be computed.

    Returns
    -------
    dict[str, float]
        Metrics dict, featuring 'approximation_ratio', 'best_found',
        'feasibility_ratio', 'p_opt', 'p_90' and 'tts'
    """

    # Compute feasibility ratio and approximation ratio of feasible samples
    feas = solution.feasibility_ratio()
    feasible_solutions = solution.filter_feasible()
    ar = feasible_solutions.expectation_value()

    metrics = {
        "approximation_ratio": ar,
        "feasibility_ratio": feas,
    }

    # Find best solution
    best_found = solution.best()
    if best_found is not None and best_found.obj_value is not None:
        metrics["best_found"] = best_found.obj_value

    # Compute p_opt and p_90
    total = solution.counts.sum()
    if optimal is not None:
        metrics["p_opt"] = (
            feasible_solutions.counts[
                np.isclose(feasible_solutions.obj_values, optimal)
            ].sum()
            / total
        )
        metrics["p_90"] = (
            feasible_solutions.counts[
                np.abs(feasible_solutions.obj_values - optimal) / np.abs(optimal) < 0.1
            ].sum()
            / total
        )

    # Compute TTS
    p_opt = metrics.get("p_opt", None)
    if p_opt is not None and circuit_depth is not None:
        if p_opt == 0:
            tts = np.inf
        elif p_opt == 1:
            tts = circuit_depth
        else:
            tts = circuit_depth * np.ceil(np.log(0.01) / np.log(1 - p_opt))
        metrics["tts"] = tts

    return metrics
