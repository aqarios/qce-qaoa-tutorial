from aqmodels import Model, Solution
from qiskit import QuantumCircuit
from qiskit.circuit.gate import np
from qiskit_aer.primitives import SamplerV2

sampler = SamplerV2()


def sample(model: Model, qc: QuantumCircuit, x: np.ndarray, shots=1000):
    """Execute quantum circuit and evaluate the objective function.

    Args:
        model: Optimization model
        qc: Parameterized quantum circuit
        x: Parameter values [beta_0, beta_1, ..., gamma_0, gamma_1, ...]
        shots: Number of quantum measurements

    Returns:
        Solution object with samples and objective values
    """
    # Assign parameter values and execute circuit
    qc_assigned = qc.assign_parameters({p: v for p, v in zip(qc.parameters, x)})
    pub = (qc_assigned,)
    result = sampler.run([pub], shots=shots).result()

    return Solution.from_counts(result[0].data.meas.get_counts(), model=model)  # type: ignore


def cost_function(
    x: np.ndarray,
    model: Model,
    qc: QuantumCircuit,
    logs: list | None = None,
    shots: int = 1000,
):
    """Cost function for classical optimization.

    This function is called repeatedly by the classical optimizer.

    Args:
        x: Current parameter values
        model: Optimization model
        qc: Quantum circuit
        logs: Optional list to store optimization history
        shots: Number of quantum measurements

    Returns:
        float: Expected value of the objective function
    """
    # Get quantum measurement results and evaluate objective
    sol = sample(model, qc, x, shots=shots)

    # Calculate expectation value (average objective value across all samples)
    energy = sol.expectation_value()

    # Log the approximation ration if logging is enabled
    if logs is not None:
        logs.append(energy)

    return energy
