import numpy as np
from luna_quantum import HigherOrder, Linear, Model, Quadratic, Vtype
from luna_quantum.transformations import IR
from qiskit.circuit import Parameter, QuantumCircuit


def cost_layer(model: Model, gamma: Parameter | float, scale: float = 1.0):
    """Implements a single QAOA cost operation.

    Parameters
    ----------
    model: Model
        The optimization model containing the objective function
    gamma: Parameter | Float
        Cost parameter (controls problem-specific rotations)

    Returns
    -------
    QuantumCircuit
        A single QAOA cost layer
    """
    qc = QuantumCircuit(len(model.variables()))

    # Mapping from variables to qubit indices
    varmap = {v: i for i, v in enumerate(model.variables())}

    for item, val in model.objective.items():
        angle = val * gamma * scale
        match item:
            case Linear(v):
                # Linear terms: single-qubit phase gate
                qc.p(angle, varmap[v])
            case Quadratic(v1, v2):
                # Quadratic terms: two-qubit controlled-phase gate applied to |11>
                qc.cp(angle, varmap[v1], varmap[v2])
            case HigherOrder(vs):
                # HigherOrder terms: muti-qubit controlled-phase gate applied to |11..11>
                qc.mcp(angle, [varmap[v] for v in vs[:-1]], varmap[vs[-1]])

    return qc


def rx_mixer_layer(num: int, beta: Parameter | float):
    """Implements a single QAOA mixer operation.

    Parameters
    ----------
    num: int
        The number of qubits
    gamma: Parameter | Float
        Cost parameter (controls problem-specific rotations)

    Returns
    -------
    QuantumCircuit
        A single QAOA cost layer
    """
    qc = QuantumCircuit(num)
    qc.rx(2 * beta, qc.qubits)
    return qc


def qaoa_layer(
    model: Model, beta: Parameter | float, gamma: Parameter | float, scale: float = 1.0
) -> QuantumCircuit:
    """Implements a single QAOA layer consisting of cost and mixer operations.

    Parameters
    ----------
    model: Model
        The optimization model containing the objective function
    beta: Parameter | Float
        Mixer parameter (controls X-rotations)
    gamma: Parameter | Float
        Cost parameter (controls problem-specific rotations)

    Returns
    -------
    QuantumCircuit
        A single QAOA layer
    """

    qc = cost_layer(model, gamma, scale=scale).compose(
        rx_mixer_layer(len(model.variables()), beta)
    )

    assert qc is not None

    return qc


def qaoa_layer_spin(
    model: Model, beta: Parameter | float, gamma: Parameter | float
) -> QuantumCircuit:
    """Implements a single QAOA layer consisting of cost and mixer operations.

    Args:
        model: The optimization model containing the objective function
        beta: Mixer parameter (controls X-rotations)
        gamma: Cost parameter (controls problem-specific rotations)

    Returns:
        QuantumCircuit: A single QAOA layer
    """
    # Quantum circuit with one qubit per variable
    qc = QuantumCircuit(len(model.variables()))
    # Mapping from variables to qubit indices
    varmap = {v: i for i, v in enumerate(model.variables())}

    for item, val in model.objective.items():
        match item:
            case Quadratic(v1, v2):
                qc.rzz(-2 * val * gamma, varmap[v1], varmap[v2])

    qc.rx(2 * beta, qc.qubits)

    return qc


def qaoa_circ(model: Model, reps: int = 1, scale: float = 1.0) -> QuantumCircuit:
    """Generates a complete QAOA circuit with specified number of layers.

    Parameters
    ----------
    ir: IR
        The intermediate representation containing the transformed objective function
        and model metadata
    reps: int
        Number of QAOA layers (p parameter)

    Returns
    -------
    QuantumCircuit
        Complete QAOA circuit with measurements
    """
    # Verify the model is suitable for QAOA
    assert model.num_constraints == 0, "QAOA requires unconstrained problems"
    assert all(v.vtype == Vtype.Binary for v in model.variables()), (
        "All variables must be binary"
    )

    # Initialize quantum circuit
    qc = QuantumCircuit(len(model.variables()))

    # Start with equal superposition state |+⟩^⊗n
    qc.h(qc.qubits)

    # Create parameter templates for a single layer
    beta = Parameter("beta")
    gamma = Parameter("gamma")
    layer_circ = qaoa_layer(model, beta, gamma, scale=scale)

    # Create individual parameters for each layer
    betas = [Parameter(f"beta_{p}") for p in range(reps)]
    gammas = [Parameter(f"gamma_{p}") for p in range(reps)]

    # Add all QAOA layers
    for beta_p, gamma_p in zip(betas, gammas):
        # Substitute the template parameters with layer-specific ones
        layer_instance = layer_circ.assign_parameters({beta: beta_p, gamma: gamma_p})
        qc = qc.compose(layer_instance)
        assert qc is not None

    # Add measurements to all qubits
    qc.measure_all()
    return qc


# =====================================================================================


def w_state(num: int):
    """Implements sequential W-state construction.

    Parameters
    ----------
    num: int
        The number of qubits that are part of the W-state.

    Returns
    -------
    QuantumCircuit
    """
    qc = QuantumCircuit(num)
    assert num >= 2
    if num == 2:
        qc.h(0)
        qc.x(1)
        qc.cx(0, 1)
    else:
        qc.x(0)
        for i in range(num - 1):
            angle = 2 * np.arccos(np.sqrt(1 / (num - i)))
            qc.cry(angle, i, i + 1)
            qc.cx(i + 1, i)
    return qc


def initial_state(ir: IR):
    """Implements the initial state for an arbitrary IR.

    Parameters
    ----------
    ir: IR
        The intermediate representation of the model transformation.

    Returns
    -------
    QuantumCircuit
    """
    qc = QuantumCircuit(len(ir.model.variables()))
    remove_one_hot = ir.cache["remove-one-hot"]
    if remove_one_hot is None:
        qc.h(qc.qubits)
    else:
        varmap = {v: i for i, v in enumerate(ir.model.variables())}
        one_hots = remove_one_hot["removed"]
        one_hot_vars = set(i for v in one_hots.values() for i in v)

        # Normal initial state
        if qc.num_qubits - len(one_hot_vars) > 0:
            qc.h([i for v, i in varmap.items() if v not in one_hot_vars])

        # W-state
        for w in one_hots.values():
            qc.compose(w_state(len(w)), [varmap[i] for i in w], inplace=True)

    return qc


def rxy(beta: Parameter | float):
    """Implements single XY rotation.

    Parameters
    ----------
    beta: float | Parameter
        The beta parameter.

    Returns
    -------
    QuantumCircuit
    """
    qc = QuantumCircuit(2)
    qc.rxx(beta, 0, 1)
    qc.ryy(beta, 0, 1)
    # qc.append(XXPlusYYGate(2 * beta).definition, (0, 1))
    return qc


def xy_mixer(num: int, beta: Parameter | float):
    """Implements the XY-ring-mixer.

    Parameters
    ----------
    num: int
        The number of qubits that are part of this ring mixer.
    beta: float | Parameter
        The beta parameter.

    Returns
    -------
    QuantumCircuit
    """
    qc = QuantumCircuit(num)
    assert num >= 2
    if num == 2:
        qc.compose(rxy(beta), (0, 1), inplace=True)
    else:
        uxy = rxy(beta)
        pairs = [(i, i + 1) for i in range(0, num - 1, 2)] + [
            (i, i + 1) for i in range(1, num - 1, 2)
        ]
        pairs.append((0, num - 1))
        for p in pairs:
            qc.compose(uxy, p, inplace=True)
    return qc


def combined_mixer(ir: IR, beta: Parameter | float):
    """Implements the combined mixer.

    Automatically applies normal X-mixer or combined mixer depending on IR

    Parameters
    ----------
    ir: IR
        The intermediate representation of the model transformation.
    beta: float | Parameter
        The beta parameter.

    Returns
    -------
    QuantumCircuit
    """
    remove_one_hot = ir.cache["remove-one-hot"]
    if remove_one_hot is None:
        return rx_mixer_layer(len(ir.model.variables()), beta)

    qc = QuantumCircuit(len(ir.model.variables()))
    varmap = {v: i for i, v in enumerate(ir.model.variables())}

    one_hots = remove_one_hot["removed"]
    one_hot_vars = set(i for v in one_hots.values() for i in v)

    # Normal initial state
    if qc.num_qubits - len(one_hot_vars) > 0:
        qc.compose(
            rx_mixer_layer(qc.num_qubits - len(one_hot_vars), beta),
            [i for v, i in varmap.items() if v not in one_hot_vars],
            inplace=True,
        )

    # W-state
    for w in one_hots.values():
        qc.compose(xy_mixer(len(w), beta), {varmap[i] for i in w}, inplace=True)

    return qc


def cost_layer_adv(ir: IR, gamma: Parameter | float):
    """Implements a single QAOA cost operation.

    Parameters
    ----------
    ir: IR
        The intermediate representation containing the transformed objective function
        and model metadata
    gamma: Parameter | Float
        Cost parameter (controls problem-specific rotations)

    Returns
    -------
    QuantumCircuit
        A single QAOA cost layer
    """
    qc = QuantumCircuit(len(ir.model.variables()))

    max_bias = ir.cache["max-bias"].val

    # Mapping from variables to qubit indices
    varmap = {v: i for i, v in enumerate(ir.model.variables())}

    items_list = list(ir.model.objective.items())
    order = ir.cache["optimal-coloring"] or range(len(items_list))
    for item, val in (items_list[i] for i in order):
        angle = val * gamma / max_bias
        match item:
            case Linear(v):
                # Linear terms: single-qubit phase gate
                qc.p(angle, varmap[v])
            case Quadratic(v1, v2):
                # Quadratic terms: two-qubit controlled-phase gate applied to |11>
                qc.cp(angle, varmap[v1], varmap[v2])
            case HigherOrder(vs):
                # HigherOrder terms: muti-qubit controlled-phase gate applied to |11..11>
                qc.mcp(angle, [varmap[v] for v in vs[:-1]], varmap[vs[-1]])

    return qc


def qaoa_layer_adv(
    ir: IR, beta: Parameter | float, gamma: Parameter | float
) -> QuantumCircuit:
    """Implements a single advanced QAOA layer consisting of cost and mixer operations.

    Parameters
    ----------
    ir: IR
        The intermediate representation containing the transformed objective function
        and model metadata
    beta: Parameter | Float
        Mixer parameter (controls X-rotations)
    gamma: Parameter | Float
        Cost parameter (controls problem-specific rotations)

    Returns
    -------
    QuantumCircuit
        A single QAOA layer
    """

    qc = cost_layer_adv(ir, gamma)
    qc = qc.compose(combined_mixer(ir, beta))
    assert qc is not None

    return qc


def qaoa_circ_adv(ir: IR, reps: int = 1) -> QuantumCircuit:
    """Generates a complete advanced QAOA circuit with specified number of layers.

    Parameters
    ----------
    ir: IR
        The intermediate representation containing the transformed objective function
        and model metadata
    reps: int
        Number of QAOA layers (p parameter)

    Returns
    -------
    QuantumCircuit
        Complete QAOA circuit with measurements
    """
    # Verify the model is suitable for QAOA
    assert ir.model.num_constraints == 0, "QAOA requires unconstrained problems"
    assert all(v.vtype == Vtype.Binary for v in ir.model.variables()), (
        "All variables must be binary"
    )

    # Start with equal superposition state |+⟩^⊗n \otimes |W>
    qc = initial_state(ir)

    # Create parameter templates for a single layer
    beta = Parameter("beta")
    gamma = Parameter("gamma")
    layer_circ = qaoa_layer_adv(ir, beta, gamma)

    # Create individual parameters for each layer
    betas = [Parameter(f"beta_{p}") for p in range(reps)]
    gammas = [Parameter(f"gamma_{p}") for p in range(reps)]

    # Add all QAOA layers
    for beta_p, gamma_p in zip(betas, gammas):
        # Substitute the template parameters with layer-specific ones
        layer_instance = layer_circ.assign_parameters({beta: beta_p, gamma: gamma_p})
        qc = qc.compose(layer_instance)
        assert qc is not None

    # Add measurements to all qubits
    qc.measure_all()
    return qc
