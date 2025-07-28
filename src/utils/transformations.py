import itertools
from typing import override

import networkx as nx
import numpy as np
import rustworkx as rx
from aqmodels import (
    Comparator,
    HigherOrder,
    Linear,
    Model,
    Quadratic,
    Solution,
    Variable,
    Vtype,
)
from aqmodels.decorators import analyse, transform
from aqmodels.transformations import (
    ActionType,
    AnalysisCache,
    MaxBiasAnalysis,
    PassManager,
    TransformationOutcome,
    TransformationPass,
)


@analyse()
def identify_one_hot(model: Model, _: AnalysisCache):
    """Identifies one-hot constraints present in the model."""
    onehots: list[int | str] = []
    for i, constraint in enumerate(model.constraints):
        if not all(
            isinstance(k, Linear) and v == constraint.rhs
            for k, v in constraint.lhs.items()
        ):
            continue
        onehots.append(constraint.name or i)
    return onehots


@analyse()
def optimal_coloring(model: Model, _cache: AnalysisCache):
    """Finds the best execution order of gates."""
    linear_first = []

    to_color: list[set[Variable]] = []
    nodes = []
    for i, (key, _) in enumerate(model.objective.items()):
        match key:
            case Linear(_):
                linear_first.append(i)
            case Quadratic(v1, v2):
                to_color.append({v1, v2})
                nodes.append(i)
            case HigherOrder(vs):
                to_color.append(set(*vs))
                nodes.append(i)

    g = rx.PyGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from_no_data(
        [
            (i, j)
            for (i, x1), (j, x2) in itertools.combinations(enumerate(to_color), r=2)
            if len(x1.intersection(x2)) > 0
        ]
    )

    res = rx.graph_greedy_color(g)

    order = linear_first + [
        nodes[i] for i, _ in sorted(res.items(), key=lambda x: x[1])
    ]

    return order


@transform(requires=["identify-one-hot"], invalidates=["identify-one-hot"])
def remove_one_hot(model: Model, cache: AnalysisCache):
    """Removes non-overlapping one-hot constraints from the model."""
    one_hots: list[int | str] = cache["identify-one-hot"]

    if len(one_hots) == 0:
        return TransformationOutcome.nothing(model)

    # Extract one hot constraint information
    one_hot_info: dict[int | str, list[Variable]] = {}
    for i in one_hots:
        c = model.constraints[i]
        vars = [item.var for item, _ in c.lhs.items() if isinstance(item, Linear)]
        one_hot_info[i] = vars

    # Find intersections of constraints
    intersections = []
    for (i, a), (j, b) in itertools.combinations(one_hot_info.items(), r=2):
        if len(set(a) & set(b)) > 0:
            intersections.append((i, j))

    # Fix intersections
    if len(intersections) != 0:
        G = nx.Graph(intersections)
        mis = nx.maximal_independent_set(G, seed=0)  # fix seed for reproducable results
        remaining = [i for i in one_hot_info.keys() if i not in mis]
        one_hot_info = {k: v for k, v in one_hot_info.items() if k in mis}
    else:
        remaining = []

    # Remove constraints
    for k, c in one_hot_info.items():
        model.constraints.remove(k)

    return TransformationOutcome(
        model,
        ActionType.DidTransform,
        {"removed": one_hot_info, "remaining": remaining},
    )


class QuadraticPenaltyPass(TransformationPass):
    """Integrates equality constraints as quadratic penalties."""

    def __init__(self, penalty_factor: float = 5.0):
        self.penalty_factor = penalty_factor

    @property
    def name(self):
        return "quadratic-penalty"

    @property
    def requires(self):
        return ["max-bias"]

    def run(self, model: Model, cache: AnalysisCache):
        max_bias = cache["max-bias"]

        penalty = self.penalty_factor * max_bias.val

        to_remove = []
        for i, c in enumerate(model.constraints):
            if c.comparator != Comparator.Eq:
                continue
            to_remove.append(c.name or i)
            model.objective += penalty * (c.lhs - c.rhs) ** 2

        if len(to_remove) == 0:
            return TransformationOutcome.nothing(model)

        for r in reversed(to_remove):
            model.constraints.remove(r)

        return TransformationOutcome(model, ActionType.DidTransform)

    @override
    def backwards(self, solution: Solution, cache: AnalysisCache):
        return solution


qubo_pipeline = PassManager(
    [
        MaxBiasAnalysis(),
        QuadraticPenaltyPass(),
        MaxBiasAnalysis(),
    ]
)

xy_pipeline = PassManager([identify_one_hot, remove_one_hot, MaxBiasAnalysis()])
