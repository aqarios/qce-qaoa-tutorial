from luna_quantum import Model, Variable, Vtype, algorithms, quicksum
from luna_quantum.solve.parameters.algorithms.base_params import (
    LinearQAOAParams,
    ScipyOptimizerParams,
)
from luna_quantum.solve.parameters.algorithms.quantum_gate.flex_qaoa import (
    OneHotParams,
    PipelineParams,
)

facilities = {
    "Munich": {"cost": 500, "capacity": 15},
    "Nuremberg": {"cost": 400, "capacity": 9},
    # "Augsburg": {"cost": 350, "capacity": 10},
}
hospitals = {
    "H2": {"location": "Munich-North", "demand": 6},
    "H3": {"location": "Nuremberg-South", "demand": 5},
    # "H4": {"location": "Augsburg-West", "demand": 3},
    # "H8": {"location": "Erlangen", "demand": 2},
}

# Transportation costs (€ per ton)
transport_costs = {
    ("H1", "Munich"): 0.10,
    ("H1", "Nuremberg"): 0.45,
    ("H1", "Augsburg"): 0.25,
    ("H1", "Regensburg"): 0.50,
    ("H2", "Munich"): 0.15,
    ("H2", "Nuremberg"): 0.40,
    ("H2", "Augsburg"): 0.30,
    ("H2", "Regensburg"): 0.55,
    ("H3", "Munich"): 0.40,
    ("H3", "Nuremberg"): 0.12,
    ("H3", "Augsburg"): 0.50,
    ("H3", "Regensburg"): 0.35,
    ("H4", "Munich"): 0.45,
    ("H4", "Nuremberg"): 0.18,
    ("H4", "Augsburg"): 0.55,
    ("H4", "Regensburg"): 0.40,
    ("H5", "Munich"): 0.20,
    ("H5", "Nuremberg"): 0.55,
    ("H5", "Augsburg"): 0.8,
    ("H5", "Regensburg"): 0.60,
    ("H6", "Munich"): 0.50,
    ("H6", "Nuremberg"): 0.35,
    ("H6", "Augsburg"): 0.60,
    ("H6", "Regensburg"): 0.10,
    ("H7", "Munich"): 0.35,
    ("H7", "Nuremberg"): 0.30,
    ("H7", "Augsburg"): 0.40,
    ("H7", "Regensburg"): 0.25,
    ("H8", "Munich"): 0.40,
    ("H8", "Nuremberg"): 0.15,
    ("H8", "Augsburg"): 0.45,
    ("H8", "Regensburg"): 0.30,
}

depot = "Munich"
m = Model()


# add variables
x = {}
y = {}
with m.environment:
    for f in facilities:
        for h in hospitals:
            x[(f, h)] = Variable(vtype=Vtype.Binary, name=f"x_{f},{h}")
        y[f] = Variable(vtype=Vtype.Binary, name=f"y_{f}")

# add cost to transport goods
m.objective += quicksum(
    x[f, h] * transport_costs[(h, f)] for f in facilities for h in hospitals
)

# add cost to open facility
m.objective += quicksum(y[f] * facilities[f]["cost"] for f in facilities)


# each hospital must be delivered
for h in hospitals:
    m.add_constraint(
        quicksum(x[f, h] for f in facilities) == 1, name=f"deliver_hospital_{h}"
    )

# at most usage of capacity
for f in facilities:
    m.add_constraint(
        quicksum(hospitals[h]["demand"] * x[f, h] for h in hospitals)
        <= facilities[f]["capacity"] * y[f],
        name=f"deliver_hospital_{f}",
    )


qaoa = algorithms.FlexQAOA(
    reps=3,
    pipeline=PipelineParams(
        indicator_function=None,
        one_hot=OneHotParams(),
    ),
    initial_params=LinearQAOAParams(delta_beta=0.2, delta_gamma=0.2),
    optimizer=ScipyOptimizerParams(method="COBYLA"),
)
qaoa_job = qaoa.run(m)
qaoa_sol = qaoa_job.result()
print(qaoa_sol)
