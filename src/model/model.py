import itertools

from luna_quantum import Model, quicksum

from model.data import ConventionCenter, Schedule, SessionChair


def optimization_model(
    convention_center: ConventionCenter,
    schedule: Schedule,
    chairs: list[SessionChair],
    satisfaction: float = 5.0,
):
    model = Model("Session chair assignment")

    # Add binary variables to the optimization model
    x = {}
    for r in schedule.rooms():
        for i in range(len(chairs)):
            # If no vtype is specified, binary vtype is selected by default
            x[r, i] = model.add_variable(f"x_{r}_{i}")

    # Objective function part 1: Minimize total distance to travel for every session chair
    distance = convention_center.distance_map
    model.objective += quicksum(
        round((4 - chair.fitness) * distance[room_a, room_b], 1)
        * x[room_a, i]
        * x[room_b, i]
        for i, chair in enumerate(chairs)
        for room_a, room_b in itertools.combinations(schedule.rooms(), r=2)
    )

    # Objective function part 2: Maximize satisfaction for chairing favorite sessions
    session_to_room_map = {v: k for k, v in schedule.items()}
    model.objective -= quicksum(
        satisfaction * x[session_to_room_map[chair.favorite], i]
        for i, chair in enumerate(chairs)
    )

    # one-hot constraints: Every room needs to have exactly one chair
    for room in schedule.rooms():
        constraint_name = f"someone_in_{room}"
        model.add_constraint(
            quicksum(x[room, i] for i in range(len(chairs))) == 1, constraint_name
        )

    return x, model


def optimization_model_with_max_capacity(
    convention_center: ConventionCenter,
    schedule: Schedule,
    chairs: list[SessionChair],
    satisfaction: float = 5.0,
    capacity: int = 3,
):
    x, model = optimization_model(convention_center, schedule, chairs, satisfaction)

    # capacity constraints for each chair
    for i in range(len(chairs)):
        model.add_constraint(
            quicksum(x[room, i] for room in schedule.rooms()) <= capacity,
            f"capacity_chair_{i}",
        )

    return x, model
