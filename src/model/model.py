import itertools

from aqmodels import Model, Variable, quicksum

from model.data import ConventionCenter, Schedule, SessionChair


def optimization_model(
    convention_center: ConventionCenter,
    schedule: Schedule,
    chairs: list[SessionChair],
    satisfaction: float = 5.0,
):
    model = Model("Session chair assignment")

    x: dict[tuple[str, int], Variable] = {}
    for room in schedule.rooms():
        for i in range(len(chairs)):
            x[room, i] = model.add_variable(f"x_{i}_{room}")

    # one-hot constraints
    for room in schedule.rooms():
        model.add_constraint(
            quicksum(x[room, i] for i in range(len(chairs))) == 1, f"chair_in_{room}"
        )

    # objectiv function: distance based
    distance = convention_center.distance_map
    model.objective += quicksum(
        round(chair.lazy * distance[room_a, room_b], 1) * x[room_a, i] * x[room_b, i]
        for i, chair in enumerate(chairs)
        for room_a, room_b in itertools.combinations(schedule.rooms(), r=2)
    )
    # objectiv function: satisfaction
    rev_map = {v: k for k, v in schedule.items()}
    model.objective -= quicksum(
        satisfaction * x[rev_map[fav], i]
        for i, chair in enumerate(chairs)
        for fav in chair.favourites
        if fav in rev_map
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
