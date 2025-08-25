import itertools

import matplotlib.pyplot as plt
import numpy as np
from luna_quantum import Result
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .data import ConventionCenter, Schedule, SessionChair, get_door


def solution_to_assignment(best: Result, chairs: list[SessionChair]):
    def _from_name(k):
        _, r, i = k.split("_")
        return chairs[int(i)].name, r

    return [_from_name(k) for k, v in best.sample.to_dict().items() if v == 1]


def plot_floor_plan(
    convention_center: ConventionCenter,
    schedule: Schedule | None = None,
    chairs: list[SessionChair] | None = None,
    assignment: list[tuple[str, str]] | None = None,
    figsize=(10, 5),
    seed: int | None = None,
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")

    rng = np.random.default_rng(seed)

    if assignment is not None and chairs is None:
        raise ValueError("Please provide session chairs.")

    for room_id, (x, y, w, h) in convention_center.rooms.items():
        facecolor = "lightgrey"
        alpha = 0.5 if schedule is not None else 1
        label = room_id
        if schedule is not None and room_id in schedule.rooms():
            session = schedule[room_id]
            label = f"{label}\n{session}"
            facecolor = "#90d5ff"
            alpha = 1

        # draw
        rect = Rectangle((x, y), w, h, edgecolor="k", facecolor=facecolor)
        ax.add_patch(rect)
        doors = convention_center.doors.get(room_id, None)
        if doors is not None:
            for door in doors:
                data = get_door(x, y, w, h, door)
                line = Line2D(*data, color="white")
                ax.add_line(line)

        ax.text(
            s=label,
            x=x + w / 2,
            y=y + h / 2,
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            alpha=alpha,
        )

    if chairs is not None:
        for chair in chairs:
            x_data = []
            y_data = []
            if assignment is not None:
                rooms = [r for c, r in assignment if c == chair.name]
                for room in rooms:
                    x, y, w, h = convention_center.rooms[room]
                    cx = x + w / 2
                    cy = y + h / 2
                    dx, dy = convention_center.door_pos[room][0]
                    x_data.append((cx + 2 * dx) / 3 + 0.05 * rng.normal())
                    y_data.append((cy + 2 * dy) / 3 + 0.05 * rng.normal())
            plt.scatter(
                x_data,
                y_data,
                s=60 * (4 - chair.fitness) + 50,
                marker=chair.marker,
                label=f"{chair.name}(fitness={chair.fitness}, fav={chair.favorite})",
                zorder=1000,
            )

    plt.legend(
        loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, labelspacing=3
    )

    ax.axis("off")
    # plt.grid()
    # plt.xticks(range(20))
    # plt.tight_layout()
    plt.ylim(
        -0.5, max(y + h for (_, y, _, h) in convention_center.rooms.values()) + 0.5
    )
    plt.xlim(
        -0.5, max(x + w for (x, _, w, _) in convention_center.rooms.values()) + 0.5
    )
    plt.tight_layout()

    return fig, ax


def plot_satisfaction(
    convention_center: ConventionCenter,
    schedule: Schedule,
    chairs: list[SessionChair],
    assignment: list[tuple[str, str]],
    satisfaction: float = 2.0,
    figsize=(3, 3),
):
    chair_map = {chair.name: chair for chair in chairs}

    rooms = {chair.name: [] for chair in chairs}
    sat = {chair.name: 0.0 for chair in chairs}
    for c, r in assignment:
        rooms[c].append(r)
        if schedule[r] == chair_map[c].favorite:
            sat[c] += satisfaction

    distance = {
        k: sum(
            convention_center.distance_map[i, j]
            for i, j in itertools.combinations(v, r=2)
        )
        or 0.0
        for k, v in rooms.items()
    }

    fig, ax = plt.subplots(figsize=figsize)

    dissat = {k: v * (4 - chair_map[k].fitness) - sat[k] for k, v in distance.items()}

    for k, v in dissat.items():
        ax.bar(k, -v)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.axhline(0, color="k", linestyle="dashed", alpha=0.5)
    plt.ylabel("Satisfaction")

    return fig, ax
