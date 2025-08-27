from __future__ import annotations

import random
from dataclasses import dataclass

import names
import networkx as nx
import numpy as np

ROOMS = {
    "027": (0.0, 0.0, 3.0, 4.0),
    "025": (3.0, 0.0, 1.0, 3.0),
    "023": (4.0, 0.0, 1.0, 3.0),
    "021": (5.0, 0.0, 1.0, 3.0),
    "019": (6.0, 0.0, 1.0, 3.0),
    "017": (7.0, 0.0, 2.0, 3.0),
    "010": (10.0, 1.0, 3.0, 2.0),
    "070": (14.0, 2.0, 2.0, 2.0),
    "065": (16.0, 2.0, 2.0, 2.0),
    "060": (18.0, 2.0, 2.0, 1.0),
    "029": (0.0, 4.0, 3.0, 1.0),
    "031": (0.0, 5.0, 3.0, 2.0),
    "020": (4.0, 4.0, 1.5, 1.5),
    "022": (4.0, 5.5, 1.5, 1.5),
    "028": (4.0, 7.0, 1.5, 1.5),
    "018": (5.5, 4.0, 2.0, 1.5),
    "016": (7.5, 4.0, 1.5, 1.5),
    "015": (7.5, 5.5, 1.5, 1.5),
    "032": (7.5, 7.0, 1.5, 1.5),
    "030": (5.5, 5.5, 2.0, 3.0),
}

ACTIVE = [
    "031",
    "029",
    "027",
    "025",
    "023",
    "021",
    "019",
    "017",
    "020",
    "018",
    "016",
    "015",
    "032",
    "030",
    "028",
    "022",
    "020",
]

DOORS = {  # Door perimeter location
    "027": [6.5],
    "029": [3.2],
    "031": [3.2],
    "025": [4.4],
    "023": [4.4],
    "021": [4.8],
    "019": [4.4],
    "017": [6.8],
    "020": [1.1, 4.9],
    "022": [5.9],
    "028": [5.9, 3.4],
    "030": [5.4, 6.8],
    "032": [4.3],
    "015": [2.5],
    "016": [0.2],
    "018": [0.2],
    "010": [5.5, 7.7],
    "070": [4.4],
    "065": [5.7],
    "060": [3.4],
}

DOOR_LEN = 0.25

HELPERS = {
    "h1": [(4, 4)],
    "h2": [(4, 8.5)],
    "h3": [(9, 4)],
    "h4": [(9, 8.5)],
    "h5": [(18, 4)],
}

EDGES = [
    ("031", "028"),
    ("031", "029"),
    ("031", "027"),
    ("031", "022"),
    ("031", "020"),
    ("031", "025"),
    ("031", "h1"),
    ("031", "h2"),
    ("029", "028"),
    ("029", "027"),
    ("029", "022"),
    ("029", "020"),
    ("029", "023"),
    ("029", "025"),
    ("029", "h1"),
    ("029", "h2"),
    ("027", "028"),
    ("027", "022"),
    ("027", "020"),
    ("027", "023"),
    ("027", "025"),
    ("027", "021"),
    ("027", "019"),
    ("027", "017"),
    ("027", "010"),
    ("027", "018"),
    ("027", "016"),
    ("027", "h1"),
    ("027", "h2"),
    ("027", "h3"),
    ("025", "028"),
    ("025", "022"),
    ("025", "020"),
    ("025", "023"),
    ("025", "021"),
    ("025", "019"),
    ("025", "017"),
    ("025", "010"),
    ("025", "018"),
    ("025", "016"),
    ("025", "h1"),
    ("025", "h2"),
    ("025", "h3"),
    ("022", "028"),
    ("022", "020"),
    ("022", "h1"),
    ("022", "h2"),
    ("028", "020"),
    ("028", "030"),
    ("028", "032"),
    ("028", "h4"),
    ("028", "h1"),
    ("028", "h2"),
    ("020", "023"),
    ("020", "021"),
    ("020", "019"),
    ("020", "017"),
    ("020", "010"),
    ("020", "018"),
    ("020", "016"),
    ("020", "070"),
    ("020", "065"),
    ("020", "h1"),
    ("020", "h2"),
    ("020", "h3"),
    ("018", "023"),
    ("018", "021"),
    ("018", "019"),
    ("018", "017"),
    ("018", "010"),
    ("018", "016"),
    ("018", "070"),
    ("018", "065"),
    ("018", "h1"),
    ("018", "h3"),
    ("016", "023"),
    ("016", "021"),
    ("016", "019"),
    ("016", "017"),
    ("016", "010"),
    ("016", "070"),
    ("016", "065"),
    ("016", "h1"),
    ("016", "h3"),
    ("023", "021"),
    ("023", "019"),
    ("023", "017"),
    ("023", "010"),
    ("023", "h1"),
    ("023", "h3"),
    ("021", "019"),
    ("021", "017"),
    ("021", "010"),
    ("021", "h1"),
    ("021", "h3"),
    ("019", "017"),
    ("019", "010"),
    ("019", "h1"),
    ("019", "h3"),
    ("017", "010"),
    ("017", "h1"),
    ("017", "h3"),
    ("010", "015"),
    ("010", "h1"),
    ("010", "h3"),
    ("h3", "h5"),
    ("015", "065"),
    ("015", "070"),
    ("015", "h3"),
    ("015", "h4"),
    ("015", "h5"),
    ("060", "h5"),
    ("065", "h3"),
    ("070", "h3"),
    ("065", "h5"),
    ("070", "h5"),
]


def get_door(x: float, y: float, w: float, h: float, i: float):
    """Transforms door perimeter to actual coordiantes."""
    if i - w < 0:  # bottom
        return (x + i, x + i + DOOR_LEN), (y, y)
    i -= w
    if i - h < 0:  # right
        return (x + w, x + w), (y + i, y + i + DOOR_LEN)
    i -= h
    if i - w < 0:  # top
        return (x + w - i, x + w - i + DOOR_LEN), (y + h, y + h)
    i -= w
    # left
    return (x, x), (y + h - i, y + h - i + DOOR_LEN)


def get_door_center(x: float, y: float, w: float, h: float, i: float):
    xv, yv = get_door(x, y, w, h, i)
    return sum(xv) / 2, sum(yv) / 2


DOOR_POS = {
    room_id: [get_door_center(*ROOMS[room_id], d) for d in doors]
    for room_id, doors in DOORS.items()
}


@dataclass
class ConventionCenter:
    rooms: dict[str, tuple[float, float, float, float]]
    doors: dict[str, list[float]]
    door_pos: dict[str, list[tuple[float, float]]]
    distance_map: dict[tuple[str, str], float]

    @staticmethod
    def generate(active: None | list[str] = None):
        if active is None:
            active = ACTIVE

        positions = {**DOOR_POS, **HELPERS}

        g = nx.Graph()

        for e in EDGES:
            if e[0] not in active and not e[0].startswith("h"):
                continue
            if e[1] not in active and not e[1].startswith("h"):
                continue
            a = positions[e[0]]
            b = positions[e[1]]
            for ai in a:
                for bi in b:
                    x1, y1 = ai
                    x2, y2 = bi
                    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    g.add_edge(*e, weight=distance)

        data = {}
        for n, e in nx.all_pairs_dijkstra_path_length(g):
            if n.startswith("h"):
                continue

            data.update(
                {
                    (n, k): round(float(v), 1)
                    for k, v in e.items()
                    if not k.startswith("h") and n != k
                }
            )

        return ConventionCenter(
            rooms={k: v for k, v in ROOMS.items() if k in active},
            doors={k: v for k, v in DOORS.items() if k in active},
            door_pos={k: v for k, v in DOOR_POS.items() if k in active},
            distance_map=data,
        )


SESSIONS = [
    "QALG",
    "QAPP",
    "QSYS",
    "QNET",
    "TUT",
    "PAN",
    "QPHO",
    "QTEM",
    "QML",
    "WKS",
    "KEY",
]  # can be extended from real data

MARKERS = ["o", "s", "d", "X", "P", "<", ">", "^", "v", "h"]


@dataclass
class SessionChair:
    name: str
    favorite: str
    fitness: float = 1.0
    marker: str = "o"

    def __post_init__(self):
        if self.fitness > 3:
            raise ValueError("Fitness needs to be less than 5.")
        elif self.fitness < 1:
            raise ValueError("Fitness needs to be more than 1.")

    @staticmethod
    def random_chairs(
        num: int, seed: int | None = None, schedule: Schedule | None = None
    ) -> list[SessionChair]:
        random.seed(seed)
        chairs = []

        sessions = SESSIONS
        if schedule is not None:
            sessions = list(schedule.values())

        fav = random.sample(sessions, num)
        for i in range(num):
            name = names.get_full_name()
            favorite = fav[i]
            fitness = random.randint(1, 3)
            profdr = random.choice(["Prof. ", "Dr. "])
            marker = random.choice(MARKERS)
            chairs.append(
                SessionChair(
                    name=profdr + name,
                    favorite=favorite,
                    fitness=fitness,
                    marker=marker,
                )
            )
        return chairs


class Schedule(dict[str, str]):
    @staticmethod
    def random(num: int, convention_center: ConventionCenter, seed: int | None = None):
        rng = np.random.default_rng(seed)
        rooms = rng.choice(
            list(convention_center.rooms.keys()), size=num, replace=False
        )
        sessions = rng.choice(SESSIONS, size=num, replace=False)

        return Schedule(dict(zip(map(str, rooms), map(str, sessions))))

    def rooms(self):
        return list(self.keys())

    def __eq__(self, other):
        """Enable equality comparison between Schedule instances."""
        if not isinstance(other, Schedule):
            return False
        return dict(self) == dict(other)

    def __hash__(self):
        """Enable Schedule to be used as dict key or in sets (optional)."""
        return hash(tuple(sorted(self.items())))