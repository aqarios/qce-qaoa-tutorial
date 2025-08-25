from luna_quantum import Solution
from qiskit.circuit.gate import np


class pretty:
    def __init__(self, sol: Solution, sort_feasible: bool = True):
        self.sol = sol
        keys = [sol.obj_values]
        if sort_feasible:
            infeasible = np.array([not x.feasible for x in sol])
            keys.append(infeasible)
        self._sorted = np.lexsort(keys)  # type: ignore
        cx = np.cumsum(list(map(len, sol.variable_names))) < 80
        self._vars = [var for var, ci in zip(sol.variable_names, cx) if ci]
        self._all_vars = cx[-1]

    def _repr_html_(self):
        s = "<table>"
        s += "<tr>"
        for x in self._vars:
            s += f"<th>{x}</th>"
        if not self._all_vars:
            s += "<th>...</th>"
        s += '<th style="border-left: solid 1px;">Obj Val.</ th>'
        s += "<th>Raw</ th>"
        s += "<th>Counts</ th>"
        s += "<th>Feasible</ th>"
        s += "<tr />"
        for i in self._sorted[:20]:
            x = self.sol[i]
            s += "<tr>"
            for b in list(x.sample)[: len(self._vars)]:
                s += f"<td>{b}</td>"
            if not self._all_vars:
                s += "<td></td>"
            s += f'<td style="border-left: solid 1px;">{x.obj_value:.2f}</ td>'
            s += f"<td>{x.raw_energy or x.obj_value:.2f}</ td>"
            s += f"<td>{x.counts}</ td>"
            s += f'<td style="color: {"blue" if x.feasible else "red"}">{x.feasible}</ td>'
            s += "</tr>"
        if len(self._sorted) > 20:
            s += (
                "</tr><td>...</td>"
                + ("<td/ >" * (len(self._vars) + self._all_vars + 3))
                + "</tr>"
            )
        return s + "</table>"
