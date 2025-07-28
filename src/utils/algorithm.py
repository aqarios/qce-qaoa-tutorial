import random

from aqmodels import Model
from aqmodels.transformations import PassManager
from luna_quantum.solve.parameters.algorithms.base_params.qaoa_circuit_params import (
    BasicQAOAParams,
    LinearQAOAParams,
    RandomQAOAParams,
)
from luna_quantum.solve.parameters.algorithms.base_params.scipy_optimizer import (
    ScipyOptimizerParams,
)
from scipy.optimize import minimize

from utils.qaoa import qaoa_circ_adv
from utils.sampling import cost_function, sample

type AnyQAOAParams = LinearQAOAParams | BasicQAOAParams | RandomQAOAParams


class MyQAOA:
    """Wraps our QAOA implementation in Luna Quantum API."""

    def __init__(
        self,
        pass_manager: PassManager,
        reps: int = 1,
        shots: int = 1000,
        initial_params: LinearQAOAParams
        | BasicQAOAParams
        | RandomQAOAParams
        | None = None,
        optimizer: ScipyOptimizerParams | None = None,
    ):
        self.pass_manager = pass_manager
        self.reps = reps
        self.shots = shots
        self.params = initial_params or LinearQAOAParams(
            delta_beta=0.5, delta_gamma=0.5
        )
        self.optimizer = optimizer or ScipyOptimizerParams()

        self.qc = None

    def _get_initial(self):
        """Generates initial QAOA circuit params from input."""
        if isinstance(self.params, LinearQAOAParams):
            delta = 1 / self.reps
            betas = [
                self.params.delta_beta * (1 - delta / 2 - delta * i)
                for i in range(self.reps)
            ]
            gammas = [
                self.params.delta_gamma * (delta / 2 + delta * i)
                for i in range(self.reps)
            ]
            return betas + gammas
        elif isinstance(self.params, BasicQAOAParams):
            return self.params.betas + self.params.gammas
        else:
            random.seed(self.params.seed)
            betas = [random.uniform(*self.params.beta_range) for _ in range(self.reps)]
            gammas = [
                random.uniform(*self.params.gamma_range) for _ in range(self.reps)
            ]
            return betas + gammas

    def _optimizer_kwargs(self):
        """Generates a scipy minimize compliant kwargs dict from input."""
        kwargs = self.optimizer.model_dump(exclude_none=True)
        kwargs["options"]["maxiter"] = kwargs.pop("maxiter")
        return kwargs

    @property
    def circuit(self):
        if self.qc is None:
            raise RuntimeError("Make sure to run algorithm before accessing circuit")
        return self.qc

    def run(self, model: Model):
        """Runs the QAOA."""
        ir = self.pass_manager.run(model)
        qc = qaoa_circ_adv(ir, self.reps)
        self.qc = qc

        log = []

        x0 = self._get_initial()

        res = minimize(
            cost_function,
            args=(ir.model, qc, log, self.shots),
            x0=x0,
            **self._optimizer_kwargs(),
        )

        sol = sample(ir.model, qc, res.x, self.shots)
        res = model.evaluate(self.pass_manager.backwards(sol, ir))

        class ResultWrapper:
            def result(self):
                return res

        return ResultWrapper()
