"""Solow model with a time-varying savings rate (Problem 2).

The class contains the model equations, simulation routine, welfare measure,
and the savings rules used in the accompanying exam notebook.
"""

from types import SimpleNamespace

import numpy as np
from scipy import optimize


class SolowModelClass:
    """Solow model with Cobb-Douglas production and savings rules."""

    def __init__(self, **kwargs):
        """Set default parameters and overwrite them with keyword arguments."""

        par = self.par = SimpleNamespace()

        # a. technology
        par.alpha = 1 / 3
        par.delta = 0.30

        # b. long-run savings rate in the two-parameter rule
        par.s_bar = 0.25

        # c. preferences
        par.beta = 1 / 1.2

        # d. simulation settings
        par.k0 = 0.10
        par.T = 100

        # e. overwrite defaults, e.g. SolowModelClass(alpha=0.25)
        for key, value in kwargs.items():
            setattr(par, key, value)

        # f. container for the most recent simulation
        self.sim = SimpleNamespace()

    def __str__(self):
        """Return a readable description of the calibration."""

        par = self.par
        text = "Solow model with:\n"
        text += f"  alpha = {par.alpha:.4f} (capital share)\n"
        text += f"  delta = {par.delta:.4f} (depreciation rate)\n"
        text += f"  s_bar = {par.s_bar:.4f} (long-run savings rate)\n"
        text += f"  beta  = {par.beta:.4f} (discount factor)\n"
        text += f"  k0    = {par.k0:.4f} (initial capital per worker)\n"
        text += f"  T     = {par.T} (number of periods)"
        return text

    def f(self, k):
        """Return output per worker, y = f(k) = k**alpha."""

        return np.asarray(k) ** self.par.alpha

    def k_next(self, k, s):
        """Return next-period capital, k_next = s*f(k) + (1-delta)*k."""

        return s * self.f(k) + (1 - self.par.delta) * k

    def steady_state(self, s=None):
        """Return analytical steady-state capital, output, and consumption."""

        par = self.par
        if s is None:
            s = par.s_bar

        k = (s / par.delta) ** (1 / (1 - par.alpha))
        y = float(self.f(k))
        c = (1 - s) * y
        return k, y, c

    def solve_steady_state(self, s=None):
        """Find the positive steady-state capital stock with Brent's method."""

        if s is None:
            s = self.par.s_bar

        objective = lambda k: self.k_next(k, s) - k
        result = optimize.root_scalar(
            objective, bracket=[1e-8, 1e6], method="brentq"
        )
        if not result.converged:
            raise RuntimeError("The steady-state root finder did not converge.")
        return result.root

    def simulate(self, s, k0=None):
        """Simulate the model for a constant or time-varying savings rate.

        Args:
            s (float or ndarray): A scalar or a path with ``par.T`` elements.
            k0 (float, optional): Initial capital. ``par.k0`` is used if omitted.

        Returns:
            SimpleNamespace: Independent arrays for s, k, y, i, and c.
        """

        par = self.par
        if k0 is None:
            k0 = par.k0

        s_vec = (
            np.full(par.T, s, dtype=float)
            if np.ndim(s) == 0
            else np.asarray(s, dtype=float).reshape(-1)
        )
        assert s_vec.size == par.T, (
            f"the savings rate must have {par.T} elements, but has {s_vec.size}"
        )
        if np.any((s_vec < 0) | (s_vec > 1)):
            raise ValueError("Every savings rate must be between zero and one.")
        if k0 <= 0:
            raise ValueError("Initial capital must be strictly positive.")

        k = np.empty(par.T)
        y = np.empty(par.T)
        investment = np.empty(par.T)
        c = np.empty(par.T)

        k[0] = k0
        for t in range(par.T):
            y[t] = self.f(k[t])
            investment[t] = s_vec[t] * y[t]
            c[t] = y[t] - investment[t]

            if t < par.T - 1:
                k[t + 1] = self.k_next(k[t], s_vec[t])

        sim = SimpleNamespace(s=s_vec, k=k, y=y, i=investment, c=c)
        self.sim = sim
        return sim

    def s_path(self, s0, phi, s_inf=None):
        """Return s_t = s_inf + (s0-s_inf)*phi**t.

        ``s_inf`` defaults to the calibrated long-run rate ``s_bar``. The same
        method therefore covers both the two- and three-parameter rules.
        """

        par = self.par
        if s_inf is None:
            s_inf = par.s_bar
        if not 0 <= s0 <= 1:
            raise ValueError("s0 must be between zero and one.")
        if not 0 <= phi < 1:
            raise ValueError("phi must be in [0,1).")
        if not 0 <= s_inf <= 1:
            raise ValueError("s_inf must be between zero and one.")

        t = np.arange(par.T)
        return s_inf + (s0 - s_inf) * phi**t

    def welfare(self, c):
        """Return discounted log utility, W = sum(beta**t * log(c_t))."""

        c = np.asarray(c, dtype=float)
        if c.ndim != 1 or c.size != self.par.T:
            raise ValueError(f"Consumption must be a path with {self.par.T} elements.")
        if np.any(c <= 0):
            return -np.inf

        discount = self.par.beta ** np.arange(self.par.T)
        return float(np.sum(discount * np.log(c)))

    def evaluate(self, s0, phi, s_inf=None):
        """Return welfare generated by the exponential savings rule."""

        s = self.s_path(s0, phi, s_inf=s_inf)
        sim = self.simulate(s)
        return self.welfare(sim.c)

    def grid_welfare(self, s0_grid, phi_grid):
        """Evaluate the two-parameter rule on a rectangular grid.

        Rows correspond to ``phi_grid`` and columns correspond to ``s0_grid``.
        """

        s0_grid = np.asarray(s0_grid, dtype=float)
        phi_grid = np.asarray(phi_grid, dtype=float)
        W = np.empty((phi_grid.size, s0_grid.size))

        for i, phi in enumerate(phi_grid):
            for j, s0 in enumerate(s0_grid):
                W[i, j] = self.evaluate(s0, phi)
        return W

    def stretched_s_path(self, s0, phi, s_inf, power):
        r"""Return a stretched-exponential savings transition.

        The alternative rule is

            s_t = s_inf + (s0-s_inf)*phi**(t**power).

        It nests the three-parameter rule at ``power=1``. For admissible
        parameters it remains between ``s0`` and ``s_inf`` and converges to
        ``s_inf``.
        """

        if not 0 <= s0 <= 1:
            raise ValueError("s0 must be between zero and one.")
        if not 0 <= phi < 1:
            raise ValueError("phi must be in [0,1).")
        if not 0 <= s_inf <= 1:
            raise ValueError("s_inf must be between zero and one.")
        if power <= 0:
            raise ValueError("power must be strictly positive.")

        t = np.arange(self.par.T)
        return s_inf + (s0 - s_inf) * phi ** (t**power)

    def evaluate_stretched(self, s0, phi, s_inf, power):
        """Return welfare generated by the stretched-exponential rule."""

        s = self.stretched_s_path(s0, phi, s_inf, power)
        sim = self.simulate(s)
        return self.welfare(sim.c)
