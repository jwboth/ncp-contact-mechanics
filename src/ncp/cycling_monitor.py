import logging
import numpy as np
import porepy as pp
from porepy.numerics.nonlinear.convergence_check import ConvergenceStatus
from abc import abstractmethod
from typing import Tuple

logger = logging.getLogger(__name__)


class NewtonWithCyclingCheck(pp.NewtonSolver):
    """Abstract class for Newton solvers with cycling check.

    Requires implementation of cycling check and reset methods.

    """

    @abstractmethod
    def reset_cycling_analysis(self): ...

    @abstractmethod
    def check_cycling(self, model) -> bool:
        """Check for cycling.

        Parameters:
            model: The model to check for cycling.

        Returns:
            bool: True if cycling is detected, False otherwise.

        """

    def check_convergence(
        self, model, nonlinear_increment: np.ndarray
    ) -> Tuple[ConvergenceStatus, dict]:
        """Check for convergence, including cycling check.

        Parameters:
            model: The model to check for convergence.
            nonlinear_increment: The nonlinear increment to check for convergence.

        Returns:
            Tuple[ConvergenceStatus, dict]: The convergence status and additional info.

        """

        # Standard convergence check
        convergence_status, info = super().check_convergence(model, nonlinear_increment)

        # Cycling check
        is_cycling = self.check_cycling(model)
        if is_cycling:
            convergence_status = ConvergenceStatus.CYCLING

        return convergence_status, info

    def solve(self, model) -> ConvergenceStatus:
        """Overwritten solve method to reset cycling analysis at the start of each solve.

        Parameters:
            model: The model to solve.

        Returns:
            ConvergenceStatus: The convergence status after solving.

        """
        self.reset_cycling_analysis()
        return super().solve(model)


class CyclingCriterion:
    """Implements a check for cycling."""

    @abstractmethod
    def fetch_cycling_objectives(self, model) -> dict[str, dict[str, np.ndarray]]:
        """Fetch objectives to monitor for cycling."""

    def reset_cycling_analysis(self):
        """Clean up all cached data for cycling analysis."""

        if hasattr(self, "cached_objectives"):
            del self.cached_objectives
        self.num_cached_objectives = 0

    def initialize_cycling_cache(self):
        """Initialize cache."""
        if not hasattr(self, "cached_objectives"):
            self.cached_objectives = {}
            self.num_cached_objectives = 0
        if "discrete" not in self.cached_objectives:
            self.cached_objectives["discrete"] = {}
        if "continuous" not in self.cached_objectives:
            self.cached_objectives["continuous"] = {}

    def clean_cycling_cache(self):
        """Make sure the cache does not grow too large."""
        for outer_key in self.cached_objectives:
            for inner_key in self.cached_objectives[outer_key]:
                assert isinstance(self.cached_objectives[outer_key][inner_key], list)
                while len(self.cached_objectives[outer_key][inner_key]) > 10:
                    self.cached_objectives[outer_key][inner_key].pop(0)

    def update_cycling_cache(self, objectives: dict) -> None:
        """Update the cache with new objectives.

        Parameters:
            objectives: The objectives to add to the cache.

        """
        # Update cache.
        for key_outer in objectives:
            for key_inner in objectives[key_outer]:
                self.cached_objectives[key_outer][key_inner].append(
                    objectives[key_outer][key_inner]
                )

        # Update number of cached objectives.
        self.num_cached_objectives = min(
            [
                len(self.cached_objectives[outer_key][inner_key])
                for outer_key in self.cached_objectives
                for inner_key in self.cached_objectives[outer_key]
            ]
        )

    def check_cycling(self, model) -> bool:
        """Check for cycling in contact states.

        Parameters:
            model: The model to check for cycling.

        Returns:
            bool: True if cycling is detected, False otherwise.

        """

        # Initialize state.
        self.initialize_cycling_cache()

        # Fetch objectives.
        objectives = self.fetch_cycling_objectives(model)
        if not hasattr(self, "previous_objectives"):
            self.previous_objectives = self.fetch_cycling_objectives(model)

        # Check for cycling based on 1% closedness
        cycling_window = 0
        for i in range(self.num_cached_objectives - 1, 1, -1):
            if (
                all(
                    [
                        np.all(
                            objectives["discrete"][key]
                            == self.cached_objectives["discrete"][key][i]
                        )
                        for key in objectives["discrete"]
                    ]
                )
                and all(
                    [
                        np.all(
                            self.cached_objectives["discrete"][key][-1]
                            == self.cached_objectives["discrete"][key][i - 1]
                        )
                        for key in self.cached_objectives["discrete"]
                    ]
                )
                # TODO: Need both checks?
                and all(
                    [
                        np.allclose(
                            objectives["continuous"][key],
                            self.cached_objectives["continuous"][key][i],
                            rtol=1e-2,
                        )
                        for key in objectives["continuous"]
                    ]
                )
                and all(
                    [
                        np.allclose(
                            self.cached_objectives["continuous"][key][-1],
                            self.cached_objectives["continuous"][key][i - 1],
                            rtol=1e-2,
                        )
                        for key in self.cached_objectives["continuous"]
                    ]
                )
            ):
                cycling_window = len(self.cached_objectives["contact_states"]) - i
                logger.info(f"Cycling detected with window {cycling_window}.")
                break

        # Conclude.
        is_cycling = cycling_window > 0

        # Monitor - pass cycling information to pp.SolverStatistics object for logging.
        model.nonlinear_solver_statistics.log_custom_data(cycling_window=cycling_window)

        # Update cache
        self.update_cycling_cache(objectives)

        # Clean up cache
        self.clean_cycling_cache()

        return is_cycling


class ContactMechanicsCyclingCriterion(CyclingCriterion):
    """Implements a check for cycling in contact mechanics."""

    def initialize_cycling_cache(self):
        """Initialize cache."""
        super().initialize_cycling_cache()

        if "contact_states" not in self.cached_objectives["discrete"]:
            self.cached_objectives["discrete"]["contact_states"] = []
        if "contact_traction" not in self.cached_objectives["continuous"]:
            self.cached_objectives["continuous"]["contact_traction"] = []
        if "displacement_jump" not in self.cached_objectives["continuous"]:
            self.cached_objectives["continuous"]["displacement_jump"] = []

    def fetch_cycling_objectives(self, model) -> dict[str, dict[str, np.ndarray]]:
        """Auxiliary function to fetch relevant objectives."""
        contact_states = model.compute_fracture_states()

        subdomains = model.mdg.subdomains(dim=model.nd - 1)
        contact_traction = model.equation_system.evaluate(
            model.contact_traction(subdomains)
        )
        displacement_jump = model.equation_system.evaluate(
            model.displacement_jump(subdomains)
        )
        return {
            "discrete": {
                "contact_states": contact_states,
            },
            "continuous": {
                "contact_traction": contact_traction,
                "displacement_jump": displacement_jump,
            },
        }
