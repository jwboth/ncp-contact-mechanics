import logging
import numpy as np
from typing import List, Optional
import porepy as pp

logger = logging.getLogger(__name__)


class CyclingMonitor:
    """Implements a check for cycling."""

    def reset_cycling_analysis(self):
        """Clean up all cached data for cycling analysis."""

        if hasattr(self, "cached_objectives"):
            del self.cached_objectives

    def initialize_cache(self):
        """Initialize cache."""
        if not hasattr(self, "cached_objectives"):
            self.cached_objectives = {}

    def clean_cache(self):
        """Make sure the cache does not grow too large."""
        for outer_key in self.cached_objectives:
            for inner_key in self.cached_objectives[outer_key]:
                assert isinstance(self.cached_objectives[outer_key][inner_key], list)
                while len(self.cached_objectives[outer_key][inner_key]) > 10:
                    self.cached_objectives[outer_key][inner_key].pop(0)

    def update_cache(self, objectives: dict):
        for key_outer in objectives:
            for key_inner in objectives[key_outer]:
                self.cached_objectives[key_outer][key_inner].append(
                    objectives[key_outer][key_inner]
                )

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """Integrate iteration export into simulation workflow.

        Order of operations is important, super call distributes the solution
        to iterate subdictionary.

        """
        super().after_nonlinear_iteration(solution_vector)
        self.check_cycling()

    def after_nonlinear_convergence(self):
        super().after_nonlinear_convergence()
        self.reset_cycling_analysis()


class ContactMechanicsCyclingMonitor(CyclingMonitor):
    """Implements a check for cycling in contact mechanics."""

    nonlinear_solver_statistics: pp.SolverStatistics
    """Solver statistics to monitor cycling."""

    equation_system: pp.ad.EquationSystem
    """Equation system to evaluate objectives."""

    mdg: pp.MixedDimensionalGrid
    """The mixed-dimensional grid."""

    nd: int
    """Number of spatial dimensions."""

    contact_traction: pp.ad.Operator
    """Contact traction operator."""

    displacement_jump: pp.ad.Operator
    """Displacement jump operator."""

    # def reset(self):
    #    """Clean up all cached data for cycling analysis."""

    #    super().reset()

    #    if "contact_states" in self.cached_objectives:
    #        del self.cached_objectives["contact_states"]
    #    if "contact_variables" in self.cached_objectives:
    #        del self.cached_objectives["contact_variables"]

    def initialize_cache(self):
        """Initialize cache."""
        super().initialize_cache()

        if "discrete" not in self.cached_objectives:
            self.cached_objectives["discrete"] = {}
        if "continuous" not in self.cached_objectives:
            self.cached_objectives["continuous"] = {}
        if "contact_states" not in self.cached_objectives["discrete"]:
            self.cached_objectives["discrete"]["contact_states"] = []
        if "contact_traction" not in self.cached_objectives["continuous"]:
            self.cached_objectives["continuous"]["contact_traction"] = []
        if "displacement_jump" not in self.cached_objectives["continuous"]:
            self.cached_objectives["continuous"]["displacement_jump"] = []

    def fetch_objectives(self) -> dict[str, dict[str, np.ndarray]]:
        """Auxiliary function to fetch relevant objectives."""
        contact_states = self.compute_fracture_states()

        subdomains = self.mdg.subdomains(dim=self.nd - 1)
        contact_traction = self.equation_system.evaluate(
            self.contact_traction(subdomains)
        )
        displacement_jump = self.equation_system.evaluate(
            self.displacement_jump(subdomains)
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

    def check_cycling(self):
        """Check for cycling in contact states."""

        # Initialize state.
        self.initialize_cache()

        # Fetch objectives.
        objectives = self.fetch_objectives()
        if not hasattr(self, "previous_objectives"):
            self.previous_objectives = self.fetch_objectives()

        # Monitor some infos.
        _ = self.monitor_discrete_changes(objectives)

        # Check for cycling based on 1% closedness
        cycling_window = 0
        for i in range(
            len(self.cached_objectives["discrete"]["contact_states"]) - 1, 1, -1
        ):
            if (
                all(
                    [
                        np.allclose(
                            objectives["discrete"][key],
                            self.cached_objectives["discrete"][key][i],
                        )
                        for key in objectives["discrete"]
                    ]
                )
                and all(
                    [
                        np.allclose(
                            self.cached_objectives["discrete"][key][-1],
                            self.cached_objectives["discrete"][key][i - 1],
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

        # Monitor.
        self.nonlinear_solver_statistics.cycling_window = cycling_window

        # Update cache
        self.update_cache(objectives)

        # Clean up cache
        self.clean_cache()

        return is_cycling

    def monitor_discrete_changes(self, objectives: dict):
        """Monitor objectives."""
        # Determine total number of contact states
        num_contact_states = [
            int(
                np.sum(
                    np.isclose(objectives["discrete"]["contact_states"], i).astype(int)
                )
            )
            for i in range(3)
        ]
        self.nonlinear_solver_statistics.num_contact_states = num_contact_states
        logger.info(f"Number of contact states: {num_contact_states}")

        # Determine change in discrete objectives in time.
        total_discrete_changes_in_time = {
            key: np.count_nonzero(
                np.logical_not(
                    np.isclose(
                        objectives["discrete"][key],
                        self.previous_objectives["discrete"][key],
                    )
                )
            )
            for key in objectives["discrete"]
        }
        self.nonlinear_solver_statistics.total_contact_state_changes_in_time = (
            total_discrete_changes_in_time["contact_states"]
        )
        logger.info(f"Total discrete changes in time: {total_discrete_changes_in_time}")

        # Determine total changes in iterations.
        total_discrete_changes = {
            key: np.count_nonzero(
                np.logical_not(
                    np.isclose(
                        objectives["discrete"][key],
                        self.cached_objectives["discrete"][key][-1],
                    )
                )
            )
            if len(self.cached_objectives["discrete"][key]) > 0
            else 0
            for key in objectives["discrete"]
        }
        self.nonlinear_solver_statistics.total_contact_state_changes = (
            total_discrete_changes["contact_states"]
        )
        logger.info(f"Total discrete changes: {total_discrete_changes}")

        # Determine detailed contact state changes
        discrete_changes = np.zeros((3, 3), dtype=int)
        if len(self.cached_objectives["discrete"]["contact_states"]) > 0:
            for i in range(3):
                for j in range(3):
                    discrete_changes[i, j] = int(
                        np.sum(
                            np.logical_and(
                                np.isclose(objectives["discrete"]["contact_states"], i),
                                np.isclose(
                                    self.cached_objectives["discrete"][
                                        "contact_states"
                                    ][-1],
                                    j,
                                ),
                            ).astype(int)
                        )
                    )
        else:
            for i in range(3):
                discrete_changes[i, i] = num_contact_states[i]
        self.nonlinear_solver_statistics.contact_state_changes = discrete_changes
        logger.info(f"Changes in states: \n{discrete_changes}")

        return total_discrete_changes
