import logging
import numpy as np
import porepy as pp
from porepy.numerics.nonlinear.convergence_check import ConvergenceStatus
from abc import abstractmethod

logger = logging.getLogger(__name__)


class NewtonWithCyclingCheck(pp.NewtonSolver):
    @abstractmethod
    def reset_cycling_analysis(self): ...

    @abstractmethod
    def check_cycling(self, model) -> bool: ...

    def check_convergence(self, model, nonlinear_increment):
        # Standard convergence check
        convergence_status, nonlinear_increment_norm, residual_norm = (
            super().check_convergence(model, nonlinear_increment)
        )

        # Cycling check
        is_cycling = self.check_cycling(model)
        if is_cycling:
            convergence_status = ConvergenceStatus.CYCLING

        return convergence_status, nonlinear_increment_norm, residual_norm

    def solve(self, model) -> ConvergenceStatus:
        self.reset_cycling_analysis()
        return super().solve(model)


class CyclingCriterion:
    """Implements a check for cycling."""

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

    def update_cycling_cache(self, objectives: dict):
        for key_outer in objectives:
            for key_inner in objectives[key_outer]:
                self.cached_objectives[key_outer][key_inner].append(
                    objectives[key_outer][key_inner]
                )
        self.update_num_cached_objectives()

    def update_num_cached_objectives(self):
        self.num_cached_objectives = min(
            [
                len(self.cached_objectives[outer_key][inner_key])
                for outer_key in self.cached_objectives
                for inner_key in self.cached_objectives[outer_key]
            ]
        )

    def check_cycling(self, model):
        """Check for cycling in contact states."""

        # Initialize state.
        self.initialize_cycling_cache()

        # Fetch objectives.
        objectives = self.fetch_cycling_objectives(model)
        if not hasattr(self, "previous_objectives"):
            self.previous_objectives = self.fetch_cycling_objectives(model)

        # Monitor some infos.
        _ = self.monitor_discrete_changes(objectives)

        # Check for cycling based on 1% closedness
        cycling_window = 0
        for i in range(self.num_cached_objectives - 1, 1, -1):
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

        # # Monitor.
        # TODO pass info somehow.
        # model.nonlinear_solver_statistics.cycling_window = cycling_window

        # Update cache
        self.update_cycling_cache(objectives)

        # Clean up cache
        self.clean_cycling_cache()

        # TODO pass info
        return is_cycling  # , cycling_window


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

    # TODO Make this part of iteration exporting? or solver statistics?
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
        # self.nonlinear_solver_statistics.num_contact_states = num_contact_states
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
        # self.nonlinear_solver_statistics.total_contact_state_changes_in_time = (
        #    total_discrete_changes_in_time["contact_states"]
        # )
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
        # self.nonlinear_solver_statistics.total_contact_state_changes = (
        #     total_discrete_changes["contact_states"]
        # )
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
        # self.nonlinear_solver_statistics.contact_state_changes = discrete_changes
        logger.info(f"Changes in states: \n{discrete_changes}")

        return total_discrete_changes
