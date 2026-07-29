"""Compute states of each fracture cell."""

from functools import partial

import numpy as np
import logging

import porepy as pp
from typing import cast

from enum import IntEnum

logger = logging.getLogger(__name__)


class FractureState(IntEnum):
    """Fracture states."""

    STICK = 0
    SLIP = 1
    OPEN = 2

    @property
    def values(self) -> list["FractureState"]:
        """Return the range of admissible states."""
        return [FractureState.STICK, FractureState.SLIP, FractureState.OPEN]

    def astype(self, dtype) -> int:
        if dtype == int:
            return self.value
        else:
            raise ValueError(f"Cannot convert FractureState to {dtype}")

    # Define != operator
    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        if isinstance(other, FractureState):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        else:
            return False


class FractureStates:
    """Criteria for fracture states, and tools for monitoring."""

    mdg: pp.MixedDimensionalGrid
    """The mixed-dimensional grid."""

    equation_system: pp.ad.EquationSystem
    """The equation system."""

    numerical: pp.NumericalConstants
    """Numerical parameters."""

    normal_component: pp.ad.Operator
    """Operator to extract the normal component of a vector."""

    tangential_component: pp.ad.Operator
    """Operator to extract the tangential component of a vector."""

    friction_bound: pp.ad.Operator
    """The friction bound operator."""

    characteristic_contact_traction: pp.ad.Function
    """Characteristic contact traction function."""

    friction_coefficient: pp.ad.Function
    """Friction coefficient function."""

    contact_traction: pp.ad.Operator
    """Contact traction operator."""

    displacement_jump: pp.ad.Operator
    """Displacement jump operator."""

    nd: int
    """Number of spatial dimensions."""

    units: pp.Units
    """Units."""

    def compute_fracture_states(
        self, concatenate: bool = True
    ) -> np.ndarray | list[np.ndarray]:
        """
        Compute states of each fracture cell, based on the textbook criteria.

        Args:
            concatenate (bool, optional): Whether to concatenate the outputs shall be merged
                across all fracture domains or reported separately.

        Returns:
            np.ndarray: The states.
            dict[str, float]: The area of each fracture state.

            If concatenate is False, returns a list of tuples, one per fracture subdomain.

        """
        subdomains = self.mdg.subdomains(dim=self.nd - 1)
        if concatenate:
            return self._compute_fracture_states(subdomains)
        else:
            return [self._compute_fracture_states([sd]) for sd in subdomains]

    def _compute_fracture_states(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """
        Compute states of each fracture cell, based on the textbook criteria.

        Args:
            subdomains (list[pp.Grid]): List of fracture subdomains.

        Returns:
            np.ndarray | The states of each fracture cell in subdomains.

        """
        # Preparations.
        states = []

        # Compute normal traction to decide: open vs closed.
        t_n: pp.ad.Operator = self.normal_component(subdomains) @ self.contact_traction(
            subdomains
        )
        t_n_eval = self.equation_system.evaluate(t_n)

        # Compute the yield criterion to decide: stick vs slip.
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        t_t = self.tangential_component(subdomains) @ self.contact_traction(subdomains)
        yield_criterion = self.friction_bound(subdomains) - f_norm(t_t)
        yield_criterion_eval = self.equation_system.evaluate(yield_criterion)

        # Determine tangential slip velocity to decide: stick vs slip.
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(
            self.tangential_component(subdomains) @ self.displacement_jump(subdomains)
        )
        u_t_increment_eval = self.equation_system.evaluate(u_t_increment)
        norm_u_t_increment_eval = np.linalg.norm(u_t_increment_eval, axis=0)

        # Determine the state of each fracture cell. Check the normal traction and the yield
        # criterion. Use consistent tolerance as in the equations to discuss boundary cases.
        tol = self.numerical.open_state_tolerance
        for tn_val, yc_val in zip(t_n_eval, yield_criterion_eval):
            if tn_val >= -tol:
                states.append(FractureState.OPEN)
            elif yc_val > tol:
                states.append(FractureState.STICK)
            elif yc_val <= tol and norm_u_t_increment_eval < tol:
                states.append(FractureState.STICK)
            elif yc_val <= tol:
                states.append(FractureState.SLIP)
            else:
                raise ValueError(f"Non-admissible state: tn={tn_val}, yc={yc_val}")

        return np.array(states)

    def num_fracture_states(self) -> dict[str, int]:
        """Compute the number of each fracture state.

        Returns:
            dict[str, int]: A dictionary with the number of each fracture state.

        """
        fracture_states = self.compute_fracture_states()
        num_fracture_states = {
            state.name: int(np.count_nonzero(fracture_states == state))
            for state in FractureState
        }
        num_fracture_states["total"] = int(len(fracture_states))
        return num_fracture_states

    def area_fracture_states(self) -> dict[str, float]:
        """Compute the area of each fracture state.

        Returns:
            dict[str, float]: A dictionary with the areas of each fracture state.

        """
        subdomains = self.mdg.subdomains(dim=self.nd - 1)
        discrete_fracture_states = self._compute_fracture_states(subdomains)
        cell_volumes = np.concatenate([sd.cell_volumes for sd in subdomains])
        area_fracture_states = {
            state.name: f"{float(np.sum(cell_volumes[discrete_fracture_states == state])):.1e}"
            for state in FractureState
        }
        area_fracture_states["total"] = f"{float(np.sum(cell_volumes)):.1e}"
        return area_fracture_states

    def num_fracture_states_diff(self, fracture_states_1, fracture_states_2) -> int:
        """Compute the number of different fracture states between two sets of states.

        Args:
            fracture_states_1 (np.ndarray): First set of fracture states.
            fracture_states_2 (np.ndarray): Second set of fracture states.

        Returns:
            int: The number of different fracture states.

        """
        if len(fracture_states_1) != len(fracture_states_2):
            raise ValueError("Fracture states must have the same length.")
        return int(np.count_nonzero(fracture_states_1 != fracture_states_2))

    def area_fracture_states_diff(self, fracture_states_1, fracture_states_2) -> float:
        """Compute the area of changing fracture states between two sets of states.

        Args:
            fracture_states_1 (np.ndarray): First set of fracture states.
            fracture_states_2 (np.ndarray): Second set of fracture states.

        Returns:
            float: The difference in area of changing fracture states.

        """
        subdomains = self.mdg.subdomains(dim=self.nd - 1)
        cell_volumes = np.concatenate([sd.cell_volumes for sd in subdomains])
        return np.sum(
            (fracture_states_1 != fracture_states_2).astype(int) * cell_volumes
        )

    def log_fracture_state_statistics(self):
        """Monitor objectives."""

        # Determine current fracture state status
        num_fracture_states = self.num_fracture_states()
        area_fracture_states = self.area_fracture_states()

        # Determine change in discrete objectives in time.
        # Caching of previous time step is build on the assumption that the
        # the counter of nonlinear iterations is increased after logging.
        fracture_states = self.compute_fracture_states()
        if (
            not hasattr(self, "previous_timestep_fracture_states")
            or self.nonlinear_solver_statistics.num_iterations == 0
        ):
            self.previous_timestep_fracture_states = fracture_states.copy()
        num_fracture_states_diff_in_time = self.num_fracture_states_diff(
            fracture_states, self.previous_timestep_fracture_states
        )
        area_fracture_states_diff_in_time = self.area_fracture_states_diff(
            fracture_states, self.previous_timestep_fracture_states
        )

        # Determine total changes in iterations.
        if not hasattr(self, "previous_iteration_fracture_states"):
            self.previous_iteration_fracture_states = fracture_states.copy()
        num_fracture_states_diff_in_iteration = self.num_fracture_states_diff(
            fracture_states, self.previous_iteration_fracture_states
        )
        area_fracture_states_diff_in_iteration = self.area_fracture_states_diff(
            fracture_states, self.previous_iteration_fracture_states
        )

        # Cache current states for next iteration.
        self.previous_iteration_fracture_states = fracture_states.copy()

        # Logging.
        logger.info(f"Number of fracture states: {num_fracture_states}")
        logger.info(f"Area of fracture states: {area_fracture_states}")
        logger.info(f"Number changes in time: {num_fracture_states_diff_in_time}")
        logger.info(
            f"Area of changing fracture states in time: {area_fracture_states_diff_in_time:.1e}"
        )
        logger.info(
            f"Number changes in iteration: {num_fracture_states_diff_in_iteration}"
        )
        logger.info(
            f"Area of changing fracture states in iteration: {area_fracture_states_diff_in_iteration:.1e}"
        )

        # Solver statistics logging - differentiated between non-fixed and fixed for
        # the time step.
        self.nonlinear_solver_statistics.log_custom_data(
            **{
                "num_fracture_states": num_fracture_states,
                "area_fracture_states": area_fracture_states,
                "num_fracture_states_changes_in_iteration": num_fracture_states_diff_in_iteration,
                "area_fracture_states_changes_in_iteration": area_fracture_states_diff_in_iteration,
            },
            append=True,
        )
        self.nonlinear_solver_statistics.log_custom_data(
            **{
                "num_fracture_states_changes_in_time": num_fracture_states_diff_in_time,
                "area_fracture_states_changes_in_time": area_fracture_states_diff_in_time,
            },
            append=False,
        )

    def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        """Hook to be called after each nonlinear iteration."""
        self.log_fracture_state_statistics()
        super().after_nonlinear_iteration(nonlinear_increment)


class NCPContactIndicators(pp.models.solution_strategy.ContactIndicators):
    def opening_indicator(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return super().opening_indicator(subdomains)

    def sliding_indicator(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Function describing the state of the sliding constraint."""

        # TODO sign?

        # Functions
        f_heaviside = pp.ad.Function(partial(pp.ad.heaviside, 0), "heaviside_function")
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        # Basis vector combinations
        # Mapping from a full vector to the tangential component
        nd_vec_to_tangential = self.tangential_component(subdomains)

        tangential_basis = self.basis(subdomains, dim=self.nd - 1)

        # Variables: The tangential component of the contact traction and the
        # displacement jump
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)
        # The time increment of the tangential displacement jump
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)
        # The friction bound
        friction_bound = self.friction_bound(subdomains)
        # The yield criterion
        yield_criterion = self.yield_criterion(subdomains)
        # Stick condition
        orthogonality = self.orthogonality(subdomains)
        c_num_to_one = self.contact_mechanics_numerical_constant_t(subdomains)
        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)
        u_t_increment_scaled_to_one = (
            scalar_to_tangential @ c_num_to_one
        ) * u_t_increment
        u_t_increment_scaled_to_one.set_name("u_t_increment_scaled_to_one")
        stick_condition = (
            orthogonality - f_norm(u_t_increment_scaled_to_one) * friction_bound
        )

        h_oi = f_heaviside(self.opening_indicator(subdomains))
        ind = stick_condition - yield_criterion

        if self.params.get("adaptive_indicator_scaling", False):
            # Base on all fracture subdomains
            all_subdomains = self.mdg.subdomains(dim=self.nd - 1)
            scale_op = self.contact_traction_estimate(all_subdomains)
            scale = self.compute_traction_norm(
                cast(np.ndarray, self.equation_system.evaluate(scale_op))
            )
            ind = ind / pp.ad.Scalar(scale)
        return ind * h_oi
