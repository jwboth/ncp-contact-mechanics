"""Compute states of each fracture cell."""

from functools import partial

import numpy as np
import logging

import porepy as pp
from typing import cast


class FractureStates:
    """Compute states of each fracture cell using the cumulative tangential"""

    def compute_fracture_states(self, concatenate: bool = True):
        """
        Compute states of each fracture cell, based on the textbook criteria.

        Value convention:
        - Stick=0
        - Slip=1
        - Open=2

        Args:
            concatenate (bool, optional): Whether to concatenate the output into a
            single vector. Defaults to True.

        """
        # Preparations.
        states = []
        subdomains = self.mdg.subdomains(dim=self.nd - 1)

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

        # Use consistent tolerance as in the equations to discuss boundary cases.
        tol = self.numerical.open_state_tolerance

        # Determine the state of each fracture cell.
        conversion = {
            "stick": 0,
            "slip": 1,
            "open": 2,
            "unknown": -1,
        }
        for tn_val, yc_val in zip(t_n_eval, yield_criterion_eval):
            if tn_val >= -tol:
                states.append(conversion["open"])
            elif yc_val > tol:
                states.append(conversion["stick"])
            elif yc_val <= tol:
                states.append(conversion["slip"])
            else:
                states.append(conversion["unknown"])
                logging.info("Should not get here.", tn_val, yc_val, tol)

        # Return in requested format
        if concatenate:
            return np.array(states)
        else:
            split_states = []
            num_cells = []
            for sd in subdomains:
                prev_num_cells = int(sum(num_cells))
                split_states.append(
                    np.array(states[prev_num_cells : prev_num_cells + sd.num_cells])
                )
                num_cells.append(sd.num_cells)
            return split_states


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
