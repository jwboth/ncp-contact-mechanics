"""Compute states of each fracture cell."""

from functools import partial

import numpy as np
from icecream import ic

import porepy as pp


class FractureStates:
    """Compute states of each fracture cell using the cumulative tangential"""

    def compute_fracture_states(self, split_output: bool = False, trial=False):
        """
        Compute states of each fracture cell, based on the textbook criteria.
        Returns a list where: Open=0, Sticking=1, Gliding=2

        Args:
            split_output (bool, optional): Whether to split the output into subdomain-corresponding vectors. Defaults to False.
            tol (float, optional): Tolerance for the yield criterion. Defaults to 1e-11.

        NOTE: Aim at using the same tolerance here as in the model.

        """
        # Active sets
        states = []
        subdomains = self.mdg.subdomains(dim=self.nd - 1)

        # Compute ingredients characterizing the normal contact state
        nd_vec_to_normal = self.normal_component(subdomains)
        nd_vec_to_tangential = self.tangential_component(subdomains)
        t_n: pp.ad.Operator = nd_vec_to_normal @ self.contact_traction(subdomains)
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(subdomains)
        c_n = self.contact_mechanics_numerical_constant(subdomains)
        gap = c_n * (u_n - self.fracture_gap(subdomains))

        # Compute the friction bound and the yield criterion
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        b = self.friction_bound(subdomains)
        t_t = self.tangential_component(subdomains) @ self.contact_traction(subdomains)
        yield_criterion = b - f_norm(t_t)
        b_eval = b.value(self.equation_system)
        yield_criterion_eval = yield_criterion.value(self.equation_system)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)

        # Combine the above into expressions that enter the equation
        ut_val = np.linalg.norm(
            u_t.value(self.equation_system).reshape((self.nd - 1, -1), order="F"),
            axis=0,
        )

        tol = self.numerical.open_state_tolerance

        # Determine the state of each fracture cell
        conversion = {
            "open": 2,
            "stick": 0,
            "slip": 1,
            "unknown": -1,
        }
        if trial:
            opening_ind = -t_n.value(self.equation_system) - gap.value(
                self.equation_system
            )
            for op, sl in zip(opening_ind, ut_val):
                if op <= tol:
                    states.append(conversion["open"])
                elif sl > tol:
                    states.append(conversion["slip"])
                else:
                    states.append(conversion["stick"])
        else:
            failure = False
            for b_val, yc_val in zip(b_eval, yield_criterion_eval):
                if b_val <= tol:
                    states.append(conversion["open"])
                elif yc_val > tol:
                    states.append(conversion["stick"])
                elif yc_val <= tol:
                    states.append(conversion["slip"])
                else:
                    states.append(conversion["unknown"])
                    if not failure:
                        print("Should not get here.", b_val, yc_val, tol)
                        failure = True

        # Split combined states vector into subdomain-corresponding vectors
        if split_output:
            split_states = []
            num_cells = []
            for sd in subdomains:
                prev_num_cells = int(sum(num_cells))
                split_states.append(
                    np.array(states[prev_num_cells : prev_num_cells + sd.num_cells])
                )
                num_cells.append(sd.num_cells)
            return split_states
        else:
            return states

    def fetch_fracture_vars(self):
        # Manage the states
        traction_states = []
        displacement_jump_states = []
        subdomains = self.mdg.subdomains(dim=self.nd - 1)

        # Variables
        traction_states.append(
            self.contact_traction(subdomains).value(self.equation_system)
        )
        displacement_jump_states.append(
            self.displacement_jump(subdomains).value(self.equation_system)
        )

        return traction_states, displacement_jump_states

    def fetch_fracture_residuals(self):
        # Manage the residuals
        normal_residuals = []
        tangential_residuals = []
        subdomains = self.mdg.subdomains(dim=self.nd - 1)

        normal_residuals.append(
            pp.momentum_balance.MomentumBalance.normal_fracture_deformation_equation(
                self, subdomains
            ).value(self.equation_system)
        )
        tangential_residuals.append(
            pp.momentum_balance.MomentumBalance.tangential_fracture_deformation_equation(
                self, subdomains
            ).value(self.equation_system)
        )

        return normal_residuals, tangential_residuals
