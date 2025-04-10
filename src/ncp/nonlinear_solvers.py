"""porepy's Newton solver with AA"""

import logging

import numpy as np
import porepy as pp
from icecream import ic

# from porepy.numerics.solvers.andersonacceleration import (
#    AdaptiveAndersonAcceleration,  # TODO
# )
from porepy.numerics.solvers.andersonacceleration import AndersonAcceleration

logger = logging.getLogger(__name__)


class AANewtonSolver(pp.NewtonSolver):
    """Newton solver with Anderson acceleration.

    In order to activate the Anderson acceleration,
    the parameter `aa_depth` must be set to a positive integer.

    """

    def _reset(self, model):
        self.adaptive_depth = 0
        if hasattr(self, "aa"):
            self.aa.reset(0)
        model.reset_cycling_analysis()
        if hasattr(self, "init_iteration"):
            del self.init_iteration
        if hasattr(self, "adaptive_depth"):
            del self.adaptive_depth
        logger.info("Reset AA")

    def solve(self, model) -> tuple[bool, int]:
        # Check if the model has a minimum fb switch, and update
        if hasattr(model, "update_min_fb_switch"):
            model.update_min_fb_switch(active_min=True)
        self.adaptive_alpha = 1.0
        self.new_cycling = False
        is_converged = super().solve(model)
        self._reset(model)
        return is_converged

    def iteration(self, model) -> np.ndarray:
        """A single Newton iteration, accelerated by a AA step.

        Parameters:
            model: The model instance specifying the problem to be solved.

        Returns:
            np.ndarray: Solution to linearized system, i.e. the update increment.

        """

        model.assemble_linear_system()
        nonlinear_increment = model.solve_linear_system()
        aa_depth = self.params["aa_depth"]
        use_aa = aa_depth > 0
        use_relaxation = aa_depth == -1
        use_adaptive_relaxation = aa_depth == -2
        use_min_fb_switch = aa_depth == -3
        use_adaptive_cnum = aa_depth == -4
        use_small_change_aa = aa_depth == -5
        use_decreasing_residuals_aa = aa_depth == -6

        if (
            use_aa
            or use_relaxation
            or use_adaptive_relaxation
            or use_min_fb_switch
            or use_adaptive_cnum
            or use_small_change_aa
            or use_decreasing_residuals_aa
        ):
            # Initialize AA
            if not hasattr(self, "aa"):
                ndofs = nonlinear_increment.size
                self.aa = AndersonAcceleration(ndofs, 0)
                logger.info("Initialize AA")

            # Monitor cycling
            if not hasattr(self, "adaptive_depth") or self.adaptive_depth <= 0:
                self.adaptive_depth = 0
                if self.adaptive_depth == 0 and hasattr(model, "cycling_window"):
                    logger.info(
                        f"Activate AA with windows size: {model.cycling_window}"
                    )
                    self.adaptive_depth = max(model.cycling_window - 1, 0)
                if self.adaptive_depth == 0 and hasattr(model, "stagnating_states"):
                    logger.info(
                        f"""Activate AA with stagnating states: """
                        f"""{model.stagnating_states}"""
                    )
                    self.adaptive_depth = aa_depth if model.stagnating_states else 0

            if use_small_change_aa:
                if hasattr(model, "small_changes") and model.small_changes:
                    self.adaptive_depth = 1
                    logger.info("Activate AA with small changes")

            if use_decreasing_residuals_aa:
                if (
                    hasattr(model, "decreasing_residuals")
                    and model.decreasing_residuals
                ):
                    self.adaptive_depth = 1
                    logger.info("Activate AA with decreased residuals")
                else:
                    self.adaptive_depth = 0

            if self.adaptive_depth > 0 and not hasattr(self, "init_iteration"):
                # Update depth
                # TODO Use depth 1?
                self.aa.reset(
                    depth=1,  # self.adaptive_depth, # TODO?
                    # depth=self.adaptive_depth,
                    base_iteration=model.nonlinear_solver_statistics.num_iteration,
                )
                logger.info(f"Current adaptive depth: {self.aa._depth}")

                # Activate initial iteration
                self.init_iteration = model.nonlinear_solver_statistics.num_iteration
                logger.info("Activate initial iteration")

            if use_adaptive_relaxation:
                # Check if new cycling event has occurred
                if hasattr(model, "cycling_window") and model.cycling_window > 0:
                    self.new_cycling = True
                #    self.adaptive_alpha *= 0.5
                #    model.reset_cycling_analysis()
                #    logger.info(f"New cycling event with alpha {self.adaptive_alpha}")
                # else:
                #    self.new_cycling = False

            if use_min_fb_switch:
                if hasattr(model, "cycling_window") and model.cycling_window > 0:
                    self.new_cycling = True

            if use_adaptive_cnum:
                if hasattr(model, "cycling_window") and model.cycling_window > 0:
                    self.new_cycling = True

            # AA split in two steps - build and application
            xk = model.equation_system.get_variable_values(iterate_index=0)
            if use_aa or use_small_change_aa or use_decreasing_residuals_aa:
                # AA
                xkp1 = self.aa.apply(
                    xk + nonlinear_increment,
                    nonlinear_increment,
                    model.nonlinear_solver_statistics.num_iteration,
                    # adaptive_depth=self.adaptive_depth,
                )
                logger.info(f"Apply AA with depth {self.aa._depth}")

                ## Deactivation of AA after some iterations
                # if hasattr(self, "adaptive_depth") and (
                #    self.adaptive_depth > 0
                #    and (
                #        model.nonlinear_solver_statistics.num_iteration
                #        - self.init_iteration
                #    )
                #    % self.adaptive_depth
                #    == self.adaptive_depth - 1
                # ):
                #    print(model.nonlinear_solver_statistics.num_iteration)
                #    logger.info("Deactivate AA")
                #    self._reset(model)

                ## Restart AA after some iterations
                # if self.aa._depth > 0 and (
                #    model.nonlinear_solver_statistics.num_iteration
                #    - self.init_iteration
                #    == 2  # self.aa._depth + 1  # self.adaptive_depth
                # ):
                #    # if self.adaptive_depth == 0 and hasattr(self, "init_iteration"):
                #    logger.info(
                #        f"""Deactivate AA in iteration """
                #        f"""{model.nonlinear_solver_statistics.num_iteration} """
                #        f"""and intial iteration {self.init_iteration}"""
                #    )
                #    self._reset(model)
            if (
                use_small_change_aa
                and hasattr(model, "small_changes")
                and not model.small_changes
            ):
                self._reset(model)

            elif use_relaxation:
                # Relaxation
                if self.adaptive_depth > 0:
                    # Random scalar between 0.5 and 1
                    # alpha = 0.3 + 0.3 * np.random.rand()
                    # alpha = 0.5
                    alpha = 0.4 + 0.4 * np.random.rand()
                    ic("apply relaxation", alpha)
                    xkp1 = xk + alpha * nonlinear_increment
                    self._reset(model)
                else:
                    xkp1 = xk + nonlinear_increment
            elif use_adaptive_relaxation:
                if self.new_cycling:
                    self.adaptive_alpha = max(0.4, 1 / (1 / self.adaptive_alpha + 1))
                logger.info(
                    f"Apply adaptive relaxation with alpha {self.adaptive_alpha}"
                )
                xkp1 = xk + self.adaptive_alpha * nonlinear_increment
            elif use_min_fb_switch:
                ic("apply_min_fb_switch", self.new_cycling)
                if self.new_cycling:
                    ic("update_min_fb_switch")
                    if hasattr(model, "update_min_fb_switch"):
                        model.update_min_fb_switch(active_min=False)
                xkp1 = xk + nonlinear_increment

            elif use_adaptive_cnum:
                if self.new_cycling:
                    old_cnum = model.get_cnum()
                    model.update_cnum(val=max(old_cnum / 10, 1e-3))
                    self.new_cycling = False
                    logger.info(f"Update cnum to {model.get_cnum()}")
                xkp1 = xk + nonlinear_increment

            nonlinear_increment = xkp1 - xk

        # Stop simulation if cycling detected and AA depth is zero
        if (
            hasattr(model, "cycling_window")
            and model.cycling_window >= 2
            and aa_depth == 0
        ):
            logger.info("Stop simulation due to cycling")
            model.is_cycling = True
            model.after_nonlinear_failure()

        rr_resn = []
        rr_rest = []
        resn = []
        rest = []
        for sd in model.mdg.subdomains(dim=model.nd - 1, return_data=False):
            rr_resn.append(
                np.linalg.norm(
                    pp.momentum_balance.MomentumBalance.normal_fracture_deformation_equation(
                        model, [sd]
                    ).value(model.equation_system)
                )
            )
            resn.append(
                np.linalg.norm(
                    model.normal_fracture_deformation_equation([sd]).value(
                        model.equation_system
                    )
                )
            )
            rr_rest.append(
                np.linalg.norm(
                    pp.momentum_balance.MomentumBalance.tangential_fracture_deformation_equation(
                        model, [sd]
                    ).value(model.equation_system)
                )
            )
            rest.append(
                np.linalg.norm(
                    model.tangential_fracture_deformation_equation([sd]).value(
                        model.equation_system
                    )
                )
            )
        try:
            logger.info(
                f"\nCheck radial return contact conditions (max n|t): {np.max(rr_resn)} | {np.max(rr_rest)}"
            )
            logger.info(
                f"Check contact conditions (max n|t): {np.max(resn)} | {np.max(rest)}"
            )
            logger.info(
                f"Check contact conditions (min n|t): {np.argmax(resn)} | {np.argmax(rest)}"
            )
        except:
            ...
        return nonlinear_increment
