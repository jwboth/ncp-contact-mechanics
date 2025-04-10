import logging

# Module-wide logger
logger = logging.getLogger(__name__)

import numpy as np

import ncp
import porepy as pp


class EuclideanConvergenceMetrics:
    def compute_nonlinear_increment_norm(
        self, nonlinear_increment: np.ndarray, split: bool = False
    ) -> float:
        return ncp.EuclideanMetric.norm(self, nonlinear_increment, split)

    def compute_residual_norm(self, residual: np.ndarray, split: bool = False) -> float:
        return ncp.EuclideanMetric.norm(self, residual, split)


class LebesgueConvergenceMetrics:
    def compute_nonlinear_increment_norm(
        self, nonlinear_increment: np.ndarray, split: bool = False
    ) -> float:
        return ncp.LebesgueMetric.variable_norm(
            self, values=nonlinear_increment, variables=None, split=split
        )

    def compute_residual_norm(self, residual: np.ndarray, split: bool = False) -> float:
        return ncp.LebesgueMetric.residual_norm(self, equations=None, split=split)


class CustomNewtonSolver(pp.NewtonSolver):
    def iteration(self, model):
        raise NotImplementedError(
            """Only used in custom_run_time_dependent_model - """
            """Aim at using the standard NewtonSolver."""
        )
        result = super().iteration(model)
        if False:
            cond = np.linalg.cond(model.linear_system[0].todense())
            if not hasattr(model, "max_cond_number"):
                model.max_cond_number = cond
            if not hasattr(model, "cond_number"):
                model.cond_number = []
            model.max_cond_number = max(model.max_cond_number, cond)
            model.cond_number.append(cond)
        return result


def custom_run_time_dependent_model(model, params: dict) -> None:
    """Run a time dependent model.

    Parameters:
        model: Model class containing all information on parameters, variables,
            discretization, geometry. Various methods such as those relating to solving
            the system, see the appropriate solver for documentation.
        params: Parameters related to the solution procedure. Why not just set these
            as e.g. model.solution_parameters?

    """
    raise NotImplementedError("Aim at using the standard run_time_dependent model.")
    # Assign parameters, variables and discretizations. Discretize time-indepedent terms
    if params.get("prepare_simulation", True):
        model.prepare_simulation()

    # Assign a solver
    solver = CustomNewtonSolver(params)

    res_init = model.equation_system.assemble(evaluate_jacobian=False)

    # Define a function that does all the work during one time step
    def time_step() -> None:
        model.time_manager.increase_time()
        model.time_manager.increase_time_index()
        logger.info(
            f"\nTime step {model.time_manager.time_index} at time"
            + f" {model.time_manager.time:.1e}"
            + f" of {model.time_manager.time_final:.1e}"
            + f" with time step {model.time_manager.dt:.1e}"
        )
        solver.solve(model)

        # Compute condition number in final iteration step
        if hasattr(model, "max_cond_number"):
            print("Maximal condition number:", model.max_cond_number)

        # Write model.cond_number to file
        if hasattr(model, "cond_number"):
            folder = params.get("folder_name", "output")
            with open(folder + "/cond_number.txt", "w") as f:
                for item in model.cond_number:
                    f.write("%s\n" % item)

        # Debugger
        model.debug()

        # CUSTOM PART
        res = model.equation_system.assemble(evaluate_jacobian=False)

        # Check old contact equations
        try:
            normal_res = []
            tangential_res = []
            for sd in model.mdg.subdomains(dim=1, return_data=False):
                normal_res.append(
                    np.linalg.norm(
                        pp.momentum_balance.MomentumBalance.normal_fracture_deformation_equation(
                            model, [sd]
                        ).value(model.equation_system)
                    )
                )
                tangential_res.append(
                    np.linalg.norm(
                        pp.momentum_balance.MomentumBalance.tangential_fracture_deformation_equation(
                            model, [sd]
                        ).value(model.equation_system)
                    )
                )

            if True:
                print(
                    "double-check residual (full absolute/full relative/normal/tangential):",
                    np.linalg.norm(res),
                    np.linalg.norm(res) / np.linalg.norm(res_init),
                    max(normal_res),
                    max(tangential_res),
                    end="",
                )
        except:
            pass

    while not model.time_manager.final_time_reached():
        time_step()

    model.after_simulation()
