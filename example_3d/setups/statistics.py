import numpy as np
import scipy.sparse as sps
import porepy as pp
import json
from dataclasses import dataclass, field
from copy import copy
import logging
from typing import Any
import time
import warnings

logger = logging.getLogger(__name__)


class LogPerformanceData:
    def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        result = super().after_nonlinear_iteration(nonlinear_increment)
        self.nonlinear_solver_statistics.log_performance_data(
            subdomains=self.mdg.subdomains(), time_index=self.time_manager.time_index
        )
        # Not converged yet.
        self.nonlinear_solver_statistics.status = "unknown"
        self.nonlinear_solver_statistics.save()
        return result

    def after_nonlinear_convergence(self) -> None:
        self.nonlinear_solver_statistics.status = "converged"
        super().after_nonlinear_convergence()
        self.nonlinear_solver_statistics.cache()

    def after_nonlinear_failure(self) -> None:
        # Check for detetcted cycling
        if hasattr(self, "is_cycling") and self.is_cycling:
            self.nonlinear_solver_statistics.status = "cycling"
            self.nonlinear_solver_statistics.save()
            raise ValueError("Cycling detected.")

        # Check if jacobian (A) is singular
        try:
            _ = self.solve_linear_system()
        # catch MatrixRankWarning
        except sps.linalg.MatrixRankWarning:
            self.nonlinear_solver_statistics.status = "singular"
            self.nonlinear_solver_statistics.save()
            raise ValueError("Matrix is singular.")

        # Check if jacobian (sparse matrix) contains nan values
        A, b = self.linear_system
        # A is a sps matrix. extract the data array and check for nan values
        if np.isnan(A.data).any():
            self.nonlinear_solver_statistics.status = "nan_matrix"
            self.nonlinear_solver_statistics.save()
            raise ValueError("Matrix contains NaN values.")
        if np.isnan(b).any():
            self.nonlinear_solver_statistics.status = "nan_residual"
            self.nonlinear_solver_statistics.save()
            raise ValueError("Right-hand side contains NaN values.")

        # Check if solution is none
        sol = self.equation_system.get_variable_values(iterate_index=0)
        if np.isnan(sol).any():
            self.nonlinear_solver_statistics.status = "nan"
            self.nonlinear_solver_statistics.save()
            raise ValueError("Solution contains NaN values.")

        # Unknown failure
        self.nonlinear_solver_statistics.status = "unknown"
        self.nonlinear_solver_statistics.save()

        # Cache the statistics object
        self.nonlinear_solver_statistics.cache()

        super().after_nonlinear_failure()

    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual: np.ndarray,
        reference_residual: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[bool, bool]:
        """Implements a convergence check, to be called by a non-linear solver.

        Parameters:
            nonlinear_increment: Newly obtained solution increment vector
            residual: Residual vector of non-linear system, evaluated at the newly
            obtained solution vector.
            reference_residual: Reference residual vector of non-linear system,
                evaluated for the initial guess at current time step.
            nl_params: Dictionary of parameters used for the convergence check.
                Which items are required will depend on the convergence test to be
                implemented.

        Returns:
            The method returns the following tuple:

            boolean:
                True if the solution is converged according to the test implemented by
                this method.
            boolean:
                True if the solution is diverged according to the test implemented by
                this method.

        """
        if not self._is_nonlinear_problem():
            # At least for the default direct solver, scipy.sparse.linalg.spsolve, no
            # error (but a warning) is raised for singular matrices, but a nan solution
            # is returned. We check for this.
            diverged = bool(np.any(np.isnan(nonlinear_increment)))
            converged: bool = not diverged
            residual_norm: float = np.nan if diverged else 0.0
            nonlinear_increment_norm: float = np.nan if diverged else 0.0
        else:
            # First a simple check for nan values.
            if np.any(np.isnan(nonlinear_increment)):
                # If the solution contains nan values, we have diverged.
                return False, True

            # Increment based norm
            nonlinear_increment_norm = self.compute_nonlinear_increment_norm(
                nonlinear_increment
            )
            # Residual based norm
            residual_norm = self.compute_residual_norm(residual)

            # Cache first solution and residual for reference
            if self.nonlinear_solver_statistics.num_iteration == 1:
                reference_solution = self.equation_system.get_variable_values(
                    iterate_index=0
                )
                self.reference_solution_norm = self.compute_nonlinear_increment_norm(
                    reference_solution
                )
                self.reference_residual_norm = residual_norm
            elif np.isnan(self.reference_residual_norm) and not np.isnan(residual_norm):
                self.reference_residual_norm = residual_norm

            # Monitor norms.
            logger.info(
                f"""Time simulated: {self.time_manager.time / self.time_manager.time_final * 100} %"""
            )
            logger.info(
                """Nonlinear abs.|rel. increment norm: """
                f"""{nonlinear_increment_norm:.2e} | """
                f"""{nonlinear_increment_norm / self.reference_solution_norm:.2e}"""
            )
            logger.info(
                """Nonlinear abs.|rel. residual norm: """
                f"""{residual_norm:.2e} | """
                f"""{residual_norm / self.reference_residual_norm:.2e}"""
            )
            converged = False
            # Check convergence requiring both the increment and residual to be small.
            if not converged and self.nonlinear_solver_statistics.num_iteration > 1:
                if not np.isnan(nl_params["nl_convergence_tol"]) and not np.isnan(
                    nl_params["nl_convergence_tol_rel"]
                ):
                    converged_inc = (
                        nonlinear_increment_norm
                        < nl_params["nl_convergence_tol"]
                        + nl_params["nl_convergence_tol_rel"]
                        * self.reference_solution_norm
                    )
                elif not np.isnan(nl_params["nl_convergence_tol"]):
                    converged_inc = (
                        nonlinear_increment_norm < nl_params["nl_convergence_tol"]
                    )
                elif not np.isnan(nl_params["nl_convergence_tol_rel"]):
                    converged_inc = (
                        nonlinear_increment_norm
                        < nl_params["nl_convergence_tol_rel"]
                        * self.reference_solution_norm
                    )
                else:
                    converged_inc = True

                # Same for residuals
                if not np.isnan(nl_params["nl_convergence_tol_res"]) and not np.isnan(
                    nl_params["nl_convergence_tol_res_rel"]
                ):
                    converged_res = (
                        residual_norm
                        < nl_params["nl_convergence_tol_res"]
                        + nl_params["nl_convergence_tol_res_rel"]
                        * self.reference_residual_norm
                    )
                elif not np.isnan(nl_params["nl_convergence_tol_res"]):
                    converged_res = residual_norm < nl_params["nl_convergence_tol_res"]
                elif not np.isnan(nl_params["nl_convergence_tol_res_rel"]):
                    converged_res = (
                        residual_norm
                        < nl_params["nl_convergence_tol_res_rel"]
                        * self.reference_residual_norm
                    )
                else:
                    converged_res = True

                converged = converged_inc and converged_res

            # Allow small residuals to be considered converged.
            if not converged and self.nonlinear_solver_statistics.num_iteration > 1:
                if not np.isnan(
                    nl_params["nl_convergence_tol_res_tight"]
                ) and not np.isnan(nl_params["nl_convergence_tol_res_rel_tight"]):
                    converged = (
                        residual_norm
                        < nl_params["nl_convergence_tol_res_tight"]
                        + nl_params["nl_convergence_tol_res_rel_tight"]
                        * self.reference_residual_norm
                    )
                elif not np.isnan(nl_params["nl_convergence_tol_res_tight"]):
                    converged = (
                        residual_norm < nl_params["nl_convergence_tol_res_tight"]
                    )
                elif not np.isnan(nl_params["nl_convergence_tol_res_rel_tight"]):
                    converged = (
                        residual_norm
                        < nl_params["nl_convergence_tol_res_rel_tight"]
                        * self.reference_residual_norm
                    )

            # Allow small increments to be considered converged.
            if not converged and self.nonlinear_solver_statistics.num_iteration > 1:
                if not np.isnan(nl_params["nl_convergence_tol_tight"]) and not np.isnan(
                    nl_params["nl_convergence_tol_rel_tight"]
                ):
                    converged = (
                        nonlinear_increment_norm
                        < nl_params["nl_convergence_tol_tight"]
                        + nl_params["nl_convergence_tol_rel_tight"]
                        * self.reference_solution_norm
                    )
                elif not np.isnan(nl_params["nl_convergence_tol_tight"]):
                    converged = (
                        nonlinear_increment_norm < nl_params["nl_convergence_tol_tight"]
                    )
                elif not np.isnan(nl_params["nl_convergence_tol_rel_tight"]):
                    converged = (
                        nonlinear_increment_norm
                        < nl_params["nl_convergence_tol_rel_tight"]
                        * self.reference_solution_norm
                    )

            # Allow nan residuals to be considered converged.
            if not converged and self.nonlinear_solver_statistics.num_iteration > 1:
                if np.isnan(residual_norm) or np.isnan(self.reference_residual_norm):
                    if not np.isnan(
                        nl_params["nl_convergence_tol_tight"]
                    ) and not np.isnan(nl_params["nl_convergence_tol_rel_tight"]):
                        converged = (
                            nonlinear_increment_norm
                            < nl_params["nl_convergence_tol_tight"]
                            + nl_params["nl_convergence_tol_rel_tight"]
                            * self.reference_solution_norm
                        )

            diverged = False

        # Log the errors (here increments and residuals)
        self.nonlinear_solver_statistics.log_error(
            nonlinear_increment_norm, residual_norm
        )

        return converged, diverged

    def _initialize_linear_solver(self) -> None:
        """Overwrite the default initialization to allow further linear solvers."""
        solver = self.params["linear_solver"]
        self.linear_solver = solver

        if solver not in [
            "scipy_sparse",
            "pypardiso",
            "umfpack",
            "scipy_splu",
            "scipy_sparse-pypardiso",
        ]:
            raise ValueError(f"Unknown linear solver {solver}")

    def solve_linear_system(self) -> np.ndarray:
        """Solve linear system.

        Default method is a direct solver. The linear solver is chosen in the
        initialize_linear_solver of this model. Implemented options are
            - scipy.sparse.spsolve with and without call to umfpack
            - pypardiso.spsolve

        See also:
            :meth:`initialize_linear_solver`

        Returns:
            np.ndarray: Solution vector.

        """
        A, b = self.linear_system
        t_0 = time.time()
        logger.debug(f"Max element in A {np.max(np.abs(A)):.2e}")
        logger.debug(
            f"""Max {np.max(np.sum(np.abs(A), axis=1)):.2e} and min
            {np.min(np.sum(np.abs(A), axis=1)):.2e} A sum."""
        )

        solver = self.linear_solver
        if solver == "pypardiso":
            # This is the default option which is invoked unless explicitly overridden
            # by the user. We need to check if the pypardiso package is available.
            try:
                from pypardiso import spsolve as sparse_solver  # type: ignore
            except ImportError:
                # Fall back on the standard scipy sparse solver.
                sparse_solver = sps.linalg.spsolve
                warnings.warn(
                    """PyPardiso could not be imported,
                    falling back on scipy.sparse.linalg.spsolve"""
                )
            x = sparse_solver(A, b)
            try:
                x = sparse_solver(A, b)
            except RuntimeError:
                x = np.zeros_like(b)
                x[:] = np.nan
        elif solver == "umfpack":
            # Following may be needed:
            # A.indices = A.indices.astype(np.int64)
            # A.indptr = A.indptr.astype(np.int64)
            x = sps.linalg.spsolve(A, b, use_umfpack=True)
        elif solver == "scipy_sparse":
            x = sps.linalg.spsolve(A, b)
        elif solver == "scipy_splu":
            lu = sps.linalg.splu(A.tocsc())
            x = lu.solve(b)
        elif solver == "scipy_sparse-pypardiso":
            # Only use pypardiso if scipy_sparse fails
            try:
                x = sps.linalg.spsolve(A, b)
            except RuntimeError:
                from pypardiso import spsolve as sparse_solver  # type: ignore

                x = sparse_solver(A, b)
        else:
            raise ValueError(
                f"AbstractModel does not know how to apply the linear solver {solver}"
            )
        logger.info(f"Solved linear system in {time.time() - t_0:.2e} seconds.")

        return np.atleast_1d(x)


class LogPerformanceDataVectorial(LogPerformanceData):
    def _tolerance_check(self, norms, reference_norms, tol_abs, tol_rel) -> bool:
        if not np.isnan(tol_abs) and not np.isnan(tol_rel):
            return all(
                [n < tol_abs + tol_rel * rn for (n, rn) in zip(norms, reference_norms)]
            )
        elif not np.isnan(tol_abs):
            return all([r < tol_abs for r in norms])
        elif not np.isnan(tol_rel):
            return all([r < tol_rel * rn for (r, rn) in zip(norms, reference_norms)])
        else:
            return True

    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual: np.ndarray,
        reference_residual: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[bool, bool]:
        """Implements a convergence check, to be called by a non-linear solver.

        Parameters:
            nonlinear_increment: Newly obtained solution increment vector
            residual: Residual vector of non-linear system, evaluated at the newly
            obtained solution vector.
            reference_residual: Reference residual vector of non-linear system,
                evaluated for the initial guess at current time step.
            nl_params: Dictionary of parameters used for the convergence check.
                Which items are required will depend on the convergence test to be
                implemented.

        Returns:
            The method returns the following tuple:

            boolean:
                True if the solution is converged according to the test implemented by
                this method.
            boolean:
                True if the solution is diverged according to the test implemented by
                this method.

        """
        # Check if problem is linear or solution contains nan values.
        if not self._is_nonlinear_problem() or np.any(np.isnan(nonlinear_increment)):
            # At least for the default direct solver, scipy.sparse.linalg.spsolve, no
            # error (but a warning) is raised for singular matrices, but a nan solution
            # is returned. We check for this.
            diverged = bool(np.any(np.isnan(nonlinear_increment)))
            converged: bool = not diverged
            residual_norm: float = np.nan if diverged else 0.0
            nonlinear_increment_norm: float = np.nan if diverged else 0.0
            # Log the errors (here increments and residuals)
            self.nonlinear_solver_statistics.log_error(
                nonlinear_increment_norm, residual_norm
            )

            return converged, diverged

        # Increment based norm
        nonlinear_increment_norms = self.compute_nonlinear_increment_norm(
            nonlinear_increment,
            split=True,
        )

        # Cache first solution as reference for relative increment norms
        if self.nonlinear_solver_statistics.num_iteration == 1:
            self.fixed_reference_nonlinear_increment_norms = [
                False for _ in nonlinear_increment_norms
            ]
            self.reference_nonlinear_increment_norms = [
                1.0 for _ in nonlinear_increment_norms
            ]
        if not all(self.fixed_reference_nonlinear_increment_norms):
            reference_solution = self.equation_system.get_variable_values(
                iterate_index=0
            )
            reference_solution_norms = self.compute_nonlinear_increment_norm(
                reference_solution, split=True
            )
        for i, fixed_reference in enumerate(
            self.fixed_reference_nonlinear_increment_norms
        ):
            if not fixed_reference:
                _norm = reference_solution_norms[i]
                if not np.isclose(_norm, 0.0) and not np.isnan(_norm):
                    self.reference_nonlinear_increment_norms[i] = _norm
                    self.fixed_reference_nonlinear_increment_norms[i] = True

        # Residual based norm
        residual_norms = self.compute_residual_norm(None, split=True)

        # Cache first non-zero residual as reference for relative residual norms
        if self.nonlinear_solver_statistics.num_iteration == 1:
            self.fixed_reference_residual_norms = [False for _ in residual_norms]
            self.reference_residual_norms = [1.0 for _ in residual_norms]
        for i, fixed_reference in enumerate(self.fixed_reference_residual_norms):
            if not fixed_reference:
                _norm = residual_norms[i]
                if not np.isclose(_norm, 0.0) and not np.isnan(_norm):
                    self.reference_residual_norms[i] = _norm
                    self.fixed_reference_residual_norms[i] = True

        # Relative norms
        relative_increment_norms = [
            n / (1 + rn)
            for n, rn in zip(
                nonlinear_increment_norms, self.reference_nonlinear_increment_norms
            )
        ]
        relative_residual_norms = [
            n / (1 + rn) for n, rn in zip(residual_norms, self.reference_residual_norms)
        ]

        # Log the (max) relative errors
        self.nonlinear_solver_statistics.log_error(
            np.max(relative_increment_norms), np.max(relative_residual_norms)
        )

        # Monitor norms.
        logger.info(
            f"""Time simulated: {self.time_manager.time / self.time_manager.time_final * 100} %"""
        )
        logger.info(
            """Nonlinear abs.|rel. increment norm: """
            f"""{(np.max(nonlinear_increment_norms)):.2e} | """
            f"""{(np.max(relative_increment_norms)):.2e}"""
        )
        logger.info(
            """Nonlinear abs.|rel. residual norm: """
            f"""{(np.max(residual_norms)):.2e} | """
            f"""{(np.max(relative_residual_norms)):.2e}"""
        )

        # Start convergence checks
        converged = False

        # Check convergence requiring both the increment and residual to be small.
        if not converged and self.nonlinear_solver_statistics.num_iteration > 1:
            converged_inc = self._tolerance_check(
                nonlinear_increment_norms,
                self.reference_nonlinear_increment_norms,
                nl_params["nl_convergence_tol"],
                nl_params["nl_convergence_tol_rel"],
            )
            converged_res = self._tolerance_check(
                residual_norms,
                self.reference_residual_norms,
                nl_params["nl_convergence_tol_res"],
                nl_params["nl_convergence_tol_res_rel"],
            )
            converged = converged_inc and converged_res
            if converged:
                print("Converged with both increments and residuals.")

        # Allow small increments to be considered converged.
        if not converged and self.nonlinear_solver_statistics.num_iteration > 1:
            converged = self._tolerance_check(
                nonlinear_increment_norms,
                self.reference_nonlinear_increment_norms,
                nl_params["nl_convergence_tol_tight"],
                nl_params["nl_convergence_tol_rel_tight"],
            )
            if converged:
                print("Converged with increments.")

        # Allow small residuals to be considered converged.
        if not converged and self.nonlinear_solver_statistics.num_iteration > 1:
            converged = self._tolerance_check(
                residual_norms,
                self.reference_residual_norms,
                nl_params["nl_convergence_tol_res_tight"],
                nl_params["nl_convergence_tol_res_rel_tight"],
            )
            if converged:
                print("Converged with residuals.")

        diverged = False

        return converged, diverged


@dataclass
class AdvancedSolverStatistics(pp.SolverStatistics):
    cache_num_iteration: list[int] = field(default_factory=list)
    """Cached number of non-linear iterations performed until current time step."""
    cache_nonlinear_increment_norms: list[list[float]] = field(default_factory=list)
    """Cached list of increment magnitudes for each non-linear iteration."""
    cache_residual_norms: list[list[float]] = field(default_factory=list)
    """Cached list of residual for each non-linear iteration."""

    time_index: int = 0
    contact_state_changes: list[list[int]] = field(default_factory=list)
    total_contact_state_changes: int = 0
    total_contact_state_changes_in_time: int = 0
    last_update_contact_states: int = 0
    num_contact_states: list[int] = field(default_factory=list)
    cache_contact_state_changes: list[list[list[int]]] = field(default_factory=list)
    cache_total_contact_state_changes: list[int] = field(default_factory=list)
    cache_num_contact_states: list[int] = field(default_factory=list)
    num_cells: list[int] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    stagnating_states: bool = False
    cycling_window: int = 0
    status: str = ""

    def cache(self) -> None:
        """Cache the statistics object."""
        self.cache_num_iteration.append(self.num_iteration)
        self.cache_nonlinear_increment_norms.append(self.nonlinear_increment_norms)
        self.cache_residual_norms.append(self.residual_norms)

    def reset(self):
        super().reset()
        self.contact_state_changes = np.zeros((3, 3), dtype=int).tolist()
        self.total_contact_state_changes = 0
        self.total_contact_state_changes_in_time = 0
        self.last_update_contact_states = 0
        self.num_contact_states = [0, 0, 0]

        self.cache_contact_state_changes = []
        self.cache_total_contact_state_changes = []
        self.cache_num_contact_states = []

        self.num_cells = []

        self.status = ""

    def log_error(self, nonlinear_increment_norm, residual_norm, **kwargs):
        super().log_error(nonlinear_increment_norm, residual_norm, **kwargs)

    def log_performance_data(self, **kwargs):
        """Collect contact mechanics related performance data."""

        self.cache_contact_state_changes.append(self.contact_state_changes)
        self.cache_total_contact_state_changes.append(self.total_contact_state_changes)
        self.cache_num_contact_states.append(self.num_contact_states)

        # Store grid stats
        max_dim = -1
        for sd in kwargs.get("subdomains"):
            max_dim = max(max_dim, sd.dim)
        self.num_cells = [0] * (max_dim + 1)
        for sd in kwargs.get("subdomains"):
            self.num_cells[sd.dim] += sd.num_cells

        # Store time step
        assert "time_index" in kwargs
        self.time_index = kwargs.get("time_index")

    def update_data(self) -> None:
        """Update the data dictionary for the current time index."""
        self.data["geometry"] = {
            "num_cells": copy(self.num_cells),
        }
        self.data[self.time_index] = {
            "status": self.status,
            "num_iteration": self.num_iteration,
            "nonlinear_increment_norms": copy(self.nonlinear_increment_norms),
            "residual_norms": copy(self.residual_norms),
            "contact_state_changes": copy(self.cache_contact_state_changes),
            "total_contact_state_changes": copy(self.cache_total_contact_state_changes),
            "total_contact_state_changes_in_time": self.total_contact_state_changes_in_time,
            "last_update_contact_states": self.last_update_contact_states,
            "num_contact_states": copy(self.cache_num_contact_states),
            "stagnating_states": self.stagnating_states,
            "cycling_window": self.cycling_window,
        }

    def save(self):
        """Save the statistics object to file."""
        self.update_data()
        # Save to file
        if self.path is not None:
            with self.path.open("w") as file:
                json.dump(self.data, file, indent=4)
