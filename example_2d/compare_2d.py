from pathlib import Path
from deepdiff import DeepDiff
from main import generate_case_name
import meshio

# Test different formulations
formulations = [
    ("picard", "rr-nonlinear", 0, "none"),
    ("picard", "rr-linear", 0, "none"),
    ("picard", "ncp-min", 0, "origin_and_stick_slip_transition"),
    ("picard", "ncp-fb-full", 0, "origin_and_stick_slip_transition"),
    ("picard", "ncp-min-scaled", 0, "origin_and_stick_slip_transition"),
    ("picard", "ncp-fb-scaled", 0, "origin_and_stick_slip_transition"),
    ("picard", "ncp-fb-full-scaled", 0, "origin_and_stick_slip_transition"),
    ("newton", "rr-nonlinear", 0, "none"),
    ("newton", "rr-linear", 0, "none"),
    ("newton", "ncp-min", 0, "origin_and_stick_slip_transition"),
    ("newton", "ncp-fb-full", 0, "origin_and_stick_slip_transition"),
    ("newton", "ncp-min-scaled", 0, "origin_and_stick_slip_transition"),
    ("newton", "ncp-fb-scaled", 0, "origin_and_stick_slip_transition"),
    ("newton", "ncp-fb-full-scaled", 0, "origin_and_stick_slip_transition"),
]
study = 1
seed = 4
passed = []
not_passed = []


for formulation in formulations:
    # Run the simulation with the specified formulation
    print(f"Testing formulation: {formulation}")
    ad_mode, mode, aa, regularization = formulation

    # Fetch the solver statistics
    folder = generate_case_name(
        study=study,
        seed=seed,
        mesh_size=50,
        dil=0.1,
        cn=1.0,
        ct=1.0,
        tol=1e-10,
        aa=0,
        regularization=regularization,
        method=mode,
        ad_mode=ad_mode,
        linear_solver="scipy_sparse",
        no_intersections=True,
        no_intersections_angle_cutoff=0.0,
        resolved_intersections=False,
        nice_geometry=False,
        regularized_start=False,
        unitary_units=True,
    )
    solver_statistics_filename = (
        Path("visualization") / folder / "solver_statistics.json"
    )
    final_solution_filename = {
        "data_1": Path("visualization") / folder / "data_1_000003.vtu",
        "data_2": Path("visualization") / folder / "data_2_000003.vtu",
        "mortar_1": Path("visualization") / folder / "data_mortar_1_000003.vtu",
    }
    reference_statistics_filename = (
        Path("visualization_miloke") / folder / "solver_statistics.json"
    )
    reference_solution_filename = {
        "data_1": Path("visualization_miloke") / folder / "data_1_000003.vtu",
        "data_2": Path("visualization_miloke") / folder / "data_2_000003.vtu",
        "mortar_1": Path("visualization_miloke") / folder / "data_mortar_1_000003.vtu",
    }

    # Initiate status
    status = True
    failure = []

    # Compare the final solution files
    diff = {}
    for key in final_solution_filename.keys():
        solution_data = meshio.read(final_solution_filename[key])
        reference_data = meshio.read(reference_solution_filename[key])
        diff[key] = DeepDiff(
            solution_data.__dict__,
            reference_data.__dict__,
            significant_digits=3,
            number_format_notation="e",
            ignore_order=True,
            ignore_numeric_type_changes=True,
        )
        if diff[key] != {}:
            print(f"Difference found in {key}: {diff[key]}")
            not_passed.append(formulation)
        else:
            passed.append(formulation)

# Print the results
# print("Passed formulations:", passed)
print("Not passed formulations:", not_passed)
