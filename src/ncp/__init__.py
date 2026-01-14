"""
NCP extension for PorePy.

isort:skip_file

"""

from .functions import min, fb, min_regularized_fb
from .fracture_states import FractureStates
from .iteration_export import CustomExporting, IterationExporting
from .auxiliary import AuxiliaryContact
from .utils import sign, isclose_times_identity, gt_times_identity  # TODO rm file
from .cycling_monitor import ContactMechanicsCyclingCriterion
# from .nonlinear_solvers import *
# from .statistics import SolverStatisticsForContactMechanics  # TODO rm rest
# from .numerics import *
# from .solution_strategy import * # TODO rm file
