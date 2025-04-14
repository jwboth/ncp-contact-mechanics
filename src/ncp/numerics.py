# TODO: Integrate into NCP after experimenting

import numpy as np
import scipy.sparse as sps
import porepy as pp
from icecream import ic

import logging

# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ScaledNCPAdapters:
    def characteristic_distance(self, subdomains):
        return self.characteristic_displacement(subdomains)


class DarcysLawAd(pp.constitutive_laws.DarcysLawAd):
    def darcy_flux_discretization(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Discretization of the Darcy flux.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Operator for the Darcy flux discretization.

        """
        if all([sd.dim < self.nd for sd in subdomains]):
            return pp.ad.TpfaAd(self.darcy_keyword, subdomains)
        else:
            return super().darcy_flux_discretization(subdomains)


class ReverseElasticModuli:
    """Same as ElasticModuli, but with reversed assignment of characteristic values."""

    def characteristic_contact_traction(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Characteristic traction [Pa].

        Parameters:
            subdomains: List of subdomains where the characteristic traction is defined.

        Returns:
            Scalar operator representing the characteristic traction.

        """
        t_char = pp.ad.Scalar(self.numerical.characteristic_contact_traction)
        t_char.set_name("characteristic_contact_traction")
        return t_char

    def characteristic_displacement(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Characteristic displacement [m].

        Parameters:
            subdomains: List of subdomains where the characteristic displacement is
                defined.

        Returns:
            Scalar operator representing the characteristic displacement.

        """
        size = pp.ad.Scalar(np.max(self.domain.side_lengths()))
        u_char = (
            self.characteristic_contact_traction(subdomains)
            * size
            / self.youngs_modulus(subdomains)
        )
        u_char.set_name("characteristic_displacement")
        return u_char


class Switch:
    def switch(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:
        """Switch between Fischer-Burmeister and min NCP formulations.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            switch: Switch as scalar.

        """
        return pp.ad.TimeDependentDenseArray(
            "active_switch", [self.mdg.subdomains()[0]]
        )

    def update_switch(self, activate: bool) -> None:
        for sd in self.mdg.subdomains(return_data=False):
            pp.set_solution_values(
                name="active_switch",
                values=np.array([int(activate)]),
                data=self.mdg.subdomain_data(sd),
                iterate_index=0,
            )
        logging.info(f"Switched to min NCP: {activate}")

    def update_time_dependent_ad_arrays(self) -> None:
        """Start with min NCP formulation."""
        super().update_time_dependent_ad_arrays()
        self.update_switch(activate=True)
