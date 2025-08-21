import porepy as pp


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


# TODO rm
# class Switch:
#    def switch(self, subdomains: list[pp.Grid]) -> pp.ad.Scalar:
#        """Switch between Fischer-Burmeister and min NCP formulations.
#
#        Parameters:
#            subdomains: List of subdomains.
#
#        Returns:
#            switch: Switch as scalar.
#
#        """
#        return pp.ad.TimeDependentDenseArray(
#            "active_switch", [self.mdg.subdomains()[0]]
#        )
#
#    def update_switch(self, activate: bool) -> None:
#        for sd in self.mdg.subdomains(return_data=False):
#            pp.set_solution_values(
#                name="active_switch",
#                values=np.array([int(activate)]),
#                data=self.mdg.subdomain_data(sd),
#                iterate_index=0,
#            )
#        logging.info(f"Switched to min NCP: {activate}")
#
#    def update_time_dependent_ad_arrays(self) -> None:
#        """Start with min NCP formulation."""
#        super().update_time_dependent_ad_arrays()
#        self.update_switch(activate=True)
