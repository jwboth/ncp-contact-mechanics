"""Cubic law for the normal permeability."""

import porepy as pp


class CubicLawNormalPermeability:
    """Introduce the cubic law for the normal permeability."""

    def normal_permeability(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Compute the normal permeability for the cubic law.

        The cubic law for the normal permeability is given by

                k_n = (A^2 / 12) * aperture,

            where A is the aperture of the fracture and k_n is the normal
            permeability.
        """
        # Compute the subdomains and mortar grid.
        subdomains = self.interfaces_to_subdomains(interfaces)

        # Compute the normal permeability.
        aperture = self.aperture(subdomains)
        normal_perm = (aperture ** pp.ad.Scalar(2)) / pp.ad.Scalar(12)

        # Project the normal permeability to the mortar grid.
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )
        return mortar_projection.secondary_to_mortar_avg @ normal_perm


# TODO: Remove this class after debugging.
class DebuggingPerm:
    """Only for debugging purposes."""

    def normal_permeability(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
        """Compute the normal permeability for the cubic law.

        The cubic law for the normal permeability is given by

                k_n = (A^2 / 12) * aperture,

            where A is the aperture of the fracture and k_n is the normal
            permeability.
        """
        # Compute the subdomains and mortar grid.
        subdomains = self.interfaces_to_subdomains(interfaces)
        size = sum(sd.num_cells for sd in subdomains)
        normal_perm = pp.wrap_as_dense_ad_array(1e-9, size, name="normal_permeability")
        mortar_projection = pp.ad.MortarProjections(
            self.mdg, subdomains, interfaces, dim=1
        )
        return mortar_projection.secondary_to_mortar_avg @ normal_perm

    def fracture_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability of fractures.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        size = sum(sd.num_cells for sd in subdomains)
        permeability = pp.wrap_as_dense_ad_array(1e-9, size, name="permeability")
        return self.isotropic_second_order_tensor(subdomains, permeability)

    def intersection_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability of intersections.

        Note that as permeability is not meaningful in 0d domains, this method will only
        impact the tangential permeability of 1d intersection lines.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        size = sum(sd.num_cells for sd in subdomains)
        permeability = pp.wrap_as_dense_ad_array(1e-9, size, name="permeability")
        return self.isotropic_second_order_tensor(subdomains, permeability)
