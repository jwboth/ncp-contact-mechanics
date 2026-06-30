import porepy as pp
from functools import partial

import porepy as pp
import numpy as np


class SOCCPContactMechanics:
    def _dot_product(self, xt, yt, tangential_basis):
        """Dot product of two vectors in the tangential basis."""
        element_wise_product = xt * yt
        if self.nd == 2:
            dot_product = element_wise_product
        elif self.nd == 3:
            e_0 = tangential_basis[0]
            e_1 = tangential_basis[-1]
            dot_product = e_0.T @ element_wise_product + e_1.T @ element_wise_product
        dot_product.set_name("dot_product")
        return dot_product

    def _jordan_product(self, xn, xt, yn, yt, tangential_basis):
        """Take the jordan product of x and y."""

        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)
        normal_component = xn * yn + self._dot_product(xt, yt, tangential_basis)
        tangential_component = (scalar_to_tangential @ xn) * yt + (
            scalar_to_tangential @ yn
        ) * xt
        return normal_component, tangential_component

    def _jordan_square(self, xn, xt, tangential_basis):
        """Take the jordan square of x."""
        return self._jordan_product(xn, xt, xn, xt, tangential_basis)

    def _jordan_square_root(self, xn, xt, f_norm):
        """Take the jordan square root of x."""

        condition = xn**2 - f_norm(xt) ** 2
        s = (pp.ad.Scalar(0.5) * (xn + condition**0.5)) ** 0.5
        normal_component = s
        tangential_component = pp.ad.Scalar(0.5) * (xt / s)
        return normal_component, tangential_component

    def _auxiliary_soccp_fb(self, subdomains: list[pp.Grid]):
        """Auxiliary quantities for the SOC-CP formulation."""

        # Variables
        nd_vec_to_normal = self.normal_component(subdomains)
        nd_vec_to_tangential = self.tangential_component(subdomains)
        t_n: pp.ad.Operator = nd_vec_to_normal @ self.contact_traction(subdomains)
        t_t: pp.ad.Operator = nd_vec_to_tangential @ self.contact_traction(subdomains)
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(subdomains)
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)
        u_t_increment: pp.ad.Operator = pp.ad.time_increment(u_t)

        # Friction
        gap = self.fracture_gap(subdomains)
        friction_coefficient = self.friction_coefficient(subdomains)

        # Functions.
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")

        # Projections.
        tangential_basis = self.basis(subdomains, dim=self.nd - 1)
        c_num = self.contact_mechanics_numerical_constant(subdomains)
        scalar_to_tangential = pp.ad.sum_projection_list(tangential_basis)

        xn = c_num * (u_n - gap + friction_coefficient * f_norm(u_t_increment))
        xt = (scalar_to_tangential @ (c_num * friction_coefficient)) * u_t_increment
        yn = -friction_coefficient * t_n
        yt = t_t

        xn_sq, xt_sq = self._jordan_square(xn, xt, tangential_basis)
        yn_sq, yt_sq = self._jordan_square(yn, yt, tangential_basis)
        xn_sq_p_yn_sq_sqrt, xt_sq_p_yt_sq_sqrt = self._jordan_square_root(
            xn_sq + yn_sq,
            xt_sq + yt_sq,
            f_norm,
        )
        fb_n = xn_sq_p_yn_sq_sqrt - (xn + yn)
        fb_t = xt_sq_p_yt_sq_sqrt - (xt + yt)
        return fb_n, fb_t

    def normal_fracture_deformation_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        """Contact mechanics equation for the normal constraints."""
        fb_n, _ = self._auxiliary_soccp_fb(subdomains)
        fb_n.set_name("normal_fracture_deformation_equation")
        return fb_n

    def tangential_fracture_deformation_equation(
        self,
        subdomains: list[pp.Grid],
    ) -> pp.ad.Operator:
        """Contact mechanics equation for the tangential constraints."""
        _, fb_t = self._auxiliary_soccp_fb(subdomains)
        fb_t.set_name("tangential_fracture_deformation_equation")
        return fb_t
