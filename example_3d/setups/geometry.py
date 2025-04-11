"""Implementation of Bedretto-like geometry"""

import numpy as np
import porepy as pp
from porepy.applications.md_grids.model_geometries import CubeDomainOrthogonalFractures


class BedrettoGeometry(CubeDomainOrthogonalFractures):
    def set_domain(self) -> None:
        """Set the cube domain."""
        bounding_box = {
            "xmin": self.units.convert_units(-2000, "m"),
            "xmax": self.units.convert_units(2000, "m"),
            "ymin": self.units.convert_units(-2000, "m"),
            "ymax": self.units.convert_units(2000, "m"),
            "zmin": self.units.convert_units(-5000, "m"),
            "zmax": self.units.convert_units(-1000, "m"),
        }
        self._domain = pp.Domain(bounding_box)

    def meshing_arguments(self) -> dict:
        mesh_args = {}
        # mesh_args["cell_size"] = self.units.convert_units(500, "m")
        # mesh_args["cell_size_fracture"] = self.units.convert_units(100, "m")
        mesh_args["cell_size"] = self.units.convert_units(1000, "m")
        mesh_args["cell_size_fracture"] = self.units.convert_units(500, "m")
        return mesh_args

    def grid_type(self) -> str:
        return "simplex"

    def tunnel_coord(self, tm) -> np.ndarray:
        """TM: tunnel depth in m. (x,y)=(0,0) corresponds to the lab."""
        deg = 317 * np.pi / 180
        xy = np.array([np.cos(deg), np.sin(deg), 0]) * self.units.convert_units(
            tm - 2200, "m"
        )
        return np.array([xy[0], xy[1], self.units.convert_units(-3000, "m")])

    def set_fractures(self) -> None:
        # Add deterministic fractures identified by Jordan (MSc thesis, 2019)

        # EW-striking fractures
        def ew_frac(tm):
            return pp.create_elliptic_fracture(
                center=self.tunnel_coord(tm),
                major_axis=self.units.convert_units(600, "m"),
                minor_axis=self.units.convert_units(600, "m"),
                major_axis_angle=0,  # TODO?
                strike_angle=0,  # E-W
                dip_angle=50 * np.pi / 180,  # TODO
                num_points=16,
                # index=0,
            )

        # Tunnel perpendicular fractures
        def tunnel_perp(tm):
            return pp.create_elliptic_fracture(
                center=self.tunnel_coord(tm),
                major_axis=self.units.convert_units(600, "m"),
                minor_axis=self.units.convert_units(600, "m"),
                major_axis_angle=0,  # TODO?
                strike_angle=(317 - 180) * np.pi / 180,  # perp to N317-E
                dip_angle=50 * np.pi / 180,
                num_points=16,
                # index=0,
            )

        # NS striking fractures
        def ns_frac(tm):
            return pp.create_elliptic_fracture(
                center=self.tunnel_coord(tm),
                major_axis=self.units.convert_units(600, "m"),
                minor_axis=self.units.convert_units(600, "m"),
                major_axis_angle=0,  # TODO?
                strike_angle=90 * np.pi / 180,  # N-S
                dip_angle=50 * np.pi / 180,
                num_points=16,
                # index=0,
            )

        # self._fractures = []
        self._fractures = [
            ew_frac(tm=1330),
            ew_frac(tm=1570),
            ew_frac(tm=1850),
            ns_frac(tm=2200),
            ns_frac(tm=2440),
            tunnel_perp(tm=1925),
        ]

        # Allow to steer number of fractures
        num_fractures = self.params.get("num_fractures")
        self._fractures = self._fractures[:num_fractures]
