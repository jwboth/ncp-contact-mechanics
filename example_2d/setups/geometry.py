import numpy as np
import porepy as pp


class GeometryFromFile:
    """2D cube with fracture network in the center."""

    params: dict
    units: pp.Units
    solid: pp.SolidConstants

    def meshing_arguments(self) -> dict:
        mesh_args = {}
        mesh_args["cell_size"] = self.units.convert_units(100, "m")
        cell_size_fracture = self.params.get("cell_size_fracture", 30)
        mesh_args["cell_size_fracture"] = self.units.convert_units(
            cell_size_fracture, "m"
        )
        return mesh_args

    def meshing_kwargs(self) -> dict:
        kwargs = super().meshing_kwargs()
        if hasattr(self, "constraint_ind"):
            kwargs["constraints"] = self.constraint_ind
        kwargs["file_name"] = self.params.get("gmsh_file_name")
        return kwargs

    def grid_type(self) -> str:
        return "simplex"

    def set_fractures(self) -> None:
        study = self.params.get("study", None)
        assert self.params.get("study") is not None, "Study not defined."
        num_fractures = [study, study]

        fracture_generator_file = self.params.get("fracture_generator_file")

        main_orientations = [np.pi / 4, -np.pi / 8]
        points = []
        # Base fracture 1 and 2 in center of domain
        x = self.units.convert_units(1500 - 150 * np.cos(main_orientations[0]), "m")
        y = self.units.convert_units(1500 - 150 * np.sin(main_orientations[0]), "m")
        x_end = self.units.convert_units(1500 + 150 * np.cos(main_orientations[0]), "m")
        y_end = self.units.convert_units(1500 + 150 * np.sin(main_orientations[0]), "m")
        points.append(np.array([[x, y], [x_end, y_end]]))
        i = 1
        x = self.units.convert_units(1400 - 150 * np.cos(main_orientations[1]), "m")
        y = self.units.convert_units(1500 - 150 * np.sin(main_orientations[1]), "m")
        x_end = self.units.convert_units(1400 + 150 * np.cos(main_orientations[1]), "m")
        y_end = self.units.convert_units(1500 + 150 * np.sin(main_orientations[1]), "m")
        points.append(np.array([[x, y], [x_end, y_end]]))

        # Read the fracture generator file
        data = np.genfromtxt(fracture_generator_file, delimiter=",")
        random_centers_1 = data[:24, :2].T
        random_perturbations_1 = data[:24, 2].T
        random_centers_2 = data[:24, 3:5].T
        random_perturbations_2 = data[:24, 5].T

        # Merge arrays for convenience
        random_centers = np.array([random_centers_1, random_centers_2])
        random_perturbations = np.vstack(
            [random_perturbations_1, random_perturbations_2]
        )

        # Add random fractures
        for i, nf in enumerate(num_fractures):
            for j in range(nf):
                # Make a random perturbation of the main orientations
                perturbation = random_perturbations[i, j]
                orientation = main_orientations[i] + perturbation
                xmid = self.units.convert_units(random_centers[i, 0, j], "m")
                ymid = self.units.convert_units(random_centers[i, 1, j], "m")
                x = xmid - self.units.convert_units(150 * np.cos(orientation), "m")
                y = ymid - self.units.convert_units(150 * np.sin(orientation), "m")
                x_end = xmid + self.units.convert_units(150 * np.cos(orientation), "m")
                y_end = ymid + self.units.convert_units(150 * np.sin(orientation), "m")
                points.append(np.array([[x, y], [x_end, y_end]]))
        self._fractures = [pp.LineFracture(pts.T) for pts in points]

    def set_domain(self) -> None:
        """Set the cube domain."""
        bounding_box = {
            "xmin": self.units.convert_units(0, "m"),
            "xmax": self.units.convert_units(3000, "m"),
            "ymin": self.units.convert_units(0, "m"),
            "ymax": self.units.convert_units(3000, "m"),
        }
        self._domain = pp.Domain(bounding_box)


def find_intersection(line1, line2) -> np.ndarray | None:
    """Find the intersection point of two lines.

    Parameters:
        line1: First line.
        line2: Second line.

    Returns:
        Intersection point if it exists, else None.

    """
    # Determine factor alpha needed to extend or shrink line1 to hit line2
    # Each line is defines as [[x, y], [x_end, y_end]]
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    # Check if the lines are parallel
    if (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) == 0:
        return None, None, None

    # Find alpha (float) such that (x1,y1) + alpha*((x2-x1), (y2-y1)) = (x3,y3) + beta*((x4-x3), (y4-y3))
    # This is equivalent to solving a linear system of equations
    A = np.array([[x2 - x1, x3 - x4], [y2 - y1, y3 - y4]])
    b = np.array([x3 - x1, y3 - y1])
    try:
        x = np.linalg.solve(A, b)
        alpha = x[0]
        beta = x[1]
        angle1 = np.arctan2(y2 - y1, x2 - x1)
        angle2 = np.arctan2(y4 - y3, x4 - x3)
        angle = angle1 - angle2
    except np.linalg.LinAlgError:
        return None, None, None
    if 0 <= alpha <= 1 and 0 <= beta <= 1:
        return alpha, beta, angle
    else:
        return None, None, None


class GeometryFromFile_SingleFracs(GeometryFromFile):
    def set_geometry(self) -> None:
        """After creating the mesh, determine fracture statistics."""
        super().set_geometry()

        # Fetch active fracture intersections
        intersections = self._fracture_intersections

        # Determine the fracture cells which are intersected by other fractures
        fracture_subdomains = self.mdg.subdomains(dim=self.nd - 1)
        fracture_intersection_cells = {sd: [] for sd in fracture_subdomains}
        for intersection in intersections:
            intersection_point, _ = intersection
            for sd in fracture_subdomains:
                # Find closest node
                for closest_node, node in enumerate(sd.nodes[: self.nd].T):
                    if not np.isclose(
                        np.linalg.norm(node - intersection_point), 0, atol=1e-10
                    ):
                        continue
                    # Find the cells which contains the node
                    # sd.cell_nodes() is a csc matrix of dimension nodes x cells, and is 1 if the node is part of the cell, i.e., find all cells in the row closest_node
                    row = sd.cell_nodes().getrow(closest_node)
                    closest_cells = row.nonzero()[1]
                    fracture_intersection_cells[sd].extend(closest_cells.tolist())

        # Build a indicator vector which is 1 in all fracture_intersection_cells
        for sd in fracture_subdomains:
            data = self.mdg.subdomain_data(sd)
            indicator = np.zeros(sd.num_cells)
            if len(fracture_intersection_cells[sd]) > 0:
                indicator[np.array(fracture_intersection_cells[sd])] = 1
            pp.set_solution_values(
                name="fracture_intersection_cells",
                values=indicator,
                data=data,
                iterate_index=0,
            )

    def _add_line_safely(self, trial_line, lines, depth=0):
        no_intersections_angle_cutoff = self.params.get(
            "no_intersections_angle_cutoff", np.pi / 2
        )
        intersections = []

        no_intersection = True
        (x, y), (x_end, y_end) = trial_line
        old_lines = lines.copy()
        # Split lines
        for _, line in enumerate(old_lines):
            alpha, _, angle = find_intersection(trial_line, line)

            # Only continue if there is an intersection
            if alpha is None or angle is None:
                continue

            # Unique angle
            angle1 = np.abs(angle)
            angle2 = np.pi - angle1
            angle = min(angle1, angle2)

            small_angle = (
                no_intersections_angle_cutoff > 0
                and angle <= no_intersections_angle_cutoff
            )
            large_angle = (
                no_intersections_angle_cutoff < 0
                and angle >= np.pi / 2 + no_intersections_angle_cutoff
            )

            # Check if the angle is small
            if small_angle or large_angle:
                # Remove the intersection by breaking the line into two.
                # But only if the intersection is not at the endpoints.
                no_intersection = False
                if 0.02 < alpha:
                    x_new_end = x + (alpha - 0.01) * (x_end - x)
                    y_new_end = y + (alpha - 0.01) * (y_end - y)
                    trial_line_left = np.array([[x, y], [x_new_end, y_new_end]])
                    new_lines_1, _ = self._add_line_safely(
                        trial_line_left, old_lines, depth + 1
                    )
                else:
                    new_lines_1 = old_lines
                if alpha < 0.98:
                    x_new_start = x + (alpha + 0.01) * (x_end - x)
                    y_new_start = y + (alpha + 0.01) * (y_end - y)
                    trial_line_right = np.array(
                        [[x_new_start, y_new_start], [x_end, y_end]]
                    )
                    new_lines, _ = self._add_line_safely(
                        trial_line_right, new_lines_1, depth + 1
                    )
                else:
                    new_lines = new_lines_1

            else:
                # Keep the intersection if the angle is large enough
                x_intersection = x + alpha * (x_end - x)
                y_intersection = y + alpha * (y_end - y)
                intersections.append(((x_intersection, y_intersection), angle))

        # Add line
        if no_intersection:
            new_lines = lines.copy()
            new_lines.append(trial_line)
        return new_lines, intersections

    def set_fractures(self):
        study = self.params.get("study", None)
        assert self.params.get("study") is not None, "Study not defined."
        num_fractures = [study, study]

        fracture_generator_file = self.params.get("fracture_generator_file")

        intersections = []

        main_orientations = [np.pi / 4, -np.pi / 8]
        points = []
        # Base fracture 1 and 2 in center of domain
        x = self.units.convert_units(1500 - 150 * np.cos(main_orientations[0]), "m")
        y = self.units.convert_units(1500 - 150 * np.sin(main_orientations[0]), "m")
        x_end = self.units.convert_units(1500 + 150 * np.cos(main_orientations[0]), "m")
        y_end = self.units.convert_units(1500 + 150 * np.sin(main_orientations[0]), "m")
        points.append(np.array([[x, y], [x_end, y_end]]))
        i = 1
        x = self.units.convert_units(1400 - 150 * np.cos(main_orientations[1]), "m")
        y = self.units.convert_units(1500 - 150 * np.sin(main_orientations[1]), "m")
        x_end = self.units.convert_units(1400 + 150 * np.cos(main_orientations[1]), "m")
        y_end = self.units.convert_units(1500 + 150 * np.sin(main_orientations[1]), "m")
        trial_line = np.array([[x, y], [x_end, y_end]])
        points, new_intersections = self._add_line_safely(trial_line, points)
        intersections = intersections + new_intersections

        # Read the fracture generator file
        data = np.genfromtxt(fracture_generator_file, delimiter=",")
        random_centers_1 = data[:24, :2].T
        random_perturbations_1 = data[:24, 2].T
        random_centers_2 = data[:24, 3:5].T
        random_perturbations_2 = data[:24, 5].T

        # Merge arrays for convenience
        random_centers = np.array([random_centers_1, random_centers_2])
        random_perturbations = np.vstack(
            [random_perturbations_1, random_perturbations_2]
        )

        # Add random fractures
        for i, nf in enumerate(num_fractures):
            for j in range(nf):
                # Make a random perturbation of the main orientations
                perturbation = random_perturbations[i, j]
                orientation = main_orientations[i] + perturbation
                xmid = self.units.convert_units(random_centers[i, 0, j], "m")
                ymid = self.units.convert_units(random_centers[i, 1, j], "m")
                x = xmid - self.units.convert_units(150 * np.cos(orientation), "m")
                y = ymid - self.units.convert_units(150 * np.sin(orientation), "m")
                x_end = xmid + self.units.convert_units(150 * np.cos(orientation), "m")
                y_end = ymid + self.units.convert_units(150 * np.sin(orientation), "m")
                trial_line = np.array([[x, y], [x_end, y_end]])
                # Check if the trial line intersects with any of the existing fractures
                points, new_intersections = self._add_line_safely(trial_line, points)
                intersections = intersections + new_intersections
        self._fractures = [pp.LineFracture(pts.T) for pts in points]

        # Cache the intersections
        self._fracture_intersections = intersections


class Geometry:
    """2D cube with fracture network in the center."""

    params: dict
    units: pp.Units
    solid: pp.SolidConstants

    def meshing_arguments(self) -> dict:
        mesh_args = {}
        mesh_args["cell_size"] = self.units.convert_units(100, "m")
        cell_size_fracture = self.params.get("cell_size_fracture", 30)
        mesh_args["cell_size_fracture"] = self.units.convert_units(
            cell_size_fracture, "m"
        )
        mesh_args["file_name"] = self.params.get("gmsh_file_name")
        return mesh_args

    def meshing_kwargs(self) -> dict:
        kwargs = super().meshing_kwargs()
        if hasattr(self, "constraint_ind"):
            kwargs["constraints"] = self.constraint_ind
        return kwargs

    def grid_type(self) -> str:
        return "simplex"

    def set_fractures(self) -> None:
        study = self.params.get("study", None)
        assert self.params.get("study") is not None, "Study not defined."

        seed = self.params.get("seed")
        np.random.seed(seed)

        num_fractures = [study, study]
        # if study == 0:
        #    num_fractures = [0, 0]

        # elif study == 1:
        #    num_fractures = [3, 3]

        # elif study == 2:
        #    num_fractures = [6, 6]

        # elif study == 3:
        #    num_fractures = [12, 12]

        # elif study == 4:
        #    num_fractures = [24, 24]

        # elif study == 50:
        #    np.random.seed(seed)
        #    num_fractures = [6, 6]

        # elif study in [60, 70]:
        #    ...

        # else:
        #    raise ValueError(f"Study {study} not recognized.")

        main_orientations = [np.pi / 4, -np.pi / 8]
        points = []
        # Base fracture 1 and 2 in center of domain
        x = self.units.convert_units(1500 - 150 * np.cos(main_orientations[0]), "m")
        y = self.units.convert_units(1500 - 150 * np.sin(main_orientations[0]), "m")
        x_end = self.units.convert_units(1500 + 150 * np.cos(main_orientations[0]), "m")
        y_end = self.units.convert_units(1500 + 150 * np.sin(main_orientations[0]), "m")
        points.append(np.array([[x, y], [x_end, y_end]]))
        i = 1
        x = self.units.convert_units(1400 - 150 * np.cos(main_orientations[1]), "m")
        y = self.units.convert_units(1500 - 150 * np.sin(main_orientations[1]), "m")
        x_end = self.units.convert_units(1400 + 150 * np.cos(main_orientations[1]), "m")
        y_end = self.units.convert_units(1500 + 150 * np.sin(main_orientations[1]), "m")
        points.append(np.array([[x, y], [x_end, y_end]]))

        # if study == 50:
        #     def add_lines_from_lists(
        #         fractures1, fractures2, perturbations1, perturbations2
        #     ):
        #         points = []
        #         for f, p in zip(fractures1, perturbations1):
        #             x = self.units.convert_units(f[0], "m")
        #             y = self.units.convert_units(f[1], "m")
        #             x_end = x + self.units.convert_units(
        #                 500 * np.cos(main_orientations[0] + p), "m"
        #             )
        #             y_end = y + self.units.convert_units(
        #                 500 * np.sin(main_orientations[0] + p), "m"
        #             )
        #             points.append(np.array([[x, y], [x_end, y_end]]))

        #         for f, p in zip(fractures2, perturbations2):
        #             x = self.units.convert_units(f[0], "m")
        #             y = self.units.convert_units(f[1], "m")
        #             x_end = x + self.units.convert_units(
        #                 500 * np.cos(main_orientations[1] + p + np.pi), "m"
        #             )
        #             y_end = y + self.units.convert_units(
        #                 500 * np.sin(main_orientations[1] + p + np.pi), "m"
        #             )
        #             points.append(np.array([[x, y], [x_end, y_end]]))
        #         return points

        #     # Manual design (start coordinates in pixels)
        #     fractures1 = [
        #         [900, 1355],
        #         [990, 1390],
        #         [1050, 1500],
        #         [1270, 1310],
        #         [1840, 1900],
        #         [1570, 2140],
        #         [1520, 2040],
        #         [1210, 2040],
        #         [1080, 1700],
        #         [1545, 1785],
        #         [1610, 1485],
        #         [1345, 1590],
        #     ]

        #     fractures2 = [
        #         [1210, 1420],
        #         [1525, 1265],
        #         [1360, 1770],
        #         [1500, 2050],
        #         [2050, 2100],
        #         [2080, 1700],
        #         [1720, 1820],
        #         [1990, 1540],
        #         [2145, 1495],
        #         [2080, 1345],
        #         [2055, 1195],
        #         [1830, 1145],
        #         # [1585, 1540],
        #     ]

        #     # Convert from pixels to meters
        #     fractures1 = [[f[0], 3000 - f[1]] for f in fractures1]
        #     fractures2 = [[f[0], 3000 - f[1]] for f in fractures2]

        #     # Perturbations
        #     perturbations1 = np.random.rand(len(fractures1)) * np.pi / 16
        #     perturbations2 = np.random.rand(len(fractures2)) * np.pi / 16

        #     points = add_lines_from_lists(
        #         fractures1, fractures2, perturbations1, perturbations2
        #     )
        #     self._fractures = [pp.LineFracture(pts.T) for pts in points]
        #     return None

        # if study == 60:
        #     # Manual design (start coordinates in pixels)
        #     fractures1 = [[1500, 1500], [1400, 1325]]

        #     fractures2 = [[1350, 1410], [1355, 1415]]

        #     # Convert from pixels to meters
        #     fractures1 = [[f[0], 3000 - f[1]] for f in fractures1]
        #     fractures2 = [[f[0], 3000 - f[1]] for f in fractures2]

        #     # Perturbations
        #     perturbations1 = [seed / 40 * np.pi, 0 * np.pi]
        #     perturbations2 = [0 * np.pi, 0 * np.pi]

        #     for f, p in zip(fractures1, perturbations1):
        #         xmid = self.units.convert_units(f[0], "m")
        #         ymid = self.units.convert_units(f[1], "m")
        #         x = xmid - self.units.convert_units(
        #             250 * np.cos(main_orientations[0] + p), "m"
        #         )
        #         y = ymid - self.units.convert_units(
        #             250 * np.sin(main_orientations[0] + p), "m"
        #         )
        #         x_end = xmid + self.units.convert_units(
        #             250 * np.cos(main_orientations[0] + p), "m"
        #         )
        #         y_end = ymid + self.units.convert_units(
        #             250 * np.sin(main_orientations[0] + p), "m"
        #         )
        #         points.append(np.array([[x, y], [x_end, y_end]]))

        #     for f, p in zip(fractures2, perturbations2):
        #         xmid = self.units.convert_units(f[0], "m")
        #         ymid = self.units.convert_units(f[1], "m")
        #         x = xmid - self.units.convert_units(
        #             250 * np.cos(main_orientations[1] + p), "m"
        #         )
        #         y = ymid - self.units.convert_units(
        #             250 * np.sin(main_orientations[1] + p), "m"
        #         )
        #         x_end = xmid + self.units.convert_units(
        #             250 * np.cos(main_orientations[1] + p), "m"
        #         )
        #         y_end = ymid + self.units.convert_units(
        #             250 * np.sin(main_orientations[1] + p), "m"
        #         )
        #         points.append(np.array([[x, y], [x_end, y_end]]))

        #     self._fractures = [pp.LineFracture(pts.T) for pts in points]
        #     return None

        # if study == 70:
        #     # Start and end points of the fractures
        #     fractures = [
        #         [[1406, 1593], [1611, 1390]],
        #         [[1158, 1420], [1361, 1216]],
        #         [[1144, 1395], [1431, 1495]],
        #         [[1283, 1245], [1551, 1359]],
        #         [[1197, 1490], [1480, 1568]],
        #         [[1225, 1627], [1390, 1390]],
        #         [[1265, 1618], [1524, 1705]],
        #         [[1752, 1252], [1584, 1500]],
        #         [[1580, 1445], [1857, 1520]],
        #         [[1172, 1783], [1433, 1883]],
        #         [[1155, 1719], [1424, 1803]],
        #         [[1220, 1787], [1375, 1554]],
        #         [[1858, 1608], [1584, 1500]],
        #         [[1553, 1294], [1335, 1500]],
        #         [[1698, 1525], [1402, 1423]],
        #         [[1739, 1530], [1527, 1772]],
        #         [[1774, 1864], [1496, 1775]],
        #         [[1808, 1761], [1524, 1705]],
        #         [[1662, 1881], [1394, 1790]],
        #         [[1448, 1378], [1272, 1630]],
        #         [[1465, 1485], [1264, 1701]],
        #         [[1515, 1503], [1338, 1737]],
        #         [[1307, 1741], [1583, 1852]],
        #         [[1592, 1653], [1422, 1792]],
        #         [[1424, 1803], [1621, 1597]],
        #         [[1418, 1850], [1587, 1627]],
        #     ]

        #     # Convert y-coord from pixel to Euclidean
        #     fractures = [
        #         [[f0, 3000 - f1], [f2, 3000 - f3]] for [[f0, f1], [f2, f3]] in fractures
        #     ]

        #     # Start from scratch
        #     points = []
        #     for f in fractures:
        #         points.append(np.array(f))
        #     self._fractures = [pp.LineFracture(pts.T) for pts in points]
        #     return None

        # Generate 24 x 24 random fractures. Depending on the study pick the first 3 x 3,
        # 6 x 6, 12 x 12, or 24 x 24. For creating the fractures, generate a random position
        # in a 100 x 100 square around the center of the domain. Then generate a random
        random_perturbations = np.random.rand(2, 24) * np.pi / 16
        random_centers = [1500 - 100 + 200 * np.random.rand(2, 24) for _ in range(2)]

        # Add random fractures
        for i, nf in enumerate(num_fractures):
            for j in range(nf):
                # Make a random perturbation of the main orientations
                perturbation = random_perturbations[i, j]
                orientation = main_orientations[i] + perturbation
                xmid = self.units.convert_units(random_centers[i][0, j], "m")
                ymid = self.units.convert_units(random_centers[i][1, j], "m")
                x = xmid - self.units.convert_units(150 * np.cos(orientation), "m")
                y = ymid - self.units.convert_units(150 * np.sin(orientation), "m")
                x_end = xmid + self.units.convert_units(150 * np.cos(orientation), "m")
                y_end = ymid + self.units.convert_units(150 * np.sin(orientation), "m")
                # if study in [40]:
                #    xmid = self.units.convert_units(
                #        1500 - 50 + np.random.rand() * 100, "m"
                #    )
                #    ymid = self.units.convert_units(
                #        1500 - 50 + np.random.rand() * 100, "m"
                #    )
                #    x = xmid - self.units.convert_units(
                #        150 * np.cos(orientations[j]), "m"
                #    )
                #    y = ymid - self.units.convert_units(
                #        150 * np.sin(orientations[j]), "m"
                #    )
                #    x_end = xmid + self.units.convert_units(
                #        150 * np.cos(orientations[j]), "m"
                #    )
                #    y_end = ymid + self.units.convert_units(
                #        150 * np.sin(orientations[j]), "m"
                #    )
                # else:
                #    x = self.units.convert_units(np.random.rand() * 500 + 1100, "m")
                #    y = self.units.convert_units(np.random.rand() * 500 + 1100, "m")
                #    # Find the end point of the fracture by following the orientation
                #    x_end = x + self.units.convert_units(
                #        300 * np.cos(orientations[j]), "m"
                #    )
                #    y_end = y + self.units.convert_units(
                #        300 * np.sin(orientations[j]), "m"
                #    )
                points.append(np.array([[x, y], [x_end, y_end]]))
        self._fractures = [pp.LineFracture(pts.T) for pts in points]

    def set_domain(self) -> None:
        """Set the cube domain."""
        bounding_box = {
            "xmin": self.units.convert_units(0, "m"),
            "xmax": self.units.convert_units(3000, "m"),
            "ymin": self.units.convert_units(0, "m"),
            "ymax": self.units.convert_units(3000, "m"),
        }
        self._domain = pp.Domain(bounding_box)


class EGS_NiceGeometry_2d(Geometry):
    """2D cube with fracture network in the center."""

    def set_fractures(self) -> None:
        study = self.params.get("study", None)
        assert self.params.get("study") is not None, "Study not defined."

        if study == 2:
            np.random.seed(seed)
            num_fractures = [3, 3]

        # NCP seems to work (but only for increased residual aperture)
        # PP also does not work with relaxation and without
        elif study == 3:
            np.random.seed(seed)
            num_fractures = [6, 6]

        elif study == 4:
            np.random.seed(seed)
            num_fractures = [12, 12]

        else:
            raise ValueError(f"Study {study} not recognized.")

        main_orientations = [np.pi / 4, -np.pi / 8]
        points = []
        # Base fracture 1
        x = self.units.convert_units(1500 - 150 * np.cos(main_orientations[0]), "m")
        y = self.units.convert_units(1500 - 150 * np.sin(main_orientations[0]), "m")
        x_end = self.units.convert_units(1500 + 150 * np.cos(main_orientations[0]), "m")
        y_end = self.units.convert_units(1500 + 150 * np.sin(main_orientations[0]), "m")
        points.append(np.array([[x, y], [x_end, y_end]]))

        def check_line(trial_line, lines):
            fracture_OK = True
            for _, line in enumerate(lines):
                _, _, angle = find_intersection(trial_line, line)
                ic(angle)
                fracture_OK = fracture_OK and (
                    angle is not None and 0.2 < np.abs(angle) < np.pi - 0.2
                )
            ic(fracture_OK)
            return fracture_OK

        # Additional fractures
        counter = [0, 0]
        for j in [1, 0]:
            while True:
                # Make a random perturbation of the main orientations
                perturbation = np.random.rand(1) * np.pi / 16
                orientations = main_orientations[j] + perturbation[0]
                # Find nf many random points in the domain [1250, 1750] x [1250, 1750]
                x = self.units.convert_units(np.random.rand() * 300 + 1350, "m")
                y = self.units.convert_units(np.random.rand() * 300 + 1350, "m")
                # Find the end point of the fracture by following the orientation
                x_end = x + self.units.convert_units(300 * np.cos(orientations), "m")
                y_end = y + self.units.convert_units(300 * np.sin(orientations), "m")
                trial_line = np.array([[x, y], [x_end, y_end]])
                if check_line(trial_line, points):
                    points.append(np.array([[x, y], [x_end, y_end]]))
                    counter[j] += 1
                if counter[j] == num_fractures[j]:
                    break

        self._fractures = [pp.LineFracture(pts.T) for pts in points]
        assert False


class EGS_SingleFracs_2d(Geometry):
    def set_fractures(self):
        study = self.params.get("study", None)
        assert self.params.get("study") is not None, "Study not defined."
        no_intersections_angle_cutoff = self.params.get(
            "no_intersections_angle_cutoff", np.pi / 2
        )

        if study == 0:
            np.random.seed(seed)
            num_fractures = [0, 0]

        elif study == 1:
            np.random.seed(seed)
            num_fractures = [1, 1]

        elif study == 2:
            np.random.seed(seed)
            num_fractures = [3, 3]

        # NCP seems to work (but only for increased residual aperture)
        # PP also does not work with relaxation and without
        elif study == 3:
            np.random.seed(seed)
            num_fractures = [6, 6]

        elif study == 4:
            np.random.seed(seed)
            num_fractures = [12, 12]

        elif study == 5:
            np.random.seed(seed)
            num_fractures = [24, 24]

        else:
            raise ValueError(f"Study {study} not recognized.")

        main_orientations = [np.pi / 4, -np.pi / 8]
        points = []
        # Base fracture 1
        x = self.units.convert_units(1500 - 150 * np.cos(main_orientations[0]), "m")
        y = self.units.convert_units(1500 - 150 * np.sin(main_orientations[0]), "m")
        x_end = self.units.convert_units(1500 + 150 * np.cos(main_orientations[0]), "m")
        y_end = self.units.convert_units(1500 + 150 * np.sin(main_orientations[0]), "m")
        points.append(np.array([[x, y], [x_end, y_end]]))
        # Base fracture 2
        x = self.units.convert_units(1400 - 150 * np.cos(main_orientations[1]), "m")
        y = self.units.convert_units(1700 - 150 * np.sin(main_orientations[1]), "m")
        x_end = self.units.convert_units(1400 + 150 * np.cos(main_orientations[1]), "m")
        y_end = self.units.convert_units(1700 + 150 * np.sin(main_orientations[1]), "m")
        points.append(np.array([[x, y], [x_end, y_end]]))

        def add_line_safely(trial_line, lines, depth=0):
            no_intersection = True
            (x, y), (x_end, y_end) = trial_line
            old_lines = lines.copy()
            # Split lines
            for _, line in enumerate(old_lines):
                alpha, _, angle = find_intersection(trial_line, line)
                angle_OK = angle is not None and (
                    np.abs(angle) <= no_intersections_angle_cutoff
                    or np.abs(angle) >= np.pi - no_intersections_angle_cutoff
                )
                if alpha is not None and angle_OK:
                    no_intersection = False
                    # Break the line into two
                    if 0.02 < alpha:
                        x_new_end = x + (alpha - 0.01) * (x_end - x)
                        y_new_end = y + (alpha - 0.01) * (y_end - y)
                        trial_line_left = np.array([[x, y], [x_new_end, y_new_end]])
                        new_lines_1 = add_line_safely(
                            trial_line_left, old_lines, depth + 1
                        )
                    else:
                        new_lines_1 = old_lines
                    if alpha < 0.98:
                        x_new_start = x + (alpha + 0.01) * (x_end - x)
                        y_new_start = y + (alpha + 0.01) * (y_end - y)
                        trial_line_right = np.array(
                            [[x_new_start, y_new_start], [x_end, y_end]]
                        )
                        new_lines = add_line_safely(
                            trial_line_right, new_lines_1, depth + 1
                        )
                    else:
                        new_lines = new_lines_1
            # Add line
            if no_intersection:
                new_lines = lines.copy()
                new_lines.append(trial_line)
            return new_lines

        # Additional fractures
        for i, nf in enumerate(num_fractures):
            # Make a random perturbation of the main orientations
            perturbation = np.random.rand(nf) * np.pi / 16
            orientations = main_orientations[i] + perturbation
            # Find nf many random points in the domain [1250, 1750] x [1250, 1750]
            for i in range(nf):
                x = self.units.convert_units(np.random.rand() * 500 + 1100, "m")
                y = self.units.convert_units(np.random.rand() * 500 + 1100, "m")
                # Find the end point of the fracture by following the orientation
                x_end = x + self.units.convert_units(300 * np.cos(orientations[i]), "m")
                y_end = y + self.units.convert_units(300 * np.sin(orientations[i]), "m")
                trial_line = np.array([[x, y], [x_end, y_end]])
                # Check if the trial line intersects with any of the existing fractures
                points = add_line_safely(trial_line, points)
        self._fractures = [pp.LineFracture(pts.T) for pts in points]


class EGS_ResolvedFracs_2d(Geometry):
    def set_fractures(self):
        study = self.params.get("study", None)
        assert self.params.get("study") is not None, "Study not defined."

        if study == 0:
            np.random.seed(seed)
            num_fractures = [0, 0]

        elif study == 1:
            np.random.seed(seed)
            num_fractures = [1, 1]

        elif study == 2:
            np.random.seed(seed)
            num_fractures = [3, 3]

        # NCP seems to work (but only for increased residual aperture)
        # PP also does not work with relaxation and without
        elif study == 3:
            np.random.seed(seed)
            num_fractures = [6, 6]

        elif study == 4:
            np.random.seed(seed)
            num_fractures = [12, 12]

        elif study == 5:
            np.random.seed(seed)
            num_fractures = [24, 24]

        else:
            raise ValueError(f"Study {study} not recognized.")

        main_orientations = [np.pi / 4, -np.pi / 8]
        points = []
        # Base fracture 1
        x = self.units.convert_units(1500 - 150 * np.cos(main_orientations[0]), "m")
        y = self.units.convert_units(1500 - 150 * np.sin(main_orientations[0]), "m")
        x_end = self.units.convert_units(1500 + 150 * np.cos(main_orientations[0]), "m")
        y_end = self.units.convert_units(1500 + 150 * np.sin(main_orientations[0]), "m")
        points.append(np.array([[x, y], [x_end, y_end]]))
        # Base fracture 2
        x = self.units.convert_units(1400 - 150 * np.cos(main_orientations[1]), "m")
        y = self.units.convert_units(1700 - 150 * np.sin(main_orientations[1]), "m")
        x_end = self.units.convert_units(1400 + 150 * np.cos(main_orientations[1]), "m")
        y_end = self.units.convert_units(1700 + 150 * np.sin(main_orientations[1]), "m")
        points.append(np.array([[x, y], [x_end, y_end]]))

        def add_line_safely(trial_line, lines, depth=0):
            no_intersection = True
            (x, y), (x_end, y_end) = trial_line
            old_lines = lines.copy()
            # Split lines
            for _, line in enumerate(old_lines):
                alpha, _, angle = find_intersection(trial_line, line)
                if alpha is not None and (
                    np.abs(angle) <= no_intersections_angle_cutoff
                    or np.abs(angle) >= np.pi - no_intersections_angle_cutoff
                ):
                    # print(alpha, depth)
                    assert depth < 20
                    cell_size_fracture = self.params.get("cell_size_fracture", 30)
                    fracture_length = np.sqrt((x_end - x) ** 2 + (y_end - y) ** 2)
                    dalpha = cell_size_fracture / fracture_length * 0.001
                    alpha0 = dalpha / 2
                    alpha1 = 1 - alpha0
                    # Break the line into two
                    if (
                        not np.isclose(alpha, 0.0, atol=1e-8)
                        and not np.isclose(alpha, 1.0, atol=1e-8)
                        and not np.isclose(alpha, 0.5, atol=1e-8)
                    ):
                        # print("split line", alpha)
                        if np.isclose(alpha, alpha0) or np.isclose(alpha, alpha1):
                            print(alpha)
                            raise ValueError("unlucky...")
                        no_intersection = False
                        # Split line [[x,y], [x_end, y_end]] into three lines using the scaling factor alpha
                        # First line is alpha 0..alpha-0.01, second line is alpha-0.01..alpha+0.01, third line is alpha+0.01..1
                        x0 = x
                        y0 = y
                        x1 = x + (alpha - alpha0) * (x_end - x)
                        y1 = y + (alpha - alpha0) * (y_end - y)
                        x2 = x + (alpha + alpha0) * (x_end - x)
                        y2 = y + (alpha + alpha0) * (y_end - y)
                        x3 = x_end
                        y3 = y_end
                        trial_line_0 = np.array([[x0, y0], [x1, y1]])
                        trial_line_1 = np.array([[x1, y1], [x2, y2]])
                        trial_line_2 = np.array([[x2, y2], [x3, y3]])
                        new_lines_0 = add_line_safely(
                            trial_line_0, old_lines, depth + 1
                        )
                        new_lines_1 = add_line_safely(
                            trial_line_1, new_lines_0, depth + 1
                        )
                        new_lines = add_line_safely(
                            trial_line_2, new_lines_1, depth + 1
                        )
                    else:
                        # print("no split", alpha, alpha0)
                        new_lines = old_lines
            # Add line
            if no_intersection:
                new_lines = lines.copy()
                new_lines.append(trial_line)
            return new_lines

        # Additional fractures
        for i, nf in enumerate(num_fractures):
            # Make a random perturbation of the main orientations
            perturbation = np.random.rand(nf) * np.pi / 16
            orientations = main_orientations[i] + perturbation
            # Find nf many random points in the domain [1250, 1750] x [1250, 1750]
            for i in range(nf):
                x = self.units.convert_units(np.random.rand() * 500 + 1100, "m")
                y = self.units.convert_units(np.random.rand() * 500 + 1100, "m")
                # Find the end point of the fracture by following the orientation
                x_end = x + self.units.convert_units(300 * np.cos(orientations[i]), "m")
                y_end = y + self.units.convert_units(300 * np.sin(orientations[i]), "m")
                trial_line = np.array([[x, y], [x_end, y_end]])
                # Check if the trial line intersects with any of the existing fractures
                points = add_line_safely(trial_line, points)
        self._fractures = [pp.LineFracture(pts.T) for pts in points]


class EGS_ConstrainedResolvedFracs_2d(Geometry):
    def set_fractures(self):
        study = self.params.get("study", None)
        assert self.params.get("study") is not None, "Study not defined."

        if study == 0:
            np.random.seed(seed)
            num_fractures = [0, 0]

        elif study == 1:
            np.random.seed(seed)
            num_fractures = [1, 1]

        elif study == 2:
            np.random.seed(seed)
            num_fractures = [3, 3]

        # NCP seems to work (but only for increased residual aperture)
        # PP also does not work with relaxation and without
        elif study == 3:
            np.random.seed(seed)
            num_fractures = [6, 6]

        elif study == 4:
            np.random.seed(seed)
            num_fractures = [12, 12]

        elif study == 5:
            np.random.seed(seed)
            num_fractures = [24, 24]

        else:
            raise ValueError(f"Study {study} not recognized.")

        main_orientations = [np.pi / 4, -np.pi / 8]
        points = []
        points_for_constraints = []
        # Base fracture 1
        x = self.units.convert_units(1500 - 150 * np.cos(main_orientations[0]), "m")
        y = self.units.convert_units(1500 - 150 * np.sin(main_orientations[0]), "m")
        x_end = self.units.convert_units(1500 + 150 * np.cos(main_orientations[0]), "m")
        y_end = self.units.convert_units(1500 + 150 * np.sin(main_orientations[0]), "m")
        points.append(np.array([[x, y], [x_end, y_end]]))
        # Base fracture 2
        x = self.units.convert_units(1400 - 150 * np.cos(main_orientations[1]), "m")
        y = self.units.convert_units(1700 - 150 * np.sin(main_orientations[1]), "m")
        x_end = self.units.convert_units(1400 + 150 * np.cos(main_orientations[1]), "m")
        y_end = self.units.convert_units(1700 + 150 * np.sin(main_orientations[1]), "m")
        points.append(np.array([[x, y], [x_end, y_end]]))

        def add_line_safely(trial_line, lines, depth=0):
            no_intersection = True
            (x, y), (x_end, y_end) = trial_line
            old_lines = lines.copy()
            # Split lines
            for _, line in enumerate(old_lines):
                alpha, _, angle = find_intersection(trial_line, line)
                if alpha is not None and (
                    np.abs(angle) <= no_intersections_angle_cutoff
                    or np.abs(angle) >= np.pi - no_intersections_angle_cutoff
                ):
                    assert depth < 20
                    cell_size_fracture = self.params.get("cell_size_fracture", 30)
                    fracture_length = np.sqrt((x_end - x) ** 2 + (y_end - y) ** 2)
                    dalpha = cell_size_fracture / fracture_length * 0.001
                    alpha0 = dalpha / 2
                    alpha1 = 1 - alpha0
                    print(f"New intersection {alpha} | {angle} | {alpha0}")
                    # Break the line into two
                    if False and (
                        not np.isclose(alpha, 0.0, atol=1e-8)
                        and not np.isclose(alpha, 1.0, atol=1e-8)
                        and not np.isclose(alpha, 0.5, atol=1e-8)
                    ):
                        # print("split line", alpha)
                        if np.isclose(alpha, alpha0) or np.isclose(alpha, alpha1):
                            print(alpha)
                            raise ValueError("unlucky...")
                        no_intersection = False
                        # Split line [[x,y], [x_end, y_end]] into three lines using the scaling factor alpha
                        # First line is alpha 0..alpha-0.01, second line is alpha-0.01..alpha+0.01, third line is alpha+0.01..1
                        x0 = x
                        y0 = y
                        x1 = x + (alpha - alpha0) * (x_end - x)
                        y1 = y + (alpha - alpha0) * (y_end - y)
                        x2 = x + (alpha + alpha0) * (x_end - x)
                        y2 = y + (alpha + alpha0) * (y_end - y)
                        x3 = x_end
                        y3 = y_end
                        trial_line_0 = np.array([[x0, y0], [x1, y1]])
                        trial_line_1 = np.array([[x1, y1], [x2, y2]])
                        trial_line_2 = np.array([[x2, y2], [x3, y3]])
                        new_lines_0 = add_line_safely(
                            trial_line_0, old_lines, depth + 1
                        )
                        new_lines_1 = add_line_safely(
                            trial_line_1, new_lines_0, depth + 1
                        )
                        new_lines = add_line_safely(
                            trial_line_2, new_lines_1, depth + 1
                        )

                    else:
                        # print("no split", alpha, alpha0)
                        new_lines = old_lines

                    x_intersection = x + alpha * (x_end - x)
                    y_intersection = y + alpha * (y_end - y)

                    (x_line, y_line), (x_end_line, y_end_line) = line
                    x_intersection_orth0 = (
                        x_intersection
                        + alpha0 * (y_end - y)
                        + alpha0 * (y_end_line - y_line)
                    )
                    y_intersection_orth0 = (
                        y_intersection
                        - alpha0 * (x_end - x)
                        - alpha0 * (x_end_line - x_line)
                    )
                    x_intersection_orth1 = (
                        x_intersection
                        - alpha0 * (y_end - y)
                        - alpha0 * (y_end_line - y_line)
                    )
                    y_intersection_orth1 = (
                        y_intersection
                        + alpha0 * (x_end - x)
                        + alpha0 * (x_end_line - x_line)
                    )
                    points_for_constraints.append(
                        np.array(
                            [
                                [x_intersection_orth0, y_intersection_orth0],
                                [x_intersection_orth1, y_intersection_orth1],
                            ]
                        )
                    )

                    x_intersection_aligned0 = (
                        x_intersection
                        - alpha0 * (x_end - x)
                        - alpha0 * (x_end_line - x_line)
                    )
                    y_intersection_aligned0 = (
                        y_intersection
                        - alpha0 * (y_end - y)
                        - alpha0 * (y_end_line - y_line)
                    )
                    x_intersection_aligned1 = (
                        x_intersection
                        + alpha0 * (x_end - x)
                        + alpha0 * (x_end_line - x_line)
                    )
                    y_intersection_aligned1 = (
                        y_intersection
                        + alpha0 * (y_end - y)
                        + alpha0 * (y_end_line - y_line)
                    )
                    ic(x_intersection_aligned0, y_intersection_aligned0)
                    ic(x_intersection_aligned1, y_intersection_aligned1)
                    points_for_constraints.append(
                        np.array(
                            [
                                [x_intersection_aligned0, y_intersection_aligned0],
                                [x_intersection_aligned1, y_intersection_aligned1],
                            ]
                        )
                    )

            # Add line
            if no_intersection:
                new_lines = lines.copy()
                new_lines.append(trial_line)
            return new_lines

        # Additional fractures
        for i, nf in enumerate(num_fractures):
            # Make a random perturbation of the main orientations
            perturbation = np.random.rand(nf) * np.pi / 16
            orientations = main_orientations[i] + perturbation
            # Find nf many random points in the domain [1250, 1750] x [1250, 1750]
            for i in range(nf):
                x = self.units.convert_units(np.random.rand() * 500 + 1100, "m")
                y = self.units.convert_units(np.random.rand() * 500 + 1100, "m")
                # Find the end point of the fracture by following the orientation
                x_end = x + self.units.convert_units(300 * np.cos(orientations[i]), "m")
                y_end = y + self.units.convert_units(300 * np.sin(orientations[i]), "m")
                trial_line = np.array([[x, y], [x_end, y_end]])
                # Check if the trial line intersects with any of the existing fractures
                points = add_line_safely(trial_line, points)
        self._fractures = [pp.LineFracture(pts.T) for pts in points]
        # self._constraints = [pp.LineFracture(pts.T) for pts in points_for_constraints]
        # self._fractures += self._constraints
        # self.constraint_ind = np.arange(
        #    len(self._fractures) - len(points_for_constraints), len(self._fractures)
        # )
