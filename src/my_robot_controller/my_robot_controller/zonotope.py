"""
zonotope.py

This module implements the Zonotope class which provides operations for creating and manipulating zonotopes.
A zonotope is defined by a center vector and a generator matrix. The class supports basic operations such as:
    - linear_transform: apply a linear transformation to the zonotope.
    - minkowski_sum: compute the Minkowski sum of two zonotopes.
    - cartesian_product: compute the Cartesian product of two zonotopes.
    - to_interval: convert the zonotope into interval bounds.

These operations are essential for representing reachable sets and uncertainties in the data-driven predictive control framework.
"""

import numpy as np
import cvxpy as cp
from typing import Tuple, List, Union


# Helper function
def interval_to_zonotope(lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> "Zonotope":
    """Converts an interval vector [l, u] into a zonotope <c, G_diag>."""
    if not isinstance(lower_bounds, np.ndarray) or not isinstance(upper_bounds, np.ndarray):
        raise TypeError("Bounds must be numpy arrays for interval_to_zonotope.")
    if lower_bounds.shape != upper_bounds.shape or lower_bounds.ndim != 1:
        raise ValueError("Bounds must be 1D and have the same shape.")
    center = (upper_bounds + lower_bounds) / 2.0
    generators_diag_elements = (upper_bounds - lower_bounds) / 2.0
    generators_diag_elements = np.maximum(generators_diag_elements, 0)
    generators = np.diag(generators_diag_elements)
    return Zonotope(center, generators)


class Zonotope:
    """Class representing a zonotope defined by a center and generators.

    Attributes:
        center (Union[np.ndarray, cp.Expression]): 1D array or CVXPY expression representing the center of the zonotope.
        generators (np.ndarray): 2D array representing the generator matrix of the zonotope;
                                 each column corresponds to one generator.
    """

    def __init__(self, center: Union[np.ndarray, cp.Expression], generators: np.ndarray) -> None:
        """
        Initialize the zonotope with a given center and generator matrix.

        Args:
            center (Union[np.ndarray, cp.Expression]): 1D array or CVXPY expression representing the zonotope center.
            generators (np.ndarray): 2D array of shape (n, m) representing the generator matrix.

        Raises:
            ValueError: If center is not a 1D array or CVXPY expression, if generators is not a 2D array,
                        or if the number of rows in generators does not match the length of center.
        """
        self.center = center

        if isinstance(center, np.ndarray):
            if center.ndim != 1:
                raise ValueError("Numeric center must be a 1D numpy array.")
            self.dim = center.shape[0]
        elif isinstance(center, cp.Expression):
            if center.ndim != 1:
                if center.ndim == 2 and center.shape[1] == 1:
                    self.center = center.flatten()
                else:
                    raise ValueError("CVXPY center must be 1D or (n,1) shaped.")
            self.dim = self.center.shape[0]
        else:
            raise TypeError("Center must be a numpy array or a CVXPY expression.")

        self.generators: np.ndarray = np.array(generators, dtype=np.float64)
        if self.generators.ndim != 2:
            if self.generators.size == 0:
                self.generators = np.empty((self.dim, 0), dtype=np.float64)
            else:
                raise ValueError("Generators must be a 2D numpy array.")

        if self.generators.shape[0] != self.dim and self.generators.shape[1] != 0:
            raise ValueError(
                f"Generators must have {self.dim} rows (matching center dimension), "
                f"or be (dim, 0) if no generators. Got G_shape={self.generators.shape}"
            )

    def linear_transform(self, L: np.ndarray) -> "Zonotope":
        """
        Compute the linear transformation of the zonotope under a mapping L.

        Given a transformation matrix L, the transformed zonotope is:
            L * Z = ⟨L * center, L * generators⟩

        Args:
            L (np.ndarray): 2D transformation matrix.

        Returns:
            Zonotope: A new zonotope that is the result of applying L to the current zonotope.

        Raises:
            ValueError: If L is not a 2D array or if the number of columns in L does not
                        match the dimension of the zonotope.
        """
        if L.ndim != 2:
            raise ValueError("Transformation matrix L must be a 2D numpy array.")
        
        n_dim: int = self.dim
        if L.shape[1] != n_dim:
            raise ValueError("The number of columns in L must match the dimension of the zonotope's center.")
        
        new_center = L @ self.center
        new_generators = L @ self.generators
        return Zonotope(new_center, new_generators)

    def minkowski_sum(self, other: "Zonotope") -> "Zonotope":
        """
        Compute the Minkowski sum of this zonotope with another.

        The Minkowski sum of zonotopes Z1 = ⟨c1, G1⟩ and Z2 = ⟨c2, G2⟩ is defined by:
            Z1 + Z2 = ⟨c1 + c2, [G1, G2]⟩

        Args:
            other (Zonotope): The other zonotope to add.

        Returns:
            Zonotope: A new zonotope representing the Minkowski sum.

        Raises:
            ValueError: If the zonotopes have centers of different dimensions.
        """
        if self.dim != other.dim:
            raise ValueError("Both zonotopes must have centers of the same dimension for the Minkowski sum.")
        
        new_center = self.center + other.center
        new_generators = np.concatenate((self.generators, other.generators), axis=1)
        return Zonotope(new_center, new_generators)

    def cartesian_product(self, other: "Zonotope") -> "Zonotope":
        """
        Compute the Cartesian product of this zonotope with another.
        Handles cases where one or both zonotopes might have CVXPY expression centers.
        Generators are assumed to be numeric.
        """
        c1_is_cvx = isinstance(self.center, cp.Expression)
        c2_is_cvx = isinstance(other.center, cp.Expression)

        if c1_is_cvx or c2_is_cvx:
            new_center = cp.hstack([self.center, other.center])
        else:
            new_center = np.concatenate((self.center, other.center), axis=0)

        n1, m1 = self.generators.shape
        n2, m2 = other.generators.shape
        top_left = self.generators
        top_right = np.zeros((n1, m2), dtype=np.float64)
        bottom_left = np.zeros((n2, m1), dtype=np.float64)
        bottom_right = other.generators
        new_generators = np.block([
            [top_left, top_right],
            [bottom_left, bottom_right]
        ])
        return Zonotope(new_center, new_generators)

    def get_interval_bounds(self) -> Tuple[Union[np.ndarray, cp.Expression], Union[np.ndarray, cp.Expression]]:
        """
        Convert the zonotope into interval bounds. Center can be CVXPY expression.
        Generators must be numeric.
        """
        if self.generators.size == 0:
            if isinstance(self.center, np.ndarray):
                delta = np.zeros_like(self.center)
            else:
                delta = np.zeros(self.center.shape)
        else:
            delta = np.sum(np.abs(self.generators), axis=1)
        lower_bound = self.center - delta
        upper_bound = self.center + delta
        return lower_bound, upper_bound

    def __add__(self, other: "Zonotope") -> "Zonotope":
        if self.dim != other.dim:
            raise ValueError("Zonotope centers must have the same dimension for Minkowski sum.")
        new_center = self.center + other.center
        if self.generators.size == 0:
            new_generators = other.generators.copy()
        elif other.generators.size == 0:
            new_generators = self.generators.copy()
        else:
            new_generators = np.concatenate((self.generators, other.generators), axis=1)
        return Zonotope(new_center, new_generators)

    def __sub__(self, other: "Zonotope") -> "Zonotope":
        if self.dim != other.dim:
            raise ValueError("Zonotope centers must have the same dimension for Minkowski difference.")
        new_center = self.center - other.center
        if self.generators.size == 0:
            new_generators = other.generators.copy()
        elif other.generators.size == 0:
            new_generators = self.generators.copy()
        else:
            new_generators = np.concatenate((self.generators, other.generators), axis=1)
        return Zonotope(new_center, new_generators)

    def scale_by_scalar(self, scalar: float) -> "Zonotope":
        new_center = self.center * scalar
        new_generators = self.generators * scalar
        return Zonotope(new_center, new_generators)

    def to_interval(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the zonotope into interval bounds.

        For each coordinate j, the zonotope is represented as:
            x[j] = center[j] + Sum_i (β_i * generators[j, i]), with β_i in [-1, 1].

        The maximum deviation in each coordinate is the sum of the absolute values of the generators,
        so the interval bounds are:
            lower_bound = center - delta, upper_bound = center + delta,
        where delta[j] = sum(|generators[j, :]|).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing (lower_bound, upper_bound) as 1D numpy arrays.
        """
        if self.generators.size == 0:
            delta: np.ndarray = np.zeros_like(self.center)
        else:
            delta: np.ndarray = np.sum(np.abs(self.generators), axis=1)
        
        lower_bound: np.ndarray = self.center - delta
        upper_bound: np.ndarray = self.center + delta
        return lower_bound, upper_bound

    def reduce_order_girard(self, target_order: int) -> "Zonotope":
        """
        Reduces the order of the zonotope using Girard's method.
        The goal is to have approximately 'target_order' generators,
        where some of these might be new axis-aligned generators from bounding.

        Args:
            target_order (int): The desired approximate number of generators in the
                                reduced zonotope. More precisely, we aim to keep
                                target_order - self.dim principal components/generators
                                and bound the rest if target_order > self.dim.
                                If target_order <= self.dim, it might result in only
                                axis-aligned box generators.

        Returns:
            Zonotope: The order-reduced zonotope.
        """
        if self.generators.shape[1] == 0 or self.generators.shape[1] <= target_order:
            # No reduction needed if no generators or already fewer/equal to target
            # Or if target_order is too small to be meaningful beyond an interval box
            if self.generators.shape[1] <= self.dim:
                return self
        num_original_gens = self.generators.shape[1]
        dim = self.dim

        # Number of original generators to keep directly.
        # The interval box for the remainder will add 'dim' generators.
        # So, we want to keep roughly (target_order - dim) original generators.
        num_gens_to_keep = max(0, target_order - dim)

        if num_gens_to_keep >= num_original_gens:
            return self

        if num_gens_to_keep < 0:
            num_gens_to_keep = 0

        # Calculate the L-infinity norm for each generator vector (column)
        l_inf_norms = np.array([np.linalg.norm(self.generators[:, i], ord=np.inf) for i in range(num_original_gens)])

        # Get indices that would sort these norms in ascending order
        sorted_indices = np.argsort(l_inf_norms)

        # Generators to keep (those with largest L-infinity norms)
        kept_generators_indices = sorted_indices[-num_gens_to_keep:] if num_gens_to_keep > 0 else []
        kept_generators = self.generators[:, kept_generators_indices] if num_gens_to_keep > 0 else np.empty((dim, 0))

        # Generators to be enclosed in an interval box
        residual_generators_indices = sorted_indices[:-num_gens_to_keep] if num_gens_to_keep > 0 else sorted_indices[:]
        residual_generators = self.generators[:, residual_generators_indices]

        # Compute the radius of the interval box for residual generators
        if residual_generators.shape[1] > 0:
            interval_box_radii = np.sum(np.abs(residual_generators), axis=1)
        else:
            interval_box_radii = np.zeros(dim)

        # Create new generators for this interval box (axis-aligned)
        interval_box_radii = np.maximum(interval_box_radii, 0)
        box_generators = np.diag(interval_box_radii)

        # Filter out zero columns from box_generators to avoid unnecessary generators
        non_zero_radii_indices = np.where(interval_box_radii > 1e-9)[0]
        if non_zero_radii_indices.size > 0:
            final_box_generators = box_generators[:, non_zero_radii_indices]
        else:
            final_box_generators = np.empty((dim, 0))

        # Combine kept generators and box generators
        if kept_generators.shape[1] > 0 and final_box_generators.shape[1] > 0:
            new_generators = np.concatenate((kept_generators, final_box_generators), axis=1)
        elif kept_generators.shape[1] > 0:
            new_generators = kept_generators
        elif final_box_generators.shape[1] > 0:
            new_generators = final_box_generators
        else:
            new_generators = np.empty((dim, 0))

        return Zonotope(self.center, new_generators)

    def polygon(self) -> np.ndarray:
        """
        Computes the vertices of the polygon representing the 2D projection of the zonotope.
        If the zonotope is of dimension > 2, it is projected onto its first two dimensions.
        This method is only applicable for zonotopes with a numeric center and dimension >= 2.

        Returns:
            np.ndarray: A 2xN array of polygon vertices.

        Raises:
            ValueError: If the zonotope is not at least 2D or has a non-numeric (CVXPY) center.
        """
        if self.dim < 2:
            raise ValueError("Polygon can only be computed for zonotopes of at least 2 dimensions.")
        if not isinstance(self.center, np.ndarray):
            raise ValueError("Polygon computation requires a numeric center.")

        c = self.center[:2]
        G = self.generators[:2, :]

        if G.shape[1] == 0:
            return c.reshape(2, 1)

        n = G.shape[1]

        xmax = np.sum(np.abs(G[0, :]))
        ymax = np.sum(np.abs(G[1, :]))

        Gnorm = np.copy(G)
        Gnorm[:, np.where(G[1, :] < 0)] = G[:, np.where(G[1, :] < 0)] * -1

        angles = np.arctan2(Gnorm[1, :], Gnorm[0, :])

        angles[np.where(angles < 0)] = angles[np.where(angles < 0)] + 2 * np.pi

        IX = np.argsort(angles)

        p = np.zeros((2, n + 1))

        for i in range(n):
            p[:, i + 1] = p[:, i] + 2 * Gnorm[:, IX[i]]

        p[0, :] = p[0, :] + xmax - max(p[0, :])
        p[1, :] = p[1, :] - ymax

        p = np.vstack((np.hstack((p[0, :], p[0, -1] + p[0, 0] - p[0, 1:])),
                       np.hstack((p[1, :], p[1, -1] + p[1, 0] - p[1, 1:]))))

        # consider center
        p[0, :] = c[0] + p[0, :]
        p[1, :] = c[1] + p[1, :]

        return p

    def __repr__(self) -> str:
        return f"Zonotope(center={self.center}, generators={self.generators})"


class MatrixZonotope:
    """Class representing a matrix zonotope.
    
    A matrix zonotope is defined by a center matrix C and a set of generator matrices G_i.
    M = { C + sum(beta_i * G_i) | -1 <= beta_i <= 1 }
    
    For implementation simplicity aligned with Def. 2 of the paper,
    the generators are stored as a single concatenated matrix:
    tilde_G = [G_1, G_2, ..., G_gamma_M]
    where each G_i is an (n x T) matrix, so tilde_G is (n x (T * gamma_M)).
    The center C is (n x T).

    Attributes:
        center (np.ndarray): 2D array representing the center matrix (n x T).
        generator_matrices (np.ndarray): 2D array representing the concatenated generator
                                         matrices (n x (T * gamma_M)).
                                         It has the same number of rows as 'center'.
        num_rows (int): Number of rows (n).
        num_cols (int): Number of columns in the center matrix (T).
        num_generators (int): Number of actual generator matrices (gamma_M).
                              This is derived if generator_matrices is shaped n x (T * gamma_M).
    """

    def __init__(self, center: np.ndarray, generator_matrices: np.ndarray) -> None:
        """
        Initialize the MatrixZonotope.

        Args:
            center (np.ndarray): 2D array (n x T) for the center matrix.
            generator_matrices (np.ndarray): 2D array (n x (T * gamma_M)) for concatenated
                                             generator matrices. If there are no generators,
                                             it can be an (n x 0) array.

        Raises:
            ValueError: If dimensions are inconsistent.
        """
        self.center: np.ndarray = np.array(center, dtype=np.float64)
        if self.center.ndim != 2:
            raise ValueError("Center matrix must be a 2D numpy array.")

        self.generator_matrices: np.ndarray = np.array(generator_matrices, dtype=np.float64)
        if self.generator_matrices.ndim != 2:
            if self.generator_matrices.size == 0: # Allow empty array [] to be reshaped
                self.generator_matrices = np.empty((self.center.shape[0], 0), dtype=np.float64)
            else:
                 raise ValueError("Generator matrices (tilde_G) must be a 2D numpy array.")
        
        if self.generator_matrices.shape[0] != self.center.shape[0] and self.generator_matrices.shape[1] != 0 :
             # Allow (0,0) for generators if center is (0, Tcol) or (Nrow, 0)
            if not (self.center.shape[0] == 0 and self.generator_matrices.shape[0] == 0):
                raise ValueError(
                    "Generator matrices must have the same number of rows (n) as the center matrix, "
                    f"or be (N_center_rows, 0) if no generators. Got center_rows={self.center.shape[0]}, "
                    f"gen_rows={self.generator_matrices.shape[0]}, gen_cols={self.generator_matrices.shape[1]}"
                )

        self.num_rows: int = self.center.shape[0]
        self.num_cols: int = self.center.shape[1] # This is T

        if self.num_cols == 0 and self.generator_matrices.shape[1] != 0:
             raise ValueError("Center matrix cannot have 0 columns if there are generator matrices with columns.")

        # If num_cols (T) is 0, num_generators is ill-defined unless generator_matrices is also 0-column.
        if self.num_cols == 0:
            if self.generator_matrices.shape[1] == 0:
                self.num_generators: int = 0
            else: # This case should have been caught by the check above
                raise ValueError("If center has 0 columns, generator_matrices must also have 0 columns.")
        else: # num_cols > 0
            if self.generator_matrices.shape[1] % self.num_cols != 0:
                raise ValueError(
                    f"The number of columns in generator_matrices ({self.generator_matrices.shape[1]}) "
                    f"must be a multiple of the number of columns in the center matrix T ({self.num_cols})."
                )
            self.num_generators: int = self.generator_matrices.shape[1] // self.num_cols

    def get_nominal_matrix(self) -> np.ndarray:
        return self.center

    def get_generator_matrix_by_index(self, index: int) -> np.ndarray:
        if not (0 <= index < self.num_generators):
            raise IndexError(f"Generator index {index} out of bounds for {self.num_generators} generators.")
        start_col = index * self.num_cols
        end_col = (index + 1) * self.num_cols
        return self.generator_matrices[:, start_col:end_col]

    def multiply_by_vector_zonotope_approx(
            self,
            vec_zono: Zonotope,
            u_interval_bounds: Union[None, Tuple[np.ndarray, np.ndarray]] = None,
            u_dim: int = 0,
        ) -> Zonotope:
        """
        Approximated multiplication M * X where M is this MatrixZonotope,
        and X is a vector Zonotope.
        X.center can be a CVXPY expression: X.center = [c_R_numeric; u_k_cvxpy_var]
        Generators of X (G_X) are numeric.

        Result Y = <c_Y, G_Y> where c_Y is CVXPY expr, G_Y is numeric.
        c_Y = C_M @ X.center
        G_Y contains:
            1. C_M @ G_X
            2. G_M^(i) @ c_R_numeric (numeric part of X.center)
            3. Zonotope representing interval hull of sum(beta_i * G_M_B^(i) @ u_k_interval)
            4. G_M^(i) @ G_X (cross terms of generators)
        """
        C_M = self.center
        c_X = vec_zono.center
        G_X = vec_zono.generators

        # 1. New center: C_M @ X.center (CVXPY expression if X.center is)
        new_center_expr = C_M @ c_X

        all_new_numeric_generators_list: List[np.ndarray] = []

        # 2. Generators from C_M @ G_X
        if G_X.size > 0:
            gens_from_CM_GX = C_M @ G_X
            all_new_numeric_generators_list.append(gens_from_CM_GX)

        for i in range(self.num_generators):
            G_M_i = self.get_generator_matrix_by_index(i)
            G_M_A_i = G_M_i[:, :-u_dim] if u_dim > 0 else G_M_i
            G_M_B_i = G_M_i[:, -u_dim:] if u_dim > 0 else np.empty((G_M_i.shape[0], 0))

            # 3. Generators from G_M^(i) @ c_X_numeric_part (numeric part of X's center)
            if not isinstance(c_X, cp.Expression):
                gen_from_GMI_cX = G_M_i @ c_X
                if gen_from_GMI_cX.ndim == 1:
                    gen_from_GMI_cX = gen_from_GMI_cX.reshape(-1, 1)
                all_new_numeric_generators_list.append(gen_from_GMI_cX)

            # 4. Generators from G_M^(i) @ G_X (cross-terms of generators)
            if G_X.size > 0:
                gens_from_GMI_GX = G_M_i @ G_X
                all_new_numeric_generators_list.append(gens_from_GMI_GX)

        # 5. Handle terms involving u_k from c_X if u_interval_bounds are provided
        c_u_eff_total = np.zeros((self.num_rows, 1))
        if u_interval_bounds is not None and u_dim > 0:
            u_lower, u_upper = u_interval_bounds
            z_uk_interval = interval_to_zonotope(u_lower.flatten(), u_upper.flatten())
            for i in range(self.num_generators):
                G_M_i = self.get_generator_matrix_by_index(i)
                G_M_B_i = G_M_i[:, -u_dim:]
                gen_from_GMBi_center_uk = G_M_B_i @ z_uk_interval.center
                if gen_from_GMBi_center_uk.ndim == 1:
                    gen_from_GMBi_center_uk = gen_from_GMBi_center_uk.reshape(-1, 1)
                all_new_numeric_generators_list.append(gen_from_GMBi_center_uk)
                if z_uk_interval.generators.size > 0:
                    gens_from_GMBi_GUK = G_M_B_i @ z_uk_interval.generators
                    all_new_numeric_generators_list.append(gens_from_GMBi_GUK)

        if not all_new_numeric_generators_list:
            final_numeric_generators = np.empty((self.num_rows, 0))
        else:
            final_numeric_generators = np.concatenate(all_new_numeric_generators_list, axis=1)

        return Zonotope(new_center_expr + c_u_eff_total.flatten(), final_numeric_generators)

    def __repr__(self) -> str:
        return (f"MatrixZonotope(center_shape={self.center.shape}, "
                f"generators_shape={self.generator_matrices.shape}, "
                f"num_actual_generators={self.num_generators})")

    def __add__(self, other: "MatrixZonotope") -> "MatrixZonotope":
        """
        Compute the Minkowski sum of this matrix zonotope with another.
        M1 + M2 = <C1 + C2, [tilde_G1, tilde_G2]>
        """
        if not isinstance(other, MatrixZonotope):
            return NotImplemented
        if self.center.shape != other.center.shape:
            raise ValueError("Center matrices must have the same shape for addition.")
        
        new_center = self.center + other.center
        
        # Concatenate generator matrices (tilde_G format)
        if self.generator_matrices.size == 0:
            new_generators = other.generator_matrices.copy()
        elif other.generator_matrices.size == 0:
            new_generators = self.generator_matrices.copy()
        else:
            new_generators = np.concatenate((self.generator_matrices, other.generator_matrices), axis=1)
            
        return MatrixZonotope(new_center, new_generators)

    def __sub__(self, other: "MatrixZonotope") -> "MatrixZonotope":
        """
        Compute the Minkowski difference M1 - M2.
        M1 - M2 = <C1 - C2, [tilde_G1, tilde_G2]>
        Note: For zonotopes, A - B = A + (-B). -B = <-CB, tilde_GB>.
        So the generators are still concatenated.
        """
        if not isinstance(other, MatrixZonotope):
            return NotImplemented
        if self.center.shape != other.center.shape:
            raise ValueError("Center matrices must have the same shape for subtraction.")
            
        new_center = self.center - other.center
        
        if self.generator_matrices.size == 0:
            new_generators = other.generator_matrices.copy() # effectively [0, G2]
        elif other.generator_matrices.size == 0:
            new_generators = self.generator_matrices.copy() # effectively [G1, 0]
        else:
            new_generators = np.concatenate((self.generator_matrices, other.generator_matrices), axis=1)
            
        return MatrixZonotope(new_center, new_generators)

    def scale(self, scalar: float) -> "MatrixZonotope":
        """
        Scale the matrix zonotope by a scalar.
        scalar * M = <scalar * C, scalar * tilde_G>
        """
        new_center = scalar * self.center
        new_generators = scalar * self.generator_matrices
        return MatrixZonotope(new_center, new_generators)

    def multiply_by_matrix(self, matrix: np.ndarray) -> "MatrixZonotope":
        """
        Post-multiply the matrix zonotope by a constant matrix P.
        M * P = <C * P, [G1*P, G2*P, ...]>
        This means tilde_G * P effectively, but tilde_G = [G1, G2, ...]
        So, new_tilde_G = [G1@P, G2@P, ...].
        Each Gi is (n_rows x num_cols_center). P must be (num_cols_center x num_cols_P).
        Resulting Gi' will be (n_rows x num_cols_P).
        New center C' = C @ P, shape (n_rows x num_cols_P)
        New tilde_G' will be (n_rows x (num_cols_P * num_generators)
        """
        if matrix.ndim != 2:
            raise ValueError("Multiplier matrix must be 2D.")
        if self.num_cols != matrix.shape[0]: # num_cols is T (cols of C and individual Gi)
            raise ValueError(
                f"Matrix zonotope's column count ({self.num_cols}) "
                f"must match multiplier matrix's row count ({matrix.shape[0]})."
            )

        new_center = self.center @ matrix
        
        if self.num_generators == 0:
            new_tilde_generators = np.empty((self.num_rows, 0), dtype=np.float64)
        else:
            # Process each G_i individually
            gen_list: List[np.ndarray] = []
            for i in range(self.num_generators):
                g_i = self.get_generator_matrix_by_index(i)
                gi_new = g_i @ matrix
                gen_list.append(gi_new)
            new_tilde_generators = np.concatenate(gen_list, axis=1) # n_rows x (num_cols_P * num_generators)
            
        return MatrixZonotope(new_center, new_tilde_generators)
