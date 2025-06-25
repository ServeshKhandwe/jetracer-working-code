"""
system_identification.py

This module implements system identification methods to learn a MatrixZonotope
representation of the system dynamics from collected trajectory data.
The learned model captures both nominal dynamics and uncertainty bounds.
"""

import numpy as np
import rospy
from typing import Tuple, Optional
from sklearn.linear_model import LinearRegression, Ridge
from zonotope import MatrixZonotope


class SystemIdentification:
    """
    System identification for learning MatrixZonotope dynamics models from data.
    """

    def __init__(self, config: dict):
        """
        Initialize system identification.
        
        Args:
            config (dict): Configuration dictionary containing noise parameters
        """
        self.config = config
        self.model_learned = False
        self.A_nominal = None
        self.B_nominal = None
        self.residuals = None

    def learn_linear_model(self, X_data: np.ndarray, U_data: np.ndarray, X_next_data: np.ndarray, 
                          regularization: float = 1e-6) -> MatrixZonotope:
        """
        Learn a linear dynamics model from data: X_{t+1} = A*X_t + B*U_t + noise
        
        Args:
            X_data: Current states (N, n_state)
            U_data: Actions (N, n_action)  
            X_next_data: Next states (N, n_state)
            regularization: Ridge regularization parameter
            
        Returns:
            MatrixZonotope representing the learned dynamics model
        """
        if X_data.shape[0] == 0:
            rospy.logerr("No training data provided for system identification")
            return self._create_fallback_model(8, 2)  # Default turtlebot dimensions
            
        n_state = X_data.shape[1]
        n_action = U_data.shape[1]
        
        rospy.loginfo(f"Learning linear model with {X_data.shape[0]} data points")
        rospy.loginfo(f"State dim: {n_state}, Action dim: {n_action}")
        
        # Prepare regression data: [X_t, U_t] -> X_{t+1}
        XU_data = np.hstack([X_data, U_data])  # (N, n_state + n_action)
        
        # Learn dynamics for each state dimension
        A_list = []
        B_list = []
        residuals_list = []
        
        for i in range(n_state):
            # Ridge regression for numerical stability
            reg = Ridge(alpha=regularization, fit_intercept=False)
            reg.fit(XU_data, X_next_data[:, i])
            
            # Extract A and B matrices
            A_row = reg.coef_[:n_state]
            B_row = reg.coef_[n_state:]
            
            A_list.append(A_row)
            B_list.append(B_row)
            
            # Compute residuals for uncertainty estimation
            predictions = reg.predict(XU_data)
            residuals = X_next_data[:, i] - predictions
            residuals_list.append(residuals)
        
        # Construct nominal matrices
        self.A_nominal = np.array(A_list)  # (n_state, n_state)
        self.B_nominal = np.array(B_list)  # (n_state, n_action)
        self.residuals = np.array(residuals_list).T  # (N, n_state)
        
        rospy.loginfo(f"Learned A matrix shape: {self.A_nominal.shape}")
        rospy.loginfo(f"Learned B matrix shape: {self.B_nominal.shape}")
        rospy.loginfo(f"Residuals shape: {self.residuals.shape}")
        
        # Create MatrixZonotope with uncertainty
        matrix_zonotope = self._create_matrix_zonotope_from_regression(n_state, n_action)
        
        self.model_learned = True
        return matrix_zonotope

    def _create_matrix_zonotope_from_regression(self, n_state: int, n_action: int) -> MatrixZonotope:
        """
        Create a MatrixZonotope from the learned regression parameters and residuals.
        """
        # Nominal dynamics matrix [A, B]
        C_M = np.hstack([self.A_nominal, self.B_nominal])  # (n_state, n_state + n_action)
        
        # Estimate uncertainty from residuals - fix the variable name and ensure proper dtype
        residual_std = np.std(self.residuals.astype(np.float64), axis=0)  # Standard deviation per state dimension
        
        # Get noise scales from config with proper fallback
        safety_config = self.config.get("safety", {})
        noise_config = safety_config.get("noise", {})
        
        # Use fallback values if noise config is missing
        process_noise_scale = noise_config.get("process_noise_scale", 0.01)
        measurement_noise_scale = noise_config.get("measurement_noise_scale", 0.005)
        
        rospy.logdebug(f"Noise configuration: process_scale={process_noise_scale}, measurement_scale={measurement_noise_scale}")
        
        # Create uncertainty generators
        # Generator 1: Process noise uncertainty
        G1 = np.zeros((n_state, n_state + n_action))
        for i in range(n_state):
            uncertainty_scale = max(float(residual_std[i]) * process_noise_scale, 0.001)  # Minimum uncertainty
            G1[i, i] = uncertainty_scale  # Diagonal uncertainty in A matrix
        
        # Generator 2: Measurement noise uncertainty  
        G2 = np.zeros((n_state, n_state + n_action))
        for i in range(n_state):
            uncertainty_scale = max(float(residual_std[i]) * measurement_noise_scale, 0.0005)
            if i < n_action:  # Add uncertainty to B matrix entries
                G2[i, n_state + i] = uncertainty_scale
        
        # Generator 3: Cross-coupling uncertainty
        G3 = np.zeros((n_state, n_state + n_action))
        avg_residual_std = float(np.mean(residual_std))
        cross_uncertainty = avg_residual_std * 0.01  # Small cross-coupling uncertainty
        
        # Add small off-diagonal uncertainties
        for i in range(min(n_state, 3)):  # Limit to avoid too many generators
            for j in range(min(n_state, 3)):
                if i != j:
                    G3[i, j] = cross_uncertainty
        
        # Concatenate all generators
        G_M = np.concatenate([G1, G2, G3], axis=1)  # (n_state, 3*(n_state + n_action))
        
        rospy.loginfo(f"MatrixZonotope center shape: {C_M.shape}")
        rospy.loginfo(f"MatrixZonotope generators shape: {G_M.shape}")
        rospy.loginfo(f"Residual statistics - mean: {np.mean(residual_std):.4f}, max: {np.max(residual_std):.4f}")
        
        return MatrixZonotope(C_M, G_M)

    def _create_fallback_model(self, n_state: int, n_action: int) -> MatrixZonotope:
        """Create a simple fallback model when no data is available."""
        rospy.logwarn("Creating fallback kinematic model (no data available)")
        
        # Simple kinematic model for turtlebot
        dt = 0.1
        A = np.eye(n_state)
        B = np.zeros((n_state, n_action))
        
        # Basic kinematic relationships
        if n_state >= 4 and n_action >= 2:
            B[0, 0] = dt  # dx/dv
            B[2, 1] = dt  # dtheta/domega
            B[3, 0] = 1.0  # v = v_cmd
        
        C_M = np.hstack([A, B])
        G_M = 0.05 * np.ones((n_state, n_state + n_action))  # Conservative uncertainty
        
        return MatrixZonotope(C_M, G_M)

    def validate_model(self, X_val: np.ndarray, U_val: np.ndarray, X_next_val: np.ndarray) -> dict:
        """
        Validate the learned model on validation data.
        
        Returns:
            dict: Validation metrics
        """
        if not self.model_learned or X_val.shape[0] == 0:
            return {"error": "No model learned or no validation data"}
        
        # Predict using learned model
        XU_val = np.hstack([X_val, U_val])
        predictions = XU_val @ np.hstack([self.A_nominal, self.B_nominal]).T
        
        # Compute metrics
        mse = np.mean((predictions - X_next_val) ** 2)
        mae = np.mean(np.abs(predictions - X_next_val))
        
        # R-squared per dimension
        r2_scores = []
        for i in range(X_next_val.shape[1]):
            ss_res = np.sum((X_next_val[:, i] - predictions[:, i]) ** 2)
            ss_tot = np.sum((X_next_val[:, i] - np.mean(X_next_val[:, i])) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            r2_scores.append(r2)
        
        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "r2_mean": float(np.mean(r2_scores)),
            "r2_scores": [float(r2) for r2 in r2_scores],
            "validation_samples": X_val.shape[0]
        }
        
        rospy.loginfo(f"Model validation: MSE={mse:.6f}, MAE={mae:.6f}, R²={np.mean(r2_scores):.4f}")
        
        return metrics
