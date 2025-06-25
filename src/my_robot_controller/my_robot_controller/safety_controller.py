"""
safety_controller.py

This module implements the SafetyController class which enforces safety for
reinforcement learning (RL) agents using a data‐driven predictive control
strategy. The safety filter adjusts the candidate action provided by the RL agent
by solving a CVXPY optimization problem so that the predicted reachable sets
remain within safe state bounds. The reachable set propagation is based on a
nominal linear system extracted from the offline-computed modelSet (a matrix
zonotope representing the set of consistent system models) and incorporates an
uncertainty margin computed from the zonotope generators and predefined noise
terms.

The SafetyController is used by the Trainer module to "shield" the RL actions,
ensuring safety during training and deployment.
"""

import cvxpy as cp
import numpy as np
import rospy
from typing import Any, Dict, List, Tuple, Union, Optional

from zonotope import Zonotope, MatrixZonotope, interval_to_zonotope


class SafetyController:
    def __init__(self, config: Dict[str, Any], modelSet: MatrixZonotope) -> None:
        self.config = config
        self.modelSet = modelSet 

        self.n_physical_state_from_model = self.modelSet.num_rows 
        self.n_action_from_model = self.modelSet.num_cols - self.n_physical_state_from_model
        
        if self.n_action_from_model <= 0:
            raise ValueError(
                f"Invalid modelSet dimensions: n_physical_state={self.n_physical_state_from_model}, "
                f"n_total_cols={self.modelSet.num_cols}. Cannot determine positive action dimension."
            )

        vector_noise_conf = self.config.get("safety", {}).get("vector_noise", {})
        
        default_z_w_diag = [0.01] * self.n_physical_state_from_model
        z_w_diag: List[float] = vector_noise_conf.get("z_w_generators_diag", default_z_w_diag)
        if len(z_w_diag) != self.n_physical_state_from_model:
            rospy.logwarn(f"Length of 'z_w_generators_diag' ({len(z_w_diag)}) in config does not match "
                           f"physical_state_dim_from_model ({self.n_physical_state_from_model}). Using default: {default_z_w_diag}")
            z_w_diag = default_z_w_diag
        self.Z_w = Zonotope(np.zeros(self.n_physical_state_from_model), np.diag(z_w_diag))

        default_z_v_diag = [0.005] * self.n_physical_state_from_model
        z_v_diag: List[float] = vector_noise_conf.get("z_v_generators_diag", default_z_v_diag)
        if len(z_v_diag) != self.n_physical_state_from_model:
            rospy.logwarn(f"Length of 'z_v_generators_diag' ({len(z_v_diag)}) in config does not match "
                           f"physical_state_dim_from_model ({self.n_physical_state_from_model}). Using default: {default_z_v_diag}")
            z_v_diag = default_z_v_diag
        self.Z_v = Zonotope(np.zeros(self.n_physical_state_from_model), np.diag(z_v_diag))

        default_z_av_diag = [0.002] * self.n_physical_state_from_model
        z_av_diag: List[float] = vector_noise_conf.get("z_av_generators_diag", default_z_av_diag)
        if len(z_av_diag) != self.n_physical_state_from_model:
            rospy.logwarn(f"Length of 'z_av_generators_diag' ({len(z_av_diag)}) in config does not match "
                           f"physical_state_dim_from_model ({self.n_physical_state_from_model}). Using default: {default_z_av_diag}")
            z_av_diag = default_z_av_diag
        self.Z_Av = Zonotope(np.zeros(self.n_physical_state_from_model), np.diag(z_av_diag))

        self.max_generators_in_reach_set = int(
            self.config.get("safety", {}).get("max_generators_in_reach_set", 20)
        )
        
        self.visualizer = None 
        self.problem_status = None # To store problem status for debugging

    def set_visualizer(self, visualizer_instance):
        self.visualizer = visualizer_instance

    def _check_zonotope_intersection(self, zono1: Zonotope, zono2: Zonotope, dims_to_check: Optional[List[int]] = None) -> bool:
        """
        Checks if two zonotopes intersect by examining their interval hull overlap
        in the specified dimensions. For reachable set checking, the first zonotope
        is flipped relative to the robot position for consistency with visualization.

        Args:
            zono1 (Zonotope): First zonotope (assumed to be reachable set).
            zono2 (Zonotope): Second zonotope (assumed to be obstacle).
            dims_to_check (Optional[List[int]]): List of dimension indices to check.
                                                If None, checks all dimensions of zono1.

        Returns:
            bool: True if their interval hulls overlap in all checked dimensions, False otherwise.
        """
        if not isinstance(zono1.center, np.ndarray) or not isinstance(zono2.center, np.ndarray):
            rospy.logwarn("Intersection check skipped: one or both zonotopes have non-numeric centers.")
            return True # Conservative: assume intersection if type is wrong

        # Get the robot's current position from the model
        robot_pos = np.zeros(zono1.dim)
        if hasattr(self, 'current_robot_pos') and self.current_robot_pos is not None:
            # Use stored robot position if available
            for i in range(min(len(self.current_robot_pos), zono1.dim)):
                robot_pos[i] = self.current_robot_pos[i]
        
        # Get interval bounds for original zonotopes
        l1, u1 = zono1.get_interval_bounds()
        l2, u2 = zono2.get_interval_bounds()

        # Create flipped bounds for the reachable set (zono1)
        flipped_l1 = l1.copy()
        flipped_u1 = u1.copy()
        
        # Flip the reachable set position relative to robot position for the first two dimensions (x,y)
        if dims_to_check and 0 in dims_to_check and 1 in dims_to_check and len(l1) >= 2 and len(u1) >= 2:
            # Calculate the center of the reachable set in x,y dimensions
            center_x = (l1[0] + u1[0]) / 2
            center_y = (l1[1] + u1[1]) / 2
            
            # Calculate width and height
            width = u1[0] - l1[0]
            height = u1[1] - l1[1]
            
            # Calculate displacement from robot position
            dx = center_x - robot_pos[0]
            dy = center_y - robot_pos[1]
            
            # Flip the displacement direction
            flipped_center_x = robot_pos[0] - dx
            flipped_center_y = robot_pos[1] - dy
            
            # Update the bounds based on the flipped center
            flipped_l1[0] = flipped_center_x - width/2
            flipped_u1[0] = flipped_center_x + width/2
            flipped_l1[1] = flipped_center_y - height/2
            flipped_u1[1] = flipped_center_y + height/2
            
            rospy.logdebug(f"Flipped reachable set: original=({center_x:.3f},{center_y:.3f}), flipped=({flipped_center_x:.3f},{flipped_center_y:.3f})")
            rospy.logdebug(f"Robot position: ({robot_pos[0]:.3f}, {robot_pos[1]:.3f})")
            rospy.logdebug(f"Original bounds: x=[{l1[0]:.3f}, {u1[0]:.3f}], y=[{l1[1]:.3f}, {u1[1]:.3f}]")
            rospy.logdebug(f"Flipped bounds: x=[{flipped_l1[0]:.3f}, {flipped_u1[0]:.3f}], y=[{flipped_l1[1]:.3f}, {flipped_u1[1]:.3f}]")
            rospy.logdebug(f"Obstacle bounds: x=[{l2[0]:.3f}, {u2[0]:.3f}], y=[{l2[1]:.3f}, {u2[1]:.3f}]")

        if dims_to_check is None:
            # Default to checking all dimensions common to both, up to zono1's dim
            # This assumes zono1 and zono2 are in the same ambient space, but collision relevant dims are checked
            num_dims = zono1.dim 
            dims_to_check = list(range(num_dims))
        
        if not dims_to_check: # No dimensions to check means no basis for collision
            return False

        # Use increased tolerance for floating point comparisons
        tolerance = 1e-5  # Increased from 1e-6
        
        overlap_count = 0
        for dim_idx in dims_to_check:
            if dim_idx >= len(flipped_l1) or dim_idx >= len(flipped_u1) or \
               dim_idx >= len(l2) or dim_idx >= len(u2):
                rospy.logwarn(f"Dimension index {dim_idx} out of bounds for interval check. Skipping this dim.")
                continue 
            
            # Check for overlap in this dimension using flipped bounds for reachable set
            # We have an overlap if:
            # NOT (flipped_u1[dim_idx] < l2[dim_idx] OR flipped_l1[dim_idx] > u2[dim_idx])
            # which is equivalent to:
            # flipped_u1[dim_idx] >= l2[dim_idx] AND flipped_l1[dim_idx] <= u2[dim_idx]
            if (flipped_u1[dim_idx] < l2[dim_idx] - tolerance) or \
               (flipped_l1[dim_idx] > u2[dim_idx] + tolerance):
                # No overlap in this dimension
                if dim_idx < 2:  # Only log for x,y dimensions to avoid spam
                    rospy.logdebug(f"No overlap in dimension {dim_idx}: flipped=[{flipped_l1[dim_idx]:.3f}, {flipped_u1[dim_idx]:.3f}], " +
                                  f"obstacle=[{l2[dim_idx]:.3f}, {u2[dim_idx]:.3f}]")
                return False
            else:
                # There is overlap in this dimension
                if dim_idx < 2:  # Only log for x,y dimensions
                    rospy.logdebug(f"Overlap in dimension {dim_idx}: flipped=[{flipped_l1[dim_idx]:.3f}, {flipped_u1[dim_idx]:.3f}], " +
                                  f"obstacle=[{l2[dim_idx]:.3f}, {u2[dim_idx]:.3f}]")
                overlap_count += 1
        
        # If we reach here, there is overlap in all specified dimensions
        if overlap_count > 0:
            rospy.loginfo(f"INTERSECTION DETECTED: Overlapping in {overlap_count} dimensions")
        return True

    def enforce_safety(
        self, 
        curr_obs_state: np.ndarray, 
        candidate_action: np.ndarray,
        obstacle_zonotopes: List[Zonotope]
    ) -> Tuple[bool, List[np.ndarray], List[Zonotope]]:
        
        n_physical_state = self.n_physical_state_from_model
        n_action = self.n_action_from_model

        if candidate_action.shape[0] != n_action:
            rospy.logerr(f"Candidate action dim ({candidate_action.shape[0]}) != model action dim ({n_action}).")
            return False, [], []
        if curr_obs_state.shape[0] < n_physical_state:
            rospy.logerr(f"Obs state dim ({curr_obs_state.shape[0]}) < phys state dim ({n_physical_state}).")
            return False, [], []
        
        # Use the EXACT current physical state passed in - no buffering or delays
        current_physical_state_np = curr_obs_state[:n_physical_state].copy()
        
        # Store the current robot position for intersection checking
        self.current_robot_pos = current_physical_state_np.copy()
        
        # Log the exact state being used for reachable set computation
        rospy.logdebug(f"Safety controller using state: pos=({current_physical_state_np[0]:.3f},{current_physical_state_np[1]:.3f}), yaw={current_physical_state_np[2]:.3f}")

        planning_horizon: int = int(self.config["safety"].get("planning_horizon", 10))
        
        action_constraints_config = self.config["safety"].get("action_constraints", {})
        env_type_for_actions = self.config.get("environment", {}).get("env_type", "turtlebot")
        action_lower_np = -1e6 * np.ones(n_action, dtype=np.float64)
        action_upper_np =  1e6 * np.ones(n_action, dtype=np.float64)
        
        specific_env_action_conf = action_constraints_config.get(env_type_for_actions.lower())
        if specific_env_action_conf:
            if env_type_for_actions.lower() == "turtlebot" and n_action == 2:
                lv_bounds_raw = specific_env_action_conf.get("longitudinal_velocity", [0.0, 0.25])
                av_bounds_raw = specific_env_action_conf.get("angular_velocity", [-0.5, 0.5])
                lv_bounds = lv_bounds_raw if isinstance(lv_bounds_raw, list) and len(lv_bounds_raw) == 2 else [0.0, 0.25]
                av_bounds = av_bounds_raw if isinstance(av_bounds_raw, list) and len(av_bounds_raw) == 2 else [-0.5, 0.5]
                # Ensure all values are numeric
                try:
                    action_lower_np = np.array([float(lv_bounds[0]), float(av_bounds[0])], dtype=np.float64)
                    action_upper_np = np.array([float(lv_bounds[1]), float(av_bounds[1])], dtype=np.float64)
                except (ValueError, TypeError) as e:
                    rospy.logwarn(f"Invalid action bounds in config: {e}. Using defaults.")
                    action_lower_np = np.array([0.0, -0.5], dtype=np.float64)
                    action_upper_np = np.array([0.25, 0.5], dtype=np.float64)
            elif env_type_for_actions.lower() == "quadrotor" and n_action == 3:
                vel_bounds_val_raw = specific_env_action_conf.get("velocity", [-5.0, 5.0])
                vel_bounds_val = vel_bounds_val_raw if isinstance(vel_bounds_val_raw, list) and len(vel_bounds_val_raw) == 2 else [-5.0, 5.0]
                try:
                    action_lower_np = np.full(n_action, float(vel_bounds_val[0]), dtype=np.float64)
                    action_upper_np = np.full(n_action, float(vel_bounds_val[1]), dtype=np.float64)
                except (ValueError, TypeError) as e:
                    rospy.logwarn(f"Invalid quadrotor velocity bounds in config: {e}. Using defaults.")
                    action_lower_np = np.full(n_action, -5.0, dtype=np.float64)
                    action_upper_np = np.full(n_action, 5.0, dtype=np.float64)
        
        u_interval_for_gen_calc = (action_lower_np, action_upper_np)

        safe_bounds_config: Dict[str, Any] = self.config["safety"].get("safe_state_bounds", {})
        default_s_lower = [-1e6] * n_physical_state
        default_s_upper = [ 1e6] * n_physical_state
        s_lower_raw = safe_bounds_config.get("lower", default_s_lower)
        s_upper_raw = safe_bounds_config.get("upper", default_s_upper)
        
        safe_lower_np_list = list(s_lower_raw[:n_physical_state]) if len(s_lower_raw) >= n_physical_state else list(s_lower_raw) + default_s_lower[len(s_lower_raw):n_physical_state]
        safe_upper_np_list = list(s_upper_raw[:n_physical_state]) if len(s_upper_raw) >= n_physical_state else list(s_upper_raw) + default_s_upper[len(s_upper_raw):n_physical_state]
        safe_lower_np = np.array(safe_lower_np_list, dtype=np.float64)
        safe_upper_np = np.array(safe_upper_np_list, dtype=np.float64)

        u_k_vars = [cp.Variable(n_action, name=f"u_{k_step}") for k_step in range(planning_horizon)]
        
        # Initialize R_0 = current state (point zonotope) using EXACT current state
        R_current_step_cvx = Zonotope(center=current_physical_state_np.copy(), 
                                      generators=np.empty((n_physical_state, 0), dtype=np.float64))
        
        # Log the initial reachable set center for debugging
        rospy.logdebug(f"Initial reachable set R_0 center: {R_current_step_cvx.center[:3].round(3)}")
        
        constraints = []
        objective = cp.Minimize(cp.sum_squares(u_k_vars[0] - candidate_action))

        is_debug_deterministic_model_case = hasattr(self.modelSet, 'is_deterministic_for_test') and self.modelSet.is_deterministic_for_test
        
        # Track center for logging - use the exact center we started with
        current_R_center_for_log = R_current_step_cvx.center.copy()

        # Main reachability computation loop - EXACTLY as in reference
        for k_step in range(planning_horizon):
            u_k_cvx_var = u_k_vars[k_step]
            
            # Create action zonotope Z_uk with CVXPY center - EXACTLY as in reference
            Z_uk_cvx = Zonotope(center=u_k_cvx_var, generators=np.empty((n_action, 0))) 
            
            # Cartesian product: X_k = R_k ⊗ Z_uk - EXACTLY as in reference
            X_k_cvx = R_current_step_cvx.cartesian_product(Z_uk_cvx) 
            
            # Matrix-vector multiplication: R_mult = modelSet * X_k - EXACTLY as in reference
            R_mult_cvx = self.modelSet.multiply_by_vector_zonotope_approx(
                X_k_cvx, u_interval_bounds=u_interval_for_gen_calc, u_dim=n_action)
            R_mult_cvx = R_mult_cvx.reduce_order_girard(self.max_generators_in_reach_set)

            # Add process noise: R_next = R_mult ⊕ Z_w ⊕ Z_v ⊖ Z_Av
            # This follows the EXACT sequence from the reference implementation
            R_next_step_intermediate_cvx = R_mult_cvx + self.Z_w
            R_next_step_intermediate_cvx = R_next_step_intermediate_cvx.reduce_order_girard(self.max_generators_in_reach_set)
            
            R_next_step_intermediate_cvx = R_next_step_intermediate_cvx + self.Z_v
            R_next_step_intermediate_cvx = R_next_step_intermediate_cvx.reduce_order_girard(self.max_generators_in_reach_set)
            
            R_next_step_cvx = R_next_step_intermediate_cvx - self.Z_Av
            R_next_step_cvx = R_next_step_cvx.reduce_order_girard(self.max_generators_in_reach_set)
            
            # Get interval bounds for constraints - EXACTLY as in reference
            lower_R_next, upper_R_next = R_next_step_cvx.get_interval_bounds()

            if is_debug_deterministic_model_case:
                rospy.logdebug(f"DEBUG_DET_MPC k_step={k_step}:")
                if isinstance(current_R_center_for_log, np.ndarray):
                     rospy.logdebug(f"  R_curr_center for this step (numeric): {current_R_center_for_log[:4].round(3)}")
                else:
                     rospy.logdebug(f"  R_curr_center for this step (CVXPY expr): {type(current_R_center_for_log)}")

                rospy.logdebug(f"  Constraint: lower_R_next >= {safe_lower_np[:min(4, n_physical_state)].round(3)}")
                rospy.logdebug(f"  Constraint: upper_R_next <= {safe_upper_np[:min(4, n_physical_state)].round(3)}")
                action_limits_str_low = str(action_lower_np[:min(2,n_action)].round(3)) if n_action > 0 else "[]"
                action_limits_str_high = str(action_upper_np[:min(2,n_action)].round(3)) if n_action > 0 else "[]"
                rospy.logdebug(f"  Action constraint for {u_k_vars[k_step].name()}: [{action_limits_str_low}, {action_limits_str_high}]")
                num_gens_r_next = R_next_step_cvx.generators.shape[1]
                if num_gens_r_next > 0:
                    rospy.logdebug(f"  R_next_step_cvx (after reduction) has {num_gens_r_next} generators.")
                else:
                    rospy.logdebug(f"  R_next_step_cvx (after reduction) has 0 generators.")

            # Add safety constraints - EXACTLY as in reference
            constraints.append(lower_R_next >= safe_lower_np)
            constraints.append(upper_R_next <= safe_upper_np)
            constraints.append(u_k_cvx_var >= action_lower_np)
            constraints.append(u_k_cvx_var <= action_upper_np)
            
            # Update for next iteration: R_{k+1} = R_next - EXACTLY as in reference
            R_current_step_cvx = R_next_step_cvx
            current_R_center_for_log = R_current_step_cvx.center

        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.OSQP, verbose=False, warm_start=True, 
                          eps_abs=1e-4, eps_rel=1e-4, polish=True)
            self.problem_status = problem.status
        except Exception as e:
            rospy.logerr("CVXPY solver exception: %s", e)
            self.problem_status = "solver_exception"
            return False, [], []

        # Always compute visualizable zonotopes using the EXACT current state
        visualizable_zonotopes: List[Zonotope] = []
        
        if self.problem_status not in ["optimal", "optimal_inaccurate"]:
            rospy.logwarn("MPC (box constraints) status: %s. Candidate: %s", self.problem_status, candidate_action.round(3))
            
            # Enhanced debugging for infeasible cases
            if self.problem_status == "infeasible" or self.problem_status == "infeasible_inaccurate":
                rospy.logwarn(f"MPC infeasible. Vars: u_k_vars[0].value = {u_k_vars[0].value}")
                rospy.logwarn(f"Current state: {current_physical_state_np[:4].round(3)}")
                rospy.logwarn(f"Action bounds: linear=[{action_lower_np[0]:.3f}, {action_upper_np[0]:.3f}], angular=[{action_lower_np[1]:.3f}, {action_upper_np[1]:.3f}]")
                rospy.logwarn(f"Safe state bounds: lower={safe_lower_np[:4].round(3)}, upper={safe_upper_np[:4].round(3)}")
                rospy.logwarn(f"Planning horizon: {planning_horizon} steps")
                
                # Check if state is already out of bounds
                state_violations = []
                for i in range(min(len(current_physical_state_np), len(safe_lower_np))):
                    if current_physical_state_np[i] < safe_lower_np[i]:
                        state_violations.append(f"state[{i}]={current_physical_state_np[i]:.3f} < lower_bound={safe_lower_np[i]:.3f}")
                    elif current_physical_state_np[i] > safe_upper_np[i]:
                        state_violations.append(f"state[{i}]={current_physical_state_np[i]:.3f} > upper_bound={safe_upper_np[i]:.3f}")
                
                if state_violations:
                    rospy.logwarn(f"Current state violates bounds: {'; '.join(state_violations)}")
                else:
                    rospy.logwarn("Current state is within bounds - infeasibility likely due to prediction constraints")
            
            rospy.loginfo("Computing nominal trajectory for visualization (MPC failed)")
            # For debugging: compute nominal trajectory starting from EXACT current state
            R_vis_current = Zonotope(center=current_physical_state_np.copy(), 
                                     generators=np.empty((n_physical_state, 0), dtype=np.float64))
            visualizable_zonotopes.append(R_vis_current.reduce_order_girard(self.max_generators_in_reach_set))
            
            # Log the visualization starting point
            rospy.logdebug(f"Visualization trajectory starting from: {R_vis_current.center[:3].round(3)}")
            
            # Propagate with candidate action repeated for visualization
            for k_step_vis in range(min(planning_horizon, 5)):  # Limit to 5 steps for performance
                u_k_nominal = candidate_action  # Use candidate action for nominal trajectory
                Z_uk_nominal = Zonotope(center=u_k_nominal, generators=np.empty((n_action, 0)))
                X_k_nominal = R_vis_current.cartesian_product(Z_uk_nominal)
                
                R_mult_nominal = self.modelSet.multiply_by_vector_zonotope_approx(
                    X_k_nominal, u_interval_bounds=u_interval_for_gen_calc, u_dim=n_action)
                R_mult_nominal = R_mult_nominal.reduce_order_girard(self.max_generators_in_reach_set)

                # EXACTLY match the reference noise addition sequence
                R_next_vis_intermediate = R_mult_nominal + self.Z_w
                R_next_vis_intermediate = R_next_vis_intermediate.reduce_order_girard(self.max_generators_in_reach_set)
                R_next_vis_intermediate = R_next_vis_intermediate + self.Z_v
                R_next_vis_intermediate = R_next_vis_intermediate.reduce_order_girard(self.max_generators_in_reach_set)
                R_next_vis = R_next_vis_intermediate - self.Z_Av
                R_next_vis = R_next_vis.reduce_order_girard(self.max_generators_in_reach_set)
                
                visualizable_zonotopes.append(R_next_vis)
                R_vis_current = R_next_vis
            
            return False, [], visualizable_zonotopes  # Return zonotopes even when MPC fails
        
        if u_k_vars[0].value is None:
             rospy.logerr("MPC solution (box) u_k_vars[0].value is None despite status: %s. Problem value: %s", 
                          self.problem_status, problem.value)
             # Still compute nominal trajectory for visualization
             R_vis_current = Zonotope(center=current_physical_state_np.copy(), 
                                      generators=np.empty((n_physical_state, 0), dtype=np.float64))
             visualizable_zonotopes.append(R_vis_current.reduce_order_girard(self.max_generators_in_reach_set))
             return False, [], visualizable_zonotopes

        safe_plan_actions: List[np.ndarray] = []
        for k_step in range(planning_horizon):
            if u_k_vars[k_step].value is None:
                if k_step == 0: 
                    rospy.logerr(f"MPC optimal but u_k_vars[0].value is None at k_step={k_step}. Aborting plan.")
                    return False, [], visualizable_zonotopes  # Return any zonotopes computed so far
                rospy.logwarn(f"MPC solution for u_k_vars[{k_step}].value is None. Truncating plan.")
                break 
            safe_plan_actions.append(np.array(u_k_vars[k_step].value).flatten())
        
        if not safe_plan_actions:
            rospy.logerr("No actions in safe_plan_actions after MPC solve, though status was %s.", self.problem_status)
            return False, [], visualizable_zonotopes

        # Always compute visualizable zonotopes using the solved actions, starting from EXACT current state
        R_vis_current = Zonotope(center=current_physical_state_np.copy(), 
                                 generators=np.empty((n_physical_state, 0), dtype=np.float64))
        visualizable_zonotopes.append(R_vis_current.reduce_order_girard(self.max_generators_in_reach_set))
        
        # Log the successful visualization starting point
        rospy.logdebug(f"Successful visualization trajectory starting from: {R_vis_current.center[:3].round(3)}")

        plan_is_truly_safe = True
        dims_for_collision_check = list(range(min(n_physical_state, 2))) 
        if not dims_for_collision_check:
             rospy.logdebug("No dimensions specified for collision checking; skipping post-validation obstacle check.")
        else:
            for k_step_vis in range(len(safe_plan_actions)): 
                u_k_solved_numeric = safe_plan_actions[k_step_vis] 
                Z_uk_numeric = Zonotope(center=u_k_solved_numeric, generators=np.empty((n_action, 0)))
                X_k_numeric = R_vis_current.cartesian_product(Z_uk_numeric)
                
                R_mult_numeric = self.modelSet.multiply_by_vector_zonotope_approx( X_k_numeric, 
                    u_interval_bounds=u_interval_for_gen_calc, u_dim=n_action)
                R_mult_numeric = R_mult_numeric.reduce_order_girard(self.max_generators_in_reach_set)

                # EXACTLY match the reference noise addition sequence for visualization
                R_next_vis_intermediate = R_mult_numeric + self.Z_w
                R_next_vis_intermediate = R_next_vis_intermediate.reduce_order_girard(self.max_generators_in_reach_set)
                R_next_vis_intermediate = R_next_vis_intermediate + self.Z_v
                R_next_vis_intermediate = R_next_vis_intermediate.reduce_order_girard(self.max_generators_in_reach_set)
                R_next_vis = R_next_vis_intermediate - self.Z_Av
                R_next_vis = R_next_vis.reduce_order_girard(self.max_generators_in_reach_set)
                
                visualizable_zonotopes.append(R_next_vis)

                for obs_idx, obs_zono in enumerate(obstacle_zonotopes):
                    if self._check_zonotope_intersection(R_next_vis, obs_zono, dims_to_check=dims_for_collision_check):
                        rospy.logwarn(f"POST-VALIDATION FAILED: Plan step {k_step_vis+1}, action {u_k_solved_numeric.round(3)}. "
                                       f"Predicted state R_k+{k_step_vis+1}|k intersects with obstacle zonotope {obs_idx}.")
                        r_lb, r_ub = R_next_vis.get_interval_bounds()
                        o_lb, o_ub = obs_zono.get_interval_bounds()
                        rospy.logdebug(f"  R_next_vis (dims {dims_for_collision_check}): LB={r_lb[dims_for_collision_check].round(3)}, UB={r_ub[dims_for_collision_check].round(3)}")
                        rospy.logdebug(f"  Obs_zono {obs_idx} (dims {dims_for_collision_check}): LB={o_lb[dims_for_collision_check].round(3)}, UB={o_ub[dims_for_collision_check].round(3)}")
                        plan_is_truly_safe = False
                        break 
                if not plan_is_truly_safe:
                    break
                R_vis_current = R_next_vis

        if not plan_is_truly_safe:
            rospy.logwarn("POST-VALIDATION failed, but returning zonotopes for debugging visualization")
            return False, [], visualizable_zonotopes 

        rospy.logdebug(f"MPC (box constraints) successful & POST-VALIDATION passed. Safe plan: {safe_plan_actions[0].round(3)}...")
        return True, safe_plan_actions, visualizable_zonotopes
