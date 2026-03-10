"""
Controllers for drone position, velocity, attitude, and arm control.
"""
import torch

from transforms import rotation_error

class Controller(object):
    def __init__(self, dt: float = 0.01, device='cpu') -> None:
        self.device = device
        self.e_vel_int = torch.zeros(3, device=device)
        self.e_pos_int = torch.zeros(3, device=device)
        self.vel_prev = torch.zeros(3, device=device)  # For derivative term
        self.dt = dt

    def position_ctrl(self,
                      p: torch.Tensor,
                      v: torch.Tensor,
                      p_des: torch.Tensor,
                      v_des: torch.Tensor,
                      acc_des=None,
                      nat_freq=None,
                      k_i=None,
                      damping=None,
                      acc_max=None
        ) -> torch.Tensor:
        """PD position controller: outputs desired acceleration in world frame.
        
        Args:
            p: Position tensor of shape (N, 3) containing [x, y, z]
            v: Velocity tensor of shape (N, 3) containing [x, y, z]
            p_des: Desired position (3,) or (N, 3)
            v_des: Desired velocity (3,) or (N, 3)
            acc_des: Desired acceleration (3,) or (N, 3), optional
            nat_freq: Natural frequency for PD control
            damping: Damping ratio for PD control
        
        Returns:
            acc: Desired acceleration in world frame (N, 3)
        """
        assert p.ndim == 2 and p.shape[-1] == 3, f"Expected position shape (N, 3), got {p.shape}"
        assert v.ndim == 2 and v.shape[-1] == 3, f"Expected velocity shape (N, 3), got {v.shape}"
        

        # INSERT_YOUR_CODE
        # Convert all inputs to tensors if they aren't already
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, device=self.device, dtype=torch.float32)
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v, device=self.device, dtype=torch.float32)
        if not isinstance(p_des, torch.Tensor):
            p_des = torch.tensor(p_des, device=self.device, dtype=torch.float32)
        if not isinstance(v_des, torch.Tensor):
            v_des = torch.tensor(v_des, device=self.device, dtype=torch.float32)
        if acc_des is not None and not isinstance(acc_des, torch.Tensor):
            acc_des = torch.tensor(acc_des, device=self.device, dtype=torch.float32)
        if nat_freq is not None and not isinstance(nat_freq, torch.Tensor):
            nat_freq = torch.tensor(nat_freq, device=self.device, dtype=torch.float32)
        if damping is not None and not isinstance(damping, torch.Tensor):
            damping = torch.tensor(damping, device=self.device, dtype=torch.float32)
        if nat_freq is None:
            nat_freq = torch.tensor([4.0, 5.0, 10.0], device=self.device)
        if damping is None:
            damping = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        if k_i is None:
            k_i = torch.tensor([5.0, 5.0, 15.0], device=self.device)

        if acc_des is None:
            acc_des = torch.zeros(3, device=self.device)

        # Ensure desired values are broadcast correctly
        if p_des.ndim == 1:
            p_des = p_des.unsqueeze(0).expand(p.shape[0], -1)
        if v_des.ndim == 1:
            v_des = v_des.unsqueeze(0).expand(p.shape[0], -1)
        if acc_des.ndim == 1:
            acc_des = acc_des.unsqueeze(0).expand(p.shape[0], -1)

        p_err = p_des - p
        v_err = v_des - v
        self.e_pos_int += p_err.squeeze(0) * self.dt
        acc = (
            p_err * nat_freq ** 2 
            + v_err * 2 * nat_freq * damping 
            + k_i * self.e_pos_int 
            + acc_des
        )

        acc = torch.clamp(acc, -acc_max, acc_max)
        return acc

    def velocity_ctrl(self,
                      v: torch.Tensor,
                      v_des: torch.Tensor,
                      acc_des: torch.Tensor = None,
                      k_p: torch.Tensor = torch.tensor([10.0, 10.0, 30.0]),
                      k_i: torch.Tensor = torch.tensor([5.0, 5.0, 5.0]),
                      k_d: torch.Tensor = torch.tensor([0.001, 0.001, 0.001])) -> torch.Tensor:
        """PID velocity controller: outputs desired acceleration in world frame.
        
        The derivative term is computed from the rate of change of velocity error,
        which provides damping without requiring acceleration measurements.
        
        Args:
            v: Velocity tensor of shape (N, 3) containing [x, y, z]
            v_des: Desired velocity (3,) or (N, 3)
            acc_des: Desired acceleration (3,) or (N, 3), optional feedforward
            k_p: Proportional gains (3,)
            k_i: Integral gains (3,)
            k_d: Derivative gains (3,) - provides damping
        Returns:
            acc: Desired acceleration in world frame (N, 3)
        """
        assert v.ndim == 2 and v.shape[-1] == 3, f"Expected velocity shape (N, 3), got {v.shape}"
        
        if acc_des is None:
            acc_des = torch.zeros(3, device=self.device)

        # Ensure desired values are broadcast correctly
        if v_des.ndim == 1:
            v_des = v_des.unsqueeze(0).expand(v.shape[0], -1)
        if acc_des.ndim == 1:
            acc_des = acc_des.unsqueeze(0).expand(v.shape[0], -1)

        v_err = v_des - v
        v_err_squeezed = v_err.squeeze(0)
        
        # Integral term
        self.e_vel_int += v_err_squeezed * self.dt
        
        # Derivative term: d(v_err)/dt ≈ (v_err - v_err_prev) / dt
        # This gives damping without needing acceleration measurements
        v_deriv = (v - self.vel_prev) / self.dt
        self.v_prev = v.clone()
        
        acc = k_p * v_err + k_i * self.e_vel_int - k_d * v_deriv + acc_des

        return acc

    def get_attitude_and_thrust(self, acc: torch.Tensor,
                                yaw_des: float,
                                R_current: torch.Tensor,
                                mass=None,
                                g=9.81) -> tuple:
        """Compute desired attitude (as rotation matrix) and thrust from acceleration.
        
        Args:
            acc: Desired acceleration in world frame (N, 3)
            yaw_des: Desired yaw angle (scalar)
            R_current: Current rotation matrix (N, 3, 3)
            mass: Vehicle mass (scalar)
            g: Gravitational acceleration (scalar)
        
        Returns:
            R_des: Desired rotation matrix (N, 3, 3)
            total_thrust: Total thrust magnitude (N,)
        """
        assert acc.ndim == 2 and acc.shape[-1] == 3, f"Expected acceleration shape (N, 3), got {acc.shape}"
        assert R_current.ndim == 3 and R_current.shape[-2:] == (3, 3), \
            f"Expected rotation matrix shape (N, 3, 3), got {R_current.shape}"
        
        if mass is None:
            raise ValueError("[ERROR]: No mass given for controller!")
        
        N = acc.shape[0]
        device = acc.device
        
        # Add gravity compensation
        gravity_comp = torch.tensor([0, 0, g], device=device)
        acc = acc + gravity_comp.unsqueeze(0).expand(N, -1)
        force = mass * acc

        # Saturate negative z axis acceleration
        force = torch.where(force[:, 2:3] < 0, 
                         torch.cat([force[:, :2], torch.zeros_like(force[:, 2:3])], dim=1),
                         force)

        # Compute thrust magnitude and direction
        norm_force = torch.linalg.norm(force, dim=-1, keepdim=True)  # (N, 1)
        
        # Default thrust direction is up
        thrust_dir = torch.where(
            norm_force < 1e-6,
            torch.tensor([[0, 0, 1]], device=device, dtype=torch.float).expand(N, -1),
            force / norm_force
        )  # (N, 3)

        # Thrust correction based on current attitude (tilt compensation).
        # If thrust is applied along the current body +Z axis, the vertical component is scaled by
        # cos(tilt) = e3_world^T (R_current e3_body). To achieve a desired force magnitude in world,
        # we need to divide by cos(tilt). This should INCREASE thrust when tilted (cos < 1).
        #
        # Clamp only to avoid singularity when nearly horizontal or inverted, but do not clamp to 1.0
        # (which would *disable* tilt compensation and can cause under-thrust during aggressive attitude).
        MIN_COS_TILT = 0.1
        z_body_in_world = R_current @ torch.tensor([0.0, 0.0, 1.0], device=device, dtype=acc.dtype)  # (N, 3)
        cos_tilt = z_body_in_world[:, 2]  # (N,)
        cos_tilt = torch.clamp(cos_tilt, min=MIN_COS_TILT)
        total_thrust = force[:, 2] / cos_tilt  # (N,)

        # Desired body z-axis in world frame
        b3 = thrust_dir  # (N, 3), normalized

        # Desired heading vector in world frame (horizontal)
        yaw_t = torch.as_tensor(yaw_des, device=device, dtype=acc.dtype)
        c_yaw = torch.stack([
            torch.cos(yaw_t).expand(N),
            torch.sin(yaw_t).expand(N),
            torch.zeros(N, device=device)
        ], dim=-1)  # (N, 3)

        # Body x-axis: project c_yaw onto plane orthogonal to b3, then normalize
        b1 = c_yaw - (c_yaw * b3).sum(dim=-1, keepdim=True) * b3  # (N, 3)
        b1_norm = torch.linalg.norm(b1, dim=-1, keepdim=True)
        # Fallback if b3 is nearly vertical and c_yaw is degenerate
        b1 = torch.where(
            b1_norm < 1e-6,
            torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(N, -1),
            b1 / b1_norm
        )

        # Body y-axis: complete the right-handed frame, then re-orthonormalize for numerical robustness.
        b2 = torch.linalg.cross(b3, b1, dim=-1)  # (N, 3)
        b2 = b2 / (torch.linalg.norm(b2, dim=-1, keepdim=True) + 1e-9)
        b1 = torch.linalg.cross(b2, b3, dim=-1)

        # Assemble R_des: columns are body axes expressed in world
        R_des = torch.stack([b1, b2, b3], dim=-1)  # (N, 3, 3)

        return R_des, total_thrust

    def attitude_ctrl(self, R: torch.Tensor,
                      omega: torch.Tensor,
                      R_des: torch.Tensor) -> torch.Tensor:
        """PD attitude controller on SO(3): outputs desired angular velocity.
        
        Args:
            R: Current rotation matrix (N, 3, 3)
            omega: Current angular velocity in body frame (N, 3)
            R_des: Desired rotation matrix (N, 3, 3)
        
        Returns:
            des_omega: Desired angular velocity (N, 3)
        """
        assert R.ndim == 3 and R.shape[-2:] == (3, 3), \
            f"Expected rotation matrix shape (N, 3, 3), got {R.shape}"
        assert omega.ndim == 2 and omega.shape[-1] == 3, \
            f"Expected angular velocity shape (N, 3), got {omega.shape}"
        assert R_des.ndim == 3 and R_des.shape[-2:] == (3, 3), \
            f"Expected desired rotation matrix shape (N, 3, 3), got {R_des.shape}"
        
        # Gains
        Kp = torch.diag(torch.tensor([10.0, 10.0, 6.0], device=self.device))
        Kd = torch.tensor([0.1, 0.1, 0.6], device=self.device)
        
        # Compute rotation error on SO(3)
        att_err = rotation_error(R_des=R_des, R=R)  # (N, 3)

        # Apply gains per-axis to each batch element:
        # (N,3) @ (3,3)^T -> (N,3)
        des_omega = (att_err @ Kp.T) - (Kd * omega)

        return des_omega

    def angular_vel_ctrl(self, ang_vel: torch.Tensor, ang_vel_des: torch.Tensor,
                         time_constant=None, i=None, feed_forward=None, 
                         use_gyro_compensation=True) -> torch.Tensor:
        """Body torque controller using angular velocity errors.
        
        Args:
            ang_vel: Current angular velocity (N, 3)
            ang_vel_des: Desired angular velocity (N, 3)
            time_constant: Response time constant for each axis (3,)
            i: Inertia diagonal (3,)
            feed_forward: Feed-forward acceleration (3,)
            use_gyro_compensation: Whether to add gyroscopic compensation
        
        Returns:
            total_torque: Control torques (N, 3)
        """
        assert ang_vel.ndim == 2 and ang_vel.shape[-1] == 3, \
            f"Expected angular velocity shape (N, 3), got {ang_vel.shape}"
        assert ang_vel_des.ndim == 2 and ang_vel_des.shape[-1] == 3, \
            f"Expected desired angular velocity shape (N, 3), got {ang_vel_des.shape}"
        
        if time_constant is None:
            time_constant = self.dt * 100 * torch.tensor([0.04, 0.04, 0.14], device=self.device)
        if i is None:
            # Use measured inertia from URDF (drone base without arms)
            i = torch.tensor([0.0221, 0.0221, 0.0408], device=self.device)
        if feed_forward is None:
            feed_forward = torch.zeros(3, device=self.device)

        # Angular velocity error
        ang_vel_error = ang_vel_des - ang_vel
        
        # Desired angular acceleration (simple P control on velocity)
        des_accel = ang_vel_error / time_constant
        
        # Inertia matrix (diagonal for symmetric body)
        inertia = torch.diag(i)
        
        # Control torque for desired acceleration
        control_torque = (des_accel @ inertia.T)
        
        if use_gyro_compensation:
            # Gyroscopic coupling compensation: ω × (I*ω)
            I_omega = (ang_vel @ inertia.T)
            gyroscopic_term = torch.linalg.cross(ang_vel, I_omega)
            
            # Add compensation
            total_torque = control_torque + gyroscopic_term
        else:
            total_torque = control_torque
        
        # Add feed-forward term
        if feed_forward is not None and not torch.all(feed_forward == 0):
            feed_forward_batch = feed_forward.unsqueeze(0).expand(ang_vel.shape[0], -1)
            total_torque = total_torque + (feed_forward_batch @ inertia.T)
        
        # Torque saturation for safety
        max_torque = 1000.0  # Nm
        total_torque = torch.clamp(total_torque, -max_torque, max_torque)
        
        return total_torque

    def reset(self):
        self.e_vel_int.zero_()
        self.e_pos_int.zero_()

    def go_to(
        self,
        loc: torch.Tensor,
        yaw_des: float,
        v_mag: torch.Tensor,
        p: torch.Tensor,
        v: torch.Tensor,
        R: torch.Tensor,
        w_body: torch.Tensor,
        mass: float,
        v_des: torch.Tensor = None,
        epsilon: float = 0.4,
        g: float = 9.81,
        acc_max: float = torch.inf):
        """ Go to a desired position and yaw and specified speed"""

        # Ensure all inputs are torch tensors
        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc, device=self.device, dtype=torch.float32)
        if not isinstance(yaw_des, torch.Tensor):
            yaw_des = torch.tensor(yaw_des, device=self.device, dtype=torch.float32)
        if not isinstance(v_mag, torch.Tensor):
            v_mag = torch.tensor(v_mag, device=self.device, dtype=torch.float32)
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, device=self.device, dtype=torch.float32)
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v, device=self.device, dtype=torch.float32)
        if not isinstance(R, torch.Tensor):
            R = torch.tensor(R, device=self.device, dtype=torch.float32)
        if not isinstance(w_body, torch.Tensor):
            w_body = torch.tensor(w_body, device=self.device, dtype=torch.float32)
        if not isinstance(v_des, torch.Tensor):
            v_des = torch.tensor(v_des, device=self.device, dtype=torch.float32)

        err = loc - p
        if torch.linalg.norm(err) < epsilon:
            if v_des is None:
                v_des = torch.zeros(3, device=self.device)
            return self.u_pos(
                p=p,
                v=v,
                p_des=loc,
                yaw_des=yaw_des,
                v_des=v_des,
                acc_des=torch.zeros(3, device=self.device),
                R=R,
                w_body=w_body,
                mass=mass, g=g, acc_max=acc_max
            )
        else:
            p_des = p + err /torch.linalg.norm(err) * v_mag * self.dt
            if v_des is None:
                v_des = err / torch.linalg.norm(err) * v_mag
            else:
                if torch.linalg.norm(v_des) > v_mag:
                    v_des = v_des / torch.linalg.norm(v_des) * v_mag
            return self.u_pos(
                p=p,
                v=v,
                p_des=p_des,
                yaw_des=yaw_des,
                v_des=v_des,
                acc_des=torch.zeros(3, device=self.device),
                R=R,
                w_body=w_body,
                mass=mass, g=g, acc_max=acc_max
            )

    def u_pos(self,
        p: torch.Tensor, v: torch.Tensor,
        p_des: torch.Tensor, yaw_des: float, v_des: torch.Tensor,
        acc_des: torch.Tensor,
        R: torch.Tensor, w_body: torch.Tensor,
        mass: float, g: float = 9.81, acc_max: float = torch.inf):
        """ Compute control using cascaded controller """
        
        # Convert all inputs to tensors if they aren't already
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, device=self.device, dtype=torch.float32)
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v, device=self.device, dtype=torch.float32)
        if not isinstance(p_des, torch.Tensor):
            p_des = torch.tensor(p_des, device=self.device, dtype=torch.float32)
        if not isinstance(v_des, torch.Tensor):
            v_des = torch.tensor(v_des, device=self.device, dtype=torch.float32)
        if acc_des is not None and not isinstance(acc_des, torch.Tensor):
            acc_des = torch.tensor(acc_des, device=self.device, dtype=torch.float32)
        if not isinstance(R, torch.Tensor):
            R = torch.tensor(R, device=self.device, dtype=torch.float32)
        if not isinstance(w_body, torch.Tensor):
            w_body = torch.tensor(w_body, device=self.device, dtype=torch.float32)
        if not isinstance(mass, torch.Tensor):
            mass = torch.tensor(mass, device=self.device, dtype=torch.float32)
        if not isinstance(g, torch.Tensor):
            g = torch.tensor(g, device=self.device, dtype=torch.float32)

        # Desired acceleration
        acc = self.position_ctrl(p, v, p_des, v_des, acc_des=acc_des, acc_max=acc_max)
        
        # Desired attitude and total thrust
        R_des, tot_thrust = self.get_attitude_and_thrust(
            acc, yaw_des=yaw_des, R_current=R, mass=mass, g=g
        )

        # Compute rates 
        rates_des = self.attitude_ctrl(
             R=R, omega=w_body, R_des=R_des
        )

        # Compute torques
        body_torques = self.angular_vel_ctrl(
            w_body, rates_des
        )
        body_forces = tot_thrust * torch.tensor([0, 0, 1.0], device=R.device)
        
        return body_torques, body_forces

    def u_vel(self,
        v: torch.Tensor,
        v_des: torch.Tensor,
        yaw_rate_des: float,
        acc_des: torch.Tensor,
        R: torch.Tensor, w_body: torch.Tensor,
        mass: float, g: float = 9.81):
        
        """ Compute control using cascaded controller """
        # Desired acceleration
        acc = self.velocity_ctrl(v, v_des, acc_des=acc_des)


        # Desired attitude and total thrust
        R_des, tot_thrust = self.get_attitude_and_thrust(
            acc, yaw_des=0.0, R_current=R, mass=mass, g=g
        )

        # Compute rates 
        rates_des = self.attitude_ctrl(
             R=R, omega=w_body, R_des=R_des
        )

        # Add desired yaw rate to rates_des
        rates_des[:, 2] += yaw_rate_des

        # Compute torques
        body_torques = self.angular_vel_ctrl(
            w_body, rates_des
        )
        body_forces = tot_thrust * torch.tensor([0, 0, 1.0], device=R.device)
        
        return body_torques, body_forces