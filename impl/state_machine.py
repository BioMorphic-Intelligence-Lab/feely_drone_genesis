import numpy as np
from enum import Enum
from .pose_ctrl import PoseCtrl
from .gripper_ctrl import GripperCtrl
from .search_pattern import (LinearSearchPattern,
                             SinusoidalSearchPattern,
                             CompositeSearchPattern)

class State(Enum):
    UNDEFINED = 0
    SEARCHING = 1
    POSITION = 2
    ROTATION = 3
    FINALIZE = 4
    PERCH = 5
    ABORT = 6

class StateMachine(object):

    def __init__(self,
                 dt, m_arm, l_arm, K, A, g, q0, 
                 p0, rot0,
                 m_total,
                 searching_pattern=None,
                 target_pos_estimate=np.array([0, 0, 1.0]),
                 target_yaw_estimate=0,
                 tau_max=1500 * np.ones(3),
                 alpha_rate=1.0/5.0):

        if searching_pattern is None:
            self.searching_pattern = (CompositeSearchPattern([
                     LinearSearchPattern(slope=[0, 0, 0.1], offset=[0, 0, 0]),
                     SinusoidalSearchPattern(amplitude=[0.5, 0.25, 0],
                                             frequency=[1.0/10.0, 1.0/5.0, 0.0],
                                             phaseshift=[0.0, 0.0, 0.0],
                                             offset=target_pos_estimate)
                    ])
            )
        else:
            self.searching_pattern = searching_pattern

        self.dt = dt
        self.t = 0
        self.target_yaw_estimate = target_yaw_estimate
        self.target_pos_estimate = target_pos_estimate
        self.alpha_rate = alpha_rate
        self.alpha = np.ones(3)
        self.state = State.SEARCHING      
        
        self.tactile_info_sw = np.zeros([10, 3, 3], dtype=float)

        self.pose_ctrl = PoseCtrl(m_total=m_total, dt=dt, g=g)
        self.gripper_ctrl = GripperCtrl(tau_max=tau_max)

        self.p0 = p0
        self.rot0 = rot0
        self.K = K
        self.q0 = q0
        self.A = A
        self.g = g
        self.l = l_arm
        self.M_g = np.array([
            [(np.sum(m_arm)*0.5*l_arm[0]), np.sum(m_arm[1:])*0.5*l_arm[1], m_arm[2]*0.5*l_arm[2]],
            [                         0.0, np.sum(m_arm[1:])*0.5*l_arm[1], m_arm[2]*0.5*l_arm[2]],
            [                         0.0,                            0.0, m_arm[2]*0.5*l_arm[2]]
        ])

    def get_des_yaw_vel(self, contacts, rot_vel=1.0):
        rows = np.sum(np.array([1.0, 3.0, 2.0]) * contacts, axis=1)
        if rows[0] == rows[2]:
            return 0.0
        elif rows[0] > rows[2]:
            return -rot_vel
        elif rows[2] > rows[0]:
            return rot_vel
        else:
            return 0.0

    def searching_position_control(self, x, v, contact):

        p_des, v_des = self.searching_pattern.get_ref_pos_vel(self.t)
        yaw_des = self.target_yaw_estimate
        tau = self.gripper_ctrl.open_to(self.alpha)
        
        pos_ctrl = self.pose_ctrl.pos_ctrl(p_des, x[:3], v_des, v[:3])
        yaw_ctrl = self.pose_ctrl.yaw_ctrl(yaw_des, x[3], 0.0, v[3])

        self.t += self.dt

        return {'pos_ctrl': pos_ctrl,
                'yaw_ctrl': yaw_ctrl,
                'tau': tau,
                'p_des': p_des,
                'yaw_des': yaw_des}

    def position_align_control(self, x, v, contact):
        
        self.t += self.dt
        
        p_des = self.target_pos_estimate
        yaw_des = self.target_yaw_estimate

        pos_ctrl = self.pose_ctrl.pos_ctrl(p_des, x[:3], 0.0, v[:3])
        yaw_ctrl = self.pose_ctrl.yaw_ctrl(yaw_des, x[3], 0.0, v[3])

        tau = self.gripper_ctrl.open_to(self.alpha)
        
        return {'pos_ctrl': pos_ctrl,
                'yaw_ctrl': yaw_ctrl,
                'tau': tau,
                'p_des': p_des,
                'yaw_des': yaw_des}
    
    def abort_control(self, x, v, contact):
        
        self.t += self.dt
        p_des = self.target_pos_estimate
        yaw_des = self.target_yaw_estimate
        
        pos_ctrl = self.pose_ctrl.pos_ctrl(p_des, x[:3], 0.0, v[:3])
        yaw_ctrl = self.pose_ctrl.yaw_ctrl(yaw_des, x[3], 0.0, v[3])

        self.alpha = np.ones(3)
        tau = self.gripper_ctrl.open_to(self.alpha)
        
        return {'pos_ctrl': pos_ctrl,
                'yaw_ctrl': yaw_ctrl,
                'tau': tau,
                'p_des': p_des,
                'yaw_des': yaw_des}

    def position_fine_alignment(self, contact, del_p=0.01):
        contact_yp = contact[0, :] | contact[2, :]
        contact_ym = contact[1, :]

        if np.sum(1 * contact_yp) > np.sum(1 * contact_ym):
            return np.array([0, del_p, 0])
        elif np.sum(1 * contact_yp) < np.sum(1 * contact_ym):
            return np.array([0, -del_p, 0])
        else:
            return np.zeros(3)

    def rotation_align_control(self, x, v, contact):
        
        self.t += self.dt

        omega_des = self.get_des_yaw_vel(contact)
        self.target_yaw_estimate += self.dt * omega_des
        
        yaw_des = self.target_yaw_estimate

        cT = np.cos(x[3])
        sT = np.sin(x[4])

        rot = np.array([[cT, -sT, 0],
                        [sT,  cT, 0],
                        [ 0,   0, 1]])
        
        if contact.any() and np.linalg.norm(self.target_pos_estimate[:2] - x[:2]) < 0.01:
            delta_p = rot @ self.position_fine_alignment(contact)
            self.target_pos_estimate += delta_p
        
        p_des = self.target_pos_estimate

        dalpha = -self.alpha_rate * self.dt * np.ones(3)
        indeces = ~contact.any(axis=1)
        self.alpha[indeces] += dalpha[indeces]
        self.alpha = np.clip(self.alpha, a_min=0.0, a_max=1.0)
        tau = self.gripper_ctrl.open_to(self.alpha)

        pos_ctrl = self.pose_ctrl.pos_ctrl(p_des, x[:3], 0.0, v[:3])
        yaw_ctrl = self.pose_ctrl.yaw_ctrl(yaw_des, x[3], omega_des, v[3])

        return {'pos_ctrl': pos_ctrl,
                'yaw_ctrl': yaw_ctrl,
                'tau': tau,
                'p_des': p_des,
                'yaw_des': yaw_des}

    def finalize_grasp_control(self, x, v, contact):
            
        self.t += self.dt

        p_des = self.target_pos_estimate
        yaw_des = self.target_yaw_estimate

        self.alpha = np.ones(3) * np.min(self.alpha)
        tau = self.gripper_ctrl.open_to(self.alpha)
        
        pos_ctrl = self.pose_ctrl.pos_ctrl(p_des, x[:3], 0.0, v[:3])
        yaw_ctrl = self.pose_ctrl.yaw_ctrl(yaw_des, x[3], 0.0, v[3])

        return {'pos_ctrl': pos_ctrl,
                'yaw_ctrl': yaw_ctrl,
                'tau': tau,
                'p_des': p_des,
                'yaw_des': yaw_des}
    
    def perch_control(self, x, v, contact):
            
        self.t += self.dt

        p_des = self.target_pos_estimate
        omega_des = self.get_des_yaw_vel(contact, rot_vel=0.01)
        self.target_yaw_estimate += self.dt * omega_des
        yaw_des = self.target_yaw_estimate

        self.alpha = np.ones(3) * np.min(self.alpha)
        tau = self.gripper_ctrl.open_to(self.alpha)
        
        pos_ctrl = np.zeros(3)
        yaw_ctrl = np.zeros(1)

        return {'pos_ctrl': pos_ctrl,
                'yaw_ctrl': yaw_ctrl,
                'tau': tau,
                'p_des': p_des,
                'yaw_des': yaw_des}
    
    def compute_gravity_tau(self, q,):
        """ Compute gravity torques tau(q) """
        C = np.array([np.cos(q[0]), np.cos(q[0] + q[1]), np.cos(q[0] + q[1] + q[2])])
        return np.linalg.norm(self.g) * self.M_g @ C

    def compute_jacobian(self, q):
        """ Compute Jacobian J_g(q) """
        S = np.array([
            [np.sin(q[0]), np.sin(q[0] + q[1]), np.sin(q[0] + q[1] + q[2])],
            [0, np.sin(q[0] + q[1]), np.sin(q[0] + q[1] + q[2])],
            [0, 0, np.sin(q[0] + q[1] + q[2])]
        ])
        J_tau = -np.linalg.norm(self.g) * self.M_g @ S
        return J_tau + self.K

    def newton_solve(self, tau_act, tol=1e-6, max_iter=100):
        """ Solve g(q) = 0 using Newton's method """
        q = np.zeros(3)

        for i in range(max_iter):
            tau = self.compute_gravity_tau(q)
            g_q = tau + self.K @ (q - self.q0) - self.A * tau_act

            if np.linalg.norm(g_q) < tol:
                return q

            J_g = self.compute_jacobian(q)
            delta_q = np.linalg.solve(J_g, -g_q)
            q += delta_q

        return q
    
    def find_steady_state_config(self, tau):
        config = self.newton_solve(tau[0])
        return config


    def forward_kinematics(self, p, joint_angles):
        """
        Computes the forward kinematics for a system with 3 arms, each with 3 links.
        
        Args:
           p: np.array (4), xyz and yaw [rad] position of base
           joint_angles: np.array (3,3), joint angles [arm, link].

        Returns:
            positions: np.array (3,3,3), containing XYZ positions of link centers.
        """
        positions = np.zeros((3, 3, 3))

        for i in range(3):  # Iterate over arms
            
            cT, sT = np.cos(p[3]), np.sin(p[3])
            R_prev = np.array([
                [cT, -sT, 0],
                [sT,  cT, 0],
                [0,    0, 1]
            ])               # Start with base rotation
            p_prev = p[:3] + R_prev @ self.p0[i, :]   # Start with base position
            R_prev = R_prev @ self.rot0[i, :, :] # Add arm rotation
            
            for j in range(3):  # Iterate over links
                # Joint rotation (assuming rotation around Z-axis for simplicity)
                cT, sT = np.cos(joint_angles[i, j]), np.sin(joint_angles[i, j])
                R_joint = np.array([
                    [1,  0,   0],
                    [0, cT, -sT],
                    [0, sT,  cT]
                ])
                # Compute current rotation
                R_current = R_prev @ R_joint
                
                # Compute current position
                p_current = p_prev + R_prev @ np.array([0, 0, self.l[j]])
                
                # Store the position
                positions[i, j] = p_current #- R_prev @ np.array([self.l[j] / 2, 0, 0])
                
                # Update for next iteration
                R_prev = R_current
                p_prev = p_current

        return positions

    def get_contact_sensor_location(self, p):

        config = self.find_steady_state_config(self.gripper_ctrl.open_to(self.alpha))
        config = np.reshape([config] * 3, [3,3])
        locs = self.forward_kinematics(p, config)

        return locs

    def get_new_ref_pos(self, x, contact):     

        cntc_sens_loc = self.get_contact_sensor_location(x[:4])
        
        ref_pos = np.zeros(3, dtype=float)
        cntc_pts = 0.0
        for i in range(len(contact)):
            for j in range(len(contact[i])):
                if contact[i, j]:
                    cntc_pts += 1.0
                    ref_pos += cntc_sens_loc[i, j, :] 

        return ref_pos / cntc_pts   


    def update_tactile_info_sw(self, contact):
        self.tactile_info_sw = np.roll(self.tactile_info_sw, shift=-1, axis=0)
        self.tactile_info_sw[-1] = contact

        return np.mean(1.0 * self.tactile_info_sw, axis=0)

    def control(self, x, v, contact):

        if self.state == State.SEARCHING:
            ctrl = self.searching_position_control(x, v, contact)
            if contact.any():
                self.target_pos_estimate = self.get_new_ref_pos(x, contact) - np.array([0, 0, 0.1])
                self.state = State.POSITION
                print("STATE CHANGE: SEARCHING -> POSITION")
        elif self.state == State.POSITION:
            ctrl = self.position_align_control(x, v, contact)
            if np.linalg.norm(x[:3] - self.target_pos_estimate) < 0.1:
                self.state = State.ROTATION
                print("STATE CHANGE: POSITION -> ROTATION")
            # Check for ABORT condition
            if np.linalg.norm(x[:3] - self.target_pos_estimate) > 1.5:
                self.target_pos_estimate += np.array([0, 0, -0.25])
                self.state = State.ABORT
                print("STATE CHANGE: POSITION -> ABORT")
        elif self.state == State.ROTATION:
            ctrl = self.rotation_align_control(x, v, contact)
            mean = self.update_tactile_info_sw(contact)
            # If each arm has at leat one pad that has had
            # consistent contact ....
            if np.any(mean > 0.9, axis=1).all():
                self.state = State.FINALIZE
                print("STATE CHANGE: ROTATION -> FINALIZE")
            # Check for ABORT condition
            if np.linalg.norm(x[:3] - self.target_pos_estimate) > 0.5:
                self.target_pos_estimate += np.array([0, 0, -0.25])
                self.state = State.ABORT
                print("STATE CHANGE: ROTATION -> ABORT")
        elif self.state == State.FINALIZE:
            ctrl = self.finalize_grasp_control(x, v, contact)
            mean = self.update_tactile_info_sw(contact)
            # If on each arm any of the firsst pads
            #  have had consitent contact ...
            if (np.any(mean[:, :2] > 0.95, axis=1).all()
                # ... and all of the tendon tensions have equalized
                or np.allclose(self.alpha, max(self.alpha), atol=1)
                ):
                self.state = State.PERCH
                print("STATE CHANGE: FINALIZE -> PERCH")
            # Check for ABORT condition
            if np.linalg.norm(x[:3] - self.target_pos_estimate) > 0.5:
                self.target_pos_estimate += np.array([0, 0, -0.25])
                self.state = State.ABORT
                print("STATE CHANGE: FINALIZE -> ABORT")
        elif self.state == State.PERCH:
            ctrl = self.perch_control(x, v, contact)
            mean = self.update_tactile_info_sw(contact)
        elif self.state == State.ABORT:
            ctrl = self.abort_control(x, v, contact)
            if np.linalg.norm(x[:3] - self.target_pos_estimate) < 0.05:
                self.state = State.SEARCHING
                print("STATE CHANGE: ABORT -> SEARCHING")
        else:
            ctrl = self.position_align_control(x, v, contact)
        
        return ctrl
