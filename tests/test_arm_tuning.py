"""
Arm actuation and stiffness tuning script.

Spawns the gripper drone sitting on the ground and steps through three
actuation levels (alpha = 0.0, 0.5, 1.0) at distinct times. Debug spheres
are drawn at the current arm link locations to visualize the motion.
"""

from __future__ import annotations

import numpy as np
import genesis as gs

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transforms import rotation_matrix_from_euler, quat_to_rotation_matrix
from feely_drone_common import GripperCtrl
from feely_drone_common.steady_state_calculator import get_contact_sensor_location
from sim_utils import read_po, setup_scene, run_simulation


def main():
    po = read_po(description="Arm actuation and stiffness tuning.")

    # Build scene with gravity, drone sitting near the ground, no extra environment
    scene, cam, gripper = setup_scene(
        po=po,
        gravity=(0, 0, -9.81),
        cam_pos=(1.0, -2.0, 1.0),
        cam_lookat=(0.0, 0.0, 0.0),
        viewer_pos=(1.0, -5.0, 2.0),
        viewer_lookat=(0.0, 0.0, 0.5),
    )

    # Place the drone so that its base is resting on the ground plane
    # (COM slightly above z=0 due to the base geometry).
    p0 = np.zeros((po.n_envs, 6 + 9), dtype=float)
    p0[:, 2] = 0.0825  # z position
    gripper.set_dofs_position(p0)
    gripper.set_dofs_velocity(np.zeros_like(gripper.get_dofs_velocity()))
    gripper.control_dofs_force(np.zeros_like(gripper.get_dofs_force()))

    # Extract Link Mass and Length from object
    # needed for computation of steady-state config
    m_arm = np.zeros(3)
    l_arm = np.zeros(3)
    for i in range(3):
        link = gripper.get_link(name=f"arm1_link{i+1}")
        m_arm[i] = link.get_mass()
        extents = link.geoms[0].get_trimesh().extents
        l_arm[i] = extents[2]

    # Extract offset positions and orientations of the arms
    # needed for Forward Kinematics
    p0_arms = np.zeros([3, 3])
    rot0_arms = np.zeros([3, 3, 3])
    for i in range(3):
        joint = gripper.get_joint(name=f"arm{i+1}_joint1")
        
        q = joint.get_quat()[0, :].reshape(1, 4)
        rot0_arms[i, :, :] = quat_to_rotation_matrix(q)

        p0_arms[i, :] = (
            np.array(joint.get_pos()[0, :]) 
            - rot0_arms [i, :, :] @ np.array([0, 0, 0.025]) 
            - p0[:, :3]
        )
      
    # Arm stiffness, tendon mapping, and gravity parameters
    K = np.diag(1.0 * np.ones(9))
    r = np.concatenate([np.diag(0.1 * np.ones(3)) for _ in range(3)], axis=1).T
    q0 = np.deg2rad(75) * np.ones(9)
    M_g = np.array([
            [(np.sum(m_arm)*0.5*l_arm[0]), np.sum(m_arm[1:])*0.5*l_arm[1], m_arm[2]*0.5*l_arm[2]],
            [                         0.0, np.sum(m_arm[1:])*0.5*l_arm[1], m_arm[2]*0.5*l_arm[2]],
            [                         0.0,                            0.0, m_arm[2]*0.5*l_arm[2]]
        ])
    A = -1.2 * np.ones(3)

    # One gripper controller per environment
    gripper_ctrl = np.array(
        [GripperCtrl(tau_max=12.50) for _ in range(po.n_envs)]
    )

    # Actuation schedule: three alpha levels, linear interpolation between times
    alpha_levels = [
        (0.0, 0.0),   # (time, level)
        (4.0, 0.5),
        (8.0, 1.0),
    ]

    def current_alpha(t: float) -> float:
        # Use discrete steps based on scheduled switch times
        for i in reversed(range(len(alpha_levels))):
            t_switch, a_val = alpha_levels[i]
            if t >= t_switch:
                return a_val
        # If t is before all steps, return first level
        return alpha_levels[0][1]

    def step_callback(step: int, t: float):
        p_full = np.array(gripper.get_dofs_position())
        
        actions = np.zeros_like(p_full)

        # Debug arrays for current and steady-state link locations
        ss_link_positions = np.zeros((po.n_envs * 9, 3))

        alpha_val = current_alpha(t)
        alpha_vec = np.array([alpha_val, alpha_val, alpha_val])

        for n in range(po.n_envs):
            
            # Compute stiffness contribution based on joint deviations
            q = p_full[n, 6:]
            stiffness_contrib = K @ (q0 - q)

            # Tendon actuation torques from desired alpha level
            
            tau_ctrl = gripper_ctrl[n].open_to(alpha_vec)

            # Assemble DOF forces: [base_forces(3), base_torques(3), joint_torques(9)]
            joint_torques = stiffness_contrib - r @ tau_ctrl
            action = np.concatenate(
                [
                    np.zeros(3),  # no external base force
                    np.zeros(3),  # no external base torque
                    joint_torques,
                ]
            )
            actions[n, :] = action


            # Steady-state link locations from analytical model
            # Use base pose (first 6 DOFs) and current alpha vector
            p_base = p_full[n, :6]
            locs_ss = get_contact_sensor_location(
                p_base,
                alpha=alpha_vec,
                M_g=M_g,
                K=K[:3, :3],
                A=A,
                p0=p0_arms,
                rot0=rot0_arms,
                l=l_arm,
            )
            for i in range(3):
                for j in range(3):
                    ss_link_positions[n * 9 + i * 3 + j, :] = locs_ss[i, j, :]

        gripper.control_dofs_force(actions)

        # Draw debug spheres: steady-state model (green)
        scene.clear_debug_objects()
        color_alpha = 1.0
        colors_ss = [(0.0, 1.0, 0.0, color_alpha) for _ in range(ss_link_positions.shape[0])]
        scene.draw_debug_spheres(ss_link_positions, radius=0.01, color=colors_ss)

    run_simulation(scene, cam, po, step_callback, record_filename="media/arm_tuning.mp4")


if __name__ == "__main__":
    main()

