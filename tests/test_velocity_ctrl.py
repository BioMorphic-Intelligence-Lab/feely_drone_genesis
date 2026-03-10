"""
Test script for velocity controller with arm control.

Tracks a figure-8 velocity trajectory while demonstrating arm bending and contraction.
"""

import torch
import genesis as gs
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controller import Controller
from transforms import quat_to_rotation_matrix, euler_from_quaternion, quaternion_from_euler
from plotting import plot_triplet, plot_position_groundtrack
from sim_utils import (
    read_po, setup_scene, run_simulation,
    build_arm_matrices, compute_hinge_points_per_arm,
    arm_dof_indices_hinge_major,
)


def main():
    po = read_po()

    # Setup scene and drone
    scene, cam, drone = setup_scene(po, gravity=(0, 0, -9.81), cam_pos=(0.0, -7.50, 3.0), cam_lookat=(0.0, 0.0, 0.5))

    # Drone parameters
    mass = drone.get_mass()
    hinge_points_per_arm = compute_hinge_points_per_arm(drone)
    px, r1, r2, ap, ar2 = arm_dof_indices_hinge_major(hinge_points_per_arm)

    # Build arm control matrices from indices (hinge-major layout)
    K, A = build_arm_matrices(
        px, r1, r2,
        arm_for_prismatic=ap, arm_for_revolute2=ar2,
    )

    # Get base link index
    base_link_idx = torch.tensor([1])

    # Set initial position and orientation
    drone.set_pos(torch.tensor([[0.0, 0.0, 1.0]]))
    drone.set_quat(quaternion_from_euler(
        torch.tensor([0]), torch.tensor([0]), torch.tensor([0])
    ))

    n_steps = int(po.T / po.dt)

    # Initialize controllers
    ctrl = Controller(dt=po.dt)

    # Data logging
    if po.plot:
        time = torch.linspace(0, po.T, n_steps)
        position = torch.zeros([n_steps, 3])
        att = torch.zeros([n_steps, 3])
        velocity = torch.zeros([n_steps, 3])
        omega = torch.zeros([n_steps, 3])
        ref_pos = torch.zeros([n_steps, 3])
        ref_att = torch.zeros([n_steps, 3])
        ref_vel = torch.zeros([n_steps, 3])
        ref_omega = torch.zeros([n_steps, 3])

    def step_callback(step, t):
        x_dofs = drone.get_dofs_position()
        v_dofs = drone.get_dofs_velocity()

        base_p = x_dofs[:, :3]
        base_v = v_dofs[:, :3]
        base_quat = drone.get_quat()
        R = quat_to_rotation_matrix(base_quat)
        base_w_body = v_dofs[:, 3:6]

        # Figure-8 velocity trajectory
        traj_time = 5.0
        traj_amplitude_x = 1.0
        traj_amplitude_y = 1.0
        omega_x = torch.tensor([2 * torch.pi / traj_time])
        omega_y = torch.tensor([2 * torch.pi / (0.5 * traj_time)])

        vx_des = traj_amplitude_x * -2 * torch.pi / traj_time * torch.sin(omega_x * t)
        ax_des = traj_amplitude_x * -omega_x**2 * torch.cos(omega_x * t)

        vy_des = traj_amplitude_y * 2 * torch.pi / (0.5 * traj_time) * torch.cos(omega_y * t)
        ay_des = traj_amplitude_y * -omega_y**2 * torch.sin(omega_y * t)

        if t < 2.0:
            vz_des = 0.5 / 2.0  # = 0.25 m/s
            az_des = 0.0
        else:
            vz_des = 0.0
            az_des = 0.0

        vel_des = torch.tensor([[vx_des, vy_des, vz_des]])
        acc_des = torch.tensor([[ax_des, ay_des, az_des]])
        yaw_rate_des = torch.atan2(vy_des, vx_des) - torch.pi / 2

        # Compute drone control wrench
        body_torques, body_forces = ctrl.u_vel(
            v=base_v, yaw_rate_des=yaw_rate_des, v_des=vel_des,
            acc_des=acc_des, R=R, w_body=base_w_body,
            mass=mass, g=9.81
        )
        world_torques = (R @ body_torques.unsqueeze(-1)).squeeze(-1)
        world_forces = (R @ body_forces.unsqueeze(-1)).squeeze(-1)

        scene.sim.rigid_solver.apply_links_external_force(
            force=world_forces.unsqueeze(0), links_idx=base_link_idx
        )
        scene.sim.rigid_solver.apply_links_external_torque(
            torque=world_torques.unsqueeze(0), links_idx=base_link_idx
        )

        # Log data
        if po.plot:
            position[step, :] = base_p
            att[step, :] = euler_from_quaternion(base_quat)
            velocity[step, :] = base_v
            omega[step, :] = base_w_body
            ref_vel[step, :] = vel_des

    run_simulation(scene, cam, po, step_callback, record_filename='media/video.mp4')

    # Plotting
    if po.plot:
        plot_position_groundtrack(time, position, filename="media/position_groundtrack.png")

        plot_triplet(
            time, att, ref_att,
            labels=["roll", "pitch", "yaw"],
            title="Attitude", filename="media/attitude.png",
            ylabel_unit=" [rad]"
        )

        plot_triplet(
            time, velocity, ref_vel,
            labels=["vx", "vy", "vz"],
            title="Linear velocity", filename="media/linear_velocity.png",
            ylabel_unit=" [m/s]"
        )

        plot_triplet(
            time, omega, ref_omega,
            labels=["ωx", "ωy", "ωz"],
            title="Angular velocity", filename="media/angular_velocity.png",
            ylabel_unit=" [rad/s]"
        )


if __name__ == "__main__":
    main()
