"""
Test script for position controller with arm control.

Tracks a figure-8 position trajectory while demonstrating arm bending and contraction.
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
from plotting import plot_triplet
from sim_utils import (
    read_po, setup_scene, run_simulation
)


def main():
    po = read_po()

    # Setup scene and drone
    scene, cam, drone = setup_scene(po, gravity=(0, 0, -9.81), cam_pos=(0.0, -7.50, 3.0), cam_lookat=(0.0, 0.0, 0.5))

    # Drone parameters
    mass = drone.get_mass()

    # Get base link index
    base_link_idx = torch.tensor([1])

    # Set initial position and orientation
    drone.set_pos(torch.tensor([[1.0, 1.0, 1.0]]))
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

        if t < 2.0:
            x_des = 1.0
            y_des = 1.0
            z_des = 1.0
        elif t < 4.0:
            x_des = -1.0
            y_des = 1.0
            z_des = 2.0
        elif t < 6.0:
            x_des = -1.0
            y_des = -1.0
            z_des = 1.0
        elif t < 8.0:
            x_des = 1.0
            y_des = -1.0
            z_des = 2.0
        elif t < 10.0:
            x_des = 1.0
            y_des = 1.0
            z_des = 2.0
        else:
            x_des = 1.0
            y_des = 1.0
            z_des = 1.0

        pos_des = torch.tensor([[x_des, y_des, z_des]])
        yaw_des = torch.tensor([0.0])

        # Compute drone control wrench
        body_torques, body_forces = ctrl.go_to(
            loc=pos_des, yaw_des=yaw_des, v_mag=2.0, 
            p=base_p, v=base_v,
            R=R, w_body=base_w_body,
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
            ref_pos[step, :] = torch.tensor([x_des, y_des, z_des])
            ref_vel[step, :] = torch.zeros(3)

    run_simulation(scene, cam, po, step_callback, record_filename='media/video.mp4')

    # Plotting
    if po.plot:
        fig = plt.figure(figsize=(10, 6))
        grid = GridSpec(3, 2, figure=fig)

        # Ground track
        ax_xy = fig.add_subplot(grid[:, 0])
        ax_xy.set_xlim([-1.2, 1.2])
        ax_xy.set_ylim([-1.5, 1.5])
        ax_xy.set_aspect('equal')
        ax_xy.plot(position[:, 0], position[:, 1])
        ax_xy.plot(ref_pos[:, 0], ref_pos[:, 1], "--", color="black")
        ax_xy.set_xlabel("x [m]")
        ax_xy.set_ylabel("y [m]")
        ax_xy.set_title("Ground track")

        # Position vs time
        labels = ["x", "y", "z"]
        for i in range(3):
            ax = fig.add_subplot(grid[i, 1])
            ax.plot(time, position[:, i])
            ax.plot(time, ref_pos[:, i], "--", color="black")
            ax.set_ylabel(f"{labels[i]} [m]")
            if i < 2:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("time [s]")

        fig.tight_layout()
        fig.savefig("media/position_groundtrack.png")
        plt.close(fig)

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
