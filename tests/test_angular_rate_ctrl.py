"""
Test script for angular rate controller.

Tests angular velocity tracking with step responses on each axis.
"""

import torch
import genesis as gs
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from controller import Controller
from transforms import quat_to_rotation_matrix, euler_from_quaternion
from plotting import plot_angular_velocity_tracking, plot_control_torques
from sim_utils import read_po, setup_scene, run_simulation


def main():
    po = read_po(description="Test angular rate controller.")

    # Setup scene and drone (no gravity for pure attitude test)
    scene, cam, drone = setup_scene(po, gravity=(0, 0, 0), cam_pos=(0.0, -5.0, 1.0), cam_lookat=(0.0, 0.0, 1.0))

    # Set initial position
    drone.set_pos(torch.tensor([[0.0, 0.0, 1.0]]))

    n_steps = int(po.T / po.dt)

    # Initialize controller
    ctrl = Controller()

    # Data logging
    if po.plot:
        time = torch.linspace(0, po.T, n_steps)
        omega = torch.zeros([n_steps, 3])
        omega_des = torch.zeros([n_steps, 3])
        torques_applied = torch.zeros([n_steps, 3])
        attitude = torch.zeros([n_steps, 3])

    def step_callback(step, t):
        # Get drone state
        base_link = drone.get_link("drone")
        base_quat = base_link.get_quat()
        R = quat_to_rotation_matrix(base_quat)
        base_w = base_link.get_ang()

        # Define desired angular velocity trajectory (step responses)
        omega_x_des = 0.2 if (0.5 < t < 1.5) else 0.0
        omega_y_des = -0.5 if (2.5 < t < 3.5) else 0.0
        omega_z_des = 1.0 if (4.5 < t < 5.5) else 0.0

        ang_vel_des = torch.tensor([[omega_x_des, omega_y_des, omega_z_des]])

        # Compute control torques using angular velocity controller
        body_torques = ctrl.angular_vel_ctrl(base_w, ang_vel_des)
        world_torques = (R @ body_torques.unsqueeze(-1)).squeeze(-1)

        # Apply torques to drone
        scene.sim.rigid_solver.apply_links_external_torque(
            torque=world_torques.unsqueeze(0),
            links_idx=torch.tensor([base_link.idx])
        )

        # Log data
        if po.plot:
            omega[step, :] = base_w
            omega_des[step, :] = ang_vel_des
            torques_applied[step, :] = body_torques
            attitude[step, :] = euler_from_quaternion(base_quat)

    run_simulation(scene, cam, po, step_callback, record_filename='media/angular_rate_test.mp4')

    # Plotting
    if po.plot:
        plot_angular_velocity_tracking(time, omega, omega_des, filename='media/angular_velocity_tracking.png')
        plot_control_torques(time, torques_applied, filename='media/control_torques.png')

        # Plot attitude evolution
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        labels = ['Roll', 'Pitch', 'Yaw']

        for i, ax in enumerate(axes):
            ax.plot(time, attitude[:, i], linewidth=2)
            ax.set_ylabel(f'{labels[i]} [rad]', fontsize=12)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time [s]', fontsize=12)
        fig.suptitle('Attitude Evolution', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig('media/attitude_evolution.png', dpi=150)
        plt.close(fig)

        print("\nPlots saved:")
        print("  - media/angular_velocity_tracking.png")
        print("  - media/control_torques.png")
        print("  - media/attitude_evolution.png")


if __name__ == "__main__":
    main()
