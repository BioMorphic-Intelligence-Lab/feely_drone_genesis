"""
Test script for attitude controller.

Tests attitude tracking with step responses on roll, pitch, and yaw axes.
"""

import torch
import genesis as gs
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controller import Controller
from transforms import quat_to_rotation_matrix, euler_from_quaternion, rotation_matrix_from_euler
from plotting import plot_attitude_tracking, plot_angular_velocity_tracking, plot_control_torques
from sim_utils import read_po, setup_scene, run_simulation


def main():
    po = read_po(description="Test attitude controller.")

    # Setup scene and drone (no gravity for pure attitude test)
    scene, cam, drone = setup_scene(po, gravity=(0, 0, 0), cam_pos=(0.0, -5.0, 1.0), cam_lookat=(0.0, 0.0, 1.0))

    # Set initial position
    drone.set_pos(torch.tensor([[0.0, 0.0, 2.0]]))

    n_steps = int(po.T / po.dt)

    # Initialize controller
    ctrl = Controller()

    # Data logging
    if po.plot:
        time = torch.linspace(0, po.T, n_steps)
        attitude = torch.zeros([n_steps, 3])
        attitude_des = torch.zeros([n_steps, 3])
        omega = torch.zeros([n_steps, 3])
        omega_des = torch.zeros([n_steps, 3])
        torques_applied = torch.zeros([n_steps, 3])

    print("Starting attitude controller test...")
    print("Commands:")
    print("  t=1.0-3.0s: Roll  = 45 deg")
    print("  t=4.0-6.0s: Pitch = 45 deg")
    print("  t=7.0-9.0s: Yaw   = 180 deg")

    def step_callback(step, t):
        # Get drone state
        base_link = drone.get_link("drone")
        base_quat = base_link.get_quat()
        base_w = base_link.get_ang()

        # Current attitude
        att = euler_from_quaternion(base_quat)
        R = quat_to_rotation_matrix(base_quat)

        # Define desired attitude trajectory (step responses)
        roll_des = torch.pi / 4 if (1.0 < t < 3.0) else 0.0
        pitch_des = torch.pi / 4 if (4.0 < t < 6.0) else 0.0
        yaw_des = torch.pi if (7.0 < t < 9.0) else torch.pi / 2

        att_des = torch.tensor([[roll_des, pitch_des, yaw_des]])
        R_des = rotation_matrix_from_euler(att_des)

        base_w_body = (R.transpose(-2, -1) @ base_w.unsqueeze(-1)).squeeze(-1)

        # Compute desired angular velocity from attitude controller
        rates_des = ctrl.attitude_ctrl(R=R, omega=base_w_body, R_des=R_des)

        # Compute control torques using angular rate controller
        torques = ctrl.angular_vel_ctrl(base_w_body, rates_des)

        # Check for NaN/Inf
        if torch.isnan(torques).any() or torch.isinf(torques).any():
            print(f"\nERROR at t={t:.3f}s: NaN/Inf detected!")
            print(f"  attitude: {att[0].numpy()}")
            print(f"  att_des:  {att_des[0].numpy()}")
            print(f"  omega:    {base_w[0].numpy()}")
            print(f"  torques:  {torques[0].numpy()}")
            return False

        # Transform torques to world frame
        world_torques = (R @ torques.unsqueeze(-1)).squeeze(-1)

        # Apply torques to drone
        scene.sim.rigid_solver.apply_links_external_torque(
            torque=world_torques.unsqueeze(0),
            links_idx=torch.tensor([base_link.idx])
        )

        # Log data
        if po.plot:
            attitude[step, :] = att
            attitude_des[step, :] = att_des
            omega[step, :] = base_w
            omega_des[step, :] = rates_des
            torques_applied[step, :] = torques

    run_simulation(scene, cam, po, step_callback, record_filename='attitude_test.mp4')

    # Plotting
    if po.plot:
        plot_attitude_tracking(time, attitude, attitude_des, filename='media/attitude_tracking.png')
        plot_angular_velocity_tracking(time, omega, omega_des, filename='media/attitude_rates.png')
        plot_control_torques(time, torques_applied, filename='media/attitude_torques.png')

        # Combined view
        fig = plt.figure(figsize=(14, 10))
        gs_layout = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs_layout[0, 0])
        ax1.plot(time, attitude[:, 0], label='Roll', linewidth=2)
        ax1.plot(time, attitude_des[:, 0], '--', color='black', linewidth=2)
        ax1.set_ylabel('Roll [rad]', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title('Attitude Tracking')

        ax2 = fig.add_subplot(gs_layout[0, 1])
        ax2.plot(time, attitude[:, 1], label='Pitch', linewidth=2)
        ax2.plot(time, attitude_des[:, 1], '--', color='black', linewidth=2)
        ax2.set_ylabel('Pitch [rad]', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_title('Attitude Tracking')

        ax3 = fig.add_subplot(gs_layout[1, 0])
        ax3.plot(time, omega[:, 0], label='ωx', linewidth=2)
        ax3.plot(time, omega_des[:, 0], '--', color='black', linewidth=2)
        ax3.set_ylabel('ωx [rad/s]', fontsize=11)
        ax3.set_xlabel('Time [s]', fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_title('Angular Rates')

        ax4 = fig.add_subplot(gs_layout[1, 1])
        ax4.plot(time, omega[:, 1], label='ωy', linewidth=2)
        ax4.plot(time, omega_des[:, 1], '--', color='black', linewidth=2)
        ax4.set_ylabel('ωy [rad/s]', fontsize=11)
        ax4.set_xlabel('Time [s]', fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_title('Angular Rates')

        fig.suptitle('Attitude Controller Performance', fontsize=16, fontweight='bold')
        fig.tight_layout()
        fig.savefig('media/attitude_summary.png', dpi=150)
        plt.close(fig)

        print("\nPlots saved:")
        print("  - media/attitude_tracking.png")
        print("  - media/attitude_rates.png")
        print("  - media/attitude_torques.png")
        print("  - media/attitude_summary.png")

        # Print final tracking errors
        print("\nFinal tracking errors:")
        final_errors = torch.abs(attitude[-100:, :] - attitude_des[-100:, :]).mean(dim=0)
        print(f"  Roll:  {final_errors[0]:.4f} rad ({final_errors[0]*180/torch.pi:.2f} deg)")
        print(f"  Pitch: {final_errors[1]:.4f} rad ({final_errors[1]*180/torch.pi:.2f} deg)")
        print(f"  Yaw:   {final_errors[2]:.4f} rad ({final_errors[2]*180/torch.pi:.2f} deg)")


if __name__ == "__main__":
    main()
