"""
Plotting utilities for simulation visualization.
"""

import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Sequence, Optional


def plot_triplet(
    time: torch.Tensor,
    data: torch.Tensor,
    ref_data: torch.Tensor,
    labels: Sequence[str],
    title: str,
    filename: str,
    ylabel_unit: str = ""
) -> None:
    """Plot three time-series with reference data.
    
    Args:
        time: Time vector of shape (n_steps,)
        data: Data tensor of shape (n_steps, 3)
        ref_data: Reference data tensor of shape (n_steps, 3)
        labels: List of 3 axis labels
        title: Plot title
        filename: Output filename
        ylabel_unit: Unit string to append to y-axis labels
    """
    fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(time, data[:, i])
        ax.plot(time, ref_data[:, i], "--", color="black")
        ax.set_ylabel(f"{labels[i]}{ylabel_unit}")

    axes[-1].set_xlabel("time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_position_groundtrack(
    time: torch.Tensor,
    position: torch.Tensor,
    ref_pos: Optional[torch.Tensor] = None,
    filename: str = "media/position_groundtrack.png",
    xlim: tuple = (-1.2, 1.2),
    ylim: tuple = (-1.5, 1.5)
) -> None:
    """Plot position groundtrack (x-y) and position vs time.
    
    Args:
        time: Time vector of shape (n_steps,)
        position: Position tensor of shape (n_steps, 3)
        ref_pos: Optional reference position of shape (n_steps, 3)
        filename: Output filename
        xlim: X-axis limits for groundtrack
        ylim: Y-axis limits for groundtrack
    """
    fig = plt.figure(figsize=(10, 6))
    grid = GridSpec(3, 2, figure=fig)

    # Ground track
    ax_xy = fig.add_subplot(grid[:, 0])
    ax_xy.set_xlim(xlim)
    ax_xy.set_ylim(ylim)
    ax_xy.set_aspect('equal')
    ax_xy.plot(position[:, 0], position[:, 1])
    if ref_pos is not None:
        ax_xy.plot(ref_pos[:, 0], ref_pos[:, 1], "--", color="black")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("Ground track")

    # Position vs time
    labels = ["x", "y", "z"]
    for i in range(3):
        ax = fig.add_subplot(grid[i, 1])
        ax.plot(time, position[:, i])
        if ref_pos is not None:
            ax.plot(time, ref_pos[:, i], "--", color="black")
        ax.set_ylabel(f"{labels[i]} [m]")
        if i < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("time [s]")

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_arm_lengths(
    time: torch.Tensor,
    arm_lengths: torch.Tensor,
    filename: str = "media/arm_lengths.png",
    nominal_length: Optional[float] = None,
    ylim: tuple = (0.14, 0.31)
) -> None:
    """Plot arm lengths over time.
    
    Args:
        time: Time vector of shape (n_steps,)
        arm_lengths: Arm lengths tensor of shape (n_steps, 4)
        filename: Output filename
        nominal_length: Optional nominal length to draw as reference line
        ylim: Y-axis limits
    """
    fig, axes = plt.subplots(4, 1, figsize=(6, 8), sharex=True)
    arm_labels = ["Arm 1", "Arm 2", "Arm 3", "Arm 4"]
    
    for i, ax in enumerate(axes):
        ax.plot(time, arm_lengths[:, i])
        if nominal_length is not None:
            ax.axhline(y=nominal_length, color='black', linestyle='--', label='Nominal')
        ax.set_ylabel(f"{arm_labels[i]} [m]")
        ax.set_ylim(ylim)
        ax.set_xlim([0.0, time[-1]])
    
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Arm Lengths")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_attitude_tracking(
    time: torch.Tensor,
    attitude: torch.Tensor,
    attitude_des: torch.Tensor,
    filename: str = "media/attitude_tracking.png"
) -> None:
    """Plot attitude tracking performance.
    
    Args:
        time: Time vector of shape (n_steps,)
        attitude: Actual attitude [roll, pitch, yaw] of shape (n_steps, 3)
        attitude_des: Desired attitude of shape (n_steps, 3)
        filename: Output filename
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['Roll', 'Pitch', 'Yaw']
    
    for i, ax in enumerate(axes):
        ax.plot(time, attitude[:, i], label='Actual', linewidth=2)
        ax.plot(time, attitude_des[:, i], '--', color='black', label='Desired', linewidth=2)
        ax.set_ylabel(f'{labels[i]} [rad]', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    axes[-1].set_xlabel('Time [s]', fontsize=12)
    fig.suptitle('Attitude Tracking', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_angular_velocity_tracking(
    time: torch.Tensor,
    omega: torch.Tensor,
    omega_des: torch.Tensor,
    filename: str = "media/angular_velocity_tracking.png"
) -> None:
    """Plot angular velocity tracking performance.
    
    Args:
        time: Time vector of shape (n_steps,)
        omega: Actual angular velocity of shape (n_steps, 3)
        omega_des: Desired angular velocity of shape (n_steps, 3)
        filename: Output filename
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['ωx', 'ωy', 'ωz']
    
    for i, ax in enumerate(axes):
        ax.plot(time, omega[:, i], label='Actual', linewidth=2)
        ax.plot(time, omega_des[:, i], '--', color='black', label='Desired', linewidth=2)
        ax.set_ylabel(f'{labels[i]} [rad/s]', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    axes[-1].set_xlabel('Time [s]', fontsize=12)
    fig.suptitle('Angular Velocity Tracking', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_control_torques(
    time: torch.Tensor,
    torques: torch.Tensor,
    filename: str = "media/control_torques.png"
) -> None:
    """Plot applied control torques.
    
    Args:
        time: Time vector of shape (n_steps,)
        torques: Control torques of shape (n_steps, 3)
        filename: Output filename
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['τx', 'τy', 'τz']
    
    for i, ax in enumerate(axes):
        ax.plot(time, torques[:, i], linewidth=2)
        ax.set_ylabel(f'{labels[i]} [Nm]', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time [s]', fontsize=12)
    fig.suptitle('Applied Control Torques', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
