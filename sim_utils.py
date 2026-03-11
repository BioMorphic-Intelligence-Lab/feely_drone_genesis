"""
Simulation utilities for Genesis setup and common operations.
"""

from __future__ import annotations

import argparse
from typing import Sequence

import torch
import genesis as gs


def _to_list(x: Sequence[int] | torch.Tensor) -> list[int]:
    if isinstance(x, torch.Tensor):
        return x.flatten().tolist()
    return list(x)


def read_po(description: str = "Simulation of the flying squid."):
    """Parse common command line arguments for simulation scripts.
    
    Args:
        description: Description for the argument parser
        
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--vis', action='store_true', 
                       help="Flag on whether the simulation is visualized.")
    parser.add_argument('--dt', type=float, default=0.002, 
                       help="Simulation step size")
    parser.add_argument('--T', type=float, default=20.0, 
                       help="Simulation end time")
    parser.add_argument('--record', action='store_true', 
                       help="Record experiment to video")
    parser.add_argument('--plot', action='store_true', 
                       help="Whether or not to plot the trial.")
    parser.add_argument('--n_envs', type=int, default=1, 
                       help="Number of parallel environments")
    parser.add_argument('--debug', action='store_true', 
                       help="Whether or not to draw debug arrows")
    parser.add_argument('--device', type=str, default='cpu', 
                       help="Device to run the simulation on: cpu")
    parser.add_argument('--full_vis', action='store_true', 
                       help="Flag on whether to use the full meshes for visualization.")
    parser.add_argument('--trace', action='store_true', 
                       help="Flag on whether to plot the drone trace visualization.")
    parser.add_argument('--trace_steps', type=int, default=1, 
                       help="At which n-th step to draw the trace.")
    parser.add_argument('--save', action='store_true', 
                       help="Flag on whether to save the raw data.")
    parser.add_argument('--angle_range', type=str, default=None, 
                       help="Space-separated list of \"min_angle max_angle step\" or None")
    parser.add_argument('--position_range', type=str, default=None, 
                       help="Space-separated list of \"min_pos max_pos step\" or None")
    parser.add_argument('--radius_range', type=str, default=None, 
                       help="Space-separated list of \"min_r max_r\" or None")
    parser.add_argument('--video_fps', type=int, default=25, 
                       help="Output FPS of the video recording")
    return parser.parse_args()


def init_genesis(device: str = 'cpu', precision: str = "32", logging_level: str = 'error'):
    """Initialize Genesis simulation backend.
    
    Args:
        device: Device to run on ('cpu' or 'cuda')
        precision: Floating point precision ('32' or '64')
        logging_level: Logging verbosity level
    """
    if str.lower(device) == 'cpu':
        gs.init(backend=gs.cpu, precision=precision, logging_level=logging_level)
    else:
        print("ERROR! Currently no other device than CPU supported")


def setup_rigid_options(contact_resolve_time: float = 0.01, iterations: int = 1000):
    """Set contact dynamics options for rigid body simulation.
    
    Args:
        contact_resolve_time: Time constant for contact resolution
        iterations: Number of solver iterations
    """
    gs.options.RigidOptions(
        contact_resolve_time=contact_resolve_time,
        iterations=iterations
    )


def setup_scene(po, gravity=(0, 0, -9.81), cam_pos=(0.0, -7.50, 3.0), cam_lookat=(0.0, 0.0, 0.5), viewer_pos=(-1.0, -15.0, 10), viewer_lookat=(0.0, 10.0, 0.5), max_FPS=60, logging_level='error', pre_build_callback=None):
    """Initialize Genesis and set up the simulation scene with a drone and camera.
    
    Args:
        po: Parsed arguments from read_po()
        gravity: Gravity vector (default: (0, 0, -9.81))
        cam_pos: Camera position (default: (0.0, -7.50, 3.0))
        cam_lookat: Camera lookat target (default: (0.0, 0.0, 0.5))
        viewer_pos: Scene viewer camera position
        viewer_lookat: Scene viewer camera target
        max_FPS: Viewer maximum FPS
        logging_level: Logging verbosity level for Genesis initialization
        pre_build_callback: Optional callback `func(scene)` called before scene.build() to add extra entities
        
    Returns:
        tuple: (scene, camera, drone_entity)
    """
    from feely_drone_common.utility import get_urdf_path

    # Initialize Genesis
    init_genesis(device=po.device, logging_level=logging_level)
    setup_rigid_options()

    # Define the scene
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=viewer_pos,
            camera_lookat=viewer_lookat,
            camera_fov=30,
            res=(960, 640),
            max_FPS=max_FPS,
        ),
        sim_options=gs.options.SimOptions(dt=po.dt, gravity=gravity),
        show_viewer=po.vis
    )

    # Init Camera
    cam = scene.add_camera(
        res=(1280, 960),
        pos=cam_pos,
        lookat=cam_lookat,
        fov=30,
        GUI=False
    )

    ground_plane = scene.add_entity(gs.morphs.Plane())

    # Add optional cyberzoo if full_vis is requested
    if getattr(po, 'full_vis', False):
        scene.add_entity(
            gs.morphs.URDF(
                file=get_urdf_path("cyberzoo.urdf"),
                pos=[-5, -5, 0.01],
                euler=[0, 0, 0],
                fixed=True,
                scale=0.025
            )
        )

    # Add drone
    urdf_file = get_urdf_path("gripper6dof.urdf") if (po.full_vis) else get_urdf_path("gripper6dof_simple.urdf")
    drone = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_file,
            fixed=False,
            merge_fixed_links=False,
        ),
        visualize_contact=getattr(po, 'debug', False)
    )

    if pre_build_callback is not None:
        pre_build_callback(scene)

    # Build the scene
    scene.build(n_envs=po.n_envs)

    return scene, cam, drone


def run_simulation(scene, cam, po, step_callback, record_filename='media/video.mp4', manage_recording=True):
    """Run the main simulation loop with rendering and recording.
    
    Args:
        scene: Genesis scene object
        cam: Genesis camera object
        po: Parsed arguments from read_po()
        step_callback: Callable taking (step, t) that implements the control and logging logic. 
                       If it returns False, the simulation loop will break.
        record_filename: Filename to save the video to if recording
        manage_recording: If True, calls cam.start_recording() before loop and cam.stop_recording() after loop
    """
    n_steps = int(po.T / po.dt)
    video_fps = getattr(po, 'video_fps', 24)
    
    if po.record and manage_recording:
        cam.start_recording()
        
    for step in range(n_steps):
        t = step * po.dt
        
        # Execute custom control and logging logic
        if step_callback(step, t) is False:
            break
        
        scene.step()
        
        if po.record and step % int(1.0 / (video_fps * po.dt)) == 0:
            cam.render()
            
    if po.record and manage_recording:
        cam.stop_recording(save_to_filename=record_filename, fps=video_fps)

