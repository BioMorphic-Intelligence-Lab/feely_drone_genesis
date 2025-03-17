import argparse
import genesis as gs
import numpy as np

from scipy.spatial.transform import Rotation as R

from impl import StateMachine

def read_po():
    parser = argparse.ArgumentParser(description="Simulation of the feely drone.")
    parser.add_argument('--vis', action='store_true', help="Flag on whether the simulation is visualized.")
    parser.add_argument('--record', action='store_true', help="Record experiment to video")
    parser.add_argument('--dt', type=float, default=0.01, help="Simulation step size")
    parser.add_argument('--T', type=float, default=35.0, help="Simulation end time")
    parser.add_argument('--debug', action='store_true', help="Activate debug visualizations")
    parser.add_argument('--n_envs', type=int, default=100, help="Number of environments used per trial.")
    parser.add_argument('--angle_range', type=str, default=None, help="Space-separated list of \"min_angle max_angle step\" or None")
    parser.add_argument('--position_range', type=str, default=None, help="Space-separated list of \"min_pos max_pos step\" or None")
    return parser.parse_args()

def main():
    
    args = read_po()

    if ((args.angle_range is None and args.position_range is None) 
        or  (args.angle_range is not None and args.position_range is not None)):
        print("ERROR! Not exactly one of --position_range and --angle_range needs was set!")
        exit()
    elif args.angle_range is not None:
        target_angles = np.arange(*np.fromstring(args.angle_range, sep=" "))
        target_positions = np.zeros([len(target_angles), 3])
        target_positions[:, 2] = 2.0 
    elif args.position_range is not None:
        positional_offsets_y = np.arange(*np.fromstring(args.position_range, sep=" "))
        positional_offsets_y = positional_offsets_y.reshape([positional_offsets_y.size, 1])
        target_positions = np.concatenate([np.zeros_like(positional_offsets_y),
                                           positional_offsets_y,
                                           2.0 * np.ones_like(positional_offsets_y)], axis=1)
        target_angles = np.zeros([len(positional_offsets_y), 1])

    gs.init(backend=gs.cpu, precision="32", logging_level='warning')

    scene = gs.Scene(
                viewer_options=
                gs.options.ViewerOptions(
                    camera_pos=(0.0, 10.0, 5),
                    camera_lookat=(0.0, 0.0, 0.5),
                    camera_fov=30,
                    res=(960, 640),
                    max_FPS=60,
                ),
                sim_options=gs.options.SimOptions(dt=args.dt),
                show_viewer=args.vis
            )

    cam = scene.add_camera(
                res    = (1280, 960),
                pos    = (0.0, 10.0, 5),
                lookat = (0.0, 0.0, 1.5),
                fov    = 30,
                GUI    = False
            )
    plane = scene.add_entity(gs.morphs.Plane())

    cylinder = scene.add_entity(
            gs.morphs.URDF(
                file="./assets/cylinder.urdf",  # Path to your URDF file
                pos=[0, 0, 2],
                euler=[0, 90, 0.0],
                fixed=True
            )
        )

    gripper = scene.add_entity(
        gs.morphs.URDF(
                file="./assets/gripper.urdf",  # Path to your URDF file
                pos=[0, 0, 0],
                euler=[0, 0, 0],
                fixed=True
            )
    )

    scene.build(n_envs=args.n_envs)

    p_ini = np.random.uniform(low=[-1.0, -1.0, 1.5], high=[1.0, 1.0, 1.5], size=[args.n_envs, 3])

    K = np.diag(100*np.ones(9))
    r = np.concatenate([np.diag(0.1 * np.ones(3)) for _ in range(3)], axis=1).T    
    q0 = np.deg2rad(75) * np.ones(9)

    # Extract Link Mass and Length from object
    # needed for computation of steady-state config
    m = np.zeros(3)
    l = np.zeros(3)
    for i in range(3):
        link = gripper.get_link(name=f"arm1_link{i+1}")
        m[i] = link.get_mass()
        extents = link.geoms[0].get_trimesh().extents
        l[i] = extents[2]

    # Extract offset positions and orientations of the arms
    # needed for Forward Kinematics
    p0 = np.zeros([3, 3])
    rot0 = np.zeros([3, 3, 3])
    for i in range(3):
        joint = gripper.get_joint(name=f"arm{i+1}_joint1")
        p0[i, :] = np.array(joint.get_pos()[0, :]) 
        q = joint.get_quat()[0, :]
        rot0[i, :, :] = R.from_quat(q, scalar_first=True).as_matrix()

    # Reset the State Machines
    sm = np.array([
        StateMachine(dt=args.dt, tau_max=1250, m_total=gripper.get_mass(),
                    m_arm=m, l_arm=l, p0=p0, rot0=rot0,
                    K=K[:3, :3], A=-0.1 * np.ones(3),
                    q0=q0[:3], g=np.array([0, 0, -9.81]),
                    target_pos_estimate=np.array([0, 0, 1.5]),
                    target_yaw_estimate=np.zeros([1]))
        for _ in range(args.n_envs)
    ])

    if args.record:
        cam.start_recording()

    # Run the monte carlo simulation
    for trial in range(len(target_angles)):
        scene.reset()
        # Set initial state gripper
        gripper.set_dofs_position(
            np.concatenate([p_ini, np.zeros([args.n_envs, 1]), np.zeros([args.n_envs, 9])], axis=1)
        )
        gripper.set_dofs_velocity(np.zeros_like(gripper.get_dofs_velocity()))
        gripper.control_dofs_force(np.zeros_like(gripper.get_dofs_force()))
        # Init the cylinder pos
        cylinder.set_pos(np.array([target_positions[trial, :] for _ in range(args.n_envs)]))
         # Ensure `euler` has shape (len(env_idx), 3)
        euler = np.stack([
            np.zeros(args.n_envs),    # X rotation (zero)
            90 * np.ones(args.n_envs),    # Y rotation (zero)
            target_angles[trial] * np.ones(args.n_envs) # Z rotation (converted to degrees)
        ], axis=1)  # Shape: (len(env_idx), 3)
        # Convert to quaternion
        quat = gs.utils.geom.xyz_to_quat(euler)
        cylinder.set_quat(quat)

        # Reset the state machines
        for n in range(args.n_envs):
            sm[n].reset()
        scene.step()

        # Init data storage arrays
        t = np.arange(0, args.T, args.dt)
        positions = np.zeros([args.n_envs, len(t), 13], dtype=float)
        velocities = np.zeros([args.n_envs, len(t), 13], dtype=float)
        input = np.zeros([args.n_envs, len(t), 7], dtype=float)
        p_des = np.zeros([args.n_envs, len(t), 3], dtype=float)
        yaw_des = np.zeros([args.n_envs, len(t), 1], dtype=float)
        state_machine_states = np.zeros([args.n_envs, len(t), 1], dtype=int)

        # Run Monte Carlo Trial with n_envs
        for k in range(int(args.T / args.dt)):

            p = np.array(gripper.get_dofs_position())
            #p += np.random.normal(loc=np.zeros_like(p),
            #                      scale=np.concatenate(([0.02, 0.02, 0.02, np.deg2rad(1)],
            #                                            np.zeros(9)))
            #)
            v = np.array(gripper.get_dofs_velocity())
            #v += np.random.normal(loc=np.zeros_like(v),
            #                      scale=np.concatenate(([0.01, 0.01, 0.01, np.deg2rad(0.1)],
            #                                            np.zeros(9)))
            #)
            actions = np.zeros_like(p)

            targets = np.zeros([args.n_envs, 3])
            reference_pos = np.zeros([args.n_envs, 3])
            contact_sensors = np.zeros([args.n_envs * 9, 3])

            for n in range(args.n_envs):

                contact = np.reshape(
                    np.linalg.norm(gripper.get_links_net_contact_force()[n, 5:, :],
                                axis=1) > 0.0,
                    [3,3]).T
            
                sm[n].update_tactile_info_sw(contact=contact)

                stiffness_contrib = K @ (q0 - p[n, 4:])

                sm_return = sm[n].control(p[n, :], v[n, :], contact)

                action = np.concatenate([sm_return['pos_ctrl'], sm_return['yaw_ctrl'],
                                        stiffness_contrib
                                        - r @ sm_return['tau']])

                actions[n, :] = action

                # Save current desired pose
                input[n, k, :] = np.concatenate([sm_return['pos_ctrl'], sm_return['yaw_ctrl'], sm_return['tau']])
                p_des[n, k, :] = sm_return['p_des']
                yaw_des[n, k, :] = sm_return['yaw_des']
                state_machine_states[n, k, :] = sm[n].state.value

                if args.debug:
                    targets[n, :] = sm[n].target_pos_estimate
                    reference_pos[n, :] = sm[n].reference_pos
                    for i in range(3):
                        for j in range(3):
                            contact_sensors[n*9 + i * 3 + j, :] = sm[n].contact_locs[i, j, :]
            
            if args.debug:
                scene.clear_debug_objects()
                scene.draw_debug_spheres(targets, radius=0.1, color=(1, 0, 0, 0.5))
                scene.draw_debug_spheres(reference_pos, radius=0.1, color=(0, 0, 1, 0.5))
                scene.draw_debug_spheres(sm[-1].searching_pattern.traj_dis, radius=0.025, color=(0.5, 0.5, 0.5, 0.5))
                scene.draw_debug_spheres(contact_sensors, radius=0.05, color=(0, 1, 0, 0.5))

            gripper.control_dofs_force(actions)
            scene.step()

            # Save current state
            t[k] = k * args.dt
            positions[:, k, :] = p
            velocities[:, k, :] = v

            if args.record:
                cam.render()

        if args.angle_range is not None:
            filename = f'logs/angle/trial_{int(target_angles[trial]):02}.npz'
        elif args.position_range is not None:
            filename = f'logs/position/trial_{float(target_positions[trial, 1]):.2f}.npz'
        # Save Data
        np.savez(filename,
                t=t,
                positions=positions,
                velocities=velocities,
                input=input,
                p_des=p_des,
                yaw_des=yaw_des,
                state_machine_states=state_machine_states)

        if args.record:
            cam.stop_recording(save_to_filename='video.mp4', fps=24)

if __name__=="__main__":
    main()