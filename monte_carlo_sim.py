import argparse
import genesis as gs
import numpy as np

from transforms import (rotation_matrix_from_euler, quat_to_rotation_matrix)
from controller import Controller
from feely_drone_common import (StateMachine, State, GripperCtrl,
                                SinusoidalSearchPattern,
                                get_urdf_path)
from sim_utils import read_po, setup_scene, run_simulation

def main():
    
    args = read_po(description="Simulation of the feely drone.")

    if not (np.sum([
        args.angle_range is not None,
        args.position_range is not None,
        args.radius_range is not None,
        args.inclination_range is not None,
    ]) == 1):
        print("ERROR! Not exactly one of --position_range, --angle_range, --inclination_range and --radius_range was set!")
        exit()
    elif args.angle_range is not None:
        target_angles = np.arange(*np.fromstring(args.angle_range, sep=" "))
        target_inclinations = np.zeros([len(target_angles), 1])
        target_positions = np.zeros([len(target_angles), 3])
        target_positions[:, 2] = 2.0 
        cylinder_radii = 0.03 * np.ones([len(target_angles)])
    elif args.inclination_range is not None:
        target_inclinations = np.arange(*np.fromstring(args.inclination_range, sep=" "))
        target_angles = np.zeros([len(target_inclinations), 1])
        target_positions = np.zeros([len(target_inclinations), 3])
        target_positions[:, 2] = 2.0 
        cylinder_radii = 0.03 * np.ones([len(target_inclinations)])
    elif args.position_range is not None:
        positional_offsets_x = np.arange(*np.fromstring(args.position_range, sep=" "))
        positional_offsets_x = positional_offsets_x.reshape([positional_offsets_x.size, 1])
        target_positions = np.concatenate([positional_offsets_x,
                                           np.zeros_like(positional_offsets_x),
                                           2.0 * np.ones_like(positional_offsets_x)], axis=1)
        target_angles = np.zeros([len(positional_offsets_x), 1])
        target_inclinations = np.zeros([len(positional_offsets_x), 1])
        cylinder_radii = 0.03 * np.ones([len(target_angles)])
    elif args.radius_range is not None:
        cylinder_radii = np.arange(*np.fromstring(args.radius_range, sep=" "), 0.005)
        target_positions = np.zeros([len(cylinder_radii), 3])
        target_positions[:, 2] = 2.0
        target_angles = np.zeros([len(cylinder_radii), 1])
        target_inclinations = np.zeros([len(cylinder_radii), 1])

    def pre_build_setup(scene_obj):
        objects = []

        for radius in cylinder_radii.flatten():
            if args.target_object == "h_bar":
                # Single H-bar target, shared across all trials
                objects.append(
                    scene_obj.add_entity(
                        gs.morphs.URDF(
                            file=get_urdf_path("h_bar.urdf"),
                            pos=[1000, 1000, 2],
                            euler=[0, 90, 0.0],
                            fixed=True,
                        )
                    )
                )
            else:
                urdf_name = f"cylinder_{radius:0.3f}.urdf"
                objects.append(
                    scene_obj.add_entity(
                        gs.morphs.URDF(
                            file=get_urdf_path(urdf_name),
                            pos=[1000, 1000, 2],
                            euler=[0, 90, 0.0],
                            fixed=True,
                        )
                    )
                )
        
        return objects

    scene, cam, gripper, objects = setup_scene(
        po=args,
        cam_pos=(1.0, -5.0, 4.0),
        cam_lookat=(0.0, 0.0, 1.0),
        viewer_pos=(1.0, -5.0, 4.0),
        viewer_lookat=(0.0, 0.0, 1.0),
        max_FPS=25,
        logging_level='warning',
        pre_build_callback=pre_build_setup
    )

    p_ini = np.random.uniform(low=[-1.0, -1.0, 0.25], high=[1.0, 1.0, 0.25], size=[args.n_envs, 3])

    K = np.diag(1.0 * np.ones(9)) # np.diag(100.0 * np.ones(9))
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
        
        q = joint.get_quat()[0, :].reshape(1, 4)
        rot0[i, :, :] = quat_to_rotation_matrix(q)

        p0[i, :] = np.array(joint.get_pos()[0, :]) - rot0[i, :, :] @ np.array([0, 0, 0.025]) 
        
    # Initial target position estimate
    init_target_pos_estimate=np.array([0, 0, 1.9])
    init_target_yaw_estimate=np.zeros([1])

    # Reset the State Machines
    sm = np.array([
        StateMachine(dt=args.dt,            # Delta T
                     m_arm=m,                   # Mass of the Arm
                     l_arm=l,                            # Length of the Arm
                     alpha_rate=0.1,                     # Opening and closing rate
                     p0=p0,                              # Offset Position of Arms
                     rot0=rot0,                          # Offset Rotation of Arms
                     K=K[:3, :3],                        # Stiffness Matrix of the arm
                     A=-1.20 * np.ones(3),                # Actuation map
                     q0=np.deg2rad(75) * np.ones(3),     # Neutral joint states
                     g=np.array([0, 0, -9.81]),          # Gravity Vector
                     target_pos_estimate=init_target_pos_estimate,
                     target_yaw_estimate=init_target_yaw_estimate,
                     searching_pattern=SinusoidalSearchPattern(
                           params=np.stack([
                                np.array([0.5, 0.5, 0]),     # Amplitude
                                np.array([2.0, 1.0, 0.0]),   # Frequency
                                np.array([0.0, 0.0, 0.0]),   # Phase Shift
                                init_target_pos_estimate - np.array([0, 0, 0.1]) # Offset
                            ]),
                            dt=args.dt,
                            vel_norm=0.25)
        )
        for _ in range(args.n_envs)
    ])

    # Init the low leverl controllers
    pose_ctrl = np.array([
        Controller(dt=args.dt)
        for _ in range(args.n_envs)
    ])
    gripper_ctrl = np.array([
        GripperCtrl(tau_max=12.50) for _ in range(args.n_envs)
    ])

    if args.record:
        cam.start_recording()

    # Run the monte carlo simulation
    for trial in range(len(target_angles)):

        scene.reset()
        
        # Set initial state gripper
        state_ini = np.concatenate([p_ini, np.zeros([args.n_envs, 3]), np.zeros([args.n_envs, 9])], axis=1)
        gripper.set_dofs_position(state_ini)
        gripper.set_dofs_velocity(np.zeros_like(gripper.get_dofs_velocity()))
        gripper.control_dofs_force(np.zeros_like(gripper.get_dofs_force()))
        
        # Init the target object position
        objects[trial].set_pos(np.array([target_positions[trial, :] for _ in range(args.n_envs)]))
        if trial > 0:
            objects[trial - 1].set_pos(np.array([[1000, 1000, 2] for _ in range(args.n_envs)]))

        if args.target_object == "h_bar":
            # Ensure `euler` has shape (len(env_idx), 3)
            euler = np.stack([
                target_inclinations[trial] * np.ones(args.n_envs),    # X rotation (inclination)
                np.zeros(args.n_envs),                                # Y rotation (zero)
                target_angles[trial] * np.ones(args.n_envs)           # Z rotation (converted to degrees)
            ], axis=1)      
        elif args.target_object == "cylinder":
            # Ensure `euler` has shape (len(env_idx), 3)
            euler = np.stack([
                target_inclinations[trial] * np.ones(args.n_envs),    # X rotation (inclination)
                np.zeros(args.n_envs),                            # Y rotation (90 degrees)
                target_angles[trial] * np.ones(args.n_envs)           # Z rotation (angle)
            ], axis=1)  
        

        quat = gs.utils.geom.xyz_to_quat(euler)
        objects[trial].set_quat(quat)

        # Reset the state machines and controllers
        for n in range(args.n_envs):
            sm[n].reset()
            gripper_ctrl[n].reset()
            pose_ctrl[n].reset()
        scene.step()

        # Init data storage arrays
        t = np.arange(0, args.T, args.dt)
        if args.save:
            positions = np.zeros([args.n_envs, len(t), 13], dtype=float)
            velocities = np.zeros([args.n_envs, len(t), 13], dtype=float)
            input = np.zeros([args.n_envs, len(t), 7], dtype=float)
            p_des = np.zeros([args.n_envs, len(t), 3], dtype=float)
            yaw_des = np.zeros([args.n_envs, len(t), 1], dtype=float)
            state_machine_states = np.zeros([args.n_envs, len(t), 1], dtype=int)
        if args.trace:
            tracesteps = args.trace_steps
            trace = np.zeros([args.n_envs, len(t) // tracesteps, 3])

        # Run Monte Carlo Trial with n_envs
        def step_callback(k, t_val):
            p_full = np.array(gripper.get_dofs_position())
            p = p_full[:, :6]
            if args.trace and k % tracesteps == 0:
                trace[:, k // tracesteps, :] = p[:, :3] - 0.05 * np.array([0, 0, 1])  # Trace at the tip of the gripper
            v_full = np.array(gripper.get_dofs_velocity())
            v = v_full[:, :6]

            actions = np.zeros_like(p_full)

            targets = np.zeros([args.n_envs, 3])
            reference_pos = np.zeros([args.n_envs, 3])
            contact_sensors = np.zeros([args.n_envs * 9, 3])

            for n in range(args.n_envs):
                rot = rotation_matrix_from_euler(p_full[n, 3:6]).reshape(1, 3, 3)

                contact = np.reshape(
                    np.linalg.norm(gripper.get_links_net_contact_force()[n, 4:, :],
                                axis=1) > 0.0,
                    [3,3]).T
            
                sm[n].update_tactile_info_sw(contact=contact)

                stiffness_contrib = K @ (q0 - p_full[n, 6:])

                sm_return = sm[n].control(p_full[n, :], v_full[n, :], contact)

         
                body_torques, body_forces = pose_ctrl[n].go_to(
                    loc=sm_return['p_des'], yaw_des=sm_return['yaw_des'],
                    v_des=sm_return['v_des'][:3], 
                    v_mag=2.0,
                    p=p_full[n, :3].reshape(1, 3), v=v_full[n, :3].reshape(1, 3),
                    R=rot, w_body=v_full[n, 3:6].reshape(1, 3), mass=gripper.get_mass(), g=9.81,
                    epsilon=0.1, acc_max=2.0
                )

                world_torques = (rot @ body_torques.unsqueeze(-1)).flatten()
                world_forces = (rot @ body_forces.unsqueeze(-1)).flatten()
                
                tau_ctrl = gripper_ctrl[n].open_to(sm_return['alpha'])
                action = np.concatenate([
                    world_forces,
                    world_torques,
                    stiffness_contrib - r @ tau_ctrl
                ])

                actions[n, :] = action

                # Save current desired pose
                if args.save:
                    input[n, k, :] = np.concatenate([world_forces, world_torques, tau_ctrl])
                    p_des[n, k, :] = sm_return['p_des']
                    yaw_des[n, k, :] = sm_return['yaw_des']
                    state_machine_states[n, k, :] = sm[n].state.value

                if args.debug:
                    targets[n, :] = sm[n].target_pos_estimate
                    reference_pos[n, :] = sm[n].reference_pos
                    for i in range(3):
                        for j in range(3):
                            contact_sensors[n*9 + i * 3 + j, :] = sm[n].contact_locs[i, j, :]
            
                if args.trace and k > tracesteps:
                    scene.draw_debug_line(start=trace[n, k // tracesteps-1, :],
                                        end=trace[n, k // tracesteps, :],
                                        color=(0.0/255.0, 166.0/255.0, 214.0/255.0, 0.8),
                                        radius=0.005)

            if args.debug:
                scene.clear_debug_objects()
                scene.draw_debug_spheres(targets, radius=0.05, color=(1, 0, 0, 0.5))
                scene.draw_debug_spheres(reference_pos, radius=0.05, color=(0, 0, 1, 0.5))
                scene.draw_debug_spheres(sm[-1].searching_pattern.traj_dis, radius=0.025, color=(0.5, 0.5, 0.5, 0.5))
                contact_marker_color = [(0, 1, 0, 0.5) for _ in range(9)]
                for i in range(9):
                    if contact[i // 3, i % 3]:
                        contact_marker_color[i] = (1, 1, 0, 0.5)
                scene.draw_debug_spheres(contact_sensors, radius=0.01,
                                         color=contact_marker_color)

            gripper.control_dofs_force(actions)

            # Save current state
            if args.save:
                t[k] = k * args.dt
                positions[:, k, :] = p_full
                velocities[:, k, :] = v_full

        run_simulation(scene, cam, args, step_callback, manage_recording=False)

        if args.save:
            if args.angle_range is not None:
                filename = f'logs/angle/trial_{int(target_angles[trial]):02}.npz'
            elif args.position_range is not None:
                filename = f'logs/position/trial_{float(target_positions[trial, 0]):.2f}.npz'
            elif args.radius_range is not None:
                filename = f'logs/radius/trial_{cylinder_radii[trial]:.3f}.npz'
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
        print("Saving video...")
        cam.stop_recording(save_to_filename='video.mp4', fps=args.video_fps)
        print("... done.")

if __name__=="__main__":
    main()