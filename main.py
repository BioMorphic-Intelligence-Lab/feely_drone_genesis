import argparse
import genesis as gs
import numpy as np

from scipy.spatial.transform import Rotation as R

from impl import StateMachine

def read_po():
    parser = argparse.ArgumentParser(description="Simulation of the feely drone.")
    parser.add_argument('--vis', action='store_true', help="Flag on whether the simulation is visualized.")
    parser.add_argument('--dt', type=float, default=0.01, help="Simulation step size")
    parser.add_argument('--T', type=float, default=10.0, help="Simulation end time")
    parser.add_argument('--yaw', type=float, default=0.0, help="Cylinder Yaw")

    parser.add_argument("--p_ini", type=str, default="0 0 0", help="Space-separated list of initial coordinates (x y z) or None")
    parser.add_argument("--yaw_ini", type=float, default=0.0, help="Initial gripper yaw")
    parser.add_argument('--rand', action='store_true', help="Randomize initial drone position")
    parser.add_argument('--record', action='store_true', help="Record experiment to video")
    parser.add_argument('--n_envs', type=int, default=1, help="Number of parallel environmnents")
    return parser.parse_args()

def main():
    
    args = read_po()

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
    #plane = scene.add_entity(gs.morphs.Plane())

    cylinder = scene.add_entity(
            gs.morphs.URDF(
                file="./assets/cylinder.urdf",  # Path to your URDF file
                pos=[0, 0, 2],
                euler=[0, 90, args.yaw],
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

    if args.rand:
        p_ini = np.random.uniform(low=[-1, -1, -0.5], high=[1, 1, 0.5], size=[args.n_envs, 3])
        yaw_ini = np.zeros([args.n_envs, 1])
    else:
        p_ini = np.fromstring(args.p_ini, sep=" ").reshape([1,3]).repeat(args.n_envs, axis=0)
        yaw_ini = args.yaw_ini * np.ones([args.n_envs, 1])

    gripper.set_dofs_position(
        np.concatenate([p_ini, yaw_ini, np.zeros([args.n_envs, 9])], axis=1)
    )

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
        p0[i, :] = np.array(joint.get_pos()[0, :]) - p_ini[0, :]
        q = joint.get_quat()[0, :]
        rot0[i, :, :] = R.from_quat(q, scalar_first=True).as_matrix()


    sm = np.array([
        StateMachine(dt=args.dt, tau_max=1250, m_total=gripper.get_mass(),
                     m_arm=m, l_arm=l, p0=p0, rot0=rot0,
                     K=K[:3, :3], A=-0.1 * np.ones(3),
                     q0=q0[:3], g=np.array([0, 0, -9.81]),
                     target_pos_estimate=np.array([0, 0, 1.5]),
                     target_yaw_estimate=yaw_ini[n, :])
        for n in range(args.n_envs)
    ])

    t = np.arange(0, args.T, args.dt)
    positions = np.zeros([args.n_envs, len(t), 13], dtype=float)
    velocities = np.zeros([args.n_envs, len(t), 13], dtype=float)
    input = np.zeros([args.n_envs, len(t), 7], dtype=float)
    p_des = np.zeros([args.n_envs, len(t), 3], dtype=float)
    yaw_des = np.zeros([args.n_envs, len(t), 1], dtype=float)
    state_machine_states = np.zeros([args.n_envs, len(t), 1], dtype=int)

    if args.record:
        cam.start_recording()

    for k in range(int(args.T / args.dt)):

        p = np.array(gripper.get_dofs_position())
        v = np.array(gripper.get_dofs_velocity())
        actions = np.zeros_like(p)

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

        gripper.control_dofs_force(actions)
        scene.step()

        # Save current state
        t[k] = k * args.dt
        positions[:, k, :] = p
        velocities[:, k, :] = v

        if args.record:
            cam.render()

    # Save Data
    np.savez(f'logs/ang_{args.yaw:.1f}.npz',
            t=t,
            positions=positions,
            velocities=velocities,
            input=input,
            p_des=p_des,
            yaw_des=yaw_des,
            state_machine_states=state_machine_states)
    
    if args.record:
        cam.stop_recording(save_to_filename='video.mp4', fps=60)

if __name__=="__main__":
    main()