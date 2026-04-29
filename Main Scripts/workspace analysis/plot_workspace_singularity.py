import numpy as np
import matplotlib.pyplot as plt
from CustomRobotClass import CustomRobot

JOINTS = 5

def plot_workspace_singularity(samples=50000, sing_threshold=0.015):
    robot = CustomRobot()
    print(f"Robot: {robot.name}")
    
    try:
        qlim = robot.qlim
        if qlim is None or np.all(qlim == 0):
            qlim = np.array([[-np.pi] * JOINTS, [np.pi] * JOINTS])
    except AttributeError:
        qlim = np.array([[-np.pi] * JOINTS, [np.pi] * JOINTS])
    
    print(f"Generating {samples} configurations...")
    q_random = np.random.uniform(qlim[0], qlim[1], (samples, JOINTS))
    
    x_sing, y_sing, z_sing = [], [], []
    x_work, y_work, z_work = [], [], []
    
    for i, q in enumerate(q_random):
        pose = robot.fkine(q).A
        px, py, pz = pose[0, 3], pose[1, 3], pose[2, 3]
        
        J = robot.jacob0(q)
        s = np.linalg.svd(J, compute_uv=False)
        min_sv = np.min(s)
        
        if min_sv < sing_threshold:
            x_sing.append(px)
            y_sing.append(py)
            z_sing.append(pz)
        else:
            if np.random.random() < 0.1:
                x_work.append(px)
                y_work.append(py)
                z_work.append(pz)
                
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1}/{samples}")

    all_z = np.concatenate([z_work, z_sing]) if len(z_work) > 0 or len(z_sing) > 0 else [0, 1]
    z_min, z_max = np.min(all_z), np.max(all_z)

    q_zero = np.zeros(JOINTS)
    try:
        poses = robot.fkine_all(q_zero)
        if isinstance(poses, list):
            joint_positions = np.array([p.t for p in poses])
        else:
            joint_positions = poses.t
        joint_positions = np.vstack([[0, 0, 0], joint_positions])
    except:
        joint_positions = None

    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    scatter_sing1 = ax1.scatter(x_sing, y_sing, z_sing, s=15, alpha=0.6, c=z_sing, cmap='viridis', vmin=z_min, vmax=z_max, linewidths=0, label='Singularities')
    ax1.scatter([0], [0], [0], color='black', s=100, marker='*', label='Base')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Workspace (Singularities)')
    if joint_positions is not None:
        ax1.plot(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2], color='black', linewidth=2, marker='o', markersize=4, label='Robot (q=0)')
    ax1.legend()
    fig1.colorbar(scatter_sing1, ax=ax1, label='Z (Height)')

    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    scatter_sing2 = ax2.scatter(x_sing, y_sing, s=10, alpha=0.6, c=z_sing, cmap='viridis', vmin=z_min, vmax=z_max, linewidths=0, label='Singularities')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (X-Y)')
    if joint_positions is not None:
        ax2.plot(joint_positions[:, 0], joint_positions[:, 1], color='black', linewidth=2, marker='o', markersize=4)
    ax2.axis('equal')
    ax2.grid(True)
    fig2.colorbar(scatter_sing2, ax=ax2, label='Z (Height)')
    
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111)
    scatter_sing3 = ax3.scatter(x_sing, z_sing, s=10, alpha=0.6, c=z_sing, cmap='viridis', vmin=z_min, vmax=z_max, linewidths=0, label='Singularities')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (X-Z)')
    if joint_positions is not None:
        ax3.plot(joint_positions[:, 0], joint_positions[:, 2], color='black', linewidth=2, marker='o', markersize=4)
    ax3.axis('equal')
    ax3.grid(True)
    fig3.colorbar(scatter_sing3, ax=ax3, label='Z (Height)')
    
    fig4 = plt.figure(figsize=(8, 6))
    ax4 = fig4.add_subplot(111, projection='3d')
    scatter_work4 = ax4.scatter(x_work, y_work, z_work, s=10, alpha=0.6, c=z_work, cmap='plasma', vmin=z_min, vmax=z_max, linewidths=0, label='Valid Workspace')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_zlabel('Z (m)')
    ax4.set_title('3D Valid Workspace')
    if joint_positions is not None:
        ax4.plot(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2], color='black', linewidth=2, marker='o', markersize=4)
    ax4.legend()
    fig4.colorbar(scatter_work4, ax=ax4, label='Z (Height)')
    
    all_x = np.concatenate([x_work, x_sing])
    all_y = np.concatenate([y_work, y_sing])
    
    if len(all_x) > 0:
        max_range = np.array([all_x.max() - all_x.min(), all_y.max() - all_y.min(), all_z.max() - all_z.min()]).max() / 2.0
        mid_x = (all_x.max() + all_x.min()) * 0.5
        mid_y = (all_y.max() + all_y.min()) * 0.5
        mid_z = (all_z.max() + all_z.min()) * 0.5
        for ax in [ax1, ax4]:
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
    plt.show()

if __name__ == '__main__':
    plot_workspace_singularity(samples=50000, sing_threshold=0.015)
