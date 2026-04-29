import numpy as np
import matplotlib.pyplot as plt
from CustomRobotClass import CustomRobot

JOINTS = 5

def plot_workspace_singularity(samples=50000, sing_threshold=0.015):
    """
    Plots the workspace boundaries using a singularity-based approach.
    Samples joint configurations, calculates the Jacobian, and identifies
    singular configurations (where the rank drops).
    These singular configurations typically define the workspace boundaries
    and internal singular surfaces.
    """
    robot = CustomRobot()
    print(f"Robot loaded: {robot.name}")
    
    # Attempt to get joint limits from the robot model
    try:
        qlim = robot.qlim
        if qlim is None or np.all(qlim == 0):
            print(f"Using default joint limits (-pi to pi) for {JOINTS} joints")
            qlim = np.array([[-np.pi] * JOINTS, [np.pi] * JOINTS])
        else:
            print("Using joint limits from URDF model")
    except AttributeError:
        print(f"Using default joint limits (-pi to pi) for {JOINTS} joints")
        qlim = np.array([[-np.pi] * JOINTS, [np.pi] * JOINTS])
    
    print(f"Generating {samples} random joint configurations...")
    # Generate random joint angles
    q_random = np.random.uniform(qlim[0], qlim[1], (samples, JOINTS))
    
    print("Calculating kinematics and Jacobians... This may take a moment.")
    
    # Lists to store points
    x_sing = []
    y_sing = []
    z_sing = []
    
    x_work = []
    y_work = []
    z_work = []
    
    for i, q in enumerate(q_random):
        # Calculate forward kinematics
        pose = robot.fkine(q).A
        px, py, pz = pose[0, 3], pose[1, 3], pose[2, 3]
        
        # 1. Find Jacobian matrix (in base frame)
        J = robot.jacob0(q)
        
        # 2. Determine singularity by checking the J(q) rank or singular values
        # Since it's a 5-DOF robot, J is a 6x5 matrix. The maximum rank is 5.
        # We check the singular values (SVD) to find configurations near singularities.
        s = np.linalg.svd(J, compute_uv=False)
        min_sv = np.min(s)
        
        # If the minimum singular value is below the threshold, the rank effectively drops (< 5)
        if min_sv < sing_threshold:
            x_sing.append(px)
            y_sing.append(py)
            z_sing.append(pz)
        else:
            # 3. Supposing robot can work everywhere inside workspace except singularities
            # We sample these to plot the "interior" valid workspace.
            # Downsample the interior points slightly to avoid cluttering the plot
            if np.random.random() < 0.1: # Keep 10% of non-singular points
                x_work.append(px)
                y_work.append(py)
                z_work.append(pz)
                
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1}/{samples} configurations...")

    print(f"Found {len(x_sing)} singular configurations (Boundaries / Internal Singularities).")
    print(f"Sampled {len(x_work)} non-singular configurations (Interior Workspace).")

    print("Plotting workspace...")
    
    all_z = np.concatenate([z_work, z_sing]) if len(z_work) > 0 or len(z_sing) > 0 else [0, 1]
    z_min, z_max = np.min(all_z), np.max(all_z)

    # Figure 1: 3D Plot (Singularities only)
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    # Plot singular points (boundaries)
    scatter_sing1 = ax1.scatter(x_sing, y_sing, z_sing, s=15, alpha=0.6, c=z_sing, cmap='viridis', vmin=z_min, vmax=z_max, linewidths=0, label='Singularities')
    
    ax1.scatter([0], [0], [0], color='black', s=100, marker='*', label='Base Origin')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Singularity Workspace (Singularities)')
    
    # Plot robot links for a default configuration
    q_zero = np.zeros(JOINTS)
    try:
        poses = robot.fkine_all(q_zero)
        if isinstance(poses, list):
            joint_positions = np.array([p.t for p in poses])
        else:
            joint_positions = poses.t
        joint_positions = np.vstack([[0, 0, 0], joint_positions])
        ax1.plot(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2], color='black', linewidth=2, marker='o', markersize=4, label='Robot Links (q=0)')
    except Exception as e:
        print(f"Could not plot robot links in 3D: {e}")
    ax1.legend()
    fig1.colorbar(scatter_sing1, ax=ax1, label='Z (Height)')

    # Figure 2: Top View (X-Y Plane) (Singularities only)
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    scatter_sing2 = ax2.scatter(x_sing, y_sing, s=10, alpha=0.6, c=z_sing, cmap='viridis', vmin=z_min, vmax=z_max, linewidths=0, label='Singularities')
    ax2.scatter([0], [0], color='black', s=100, marker='*', label='Base Origin')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Singularity Top View (X-Y) (Singularities)')
    try:
        ax2.plot(joint_positions[:, 0], joint_positions[:, 1], color='black', linewidth=2, marker='o', markersize=4, label='Robot Links (q=0)')
    except NameError:
        pass
    ax2.axis('equal')
    ax2.grid(True)
    ax2.legend()
    fig2.colorbar(scatter_sing2, ax=ax2, label='Z (Height)')
    
    # Figure 3: Side View (X-Z Plane) (Singularities only)
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111)
    scatter_sing3 = ax3.scatter(x_sing, z_sing, s=10, alpha=0.6, c=z_sing, cmap='viridis', vmin=z_min, vmax=z_max, linewidths=0, label='Singularities')
    ax3.scatter([0], [0], color='black', s=100, marker='*', label='Base Origin')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Singularity Side View (X-Z) (Singularities)')
    try:
        ax3.plot(joint_positions[:, 0], joint_positions[:, 2], color='black', linewidth=2, marker='o', markersize=4, label='Robot Links (q=0)')
    except NameError:
        pass
    ax3.axis('equal')
    ax3.grid(True)
    ax3.legend()
    fig3.colorbar(scatter_sing3, ax=ax3, label='Z (Height)')
    
    # Figure 4: 3D Plot (Valid Workspace Points Only)
    fig4 = plt.figure(figsize=(8, 6))
    ax4 = fig4.add_subplot(111, projection='3d')
    scatter_work4 = ax4.scatter(x_work, y_work, z_work, s=10, alpha=0.6, c=z_work, cmap='plasma', vmin=z_min, vmax=z_max, linewidths=0, label='Valid Workspace')
    
    ax4.scatter([0], [0], [0], color='black', s=100, marker='*', label='Base Origin')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_zlabel('Z (m)')
    ax4.set_title('3D Valid Workspace (Interior)')
    
    try:
        ax4.plot(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2], color='black', linewidth=2, marker='o', markersize=4, label='Robot Links (q=0)')
    except NameError:
        pass
    ax4.legend()
    fig4.colorbar(scatter_work4, ax=ax4, label='Z (Height)')
    
    # Match axes scaling for 3D plots
    all_x = np.concatenate([x_work, x_sing])
    all_y = np.concatenate([y_work, y_sing])
    
    if len(all_x) > 0:
        max_range = np.array([all_x.max() - all_x.min(), 
                              all_y.max() - all_y.min(), 
                              all_z.max() - all_z.min()]).max() / 2.0
        mid_x = (all_x.max() + all_x.min()) * 0.5
        mid_y = (all_y.max() + all_y.min()) * 0.5
        mid_z = (all_z.max() + all_z.min()) * 0.5
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)

        # Apply same scaling to the valid workspace plot
        ax4.set_xlim(mid_x - max_range, mid_x + max_range)
        ax4.set_ylim(mid_y - max_range, mid_y + max_range)
        ax4.set_zlim(mid_z - max_range, mid_z + max_range)
        
    plt.show()

if __name__ == '__main__':
    plot_workspace_singularity(samples=50000, sing_threshold=0.015)
