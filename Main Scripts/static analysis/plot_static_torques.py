import numpy as np
import matplotlib.pyplot as plt
import spatialmath as sm
from CustomRobotClass import CustomRobot

def plot_gravity_torques():
    robot = CustomRobot()
    print(f"Robot: {robot.name}")

    cell_data = []
    row_labels = []
    col_labels = [f"Joint {i}" for i in range(1, robot.n + 1)]
    
    print(f"\n{'Config':<10} | " + " | ".join([f"{col:<8}" for col in col_labels]))
    print("-" * 75)

    for config_name, q_target in robot.configs.items():
        tau_manual = np.zeros(robot.n)
        g = 9.81

        for link in robot.links:
            if link.m == 0:
                continue
            
            W_gravity = np.array([0, 0, -link.m * g, 0, 0, 0])
            J_com = robot.jacob0(q_target, end=link.name, tool=sm.SE3(link.r))
            tau_link = -J_com.T @ W_gravity
            
            num_joints = len(tau_link)
            tau_manual[:num_joints] += tau_link

        tau_manual = np.abs(tau_manual)
        row_data = [f"{val:.3f}" for val in tau_manual]
        cell_data.append(row_data)
        row_labels.append(config_name)
        
        print(f"{config_name:<10} | " + " | ".join([f"{val:<8}" for col, val in zip(col_labels, row_data)]))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=cell_data,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     loc='center',
                     cellLoc='center')
    
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    
    ax.set_title(f'Static Holding Torques for {robot.name} (Nm)\n', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_gravity_torques()
