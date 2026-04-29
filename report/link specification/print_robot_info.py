import numpy as np
import matplotlib.pyplot as plt
from CustomRobotClass import CustomRobot
import spatialmath.base.symbolic as sym

def print_robot_info():
    """
    Loads the custom robot class and prints its kinematics.
    Displays this information in formatted text tables and Matplotlib tables.
    """
    robot = CustomRobot()
    print(f"Robot loaded: {robot.name}")
    print("-" * 50)
    
    # --- TABLE 1: Joint Info ---
    cell_data_1 = []
    col_labels_1 = ["Link / Joint", "Type", "Limits (Min, Max)"]
    
    # --- TABLE 2: ETS Kinematics ---
    cell_data_2 = []
    col_labels_2 = ["Link / Joint", "ETS (Elementary Transform Sequence)"]
    
    for link in robot.links:
        # 1. Type
        if link.isrevolute:
            j_type = "Revolute"
        elif link.isprismatic:
            j_type = "Prismatic"
        else:
            j_type = "Fixed"
            
        # 2. Limits
        if link.qlim is not None and (link.isrevolute or link.isprismatic):
            q_min, q_max = link.qlim
            if link.isrevolute:
                limits = f"[{np.rad2deg(q_min):.2f}\u00b0, {np.rad2deg(q_max):.2f}\u00b0]"
            else:
                limits = f"[{q_min:.3f}m, {q_max:.3f}m]"
        else:
            limits = "None"
            
        # 3. Kinematics (ETS)
        if hasattr(link, 'ets'):
            ets_str = str(link.ets)
            if not ets_str:
                 ets_str = "Base/Fixed"
        else:
            ets_str = "N/A"
            
        name_str = f"{link.name}"
        
        # Add to Table 1
        cell_data_1.append([name_str, j_type, limits])
        
        # Add to Table 2
        # Break long ETS strings into multiple lines for the plot so they fit nicely
        wrap_width = 40
        if len(ets_str) > wrap_width:
            # Simple text wrap
            ets_wrap = "\n".join([ets_str[i:i+wrap_width] for i in range(0, len(ets_str), wrap_width)])
        else:
            ets_wrap = ets_str
        cell_data_2.append([name_str, ets_wrap])

    # --- Print to Console ---
    print("\n" + "{:<20} | {:<12} | {:<20}".format(*col_labels_1))
    print("-" * 58)
    for row in cell_data_1:
        print("{:<20} | {:<12} | {:<20}".format(*row))

    print("\n" + "{:<20} | {:<60}".format(*col_labels_2))
    print("-" * 85)
    for row in cell_data_2:
        # Remove newlines for console output to keep it on one line
        console_ets = row[1].replace("\n", "")
        print("{:<20} | {:<60}".format(row[0], console_ets))
    print("\n" + "-" * 85 + "\n")

    # --- Print Detailed ET Matrices (Numerical q=0 for quick visual) ---
    print("DETAILED ETS TRANSFORMATION MATRICES (Evaluated at q=0)")
    print("=" * 60)
    for link in robot.links:
        if not hasattr(link, 'ets') or not link.ets:
            continue
        
        print(f"\nLink: {link.name}")
        print("-" * 30)
        
        for i, et in enumerate(link.ets):
            # Evaluate the matrix. If it's a joint, use q=0
            matrix = et.A(0) if et.isjoint else et.A()
            
            # Formatting the matrix for clean printing
            matrix_str = np.array2string(matrix, precision=4, suppress_small=True, separator=', ')
            
            # Print ET index and its symbolic representation
            et_repr = str(et)
            print(f"  ET[{i}]: {et_repr}")
            
            # Indent the matrix lines for better structure
            indented_matrix = "      " + matrix_str.replace("\n", "\n      ")
            print(indented_matrix)
    print("\n" + "=" * 60 + "\n")

    # --- SYMBOLIC LINK MATRICES & LATEX EXPORT ---
    print("SYMBOLIC LINK-TO-LINK MATRICES")
    print("=" * 60)
    
    md_content = "# Robot General Symbolic Transformation Matrices\n\n"
    md_content += f"**Robot:** {robot.name}\n\n"
    md_content += "This file contains the general symbolic transformation matrices for each link, "
    md_content += "using joint variables $\\theta_i$. These matrices represent the kinematic model in its most general form.\n\n"

    def clean_symbolic_matrix(matrix):
        """Clean up numerical noise and round coefficients in symbolic matrices."""
        import sympy
        cleaned = np.copy(matrix)
        for i in range(cleaned.shape[0]):
            for j in range(cleaned.shape[1]):
                cell = cleaned[i, j]
                if isinstance(cell, (float, int, np.number)):
                    if abs(cell) < 1e-6:
                        cleaned[i, j] = 0
                    elif abs(cell - round(float(cell))) < 1e-6:
                        cleaned[i, j] = int(round(float(cell)))
                    else:
                        cleaned[i, j] = round(float(cell), 2)
                elif hasattr(cell, 'atoms'): # Sympy expression
                    expr = cell
                    # Round all numerical atoms
                    for a in expr.atoms(sympy.Number):
                        if isinstance(a, sympy.Float):
                            if abs(a) < 1e-6:
                                expr = expr.subs(a, 0)
                            elif abs(a - round(float(a))) < 1e-6:
                                expr = expr.subs(a, int(round(float(a))))
                            else:
                                expr = expr.subs(a, round(float(a), 2))
                    cleaned[i, j] = sympy.simplify(expr)
        return cleaned

    def matrix_to_latex_symbolic(matrix, label, sub=None, sup=None):
        if sub is not None and sup is not None:
            latex = f"## $T_{{{sub}}}^{{{sup}}}$ ({label})\n"
        else:
            latex = f"## {label}\n"
        
        latex += "$$T = \\begin{pmatrix} "
        rows = []
        for row in matrix:
            row_strs = []
            for val in row:
                if isinstance(val, (float, int, np.number)):
                    s = "{:.4f}".format(val).rstrip('0').rstrip('.')
                    if s == "" or s == "-": s = "0"
                else:
                    # Convert sympy object to latex string
                    s = sym.sympy.latex(val)
                row_strs.append(s)
            rows.append(" & ".join(row_strs))
        
        latex += " \\\\ ".join(rows) + " \\end{pmatrix}$$\n\n"
        return latex

    T_cumulative_sym = np.eye(4, dtype=object)
    link_count = 0
    joint_vars = []
    
    for i, link in enumerate(robot.links):
        # Determine if this link has a joint variable
        if link.isjoint:
            link_count += 1
            var_name = f"\\theta_{link_count}"
            q_var = sym.symbol(var_name)
            joint_vars.append(q_var)
            # Get symbolic matrix
            matrix = link.A(q_var)
        else:
            # Fixed link or base
            try:
                matrix = link.A(0)
            except:
                if hasattr(link, 'ets') and link.ets:
                    matrix = link.ets.eval([]) # No variables
                else:
                    continue

        # Convert to numpy array if it's an SE3/ET object
        if hasattr(matrix, 'A'):
            matrix = matrix.A
        
        # Clean numeric noise
        matrix = clean_symbolic_matrix(matrix)

        # Update cumulative transformation
        T_cumulative_sym = T_cumulative_sym @ matrix

        # Console Print (Brief symbolic repr)
        print(f"Link: {link.name} (Symbolic)")
        print(f"      [Symbolic matrix generated for {link.name}]")
        
        # LaTeX for Markdown
        if link.parent is None:
            md_content += matrix_to_latex_symbolic(matrix, f"Base: {link.name}")
        else:
            # Label as T_{current}^{parent}
            md_content += matrix_to_latex_symbolic(matrix, f"Link: {link.name}", sub=link_count, sup=link_count-1)

    # Final cumulative matrix
    md_content += "---\n\n"
    md_content += f"## General Final Transformation Matrix $T_{{{link_count}}}^{{0}}$\n"
    md_content += f"The following matrix represents the symbolic transformation from the base to the end-effector:\n"
    md_content += "$$T_{" + str(link_count) + "}^{0} = " + " ".join([f"T_{{{i}}}^{{{i-1}}}" for i in range(1, link_count + 1)]) + "$$\n\n"
    
    print("\nSimplifying final cumulative matrix (this may take a moment)...")
    T_final_clean = clean_symbolic_matrix(T_cumulative_sym)
    
    md_content += matrix_to_latex_symbolic(T_final_clean, "General Symbolic End-Effector Pose")

    # Save to Markdown
    with open("link_matrices.md", "w") as f:
        f.write(md_content)
    
    print(f"\n[Info] General symbolic matrices saved to 'link_matrices.md'")
    print("\n" + "=" * 60 + "\n")
    
    # --- Plotting Tables ---
    # Table 1: Joint Info
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    if hasattr(fig1.canvas.manager, 'set_window_title'):
        fig1.canvas.manager.set_window_title(f'Robot Info - Link/Joint Specification: {robot.name}')
        
    ax1.axis('tight')
    ax1.axis('off')
    table1 = ax1.table(cellText=cell_data_1, colLabels=col_labels_1, loc='center', cellLoc='center')
    table1.scale(1, 1.8)
    table1.auto_set_font_size(False)
    table1.set_fontsize(11)
    
    for (row, col), cell in table1.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#d3d3d3')
        elif row % 2 == 0:
            cell.set_facecolor('#f2f2f2')
            
    ax1.set_title(f'Link/Joint Specification', fontweight='bold', fontsize=14)
    fig1.tight_layout()
    
    # Table 2: ETS Info
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    if hasattr(fig2.canvas.manager, 'set_window_title'):
        fig2.canvas.manager.set_window_title(f'Robot Info - ETS: {robot.name}')

    ax2.axis('tight')
    ax2.axis('off')
    table2 = ax2.table(cellText=cell_data_2, colLabels=col_labels_2, loc='center', cellLoc='center')
    table2.scale(1, 3.5) # Make cells taller for wrapped text
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    
    for (row, col), cell in table2.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#d3d3d3')
        elif row % 2 == 0:
            cell.set_facecolor('#f2f2f2')
            
        # Left align the ETS column for better readability
        if col == 1 and row > 0:
            cell.set_text_props(ha='left')
            
    ax2.set_title(f'Elementary Transform Sequences (ETS)', fontweight='bold', fontsize=14)
    fig2.tight_layout()
    
    plt.show()

if __name__ == '__main__':
    print_robot_info()
