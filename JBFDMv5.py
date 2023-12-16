import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_square_grid(length, width, spacing):
    i = np.arange(0, length + spacing, spacing)
    alpha = np.arange(0, width + spacing, spacing)
    grid_i, grid_alpha = np.meshgrid(i, alpha)
    return grid_i, grid_alpha

def calculate_coefficients(epsilon, phi, epsilon_dot, phi_dot, omega, theta, spacing):
    H = 1 - epsilon * np.cos(theta - phi)
    Hip=1 - epsilon * np.cos(theta+0.5*spacing - phi)
    Him=1 - epsilon * np.cos(theta-0.5*spacing - phi)
    A = (Hip[:-1] ** 3)
    B = (H[:-1] ** 3)
    D = (Him[1:] ** 3)
    C = -(A + D + 2 * B)
    E = B
    F = spacing ** 2 * (epsilon * np.sin(theta - phi) - (2/omega) * (epsilon * phi_dot * np.sin(theta - phi) + epsilon_dot * np.cos(theta - phi)))
    return A, B, C, D, E, F, H, Hip, Him

def conventional_FDM_iteration_with_boundary_conditions(P, A, B, C, D, E, F, max_iterations, relaxation_coefficient, min_relative_error):
    for iteration in range(max_iterations):
        P_previous = P.copy()
        for i in range(1, len(P)-1):
            for j in range(1, len(P[i])-1):
                P[i, j] = (F[i, j] - A[i, j] * P[i-1, j] - B[i, j-1] * P[i, j-1] - D[i, j] * P[i+1, j] - E[i, j+1] * P[i, j+1]) / C[i, j]

        # Apply boundary conditions
        P[:, 0] = 0  # ¯P_(1…N,0)=0
        P[0, 1:-1] = 0  # ¯P_(1, 1…M)=0
        P[:, -1] = 0  # ¯P_(1…N, M+1)=0

        # Apply relaxation coefficients
        P = relaxation_coefficient * P + (1-relaxation_coefficient) * P_previous

        # Check for convergence
        relative_error = np.abs((P - P_previous) / np.maximum(P_previous, 1e-10))  # Avoid divide by zero
        relative_error = np.nan_to_num(relative_error)  # Replace NaN with zero

        if np.max(relative_error) < min_relative_error:
            break

    return P

def simulate_pressure_distribution(epsilon, phi, epsilon_dot, phi_dot, omega, Ld, spacing,max_iterations, relaxation_coefficient, min_relative_error):
    # Create square grid
    spacing_i=spacing
    spacing_alpha=spacing
    theta, z = create_square_grid(2 * np.pi, 2*Ld, spacing_i)

    # Calculate coefficients and simulate pressure distribution
    A, B, C, D, E, F, H, Hip, Him= calculate_coefficients(epsilon, phi, epsilon_dot, phi_dot, omega, theta, spacing_alpha)
    P_initial = np.zeros_like(theta)  # Initial guess for pressure
    P_solution = conventional_FDM_iteration_with_boundary_conditions(
        P_initial, A, B, C, D, E, F, max_iterations, relaxation_coefficient, min_relative_error
    )
    P_solution_filtered = [[max(0, value) for value in row] for row in P_solution]
    P_solution_filtered = np.array(P_solution_filtered).reshape(theta.shape)
    return theta, z, P_solution, P_solution_filtered

def calculate_bearing_force(epsilon, phi, epsilon_dot, phi_dot, omega, Ld, Bearing_clearence, Bearing_radius, miu, Atmospheric_pressure, spacing, min_relative_error, relaxation_coefficient, max_iterations,plotflag):
  """
  Calculates the bearing force and Sommerfeld number for a journal bearing.

  Args:
    epsilon: Eccentricity ratio.
    phi: Phase angle.
    epsilon_dot: Rate of change of eccentricity ratio.
    phi_dot: Rate of change of phase angle.
    omega: Shaft rotational speed (rad/s).
    Ld: Bearing length to Diameter ratio.
    Bearing_clearence: Bearing clearance (m).
    Bearing_radius: Bearing radius (m).
    miu: Dynamic viscosity of the lubricant (Pa·s).
    Atmospheric_pressure: Atmospheric pressure (Pa).
    spacing: Dimensionless Grid spacing for pressure calculation.
    min_relative_error: Minimum relative error for pressure convergence.
    relaxation_coefficient: Relaxation coefficient for pressure calculation.
    max_iterations: Maximum number of iterations for pressure calculation.
    plotflag = 1 for ploting o for not ploting
  Returns:
    Bearing_number: Bearing number (dimensionless).
    Sm: Sommerfeld number (dimensionless).
    Bearing_force: Bearing force magnitude (N).
  """

  # Calculate bearing number
  Bearing_number = 6 * miu * omega * Bearing_radius * Bearing_radius / (Atmospheric_pressure * Bearing_clearence ** 2)

  # Simulate pressure distribution
  theta, z, P_solution, P_solution_filtered = simulate_pressure_distribution(epsilon, phi, epsilon_dot, phi_dot, omega, Ld, spacing, max_iterations, relaxation_coefficient, min_relative_error)

  # Calculate force components from pressure distribution
  F_x, F_y = 0, 0
  for i in range(len(P_solution_filtered)):
    for j in range(len(P_solution_filtered[0])):
      if P_solution_filtered[i][j] > 0:
        F_x += P_solution_filtered[i][j] * spacing ** 2 * np.cos(theta[i, j])
        F_y += P_solution_filtered[i][j] * spacing ** 2 * np.sin(theta[i, j])

  # Calculate bearing force magnitude
  Bearing_force= Atmospheric_pressure*Bearing_radius*Bearing_radius*Bearing_number*np.sqrt(F_x**2+F_y**2)

  # Calculate Sommerfeld number
  Sm = (0.5 / np.pi) * (((Bearing_radius - Bearing_clearence) / Bearing_clearence) ** 2) * miu * omega * Ld * 4 * Bearing_radius * Bearing_radius / Bearing_force
  attitude_angle=math.degrees(math.atan(-F_y/F_x))
  F_reference=(miu*omega*Bearing_radius*(Ld*2*Bearing_radius)**3)/(2*Bearing_clearence**2)
  Dimensionless_Bearing_force=Bearing_force/F_reference
  # Print calculated values
  print(f"Bearing Number = {Bearing_number}")
  print(f"Sommerfeld Number = {Sm}")
  print(f"Dimensionless Bearing Force = {Dimensionless_Bearing_force}")
  print(f"Reference Force = {F_reference} N")
  print(f"Bearing Force = {Bearing_force} N")
  print(f"Attitude angle = {attitude_angle} degrees")
  if plotflag==1:
    plot_pressure_distribution(theta, z, P_solution_filtered)
    
  # Return calculated values
  return Bearing_number, Sm, Bearing_force, Dimensionless_Bearing_force, F_reference, attitude_angle

def plot_pressure_distribution(theta, z, P_solution_filtered, title="Dimensionless excess pressure distribution", cmap="jet", aspect="auto"):
  """
  Creates a 3D surface plot of the dimensionless excess pressure distribution on a bearing surface.

  Args:
    theta: Array of angular coordinates (radians).
    z: Array of axial coordinates (normalized).
    P_solution_filtered: Array of dimensionless excess pressure values.
    title: Optional title for the plot (default: "Dimensionless excess pressure distribution").
    cmap: Optional colormap for the surface plot (default: "jet").
    aspect: Optional aspect ratio for the plot axes (default: "auto").

  Returns:
    None (the plot is displayed directly).
  """

  # Create a figure and 3D subplot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection="3d", **{'aspect': aspect})

  # Plot the surface
  surface = ax.plot_surface(theta, z, P_solution_filtered, cmap=cmap)

  # Add labels and title
  ax.set_xlabel("Theta")
  ax.set_ylabel("Z")
  ax.set_zlabel("Dimensionless excess pressure")
  ax.set_title(title)

  # Add a colorbar
  fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)

  # Show the plot
  plt.show()

# Example usage
epsilon = 0.5
phi = 0
epsilon_dot = 0
phi_dot = 0
omega = 2000*np.pi
Ld = 1
Bearing_clearence=0.001
Bearing_radius=0.026
miu=0.25
Atmospheric_pressure=101325
spacing = 0.05
min_relative_error=1e-4
relaxation_coefficient=0.3
max_iterations=3000

Bearing_number, Sm, Bearing_force, Dimensionless_Bearing_force, F_reference, attitude_angle = calculate_bearing_force(epsilon, phi, epsilon_dot, phi_dot, omega, Ld, Bearing_clearence, Bearing_radius, miu, Atmospheric_pressure, spacing, min_relative_error, relaxation_coefficient, max_iterations,0)

