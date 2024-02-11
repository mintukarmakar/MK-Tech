import numpy as np
import math
import time
from scipy.optimize import fmin
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

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
    # Apply initiation
    P[:, :]=0.5
    # Apply boundary conditions
    P[0,:] = 0  # Set first row to 0
    P[-1,:] = 0  # Set last row to 0
    P[:, 0] = 0  # Set first column to 0
    P[:, -1] = 0  # Set last column to 0    
    for iteration in range(max_iterations):
        P_previous = P.copy()
        for i in range(1, len(P)-1):
            for j in range(1, len(P[i])-1):
                P[i, j] = (F[i, j] - A[i, j] * P[i-1, j] - B[i, j-1] * P[i, j-1] - D[i, j] * P[i+1, j] - E[i, j+1] * P[i, j+1]) / C[i, j]

        # Apply boundary conditions
        P[0,:] = 0  # Set first row to 0
        P[-1,:] = 0  # Set last row to 0
        P[:, 0] = 0  # Set first column to 0
        P[:, -1] = 0  # Set last column to 0
        # Apply relaxation coefficients
        P = relaxation_coefficient * P + (1-relaxation_coefficient) * P_previous

        # Check for convergence
        relative_error = np.where(np.logical_or(np.abs(P) < 0.00000001, np.abs(P_previous) < 0.00000001),
                         0,
                         ((P - P_previous) / np.maximum(np.abs(np.maximum(P_previous, P)), 0.0001))).astype(np.float64)  # Cast to float64  to Avoid divide by zero
        relative_error = np.nan_to_num(relative_error)  # Replace NaN with zero
        #relative_error = np.abs((P - P_previous) / np.maximum(P_previous, P)).astype(np.float64)  # Cast to float64
        rmse_relative_error = np.sqrt(np.mean(relative_error ** 2))
        
        max_value = np.max(np.abs(relative_error))
        if max_value > 1:
            relative_error /= max_value  # Scale to values between -1 and 1
            rmse_relative_error = np.sqrt(np.mean(relative_error ** 2)) * max_value  # Rescale RMSE

        #if np.max(relative_error) < min_relative_error:
        #    break
        if rmse_relative_error < min_relative_error:
            break    
    return P, iteration, rmse_relative_error

def simulate_pressure_distribution(epsilon, phi, epsilon_dot, phi_dot, omega, Ld, spacing,max_iterations, relaxation_coefficient, min_relative_error):
    # Create square grid
    spacing_i=spacing
    spacing_alpha=spacing
    theta, z = create_square_grid(2 * np.pi, 2*Ld, spacing_i)

    # Calculate coefficients and simulate pressure distribution
    A, B, C, D, E, F, H, Hip, Him= calculate_coefficients(epsilon, phi, epsilon_dot, phi_dot, omega, theta, spacing_alpha)
    P_initial = np.zeros_like(theta)  # Initial guess for pressure
    P_solution, iteration, rmse_relative_error = conventional_FDM_iteration_with_boundary_conditions(
        P_initial, A, B, C, D, E, F, max_iterations, relaxation_coefficient, min_relative_error
    )
    P_solution_filtered = [[max(0, value) for value in row] for row in P_solution] # Considering half sommerfield condition
    P_solution_filtered = np.array(P_solution_filtered).reshape(theta.shape)
    return theta, z, P_solution, P_solution_filtered,iteration, rmse_relative_error

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
  theta, z, P_solution, P_solution_filtered,iteration, rmse_relative_error = simulate_pressure_distribution(epsilon, phi, epsilon_dot, phi_dot, omega, Ld, spacing, max_iterations, relaxation_coefficient, min_relative_error)

  # Calculate force components from pressure distribution
  F_x, F_y = 0, 0
  for i in range(theta.shape[0]):
    for j in range(theta.shape[1]):
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
  print(f"Iteration completed = {iteration} Numbers")
  print(f"RMSE of Relative error of pressure estimation = {rmse_relative_error}")
  
  if plotflag==1:
    plot_pressure_distribution(theta, z, P_solution_filtered)
    
  # Return calculated values
  return Bearing_number, Sm, Bearing_force, Dimensionless_Bearing_force, F_reference, attitude_angle, theta, z, P_solution, P_solution_filtered,iteration, rmse_relative_error

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

def error_correction(absolute_error, relative_error, total_time_resource, data, function_to_correct, *args, **kwargs):
    # 1. Initialize variables
    error = data+absolute_error*(1+np.abs(relative_error))  # Measure initial error
    args_list = list(args)  # Create a mutable copy
    newspace=args_list[10]
    newiteration=args_list[13]
    start_time = time.time()
    # 2. Iteration loop
    while np.any(error > absolute_error) or np.any(error / data > relative_error):
        
        if time.time() - start_time > 1+total_time_resource:
            plot_pressure_distribution(theta, z, P_solution_filtered, title="Dimensionless excess pressure distribution", cmap="jet", aspect="auto")
            raise TimeoutError("Exceeded computation time limit")
        else: 
            # 3. Perform error correction step
            newspace = 1.0 * newspace  # Modify the 10th element
            newiteration = int(1.5 * newiteration)  # Modify the 13th element
            args_list[10]=newspace
            args_list[13]=newiteration
            Bearing_number, Sm, Bearing_force, Dimensionless_Bearing_force, F_reference, attitude_angle, theta, z, P_solution, P_solution_filtered,iteration, rmse_relative_error = correction_step(data, function_to_correct, *args_list, **kwargs)
            corrected_data=Bearing_force
            # 4. Update error and data
            error = calculate_error(corrected_data, data)  # Assuming a present data is reference exists
            data = corrected_data
            print(f"Absolute error of Force = {error} N")
            print(f"Relative error of Force = {error / data}") 
            print(f"Time lapsed for error correction = {(time.time() - start_time)}")
    # 5. Return corrected data
    return Bearing_number, Sm, Bearing_force, Dimensionless_Bearing_force, F_reference, attitude_angle, theta, z, P_solution, P_solution_filtered,iteration, rmse_relative_error
    
def correction_step(data, function_to_correct, *args, **kwargs):
    """Applies the given correction function to the data."""
    Bearing_number, Sm, Bearing_force, Dimensionless_Bearing_force, F_reference, attitude_angle, theta, z, P_solution, P_solution_filtered,iteration, rmse_relative_error = function_to_correct(*args, **kwargs)
    #corrected_data=Bearing_force
    return Bearing_number, Sm, Bearing_force, Dimensionless_Bearing_force, F_reference, attitude_angle, theta, z, P_solution, P_solution_filtered,iteration, rmse_relative_error
    
def calculate_error(corrected_data, reference_data):
    """ Error calculation logic here."""
    corrected_data = np.array(corrected_data)
    reference_data = np.array(reference_data)
    return np.abs(corrected_data - reference_data)


# Example usage also validated with http://dx.doi.org/10.1016/j.triboint.2012.08.011

epsilon = 0.5
phi = 0
epsilon_dot = 0
phi_dot = 0
omega = 2000*np.pi
Ld = 1
Bearing_clearence=0.001
Bearing_radius=0.026
miu=0.025
Atmospheric_pressure=101325
spacing = Ld/20
min_relative_error=1e-4
relaxation_coefficient=0.3
max_iterations=3000

absolute_error_force=0.1
relative_error_force=0.01
total_time_resource=60
flag=0
Bearing_force0=300

def my_function(x, Bearing_force0, phi, epsilon_dot, phi_dot, omega, Ld, Bearing_clearence, Bearing_radius, miu, Atmospheric_pressure, spacing, min_relative_error, relaxation_coefficient, max_iterations, flag):
    # Your function to minimize here
    Bearing_number, Sm, Bearing_force, Dimensionless_Bearing_force, F_reference, attitude_angle, theta, z, P_solution, P_solution_filtered, iteration, rmse_relative_error = calculate_bearing_force(x, phi, epsilon_dot, phi_dot, omega, Ld, Bearing_clearence, Bearing_radius, miu, Atmospheric_pressure, spacing, min_relative_error, relaxation_coefficient, max_iterations, flag)

    return np.abs(Bearing_force0 - Bearing_force)

# Initial guess for the minimum
x0 = np.array([0.5])

# Find the minimum
x_min= fmin(my_function, x0, args=(Bearing_force0, phi, epsilon_dot, phi_dot, omega, Ld, Bearing_clearence, Bearing_radius, miu, Atmospheric_pressure, spacing, min_relative_error, relaxation_coefficient, max_iterations, flag),xtol=0.02, ftol=1, maxiter=None, maxfun=None, full_output=0, disp=1, retall=0, callback=None, initial_simplex=None)

print("Eccentricity found at:", x_min)
