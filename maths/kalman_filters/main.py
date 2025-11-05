import numpy as np

from .plots import plot_trajectory_data
from .kalman_1d import kalman_filter_1d


def run_1d_example():
    """Runs a 1D Kalman filter example and plots the results."""
    np.random.seed(42) # For reproducibility
    n_timesteps = 50
    dt = 1.0

    # True states (e.g., a car moving with almost constant velocity)
    true_velocity = 2.0
    true_initial_position = 0.0
    true_positions = np.zeros(n_timesteps)
    true_positions[0] = true_initial_position
    for i in range(1, n_timesteps):
        true_positions[i] = true_positions[i-1] + true_velocity * dt
    
    # Simulate some process noise (e.g. slight changes in velocity)
    process_noise_std = 0.1
    true_positions_noisy = true_positions + np.random.randn(n_timesteps) * process_noise_std * (np.arange(n_timesteps) * dt) # noise effect accumulates

    # Measurements (noisy observations of position)
    measurement_noise_std = 1.5
    measurements = true_positions_noisy + np.random.randn(n_timesteps) * measurement_noise_std

    # Kalman Filter parameters
    R_val = measurement_noise_std**2  # Measurement noise variance
    Q_val = 0.01 # Process noise variance parameter (tune this)
                 # This Q_val is used to construct the Q matrix. 
                 # A smaller Q_val means we trust our model more.
                 # A larger Q_val means we think the underlying system is more unpredictable.

    # Run Kalman Filter
    estimated_states_kf, _ = kalman_filter_1d(measurements, R_val=R_val, Q_val=Q_val, dt=dt)

    # Plotting
    # We are interested in the estimated position, which is the first component of the state vector
    estimated_positions_kf = estimated_states_kf[:, 0]
    
    plot_trajectory_data(true_positions_noisy, measurements, estimated_positions_kf, title="1D Kalman Filter (Position)")





def run_2d_example():
    """Runs a 2D Kalman filter example for trajectory prediction and plots the results."""
    
    np.random.seed(np.random.randint(1000000))
    n_timesteps = 100
    dt = 0.5 # Time step

    # True trajectory (e.g., a vehicle moving in a curve with some acceleration)
    true_states = np.zeros((n_timesteps, 4)) # [x, y, vx, vy]
    true_states[0] = [0, 0, 2, 2] # Initial state [x, y, vx, vy]

    # Simulate a curved path with some changing acceleration
    ax_profile = np.sin(np.linspace(0, 8*np.pi, n_timesteps)) * 0.2
    ay_profile = np.cos(np.linspace(0, np.pi, n_timesteps)) * 0.1 - 0.05

    for k in range(1, n_timesteps):
        # Update velocity with acceleration
        true_states[k, 2] = true_states[k-1, 2] + ax_profile[k-1] * dt
        true_states[k, 3] = true_states[k-1, 3] + ay_profile[k-1] * dt
        # Update position with new velocity
        true_states[k, 0] = true_states[k-1, 0] + true_states[k-1, 2] * dt + 0.5 * ax_profile[k-1] * dt**2
        true_states[k, 1] = true_states[k-1, 1] + true_states[k-1, 3] * dt + 0.5 * ay_profile[k-1] * dt**2

    # Add some process noise to the true trajectory (optional, Q handles this in filter)
    process_noise_pos_std = 0.05
    true_states[:, 0] += np.random.randn(n_timesteps) * process_noise_pos_std
    true_states[:, 1] += np.random.randn(n_timesteps) * process_noise_pos_std

    # Measurements (noisy observations of x and y positions)
    measurement_noise_std_x = 4.8
    measurement_noise_std_y = 4.8
    measurements = np.zeros((n_timesteps, 2))
    measurements[:, 0] = true_states[:, 0] + np.random.randn(n_timesteps) * measurement_noise_std_x
    measurements[:, 1] = true_states[:, 1] + np.random.randn(n_timesteps) * measurement_noise_std_y

    # Kalman Filter parameters
    R_diag_vals = [measurement_noise_std_x**2, measurement_noise_std_y**2]
    process_noise_std = 0.1 # Standard deviation of acceleration noise (sigma_a)
                             # Tune this: higher means more uncertainty in motion model

    # Run Kalman Filter
    estimated_states_kf, _ = kalman_filter_2d(
        measurements,
        R_diag_vals=R_diag_vals,
        process_noise_std=process_noise_std,
        dt=dt
    )

    # Plotting
    # plot_trajectory_data expects true_states to be (n_timesteps, 2) if only plotting x,y for true
    # or (n_timesteps, 4) and it will use the first two columns.
    plot_trajectory_data(true_states, measurements, estimated_states_kf, title="2D Kalman Filter (Trajectory)")

def test_matplotlib():
    """A simple test function to check if Matplotlib is working."""
    try:
        print("Testing Matplotlib basic functionality...")
        plt.figure()
        plt.plot([1, 2, 3], [1, 4, 9])
        plt.title("Simple Matplotlib Test")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        # Since 'Agg' is non-interactive, we save the figure instead of showing it.
        test_plot_filename = "matplotlib_test_plot.png"
        plt.savefig(test_plot_filename)
        print(f"Matplotlib test plot saved to {test_plot_filename}")
        plt.close() # Close the figure
        print("Matplotlib basic functionality test successful.")
        return True
    except Exception as e:
        print(f"Matplotlib basic functionality test FAILED: {e}")
        return False

# --- Helper Functions ---
def normalize_angle(angle):
    """Normalize an angle to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

# --- 2D Extended Kalman Filter (Coordinated Turn Model) ---
def extended_kalman_filter_2d(measurements, R_diag_vals, process_noise_accel_std, process_noise_yaw_accel_std, dt=1.0, initial_x_hat=None, initial_P=None):
    """
    Implements a 2D Extended Kalman Filter for vehicle trajectory using a Coordinated Turn (CT) model.
    State vector: x_hat = [x_pos, y_pos, speed, yaw (psi), yaw_rate (omega)]^T

    Args:
        measurements (np.ndarray): Array of measurements, shape (n_timesteps, 2) for [x_meas, y_meas].
        R_diag_vals (list or np.ndarray): Diagonal values for the measurement noise covariance R.
                                          Example: [std_x_meas**2, std_y_meas**2].
        process_noise_accel_std (float): Standard deviation of the vehicle's linear acceleration noise.
        process_noise_yaw_accel_std (float): Standard deviation of the vehicle's yaw acceleration noise.
        dt (float): Time step.
        initial_x_hat (np.ndarray, optional): Initial state estimate [x, y, v, psi, omega].
                                              Defaults to [measurements[0,0], measurements[0,1], 0, 0, 0].
        initial_P (np.ndarray, optional): Initial error covariance matrix (5x5).
                                          Defaults to np.eye(5).

    Returns:
        tuple: (estimated_states, estimated_covariances)
            estimated_states (np.ndarray): Array of estimated states [x, y, v, psi, omega] for each time step.
            estimated_covariances (np.ndarray): Array of error covariance matrices (5x5) for each time step.
    """
    n_timesteps = measurements.shape[0]
    n_states = 5  # [x, y, v, psi, omega]

    x_hat = np.zeros((n_timesteps, n_states))
    P = np.zeros((n_timesteps, n_states, n_states))

    # Measurement matrix H (linear, measures x and y directly)
    # z_k = H * x_k (where x_k contains [x,y,...])
    # [x_meas] = [1 0 0 0 0] * [x_pos_k]
    # [y_meas]   [0 1 0 0 0]   [y_pos_k]
    #                            [speed_k]
    #                            [yaw_k  ]
    #                            [yaw_rate_k]
    H = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0]])

    # Measurement noise covariance R
    R = np.diag(R_diag_vals)

    # Process noise covariance Q
    # Q = E[w_k * w_k^T] where w_k is process noise.
    # We model noise in linear acceleration (affecting speed) and yaw acceleration (affecting yaw and yaw_rate).
    # Let sigma_a = process_noise_accel_std
    # Let sigma_yaw_dd = process_noise_yaw_accel_std (yaw double dot / yaw acceleration)
    q_pos_eps = (0.01 * dt)**2 # Small noise for numerical stability / model imperfection for position
    
    Q = np.diag([
        q_pos_eps,  # variance for x
        q_pos_eps,  # variance for y
        (process_noise_accel_std * dt)**2, # variance for speed (v_k = v_{k-1} + a*dt)
        (process_noise_yaw_accel_std * dt**3) / 3.0, # variance for yaw (from yaw accel)
        (process_noise_yaw_accel_std * dt)**2  # variance for yaw_rate (omega_k = omega_{k-1} + yaw_accel*dt)
    ])
    # Off-diagonal terms for yaw and yaw_rate coupling from yaw_accel noise
    Q[3,4] = (process_noise_yaw_accel_std**2 * dt**2) / 2.0
    Q[4,3] = (process_noise_yaw_accel_std**2 * dt**2) / 2.0
    
    # Initial state estimate
    if initial_x_hat is None:
        # Estimate initial yaw based on first two measurements if possible (requires at least 2 points)
        initial_yaw = 0.0
        if n_timesteps > 1:
            dx = measurements[1,0] - measurements[0,0]
            dy = measurements[1,1] - measurements[0,1]
            if abs(dx) > 1e-6 or abs(dy) > 1e-6: # Avoid atan2(0,0)
                 initial_yaw = np.arctan2(dy, dx)
        x_hat[0] = np.array([measurements[0, 0], measurements[0, 1], 0, initial_yaw, 0]) # x, y, v, psi, omega
    else:
        x_hat[0] = initial_x_hat
    x_hat[0,3] = normalize_angle(x_hat[0,3])

    # Initial error covariance
    if initial_P is None:
        P[0] = np.eye(n_states) * 1.0 
        P[0,2,2] = 10.0 # Higher uncertainty for initial speed
        P[0,3,3] = (np.pi/4)**2 # Higher uncertainty for initial yaw
        P[0,4,4] = (np.pi/8)**2 # Higher uncertainty for initial yaw rate
    else:
        P[0] = initial_P

    # EKF loop
    for k in range(1, n_timesteps):
        # --- Prediction Step ---
        # Previous state
        x_prev = x_hat[k-1, 0]
        y_prev = x_hat[k-1, 1]
        v_prev = x_hat[k-1, 2]
        psi_prev = x_hat[k-1, 3]
        omega_prev = x_hat[k-1, 4]

        # State transition function f(x_hat_{k-1|k-1})
        x_pred_k = np.zeros(n_states)
        F_k = np.zeros((n_states, n_states))

        # Handle division by zero if omega_prev is very small
        if abs(omega_prev) < 1e-5:
            # Straight line motion
            x_pred_k[0] = x_prev + v_prev * np.cos(psi_prev) * dt
            x_pred_k[1] = y_prev + v_prev * np.sin(psi_prev) * dt
            # Jacobian F_k for straight line
            F_k[0,0] = 1; F_k[0,2] = np.cos(psi_prev)*dt; F_k[0,3] = -v_prev*np.sin(psi_prev)*dt;
            F_k[1,1] = 1; F_k[1,2] = np.sin(psi_prev)*dt; F_k[1,3] =  v_prev*np.cos(psi_prev)*dt;
        else:
            # Turning motion
            factor = v_prev / omega_prev
            x_pred_k[0] = x_prev + factor * (np.sin(psi_prev + omega_prev * dt) - np.sin(psi_prev))
            x_pred_k[1] = y_prev + factor * (-np.cos(psi_prev + omega_prev * dt) + np.cos(psi_prev))
            # Jacobian F_k for turning motion
            F_k[0,0] = 1
            F_k[0,2] = (np.sin(psi_prev + omega_prev*dt) - np.sin(psi_prev)) / omega_prev
            F_k[0,3] = factor * (np.cos(psi_prev + omega_prev*dt) - np.cos(psi_prev))
            F_k[0,4] = -factor/omega_prev * (np.sin(psi_prev + omega_prev*dt) - np.sin(psi_prev)) + \
                         factor * (np.cos(psi_prev + omega_prev*dt) * dt)
            F_k[1,1] = 1
            F_k[1,2] = (-np.cos(psi_prev + omega_prev*dt) + np.cos(psi_prev)) / omega_prev
            F_k[1,3] = factor * (np.sin(psi_prev + omega_prev*dt) - np.sin(psi_prev))
            F_k[1,4] = -factor/omega_prev * (-np.cos(psi_prev + omega_prev*dt) + np.cos(psi_prev)) + \
                         factor * (np.sin(psi_prev + omega_prev*dt) * dt)

        x_pred_k[2] = v_prev  # Speed is assumed constant + process noise
        x_pred_k[3] = normalize_angle(psi_prev + omega_prev * dt) # Yaw update
        x_pred_k[4] = omega_prev # Yaw rate is assumed constant + process noise
        
        # Common terms for Jacobian F_k
        F_k[2,2] = 1
        F_k[3,3] = 1; F_k[3,4] = dt
        F_k[4,4] = 1
        
        # Predicted state estimate: x_hat_{k|k-1}
        x_hat_pred = x_pred_k
        
        # Predicted error covariance: P_{k|k-1} = F_k * P_{k-1|k-1} * F_k^T + Q
        P_pred = F_k @ P[k-1] @ F_k.T + Q

        # --- Update Step ---
        # Measurement: z_k (this is `measurements[k]`, which is [x_meas, y_meas])
        z_k = measurements[k]
        
        # Measurement prediction: h(x_hat_{k|k-1})
        # h(x) simply extracts x and y position
        z_pred = H @ x_hat_pred 
        
        # Measurement residual (innovation): y_k = z_k - h(x_hat_{k|k-1})
        y_k = z_k - z_pred
        
        # Residual covariance (innovation covariance): S_k = H * P_{k|k-1} * H^T + R
        S_k = H @ P_pred @ H.T + R
        
        # Optimal Kalman gain: K_k = P_{k|k-1} * H^T * S_k^{-1}
        # Use pseudo-inverse for S_k if it's ill-conditioned, though unlikely for 2x2 R
        K_k = P_pred @ H.T @ np.linalg.pinv(S_k)
        
        # Updated state estimate: x_hat_{k|k} = x_hat_{k|k-1} + K_k * y_k
        x_hat[k] = x_hat_pred + K_k @ y_k
        x_hat[k,3] = normalize_angle(x_hat[k,3]) # Normalize yaw angle
        
        # Updated error covariance: P_{k|k} = (I - K_k * H) * P_{k|k-1}
        I = np.eye(n_states)
        P[k] = (I - K_k @ H) @ P_pred
        # P[k] = (I - K_k @ H) @ P_pred @ (I - K_k @ H).T + K_k @ R @ K_k.T # Joseph form (more stable)

    return x_hat, P

def run_ekf_2d_example():
    """Runs a 2D EKF example for trajectory prediction and plots the results."""
    np.random.seed(int(np.random.rand() * 10000)) # Randomize seed for different trajectories
    n_timesteps = 200
    dt = 0.1 # Time step

    # True trajectory generation (Coordinated Turn model)
    # Initial true state: [x, y, v, psi, omega]
    true_initial_state = np.array([0.0, 0.0, 5.0, np.pi/4, np.deg2rad(10.0)]) # Start with a turn
    true_states_ekf = np.zeros((n_timesteps, 5))
    true_states_ekf[0] = true_initial_state

    # Simulate dynamic yaw rate (e.g., straighten out then turn other way)
    yaw_rate_profile = np.ones(n_timesteps) * true_initial_state[4]
    yaw_rate_profile[n_timesteps//3:] = np.deg2rad(-5.0) # Turn other way
    yaw_rate_profile[2*n_timesteps//3:] = np.deg2rad(0.0) # Straighten
    
    # Process noise parameters for true trajectory simulation (can be different from EKF's assumptions)
    sim_accel_noise_std = 0.2 # m/s^2
    sim_yaw_accel_noise_std = np.deg2rad(1.0) # rad/s^2

    for k in range(1, n_timesteps):
        x_prev, y_prev, v_prev, psi_prev, omega_prev = true_states_ekf[k-1]
        
        # Update omega based on profile and add noise
        current_omega = yaw_rate_profile[k-1] + np.random.randn() * sim_yaw_accel_noise_std * np.sqrt(dt)
        # Update speed with some noise
        current_v = v_prev + np.random.randn() * sim_accel_noise_std * np.sqrt(dt)
        current_v = max(0.1, current_v) # Ensure speed is positive

        if abs(current_omega) < 1e-5:
            dx = current_v * np.cos(psi_prev) * dt
            dy = current_v * np.sin(psi_prev) * dt
        else:
            factor = current_v / current_omega
            dx = factor * (np.sin(psi_prev + current_omega * dt) - np.sin(psi_prev))
            dy = factor * (-np.cos(psi_prev + current_omega * dt) + np.cos(psi_prev))
        
        true_states_ekf[k,0] = x_prev + dx
        true_states_ekf[k,1] = y_prev + dy
        true_states_ekf[k,2] = current_v
        true_states_ekf[k,3] = normalize_angle(psi_prev + current_omega * dt)
        true_states_ekf[k,4] = current_omega

    # Measurements (noisy observations of x and y positions)
    measurement_noise_std = 1.5 # meters
    measurements_ekf = true_states_ekf[:, :2] + np.random.randn(n_timesteps, 2) * measurement_noise_std

    # EKF parameters
    R_diag_vals_ekf = [measurement_noise_std**2, measurement_noise_std**2]
    ekf_process_noise_accel_std = 0.5 # EKF's belief about accel noise
    ekf_process_noise_yaw_accel_std = np.deg2rad(2.0) # EKF's belief about yaw accel noise

    # Run EKF
    estimated_states_ekf, _ = extended_kalman_filter_2d(
        measurements_ekf,
        R_diag_vals=R_diag_vals_ekf,
        process_noise_accel_std=ekf_process_noise_accel_std,
        process_noise_yaw_accel_std=ekf_process_noise_yaw_accel_std,
        dt=dt,
        # Optional: provide initial state and covariance if known better
        # initial_x_hat=np.array([measurements_ekf[0,0], measurements_ekf[0,1], 3.0, np.arctan2(measurements_ekf[1,1]-measurements_ekf[0,1], measurements_ekf[1,0]-measurements_ekf[0,0]), 0.0]),
        # initial_P = np.diag([1.0, 1.0, 2.0**2, (np.pi/8)**2, (np.deg2rad(5.0))**2]) 
    )

    # Plotting
    # The plot_trajectory_data function expects true_states and estimated_states to have x,y as first two columns.
    plot_trajectory_data(true_states_ekf, measurements_ekf, estimated_states_ekf, title="2D Extended Kalman Filter (Coordinated Turn)")

if __name__ == '__main__':
    print("Kalman filter exploration script.")
    
    if not test_matplotlib():
        print("Matplotlib test failed. Please check your Matplotlib installation and backend.")
        print("Exiting due to Matplotlib issues.")
        # exit() # Optionally exit if the test fails
    else:
        print("Proceeding with Kalman filter examples...")
        # run_1d_example()
        run_2d_example() # Run the 2D linear example
        # run_ekf_2d_example() # Run the 2D EKF example
