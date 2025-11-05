import numpy as np

def kalman_filter_1d(measurements: np.ndarray, R_val:float, Q_val:float, dt:float=1.0, initial_x_hat:np.ndarray=None, initial_P:np.ndarray=None):
    """
    Implements a 1D Kalman Filter (tracking position and velocity).

    Args:
        measurements (np.ndarray): 1D array of position measurements.
        R_val (float): Measurement noise variance (scalar).
        Q_val (float): Process noise variance (scalar for a simple model).
        dt (float): Time step.
        initial_x_hat (np.ndarray, optional): Initial state estimate [position, velocity]. 
                                              Defaults to [measurements[0], 0].
        initial_P (np.ndarray, optional): Initial error covariance matrix. 
                                          Defaults to np.eye(2).

    Returns:
        tuple: (estimated_states, estimated_covariances)
            estimated_states (np.ndarray): Array of estimated states [position, velocity] for each time step.
            estimated_covariances (np.ndarray): Array of error covariance matrices for each time step.
    """
    n_timesteps = len(measurements)
    
    # State vector: [position, velocity]
    # x_hat = [p_hat, v_hat]^T
    x_hat = np.zeros((n_timesteps, 2))
    P = np.zeros((n_timesteps, 2, 2))

    # State transition matrix F
    # x_k = F * x_{k-1}
    # [p_k] = [1 dt] * [p_{k-1}]
    # [v_k]   [0  1]   [v_{k-1}]
    F = np.array([[1, dt],
                  [0, 1]])

    # Measurement matrix H
    # z_k = H * x_k
    # We only measure position: z_k = [1 0] * [p_k, v_k]^T
    H = np.array([[1, 0]])

    # Measurement noise covariance R
    # Assumes scalar measurement, so R is 1x1
    R = np.array([[R_val]])

    # Process noise covariance Q
    # This is a simplified Q. A more accurate Q for constant velocity would be derived 
    # from a continuous noise model: G * G^T * sigma_a^2, where G = [dt^2/2, dt]^T and sigma_a is acceleration noise.
    # For simplicity, we use a diagonal Q, implying noise independently affects position and velocity updates.
    # Let's assume Q_val affects the variance of the acceleration (or changes in velocity).
    # A common form for Q assuming discrete white noise acceleration is:
    # Q = [[dt^4/4, dt^3/2],
    #      [dt^3/2, dt^2  ]] * sigma_a_sq
    # For this example, we use a simpler Q related to Q_val directly affecting variance in state vars.
    # This can be tuned. A small Q means we trust our model more.
    Q = np.array([[Q_val * (dt**4)/4, Q_val * (dt**3)/2],
                  [Q_val * (dt**3)/2, Q_val * (dt**2)  ]])
    # Or a simpler diagonal one if Q_val is meant to represent a general uncertainty
    # Q = np.eye(2) * Q_val # This would be too simplistic

    # Initial state estimate
    if initial_x_hat is None:
        x_hat[0] = np.array([measurements[0], 0]) # Assume initial velocity is 0
    else:
        x_hat[0] = initial_x_hat

    # Initial error covariance
    if initial_P is None:
        P[0] = np.eye(2) * 1.0 # Start with some uncertainty
    else:
        P[0] = initial_P

    for k in range(1, n_timesteps):
        # --- Prediction Step ---
        # Predicted state estimate: x_hat_{k|k-1} = F * x_hat_{k-1|k-1}
        x_hat_pred = F @ x_hat[k-1]
        
        # Predicted error covariance: P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        P_pred = F @ P[k-1] @ F.T + Q

        # --- Update Step ---
        # Measurement: z_k (this is `measurements[k]`)
        z_k = measurements[k]
        
        # Measurement residual (innovation): y_k = z_k - H * x_hat_{k|k-1}
        y_k = z_k - H @ x_hat_pred
        
        # Residual covariance (innovation covariance): S_k = H * P_{k|k-1} * H^T + R
        S_k = H @ P_pred @ H.T + R
        
        # Optimal Kalman gain: K_k = P_{k|k-1} * H^T * S_k^{-1}
        K_k = P_pred @ H.T @ np.linalg.inv(S_k)
        
        # Updated state estimate: x_hat_{k|k} = x_hat_{k|k-1} + K_k * y_k
        x_hat[k] = x_hat_pred + K_k @ y_k.reshape(-1,1) # Ensure y_k is a column vector for matmul
        x_hat[k] = x_hat[k].flatten() # Ensure x_hat[k] remains a 1D array of 2 elements

        # Updated error covariance: P_{k|k} = (I - K_k * H) * P_{k|k-1}
        # Using Joseph form for better numerical stability: P_k = (I - K_k H) P_{k|k-1} (I - K_k H)^T + K_k R K_k^T
        I = np.eye(F.shape[0])
        P[k] = (I - K_k @ H) @ P_pred
        # P[k] = (I - K_k @ H) @ P_pred @ (I - K_k @ H).T + K_k @ R @ K_k.T # Joseph form

    return x_hat, P