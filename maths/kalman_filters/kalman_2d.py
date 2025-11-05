import numpy as np


def kalman_filter_2d(
    measurements: np.ndarray,
    R_diag_vals: np.ndarray,
    process_noise_std: float,
    dt: float = 1.0,
    initial_x_hat: np.ndarray = None,
    initial_P: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Implements a 2D Kalman Filter for vehicle trajectory prediction.
    State vector: [x_pos, y_pos, x_vel, y_vel]^T

    Args:
        measurements (np.ndarray): Array of measurements, shape (n_timesteps, 2) for [x_meas, y_meas].
        R_diag_vals (list or np.ndarray): Diagonal values for the measurement noise covariance R.
                                          Example: [R_x_pos**2, R_y_pos**2].
        process_noise_std (float): Standard deviation of the acceleration noise (sigma_a).
                                   Used to construct the process noise covariance Q.
        dt (float): Time step.
        initial_x_hat (np.ndarray, optional): Initial state estimate [x, y, vx, vy].
                                              Defaults to [measurements[0,0], measurements[0,1], 0, 0].
        initial_P (np.ndarray, optional): Initial error covariance matrix (4x4).
                                          Defaults to np.eye(4).

    Returns:
        tuple: (estimated_states, estimated_covariances)
            estimated_states (np.ndarray): Array of estimated states [x, y, vx, vy] for each time step.
            estimated_covariances (np.ndarray): Array of error covariance matrices (4x4) for each time step.
    """
    n_timesteps = measurements.shape[0]
    n_states = 4  # [x, y, vx, vy]

    x_hat = np.zeros((n_timesteps, n_states))
    P = np.zeros((n_timesteps, n_states, n_states))

    # State transition matrix F (constant velocity model)
    # Used to predict the next state from the current one.
    # It is only the output of the 'system' for now.
    # x_k = F * x_{k-1}
    # [px_k] = [1  0  dt  0] * [px_{k-1}]
    # [py_k]   [0  1  0  dt]   [py_{k-1}]
    # [vx_k]   [0  0  1   0]   [vx_{k-1}]
    # [vy_k]   [0  0  0   1]   [vy_{k-1}]
    F = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # Measurement matrix H
    # We measure x and y positions
    # z_k = H * x_k
    # [x_measured_k] = [1  0  0  0] * [px_k]
    # [y_measured_k]   [0  1  0  0]   [py_k]
    #                                 [vx_k]
    #                                 [vy_k]
    H = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]
    )

    # Measurement noise covariance R
    # Assumes measurements of x_pos and y_pos are independent (i.e. not correlated).
    # We extract only the diagonal elements.
    R = np.diag(R_diag_vals)

    # Process noise covariance Q
    # Based on random acceleration model. Q = G * Q_accel * G^T
    # where G maps acceleration noise to state noise.
    # G = [[dt^2/2, 0      ],
    #      [0,       dt^2/2 ],
    #      [dt,      0      ],
    #      [0,       dt     ]]
    # Q_accel = [[sigma_ax^2, 0         ],
    #            [0,          sigma_ay^2]]
    # Assuming sigma_ax = sigma_ay = process_noise_std (sigma_a)
    # Q = sigma_a^2 * [[dt^4/4, 0,      dt^3/2, 0     ],
    #                  [0,      dt^4/4, 0,      dt^3/2],
    #                  [dt^3/2, 0,      dt^2,   0     ],
    #                  [0,      dt^3/2, 0,      dt^2  ]]
    sigma_a_sq = process_noise_std**2
    Q = sigma_a_sq * np.array(
        [
            [(dt**4) / 4, 0, (dt**3) / 2, 0],
            [0, (dt**4) / 4, 0, (dt**3) / 2],
            [(dt**3) / 2, 0, dt**2, 0],
            [0, (dt**3) / 2, 0, dt**2],
        ]
    )

    # Initial state estimate
    if initial_x_hat is None:
        x_hat[0] = np.array(
            [measurements[0, 0], measurements[0, 1], 0, 0]
        )  # Assume zero initial velocity
    else:
        x_hat[0] = initial_x_hat

    # Initial error covariance
    if initial_P is None:
        P[0] = np.eye(n_states) * 1.0
    else:
        P[0] = initial_P

    for k in range(1, n_timesteps):
        # --- Prediction Step ---

        # Predicted state estimate: x_hat_{k|k-1} = F * x_hat_{k-1|k-1}
        x_hat_pred = F @ x_hat[k - 1]

        # Predicted error covariance: P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        P_pred = F @ P[k - 1] @ F.T + Q

        # --- Update Step ---
        # Measurement: z_k (this is `measurements[k]`, which is [x_meas, y_meas])
        z_k = measurements[k]

        # Measurement residual (innovation): y_k = z_k - H * x_hat_{k|k-1}
        y_k = z_k - H @ x_hat_pred

        # Residual covariance (innovation covariance): S_k = H * P_{k|k-1} * H^T + R
        S_k = H @ P_pred @ H.T + R

        # Optimal Kalman gain: K_k = P_{k|k-1} * H^T * S_k^{-1}
        K_k = P_pred @ H.T @ np.linalg.inv(S_k)

        # Updated state estimate: x_hat_{k|k} = x_hat_{k|k-1} + K_k * y_k
        x_hat[k] = x_hat_pred + K_k @ y_k

        # Updated error covariance: P_{k|k} = (I - K_k * H) * P_{k|k-1}
        # Using Joseph form for better numerical stability is recommended for real applications
        I = np.eye(n_states)
        P[k] = (I - K_k @ H) @ P_pred
        # P[k] = (I - K_k @ H) @ P_pred @ (I - K_k @ H).T + K_k @ R @ K_k.T # Joseph form

    return x_hat, P
