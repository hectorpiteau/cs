
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_trajectory_data(true_states, measurements, estimated_states, title="Results", show=True):
    """
    Plots the ground truth states then measurements and estimated states in the same plot.
    Handles both 1D and 2D state vectors.

    Args:
        true_states (np.ndarray): Array of true states (ground truth).
                                  For 1D: shape (n_timesteps,)
                                  For 2D: shape (n_timesteps, 2) or (n_timesteps, 4) for [x, y] or [x, y, vx, vy] (vx vy : velocities)
        measurements (np.ndarray): Array of measurements (observations).
                                   For 1D: shape (n_timesteps,)
                                   For 2D: shape (n_timesteps, 2) for [x_meas, y_meas]
        estimated_states (np.ndarray): Array of estimated states (predictions output of the system).
                                       Shape similar to true_states.
        title (str): Title of the plot.
    """
    n_timesteps = true_states.shape[0]
    is_1d = len(true_states.shape) == 1 or true_states.shape[1] == 1

    plt.figure(figsize=(12, 8))

    if is_1d:
        plt.plot(range(n_timesteps), true_states, 'g-', label='True State')
        plt.plot(range(n_timesteps), measurements, 'bx', label='Measurements')
        plt.plot(range(n_timesteps), estimated_states, 'r--', label='Estimated State')
        plt.xlabel('Time Step')
        plt.ylabel('Position')
        plt.legend()
        plt.title(title + " (1D)")
        plt.grid(True)
    else:
        # Assuming the first two components are x, y positions
        plt.plot(true_states[:, 0], true_states[:, 1], 'g-', label='True Trajectory')
        if measurements is not None and measurements.shape[1] >= 2:
            plt.plot(measurements[:, 0], measurements[:, 1], 'bx', label='Measurements')
        plt.plot(estimated_states[:, 0], estimated_states[:, 1], 'r--', label='Estimated Trajectory')
        
        # Mark start and end points
        plt.plot(true_states[0, 0], true_states[0, 1], 'go', markersize=10, label='True Start', alpha=0.5)
        plt.plot(true_states[-1, 0], true_states[-1, 1], 'gs', markersize=10, label='True End')
        plt.plot(estimated_states[0, 0], estimated_states[0, 1], 'ro', markersize=10, label='Estimated Start')
        plt.plot(estimated_states[-1, 0], estimated_states[-1, 1], 'rs', markersize=10, label='Estimated End')

        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.title(title + " (2D Trajectory)")
        plt.axis('equal')
        plt.grid(True)

    # Save the figure before trying to show it
    fig_filename = title.replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    plt.savefig(fig_filename, dpi=800) # todo: test with other dpi values..
    
    print(f"[info] Plot saved to {fig_filename}")

    if show:
        plt.show()
    plt.close()
