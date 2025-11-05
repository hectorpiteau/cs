import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend
import matplotlib.pyplot as plt


def f(x, y):
    """
    Function to evaluate at each grid point.
    
    Args:
        x: x-coordinate(s)
        y: y-coordinate(s)
    
    Returns:
        Result of x * y
    """
    res = np.floor(x/y)
    
    res[ res == 4 ] = 100
    return res


def create_2d_grid(n_samples=100):
    """
    Create a 2D grid of samples uniformly distributed in [0, 1] x [0, 1].
    
    Args:
        n_samples: Number of samples along each axis
    
    Returns:
        X, Y: 2D arrays of shape (n_samples, n_samples) containing the coordinates
    """
    # Create 1D arrays of uniformly spaced samples in [0, 1]
    x = np.linspace(0, 1, n_samples)
    y = np.linspace(0, 1, n_samples)
    
    # Create 2D meshgrid from 1D arrays
    X, Y = np.meshgrid(x, y)
    
    return X, Y


def plot_2d_function(X, Y, Z, title="2D Function Plot"):
    """
    Plot a 2D function as a color gradient with colorbar.
    
    Args:
        X: 2D array of x-coordinates
        Y: 2D array of y-coordinates
        Z: 2D array of function values
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create the color plot
    im = ax.imshow(Z, extent=[0, 1, 0, 1], origin='lower', 
                   cmap='viridis', aspect='auto', interpolation='bilinear')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='f(x, y) = x * y')
    
    # Set labels and title
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()


def main():
    # Number of samples along each axis
    n_samples = 100
    
    # Create the 2D grid
    X, Y = create_2d_grid(n_samples)
    
    # Evaluate the function at each grid point
    Z = f(X, Y)
    
    # Plot the results
    plot_2d_function(X, Y, Z, title=f"2D Function Plot: f(x, y) = x * y (Grid: {n_samples}x{n_samples})")
    
    # Optional: Print some statistics
    print(f"Grid shape: {Z.shape}")
    print(f"Min value: {Z.min():.4f}")
    print(f"Max value: {Z.max():.4f}")
    print(f"Mean value: {Z.mean():.4f}")


if __name__ == "__main__":
    main()

