import numpy as np
import sys

def _monotone_natural_spline(Xout, X, Y):
    """Interpolates (X, Y) using a monotonicity-preserving cubic spline.

    This function first attempts to interpolate the data using a natural
    cubic spline. It then checks for any intervals where the spline is not
    monotonic and replaces those sections with a PCHIP (Piecemeal Cubic
    Hermite Interpolating Polynomial) interpolator, which guarantees
    monotonicity.

    Args:
        Xout (np.ndarray): The x-coordinates at which to evaluate the spline.
        X (np.ndarray): The x-coordinates of the data points (must be increasing).
        Y (np.ndarray): The y-coordinates of the data points (must be monotonic).

    Returns:
        np.ndarray: The interpolated y-values corresponding to `Xout`. 
        
    Raises:
        ImportError if scipy is not installed.
    """
    try:
        from scipy.interpolate import CubicSpline, PchipInterpolator
    except ImportError:
        raise ImportError("Error: scipy is required for spline interpolation.", file=sys.stderr)

    spline = CubicSpline(X, Y, bc_type='natural')
    pchip = PchipInterpolator(X, Y)
    
    Yout = spline(Xout)
    
    Yp  = spline(X, 1)
    dYout = spline(Xout, 1)
        
    idx = np.searchsorted(X, Xout) - 1
    idx = np.clip(idx, 0, len(X)-2)
    
    for i in range(len(X)-1):
        mask = (idx == i) & ((dYout < 0) | (Yp[i] <= 0) | (Yp[i+1] <= 0))
        if np.any(mask):
            mask = (idx == i)
            Yout[mask] = pchip(Xout[mask])
             
    return Yout
    
def step_pdf_from_quantiles(quantiles):
    """Reconstructs a stepwise probability density function (PDF) from its quantiles.

    This method assumes that the probability is distributed uniformly between
    any two consecutive quantiles. It calculates the height of the probability
    density steps based on this assumption.

    Args:
        quantiles (np.ndarray): A 1D array of monotonic quantile locations.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - z_steps_extended (np.ndarray): The x-coordinates (redshift) of the
              step function edges.
            - p_steps_extended (np.ndarray): The y-coordinates (probability
              density) of the step function.
    """

    Nq = len(quantiles)
    # Add a small epsilon to avoid division by zero for delta-like functions
    dz = quantiles[1:] - quantiles[:-1] + 1e-9
    p_steps = (1.0 / (Nq - 1)) / dz
    z_steps = quantiles
    z_steps_extended = np.concatenate(([z_steps[0]-0.001],z_steps,[z_steps[-1]+0.001]))
    p_steps_extended = np.concatenate(([0],p_steps,[0]))
    return z_steps_extended, p_steps_extended
    
def plot_from_quantiles(quantiles, output_filename, markers=None, source_id=None, method='all'):
    """Generates and saves a plot of a single PDF from its quantiles.

    Reconstructs a PDF using one or more methods ('steps', 'spline') and
    saves the resulting plot to a file. It can also overplot vertical lines
    to mark specific quantities of interest.

    Args:
        quantiles (np.ndarray): The array of quantile values for one PDF.
        output_filename (str): The path where the plot file will be saved.
        markers (dict, optional): A dictionary of {name: value} to mark with
            vertical lines on the plot. Defaults to None.
        source_id (str, optional): An identifier for the source, used in the
            plot title. Defaults to None.
        method (str, optional): The PDF reconstruction method to plot. Can be
            'steps', 'spline', or 'all'. Defaults to 'all'.
            
    Raises:
        ImportError if matplotlib is not installed.
                
    """
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Error: matplotlib is required for plotting.", file=sys.stderr)

    from .decode import quantiles_to_binned
    
    plt.figure(figsize=(8, 6))

    if method == 'steps' or method == 'all':
        z_steps, p_steps = step_pdf_from_quantiles(quantiles)
        plt.step(z_steps[:-1], p_steps, where='post', label='PDF (steps)')

    if method == 'spline' or method == 'all':
        zvector = np.linspace(quantiles[0], quantiles[-1], 500)
        pdf = quantiles_to_binned(quantiles, zvector=zvector, method='spline')
        plt.plot(zvector, pdf, label='PDF (spline)')

    if markers:
        linestyles = [':', '--', '-.']
        colors = [f'C{i}' for i in range(1, 10)] 

        for i, (name, value) in enumerate(markers.items()):
            if value is not None and np.isfinite(value):
                style = linestyles[i % len(linestyles)]
                color = colors[(i+2) % len(colors)]
                plt.axvline(x=value, linestyle=style, color=color, label=f'{name} = {value:.4f}', alpha=0.9)

    plt.xlabel('Redshift (z)')
    plt.ylabel('Probability Density P(z)')
    
    title = 'Reconstructed PDF'
    if source_id:
        title += f' for Source {source_id}'
    plt.title(title)
    
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_filename)
    plt.close()