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

    # Use a powerlaw instead of a spline to interpolate the first and last segments.
    # This prevents the discontinuity in P(z) often found with Pchip.
    Qstep = 1./(len(X)-1) # step between quantiles
    infirst = (Xout >= X[0]) & (Xout < X[1])
    x = (Xout[infirst] - X[0])/(X[1]-X[0]) # normalized variable ranges from 0 to 1
    Yout[infirst] = Y[0] + Qstep * x**(Yp[1]*(X[1]-X[0])/Qstep)
    
    inlast = (Xout > X[-2]) & (Xout <= X[-1])
    x = (Xout[inlast] - X[-2])/(X[-1] - X[-2])
    Yout[inlast] = Y[-1] - Qstep * (1-x) ** (Yp[-2]*(X[-1]-X[-2])/Qstep)
               
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
    
def plot_from_quantiles(quantiles, output_filename=None, interactive=False, markers=None, source_id=None, method='all', units='redshift', labels=None):
    """Generates and saves or displays a plot of one or more PDFs from their quantiles.

    Reconstructs one or multiple PDFs using one or more methods ('steps', 'spline') and
    saves the resulting plot to a file. It can also overplot vertical lines
    to mark specific quantities of interest.

    Args:
        quantiles (np.ndarray or list): The array of quantile values for one PDF,
            or a list of arrays for multiple PDFs.
        output_filename (str, optional): The path to save the plot file. Required
            if `interactive` is False. Defaults to None.
        interactive (bool, optional): If True, display the plot in an interactive
            window instead of saving. Defaults to False.
        markers (dict, optional): A dictionary of {name: value} to mark with
            vertical lines. Defaults to None.
        source_id (str, optional): An identifier for the source, used in the
            plot title. Defaults to None.
        method (str, optional): The PDF reconstruction method to plot. Can be
            'steps', 'spline', or 'all'. Defaults to 'all'.
        labels (str or list, optional): A label string for a single PDF or a list of
            label strings for multiple PDFs to display in the legend.
            
    Raises:
        ImportError if matplotlib is not installed.
        ValueError: If not in interactive mode and output_filename is not provided.          
    """
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        # Re-raise the error so the calling function can handle it.
        raise ImportError("matplotlib is required for plotting.")
        
    from .decode import quantiles_to_binned
    
    # Ensure inputs are lists for iteration
    if isinstance(quantiles, np.ndarray) and quantiles.ndim == 1:
        quantiles_list = [quantiles]
        labels_list = [labels] if isinstance(labels, str) else (labels or [None])
    else:
        quantiles_list = quantiles
        labels_list = labels or [None] * len(quantiles_list)

    fig, ax = plt.subplots(figsize=(8, 6))

    spline_lines = []

    for q, label in zip(quantiles_list, labels_list):
        if method == 'all':
            steps_label = 'PDF (steps)' if label is None else f'{label} (steps)'
            spline_label = 'PDF (spline)' if label is None else f'{label} (spline)'
        else:
            steps_label = 'PDF' if label is None else label
            spline_label = 'PDF' if label is None else label

        if method == 'steps' or method == 'all':
            z_steps, p_steps = step_pdf_from_quantiles(q)
            ax.step(z_steps[:-1], p_steps, where='post', label=steps_label)

        if method == 'spline' or method == 'all':
            zvector, pdf = quantiles_to_binned(q, Nbins=500, method='spline', renormalize=False)
            line, = ax.plot(zvector, pdf, label=spline_label)
            spline_lines.append((q, line))

    if markers:
        linestyles = [':', '--', '-.']
        colors = [f'C{i}' for i in range(1, 10)] 

        for i, (name, value) in enumerate(markers.items()):
            if value is not None and np.isfinite(value):
                style = linestyles[i % len(linestyles)]
                color = colors[(i+2) % len(colors)]
                if units == 'redshift':
                    xvalue = value
                elif units == 'zeta':
                    xvalue = np.log(1+value)    
                plt.axvline(x=xvalue, linestyle=style, color=color, label=f'{name} = {value:.4f}', alpha=0.9)

    if units == 'redshift':
        ax.set_xlabel('Redshift (z)')
        ax.set_ylabel('Probability Density P(z)')
    elif units == 'zeta':
        ax.set_xlabel('ζ = ln(1+z)')
        ax.set_ylabel('Probability Density P(ζ)')
    
    title = 'Reconstructed PDF'
    if source_id:
        title += f' for Source {source_id}'
    ax.set_title(title)
    
    ax.grid(True, alpha=0.4)
    ax.legend()

    if interactive:
        def on_zoom(axes):
            """Callback function to execute when the x-axis limits change."""
            # Get the new visible redshift range
            zmin, zmax = axes.get_xlim()
            
            # Re-calculate splines for all plotted PDFs on the new high-res grid
            for q, line in spline_lines:
                new_zvector, new_pdf = quantiles_to_binned(q, Nbins=500, z_min=zmin, z_max=zmax, method='spline', force_range=True, renormalize=False)
                line.set_data(new_zvector, new_pdf)
            
            # Rescale axis dynamically to updated spline objects
            if spline_lines:
                axes.relim()
                axes.autoscale_view(scalex=False, scaley=True)
            
            # Redraw the canvas
            axes.figure.canvas.draw_idle()

        # Connect the callback function to the 'xlim_changed' event
        ax.callbacks.connect('xlim_changed', on_zoom)
        
        plt.tight_layout()
        plt.show()
    else:
        if output_filename is None:
            plt.close() # Close the figure to avoid memory leaks
            raise ValueError("output_filename must be provided when not in interactive mode.")
        plt.tight_layout()            
        plt.savefig(output_filename)
        plt.close()

def combine_pdfs(packet1, packet2, method='conflate', length=80, tolerance=0.001):
    """Combines two coldpress-encoded PDFs into a single encoded PDF.

    Decodes two PDFs into probability densities, combines them using the
    specified method ('conflate', 'average', 'correlate'), normalizes,
    and encodes the result.

    Args:
        packet1 (bytes): Byte packet of the first PDF.
        packet2 (bytes): Byte packet of the second PDF.
        method (str): Combination method ('conflate', 'average', 'correlate').
        length (int): Packet length in bytes for encoding.
        tolerance (float): Tolerance for validation during encoding.

    Returns:
        np.ndarray or float or None: A 1D >i4 array representing the encoded combined PDF,
            a float representing the p-value if method is 'correlate', or None if 
            conditions for combination fail (e.g. no overlap for conflation).
    """
    from .decode import decode_quantiles, quantiles_to_density
    from .encode import density_to_quantiles, encode_quantiles

    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    q1 = decode_quantiles(packet1, units='redshift')
    q2 = decode_quantiles(packet2, units='redshift')

    if method == 'conflate':
        zmin = max(q1[0], q2[0])
        zmax = min(q1[-1], q2[-1])
    else:
        zmin = min(q1[0], q2[0])
        zmax = max(q1[-1], q2[-1])

    if zmax <= zmin and method == 'conflate':
        return None

    z_grid = np.linspace(zmin, zmax, 1000)
    p1 = quantiles_to_density(q1, zvector=z_grid, method='spline', force_range=True)
    p2 = quantiles_to_density(q2, zvector=z_grid, method='spline', force_range=True)

    if method == 'conflate':
        c = p1 * p2
    elif method == 'average':
        c = 0.5 * (p1 + p2)
    elif method == 'correlate':
        c = np.correlate(p1, p2, mode='full')
        # Center of mode='full' for two equal-length arrays is len(p2) - 1
        lag_zero_idx = len(p2) - 1
        p_0 = c[lag_zero_idx]
        # Calculate p-value by summing the probability mass in the tails where density <= p(delta_z=0)
        return float(np.sum(c[c <= p_0]) / np.sum(c))
    else:
        raise ValueError(f"Unknown combination method: {method}")

    area = trapz(c, z_grid)
    if area > 0:
        c /= area
        Nq = length - 8
        while Nq >= length / 3:
            q_c = density_to_quantiles(z_grid, c, Nquantiles=Nq)
            try:
                _, enc_packet = encode_quantiles(q_c, packetsize=length, tolerance=tolerance, validate=True, units='redshift')
                return np.frombuffer(enc_packet, dtype='>i4')
            except ValueError:
                Nq -= 2

    return None