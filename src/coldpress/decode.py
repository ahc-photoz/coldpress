import numpy as np
import struct
import sys

def decode_quantiles(packet):
    """Decodes a byte packet back into an array of quantiles.

    This function unpacks a compressed byte representation created by
    `encode_quantiles`. It reconstructs the quantile locations, including
    the endpoints (zmin, zmax). 

    Args:
        packet (bytes): The byte string packet to decode.

    Returns:
        np.ndarray: A 1D array of float values representing the decoded
            quantile locations. 
    """ 
    a_shift = 0.020202707 # allows to encode negative redshifts down to -0.02 as an integer
    df = 4.0e-5 # resolution in log(1+z) for the redshift of the first quantile
    eps_min = 1.0e-7 # minimum epsilon value that can be encoded
    eps_beta = 0.03 # exponential term that determines the maximum epsilon of the encoding
   
    eps_byte = packet[0]
    eps = eps_min*np.exp(eps_beta*eps_byte)

    xmin_int = struct.unpack('<H', packet[1:3])[0]   
    logq_min = xmin_int*df - a_shift

    payload = packet[3:]
    zs = [logq_min]
    i = 0
    length = len(payload)
    while i < length:
        b = payload[i]
        if (b == 0) and (max(payload[i:]) == 0): # end if just trailing zeros remain
            break
        if b == 0: # prevent two quantiles from having exactly the same z by applying tiny offsets
            zs.append(zs[-1] + 0.05*eps)
            zs[-2] -= 0.05*eps
            i += 1
        else:
            if b < 255:
                d = b
                i += 1
            else:
                d = struct.unpack('>H', payload[i+1:i+3])[0]
                i += 3
            zs.append(zs[-1] + d * eps)

    # convert from log(1+z) to z
    return np.exp(np.array(zs))-1

def quantiles_to_binned(z_quantiles, dz=None, Nbins=None, z_min=None, z_max=None, zvector=None, method='linear', force_range=False):
    """Converts quantile locations into a binned probability density function (PDF).

    Reconstructs a regularly gridded PDF from an array of quantile locations.
    The output grid can be defined explicitly via `zvector`, or by specifying
    the number of bins (`Nbins`) or bin width (`dz`) along with an optional
    range (`z_min`, `z_max`).

    Args:
        z_quantiles (np.ndarray): A 1D array of monotonic quantile locations.
        dz (float, optional): The width of the redshift bins for the output grid.
        Nbins (int, optional): The number of bins for the output grid.
        z_min (float, optional): The minimum redshift for the output grid. If
            None, inferred from `z_quantiles`.
        z_max (float, optional): The maximum redshift for the output grid. If
            None, inferred from `z_quantiles`.
        zvector (np.ndarray, optional): A 1D array defining the centers of the
            redshift bins. Takes precedence over other grid arguments.
        method (str, optional): Interpolation method to reconstruct the
            cumulative distribution function. Either 'linear' or 'spline'.
            Defaults to 'linear'.
        force_range (bool, optional): If True, allows the PDF to be truncated
            if its range exceeds the specified grid. If False, raises a
            ValueError. Defaults to False.

    Returns:
        np.ndarray or tuple[np.ndarray, np.ndarray]:
            If `zvector` is provided, returns the binned PDF (1D `np.ndarray`).
            Otherwise, returns a tuple containing the generated redshift grid
            (`z_grid`) and the corresponding binned PDF.

    Raises:
        ValueError: If grid arguments are specified incorrectly, if the PDF
            range exceeds the grid and `force_range` is False, or if an
            unknown interpolation method is given.
    """
    
    if method == 'spline':
        from .utils import _monotone_natural_spline

    # --- sanity checks for conflicting arguments ---
    if force_range and zvector is None and z_min is None and z_max is None:
        raise ValueError("force_range=True is only meaningful when an explicit range is provided via 'zvector' or 'z_min'/'z_max'.")
    if zvector is not None and (dz is not None or Nbins is not None):
        raise ValueError("Cannot specify 'dz' or 'Nbins' when 'zvector' is provided.")
    if dz is not None and Nbins is not None:
        raise ValueError("Cannot specify both 'dz' and 'Nbins' simultaneously.")
        
    zq = np.asarray(z_quantiles)
    
    # Grid creation logic
    if zvector is not None:
        z_grid = zvector
        dz = zvector[1]-zvector[0]
    else:
        # Determine range boundaries, inferring from data if not provided
        range_min = zq[0] if z_min is None else z_min
        range_max = zq[-1] if z_max is None else z_max

        if Nbins is not None:
            # Handle case of a delta function to avoid a zero-width range
            if range_min == range_max:
                range_min -= 0.0001
                range_max += 0.0001
            z_grid = np.linspace(range_min, range_max, Nbins)
            dz = z_grid[1]-z_grid[0]
        elif dz is not None:
            # Snap auto-calculated range to the dz grid if z_min/z_max not provided
            if z_min is None: range_min = dz * np.floor(zq[0] / dz)
            if z_max is None: range_max = dz * np.ceil(zq[-1] / dz)
            z_grid = np.arange(range_min, range_max + dz/2, dz)
        else:
            raise ValueError("Must provide one of 'zvector', 'Nbins', or 'dz'.")

    # Perform the range check on the final z_grid
    eps = 1e-10
    zmin_q, zmax_q = zq[0]+dz/2+eps, zq[-1]-dz/2-eps
    zmin_grid, zmax_grid = z_grid[0], z_grid[-1]

    if zmin_q < zmin_grid or zmax_q > zmax_grid:
        if not force_range:
            raise ValueError(f"Decoded redshift range [{zmin_q:.3f}, {zmax_q:.3f}] "
                             f"exceeds the target grid range [{zmin_grid:.3f}, {zmax_grid:.3f}]. "
                             "Use force_range=True to override.")
        print(f"Warning: PDF range [{zmin_q:.3f}, {zmax_q:.3f}] truncated to grid range [{zmin_grid:.3f}, {zmax_grid:.3f}].", file=sys.stderr)
    
    # Proceed with calculations
    M = len(zq)
    Fq = np.linspace(0.0, 1.0, M)
    dz_eff = z_grid[1] - z_grid[0] if len(z_grid) > 1 else 0
    
    edges = np.empty(len(z_grid)+1, dtype=float)
    edges[0] = z_grid[0] - dz_eff/2
    edges[1:] = z_grid + dz_eff/2

    if method == 'linear':
        F_grid = np.interp(edges, zq, Fq, left=0.0, right=1.0)
    elif method == 'spline':
        F_grid = np.zeros(len(edges))
        F_grid[edges < zq[0]] = 0.
        F_grid[edges > zq[-1]] = 1.
        inside = (edges >= zq[0]) & (edges <= zq[-1])
        F_inside = _monotone_natural_spline(edges[inside], zq, Fq)
        if F_inside.size > 0 and F_inside[-1] > 0:
            F_grid[inside] = F_inside / F_inside[-1]
    else:
        raise ValueError(f"Unknown interpolation method '{method}'. Choose 'linear' or 'spline'.")
        
    pdf = F_grid[1:] - F_grid[:-1]
        
    pdf_sum = np.sum(pdf * dz_eff)
    if pdf_sum > 0:
        pdf /= pdf_sum
        
    # Return both the grid and the PDF if the grid was generated internally
    if zvector is None:
        return z_grid, pdf
    else:
        return pdf
        
def quantiles_to_samples(z_quantiles, Nsamples=100, method='linear'):
    """Generates random samples from a PDF defined by its quantiles.

    Uses inverse transform sampling to draw random samples. The cumulative
    distribution function is reconstructed from the quantiles using either
    linear or spline interpolation.

    Args:
        z_quantiles (np.ndarray): A 1D array of monotonic quantile locations.
        Nsamples (int, optional): The number of random samples to generate.
            Defaults to 100.
        method (str, optional): Interpolation method to reconstruct the CDF.
            Either 'linear' or 'spline'. Defaults to 'linear'.

    Returns:
        np.ndarray: A 1D array of `Nsamples` random draws from the PDF.
    """
    
    if method == 'spline':
        from .utils import _monotone_natural_spline
  
    zq = np.asarray(z_quantiles)
    M = len(zq)
    Fq = np.linspace(0.0, 1.0, M)
   
    u = np.random.uniform(0, 1, size=Nsamples)

    if method == 'linear':
        samples = np.interp(u, Fq, zq)
    if method == 'spline':
        samples = _monotone_natural_spline(u, Fq, zq)
            
    return samples
    
def decode_to_binned(int32col, zvector, force_range=False, method='linear'):
    """Decodes a column of compressed PDFs into a 2D array of binned PDFs.

    This is a batch processing function that iterates over a column of
    compressed byte packets, decodes each one into quantiles, and then
    converts the quantiles into a binned PDF on the specified `zvector` grid.

    Args:
        int32col (np.ndarray): A 2D numpy array of type int32, where each
            row is a compressed PDF packet.
        zvector (np.ndarray): A 1D array defining the centers of the redshift
            bins for the output PDFs.
        force_range (bool, optional): If True, allows PDFs to be truncated
            if their range exceeds the `zvector` grid. Defaults to False.
        method (str, optional): Interpolation method ('linear' or 'spline')
            for PDF reconstruction. Defaults to 'linear'.

    Returns:
        np.ndarray: A 2D float32 array where each row is a reconstructed
            binned PDF corresponding to a row in `int32col`.

    Raises:
        ValueError: If decoding or binning fails for any source and
            `force_range` is False.
    """
    
    Nsources = int32col.shape[0]
    PDF = np.zeros((Nsources,len(zvector)),dtype=np.float32)
    
    for i in range(Nsources):
        if np.any(int32col[i] != 0):
            packet = int32col[i].tobytes()
            qrecovered = decode_quantiles(packet)

            try:
                PDF[i] = quantiles_to_binned(qrecovered, zvector=zvector, method=method, force_range=force_range)
            except ValueError as e:
                raise ValueError(f"Source {i}: {e}") from e

    return PDF
    
def decode_to_samples(int32col, Nsamples=None, method='linear'):
    """Decodes a column of compressed PDFs into an array of random samples.

    This is a batch processing function that iterates over a column of
    compressed byte packets, decodes each one into quantiles, and then
    generates random samples from the reconstructed PDF.

    Args:
        int32col (np.ndarray): A 2D numpy array of type int32, where each
            row is a compressed PDF packet.
        Nsamples (int, optional): The number of random samples to generate
            for each PDF.
        method (str, optional): Interpolation method ('linear' or 'spline')
            for PDF reconstruction. Defaults to 'linear'.

    Returns:
        np.ndarray: A 2D float32 array of shape (Nsources, Nsamples)
            containing random samples. Rows for invalid/empty input packets
            will contain NaNs.

    Raises:
        ValueError: If decoding or sampling fails for any source.
    """
    
    Nsources = int32col.shape[0]
    samples = np.full((Nsources,Nsamples),np.nan,dtype=np.float32)
    
    for i in range(Nsources):
        if np.any(int32col[i] != 0):
            packet = int32col[i].tobytes()
            qrecovered = decode_quantiles(packet)

            try:
                samples[i] = quantiles_to_samples(qrecovered, Nsamples=Nsamples, method=method)
            except ValueError as e:
                raise ValueError(f"Source {i}: {e}") from e

    return PDF
    
