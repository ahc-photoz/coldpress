import numpy as np
import struct
import sys
from .constants import NEGATIVE_Z_OFFSET, LOG_DZ, EPSILON_MIN, EPSILON_BETA

def decode_quantiles(packet, units='redshift'):
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
    # reconstruct epsilon and zmin
    eps_byte = packet[0]
    eps = EPSILON_MIN*np.exp(EPSILON_BETA*eps_byte)

    xmin_int = struct.unpack('<H', packet[1:3])[0]   
    logq_min = xmin_int*LOG_DZ - NEGATIVE_Z_OFFSET
   
    # decode the quantized jumps between consecutive quantiles
    payload = packet[3:]
    jumps = []
    i = 0
    length = len(payload)
    while i < length:
        b = payload[i]
        if (b == 0) and (max(payload[i:]) == 0): # end if just trailing zeros remain
            break
        if b < 255:
            d = b
            i += 1
        else:
            d = struct.unpack('>H', payload[i+1:i+3])[0]
            i += 3
        jumps.append(d)
        
    # remove seesaw pattern due to small jump values at high P(z)
    new_jumps = np.array(jumps).astype(float)
    insaw = False
    for i in range(1,len(jumps)-1):
        if not insaw and (jumps[i] < 15) and (abs(jumps[i]-jumps[i-1]) <= 1):
            insaw = True
            init_saw = i-1
            level = jumps[init_saw]
            continue
        if insaw and ((jumps[i] > 15) or (abs(jumps[i]-level) > 1)):
            insaw = False
            end_saw = i
            if end_saw - init_saw > 3:
                new_jumps[init_saw:end_saw] = np.mean(jumps[init_saw:end_saw])
        
    # remove any remaining zero-valued jumps by taking 1 unit from the next non-zero jump
    if np.min(new_jumps) == 0:
        in_strike = False
        for i in range(0,len(new_jumps)):
            if not in_strike and (new_jumps[i] == 0):
                in_strike = True
                strike_init = i
                continue
            if in_strike and (new_jumps[i] > 0): 
                if new_jumps[i] <= 1: # average from start of strike including this one
                    new_jumps[strike_init:i+1] = new_jumps[i]/(i+1-strike_init)
                else: # take one unit from this jump and split it among the previous zeros
                    new_jumps[strike_init:i] = 1./(i-strike_init) 
                    new_jumps[i] -= 1
                in_strike = False
                           
    if np.min(new_jumps) <= 0:
        print('Something went wrong. There should be no zero or negative jumps.')
        import code
        code.interact(local=locals())
        
    # check that probability is conserved
    if abs(np.sum(jumps)-np.sum(new_jumps)) > 0.01:
        print('Something went wrong while correcting decoded jump values!')
        import code
        code.interact(local=locals())      
                
    # convert jumps to quantile of zeta        
    zs = np.empty(new_jumps.size + 1, dtype=float)
    zs[0] = logq_min
    zs[1:] = logq_min + np.cumsum(new_jumps) * eps

    # return quantiles in the requested units
    if units == 'zeta':
        return zs
    elif units == 'redshift':
        return np.exp(zs)-1          
    else:
        print("Error: unknown units: {units}. Valid values: 'redshift','zeta'")
        sys.exit(1)
        
def quantiles_to_binned(z_quantiles, dz=None, Nbins=None, z_min=None, z_max=None, zvector=None, method='linear', force_range=False, renormalize=True):
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
        renormalize (bool, optional): If True, normalize the PDF integral to 1
            in the truncated range. If False, normalize to 1 in the original range.    

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
    eps = dz/100
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
        if renormalize:
            F_inside /= F_inside[-1]
        if F_inside.size > 0 and F_inside[-1] > 0:
            F_grid[inside] = F_inside
    else:
        raise ValueError(f"Unknown interpolation method '{method}'. Choose 'linear' or 'spline'.")
        
    pdf = (F_grid[1:] - F_grid[:-1]) 
    if renormalize:   
        pdf_sum = np.sum(pdf * dz_eff)
        if pdf_sum > 0:
            pdf /= pdf_sum
    else:
        pdf_sum = dz_eff
        pdf /= pdf_sum
     
    # Return both the grid and the PDF if the grid was generated internally
    if zvector is None:
        return z_grid, pdf
    else:
        return pdf
     
def quantiles_to_density(z_quantiles, dz=None, Nbins=None, z_min=None, z_max=None, zvector=None, method='linear', force_range=False, renormalize=True):
    """Converts quantile locations into a probability density sampled in a grid.

    Reconstructs a gridded PDF P(z) from an array of quantile locations.
    The output grid can be defined explicitly via `zvector`, or by specifying
    the number of bins (`Nbins`) or bin width (`dz`) along with an optional
    range (`z_min`, `z_max`).

    Args:
        z_quantiles (np.ndarray): A 1D array of monotonic quantile locations.
        dz (float, optional): The separation between redshif values in the output grid.
        Nbins (int, optional): The number of elements for the output grid.
        z_min (float, optional): The minimum redshift for the output grid. If
            None, inferred from `z_quantiles`.
        z_max (float, optional): The maximum redshift for the output grid. If
            None, inferred from `z_quantiles`.
        zvector (np.ndarray, optional): A 1D array containing the redshift grid. 
        Takes precedence over other grid arguments.
        method (str, optional): Interpolation method to reconstruct the
            cumulative distribution function. Either 'linear' or 'spline'.
            Defaults to 'linear'.
        force_range (bool, optional): If True, allows the PDF to be truncated
            if its range exceeds the specified grid. If False, raises a
            ValueError. Defaults to False.
        renormalize (bool, optional): If True, normalize the PDF integral to 1
            in the truncated range. If False, normalize to 1 in the original range.    

    Returns:
        np.ndarray or tuple[np.ndarray, np.ndarray]:
            If `zvector` is provided, returns the probability density P(z)
            at the redshifts specified by `zvector` (1D `np.ndarray`).
            Otherwise, returns a tuple containing the generated redshift grid
            (`z_grid`) and the corresponding P(z).

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
    dz_eff = z_grid[1] - z_grid[0]
    delta = dz_eff*0.01
        
    if method == 'linear':
        F_minus = np.interp(z_grid-delta, zq, Fq, left=0.0, right=1.0)
        F_plus = np.interp(z_grid+delta, zq, Fq, left=0.0, right=1.0)        
    elif method == 'spline':
        F_minus = np.zeros(len(z_grid))
        F_plus = np.zeros(len(z_grid))
        F_minus[z_grid-delta < zq[0]] = 0.
        F_minus[z_grid-delta > zq[-1]] = 1.
        F_plus[z_grid+delta < zq[0]] = 0.
        F_plus[z_grid+delta > zq[-1]] = 1.
        minus_inside = ((z_grid-delta) >= zq[0]) & ((z_grid-delta) <= zq[-1])
        plus_inside = ((z_grid+delta) >= zq[0]) & ((z_grid+delta) <= zq[-1])
        F_minus_inside = _monotone_natural_spline(z_grid[minus_inside]-delta, zq, Fq)
        F_plus_inside = _monotone_natural_spline(z_grid[plus_inside]+delta, zq, Fq)
#         if renormalize:
#             F_minus_inside /= F_minus_inside[-1]
#             F_plus_inside /= F_plus_inside[-1]
        if F_minus_inside.size > 0 and F_minus_inside[-1] > 0:
            F_minus[minus_inside] = F_minus_inside
        if F_plus_inside.size > 0 and F_plus_inside[-1] > 0:
            F_plus[plus_inside] = F_plus_inside
    else:
        raise ValueError(f"Unknown interpolation method '{method}'. Choose 'linear' or 'spline'.")
        
    pdf = 0.5*(F_plus - F_minus)/delta
     
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
    
def decode_to_density(int32col, zvector, force_range=False, method='linear'):
    """Decodes a column of compressed PDFs into a 2D array with one P(z) per row

    This is a batch processing function that iterates over a column of
    compressed byte packets, decodes each one into quantiles, and then
    converts the quantiles into P(z) sampled on the specified `zvector` grid.

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
            P(z) corresponding to a row in `int32col`.

    Raises:
        ValueError: If decoding or interpolation fails for any source and
            `force_range` is False.
    """
    
    Nsources = int32col.shape[0]
    PDF = np.zeros((Nsources,len(zvector)),dtype=np.float32)
    
    for i in range(Nsources):
        if np.any(int32col[i] != 0):
            packet = int32col[i].tobytes()
            qrecovered = decode_quantiles(packet)

            try:
                PDF[i] = quantiles_to_density(qrecovered, zvector=zvector, method=method, force_range=force_range)
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
    
