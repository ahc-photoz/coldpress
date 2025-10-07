import numpy as np
import struct
from .constants import NEGATIVE_Z_OFFSET, LOG_DZ, EPSILON_MIN, EPSILON_BETA, Q0_ZMIN, Q0_ZMAX, Q0_ZETAMIN, Q0_ZETAMAX
import sys
import time


def samples_to_quantiles(sorted_samples, Nquantiles=100, target_quantiles=None):
    """Calculates quantiles from a set of random samples of a PDF.

    This low-level function takes sorted random samples from a probability 
    distribution and computes the values that correspond to evenly spaced quantiles
    (e.g., percentiles). 
    Warning: samples are expected to be sorted by increasing value and
    are not checked for non-finite or out-of-range values, which must be removed
    in advance.  

    Args:
        samples (np.ndarray): A 1D array of random samples from the PDF.
        Nquantiles (int, optional): The number of quantiles to compute.
            Defaults to 100.

    Returns:
        np.ndarray: A 1D array of `Nquantiles` values representing the
            quantile locations.
    """   
    if target_quantiles is None:     
        target_quantiles = np.linspace(0.0, 1.0, Nquantiles) # target probability for quantiles 
    return np.quantile(sorted_samples, target_quantiles, method='linear')


def binned_to_quantiles(z_grid, Pz, Nquantiles=100, target_quantiles=None):
    """Calculates quantiles from a binned probability density function (PDF).

    This function computes the cumulative distribution function (CDF) from a
    binned PDF and then interpolates it to find the redshift values
    corresponding to a set of evenly spaced quantiles.

    Args:
        z_grid (np.ndarray): A 1D array of the redshift bin centers.
        Pz (np.ndarray): A 1D array of the probability density in each bin.
        Nquantiles (int, optional): The number of quantiles to compute.
            Defaults to 100.

    Returns:
        np.ndarray: A 1D array of `Nquantiles` values representing the
            quantile locations.
    """
    nonzero = np.where(Pz > 0)[0]

    if len(nonzero) == 1: # special case, Pz is a single delta 
        dz = z_grid[1]-z_grid[0]
        qs = np.linspace(z_grid[nonzero[0]]-dz/2,z_grid[nonzero[0]]+dz/2,Nquantiles)
        return qs

    # remove trailing and leading zeros in Pz    
    imin = np.min(nonzero)
    imax = np.max(nonzero)
    z_grid = z_grid[imin:imax+1]
    Pz = Pz[imin:imax+1]

    # compute edges of bins
    dz = z_grid[1] - z_grid[0]
    edges = np.empty(len(z_grid)+1, dtype=float)
    edges[0] = z_grid[0] - dz/2
    edges[1:] = z_grid + dz/2

    # Build CDF at edges: cdf_edges[0]=0; then cdf_edges[i+1] = cdf_edges[i] + p[i]*dz
    cdf_edges = np.empty_like(edges)
    cdf_edges[0] = 0.0
    cdf_edges[1:] = np.cumsum(Pz * dz)

    # Normalize to ensure the final CDF is exactly 1
    cdf_edges /= cdf_edges[-1]

    # compute quantiles
    if target_quantiles is None:
        target_quantiles = np.linspace(0.0, 1.0, Nquantiles) # target probability for quantiles
    qs = np.interp(target_quantiles, cdf_edges, edges)

    return qs
    
def density_to_quantiles(zvector, pdf_density, Nquantiles=100, upsample_factor=10, target_quantiles=None):
    """
    Calculates quantile redshifts from a PDF sampled on a grid.

    This function assumes the PDF is defined by its density values at specific
    redshifts and that it can be modeled with linear interpolation between them.

    Args:
        zvector (np.ndarray): Array of redshift values where the PDF is sampled.
        pdf_density (np.ndarray): Array of probability density values P(z) at each zvector point.
        Nquantiles (int, optional): The number of quantiles to compute. 
        upsample_factor (int, optional): Factor by which to up-sample the PDF grid
                                         for more accurate CDF inversion. 

    Returns:
        np.ndarray: An array of redshift values corresponding to the quantiles.
    """
    
    # remove leading and trailing zeros in the probability density
    nonzero_indices = np.where(pdf_density > 0)[0]
    
    first_idx = nonzero_indices[0]
    last_idx = nonzero_indices[-1]
    
    # Define slice to include one adjacent zero on each side (if exists)
    slice_start = max(0, first_idx - 1)
    slice_end = min(len(pdf_density), last_idx + 2) 

    z_trimmed = zvector[slice_start:slice_end]
    pdf_trimmed = pdf_density[slice_start:slice_end]

    # upsample
    num_points = (len(z_trimmed) - 1) * upsample_factor + 1
    z_hires = np.linspace(z_trimmed[0], z_trimmed[-1], num_points)
    pdf_hires = np.interp(z_hires, z_trimmed, pdf_trimmed)
    
    # 1. Normalize the PDF so that its integral is 1.
    total_area = np.trapz(pdf_hires, z_hires)
            
    normalized_pdf = pdf_hires / total_area

    # 2. Compute the Cumulative Distribution Function (CDF) 
    # It calculates the area of each trapezoidal segment and finds the cumulative sum.
    segment_areas = 0.5 * (normalized_pdf[1:] + normalized_pdf[:-1]) * np.diff(z_hires)
    cdf = np.concatenate(([0], np.cumsum(segment_areas)))
    
    # Ensure the CDF ends at exactly 1.0 to correct for floating-point inaccuracies.
    if cdf[-1] > 0:
        cdf /= cdf[-1]

    # 3. Define the target probabilities for the desired quantiles.
    if target_quantiles is None:
        target_quantiles = np.linspace(0, 1, Nquantiles)

    # 4. Interpolate the inverted CDF to find the redshift for each target quantile.
    quantile_redshifts = np.interp(target_quantiles, cdf, z_hires)

    return quantile_redshifts    


def encode_quantiles(quantiles, packetsize=80, validate=True, tolerance=0.0001, units='redshift'):
    """Encodes an array of quantiles into a compact byte packet.

    Compresses an array of quantile locations into a fixed-size byte array.
    The encoding strategy uses a scaling factor (`epsilon`) and represents
    most gaps between quantiles with a single byte. Larger gaps are
    represented with three bytes.

    The packet structure is:
    - 1 byte: Epsilon scaling factor.
    - 2 bytes: log(1+zmin) (encoded as uint16).
    - Remaining bytes: Payload of encoded gaps, padded with zeros.

    Args:
        quantiles (np.ndarray): A 1D array of monotonic quantile locations.
        packetsize (int, optional): The total size of the output byte packet.
            Defaults to 80.
        validate (bool, optional): If True, decodes the packet immediately to
            verify that the reconstructed quantiles are within the specified
            tolerance. Defaults to True.
        tolerance (float, optional): The maximum allowed absolute difference
            between original and recovered quantiles during validation.
            Defaults to 0.0001.

    Returns:
        tuple[int, bytes]: A tuple containing:
            - L (int): The length of the generated payload (excluding header).
            - packet (bytes): The final compressed byte packet.

    Raises:
        ValueError: If the number of quantiles is too large for the packet
            size, if a suitable epsilon cannot be found, if the generated
            payload exceeds the packet size, or if validation fails.
    """

    from .decode import decode_quantiles # Local import to avoid circular dependency at module level

    Nq = len(quantiles)
    if Nq > packetsize-2:
         raise ValueError(f'Error: cannot fit {Nq} quantiles in an {packetsize}-bytes packet')

    if units == 'redshift':
        logq = np.log(1+quantiles) # convert quantiles to log(1+z) scale 
    elif units == 'zeta':
        logq = quantiles # input quantiles are already in zeta units
    else:
        print("Error: unknown units: {units}. Valid values: 'redshift','zeta'")
        sys.exit(1)
            
    # encode the first quantile as uint16
    xmin_int = int(np.floor((logq[0]+NEGATIVE_Z_OFFSET)/LOG_DZ))

    # recompute true log(1+z) of the encoded value
    logq_min = xmin_int*LOG_DZ - NEGATIVE_Z_OFFSET
    
    # update logq with encoded value for first quantile
    logq[0] = logq_min
    
    # find optimal value for epsilon
    gaps = np.sort(logq[1:]-logq[:-1])

    max_big_gaps = ((packetsize-3)-(Nq-1)) // 2 # maximum number of big gaps that fit in packet for Nq quantiles
    eps_min2 = gaps[-(max_big_gaps+1)]/254 # all but n=max_big_gaps gaps must fit in 1 byte (and value 255 is reserved)            
    eps_min3 = gaps[-1]/(256**2 -1) # the largest gap must fit in a 3-byte big gap
   # print(f'target: {tolerance}  1-byte eps: {eps_min2} 3-byte eps: {eps_min3}')
    eps_target = np.max([EPSILON_MIN,eps_min2,eps_min3]) # target for epsilon
    if eps_target > 2*tolerance:
        raise ValueError(f'Error: epsilon={eps_target} is larger than 2*tolerance.')

    eps_byte = int(np.ceil(np.log(eps_target/EPSILON_MIN)/EPSILON_BETA)) # byte encoding for epsilon
    if eps_byte > 255:
        # epsilon is too large. We need to try with fewer quantiles to fit more big gaps
        raise ValueError(f'Error: epsilon={eps_target} is too large for 1 byte encoding.')
        
    eps = EPSILON_MIN*np.exp(EPSILON_BETA*eps_byte) # actual epsilon represented by the encoded value
        
    # build payload from the quantiles 1 to N-1
    payload = bytearray()
    prev = logq_min
    for z in logq[1:]:
        d = int(round((z - prev) / eps))
        if 0 <= d <= 254:
            payload.append(d)
        else:
            payload.append(255)
            payload += struct.pack('>H', d)
        prev = prev + d * eps

    L = len(payload)
    if L > packetsize-3: 
        raise ValueError(f'Error: payload of length {L} does not fit in packet of size {packetsize}.')

    packet = bytearray(packetsize)
    packet[0] = eps_byte
    packet[1:3] = struct.pack('<H', xmin_int)            
    packet[3:3+L] = payload

    if validate: 
        qrecovered = decode_quantiles(packet,units='zeta')
        if len(qrecovered) != len(logq):
           print('Error: packet decodes to wrong number of quantiles!')
           print('packet: ',[int(x) for x in packet])
           print('recovered quantiles (zeta units):')
           print(qrecovered)
           print('original quantiles (zeta units):')
           print(logq)
           sys.exit(1)
           
        shift = logq[1:]-qrecovered[1:]
        if max(abs(shift)) > tolerance:
            #raise ValueError(f'Error: shift in quantiles exceeds tolerance = {tolerance:.1g}.')
            import code
            code.interact(local=locals())
            
    return L, bytes(packet)

def _optimize_quantile_count(data, packetsize=80, min_big_jumps=0, max_big_jumps=10, d_threshold=15, units='redshift'):
    """Finds the optimal number of quantiles for encoding each PDF."""
    
    print('Finding optimal number of quantiles for each source...')
    
    qcount_min = packetsize-2*(max_big_jumps+1) # minimum number of quantiles to test
    qcount_max = packetsize-2*(min_big_jumps+1) # maximum number of quantiles to test

    qcount_values = np.arange(qcount_min,qcount_max+1,2,dtype=int)
    qcount_cumul = np.concatenate(([0],np.cumsum(qcount_values)))
    eps_max = EPSILON_MIN*np.exp(EPSILON_BETA*255) # maximum epsilon that can be encoded 
    
    scores = np.zeros((data['PDF'].shape[0],len(qcount_values))) # array to contain scores as a function of quantile count
    
    # generate all necessary quantile values at once
    all_quantiles = np.concatenate([np.linspace(0,1,N) for N in qcount_values])
     
    for i in range(data['PDF'].shape[0]):
        if data['format'] == 'PDF_histogram':    
            zquantiles = binned_to_quantiles(data['zvector'],data['PDF'][i],target_quantiles=all_quantiles)
        if data['format'] == 'PDF_density':    
            zquantiles = density_to_quantiles(data['zvector'],data['PDF'][i],target_quantiles=all_quantiles)
        if data['format'] == 'samples':
            valid = np.isfinite(data['PDF'][i])
            zquantiles = samples_to_quantiles(data['PDF'][i][valid],target_quantiles=all_quantiles)       
         
        # compute score for each quantile count
        for j, Nq in enumerate(qcount_values):  
            if units == 'redshift':  
                logq = np.log(1+zquantiles[qcount_cumul[j]:qcount_cumul[j+1]]) # convert quantiles to log(1+z) scale 
            elif units == 'zeta':
                logq = zquantiles[qcount_cumul[j]:qcount_cumul[j+1]]
            else:
                print("Error: unknown units: {units}. Valid values: 'redshift','zeta'")
                sys.exit(1)
                    
            gaps = np.sort(logq[1:]-logq[:-1]) # sorted list of quantile gaps
            
            if gaps[0] <= 0.:
                print('Error: Zero gap found!')
                sys.exit(1)
                
            max_big_gaps = ((packetsize-3)-(Nq-1)) // 2 # maximum number of big gaps that fit in packet for Nq quantiles
            eps_min2 = gaps[-(max_big_gaps+1)]/254 # all but n=max_big_gaps gaps must fit in 1 byte (and value 255 is reserved)            
            eps_min3 = gaps[-1]/(256**2 -1) # the largest gap must fit in a 3-byte big gap
            eps_target = np.max([EPSILON_MIN,eps_min2,eps_min3]) # target for epsilon
            if (eps_target > eps_max): # epsilon is too high, do not keep adding quantiles
                if (j > 0):
                    break
            
            if (j > 0) & (eps_target < d_threshold*gaps[0]): # a gap is too narrow, do not keep adding quantiles
                break
                        
            scores[i,j] = -np.log(eps_target)*Nq
                                        
    best_qcount = qcount_values[np.argmax(scores,axis=1)] 
     
    return best_qcount    

def _batch_encode(data, ini_quantiles=72, packetsize=80, tolerance=0.0002, validate=True, optimize=True, units='redshift'):
    """Internal helper function for batch encoding of PDFs.

    This function orchestrates the encoding process for a batch of PDFs,
    which can be provided as binned histograms or random samples. It
    dynamically adjusts the number of quantiles for each PDF to find an 
    optimal fit within the specified packet size.

    Args:
        data (dict): A dictionary containing the PDF data and format.
            Expected keys: 'format' ('PDF_histogram' or 'samples'),
            and either 'PDF' and 'zvector' or 'samples'.
        ini_quantiles (int, optional): The initial number of quantiles to try
            encoding. Defaults to 72.
        packetsize (int, optional): The target size of the output byte packet.
            Defaults to 80.
        tolerance (float, optional): The tolerance for validation, passed to
            `encode_quantiles`.
        validate (bool, optional): whether to validate or not the packet after encoding. 
            Passed to `encode_quantiles`.
        optimize (bool, optional): whether to optimize the number of quantiles for each
            source (otherwise it will try a decreasing number of quantiles starting from 
            ini_quantiles until a valid solution is found).
                 
    Returns:
        np.ndarray: A 2D numpy array of type `>i4` (big-endian 4-byte
            integer), where each row is a compressed PDF packet.

    Raises:
        ValueError: If `packetsize` is not a multiple of 4 or if
            `ini_quantiles` is too large.
    """   
    if packetsize % 4 != 0:
        raise ValueError(f"Error: packetsize must be a multiple of 4, but got {packetsize}.")

    NPDFs = data['PDF'].shape[0]
    int32col = np.zeros((NPDFs,packetsize//4),dtype='>i4') 

    start = time.process_time()

    if optimize:
        Nquantiles_array = _optimize_quantile_count(data, packetsize=packetsize, units=units)
    else:
        if packetsize - ini_quantiles < 2:
            raise ValueError('Error: ini_quantiles must be at most packetsize-2')
           
    for i in range(NPDFs):
        if optimize:
            Nquantiles = Nquantiles_array[i]
        else:
            Nquantiles = ini_quantiles
            
        lastgood = None
        while True:
            if data['format'] == 'PDF_histogram':    
                quantiles = binned_to_quantiles(data['zvector'],data['PDF'][i],Nquantiles=Nquantiles)
            if data['format'] == 'PDF_density':    
                quantiles = density_to_quantiles(data['zvector'],data['PDF'][i],Nquantiles=Nquantiles)
            if data['format'] == 'samples':
                valid = np.isfinite(data['PDF'][i])
                quantiles = samples_to_quantiles(data['PDF'][i][valid],Nquantiles=Nquantiles)
            try:
                payload_length, packet = encode_quantiles(quantiles,packetsize=packetsize,tolerance=tolerance,validate=validate,units=units)
                break
            except ValueError as e:
                if 'packet decodes' in str(e):
                    print(e,file=sys.stderr)
                    sys.exit(1)          
                else:
                    Nquantiles -= 2
                   # print(f'Retrying source {i} with {Nquantiles} quantiles.')
                    if Nquantiles < packetsize/3:
                        print(f"WARNING: The PDF for source #{i} could not be encoded!")
                        if data['format'] == 'samples':
                            print(f"samples z range: {np.nanmin(data['PDF'][i]):.4f} {np.nanmax(data['PDF'][i]):.4f}")   
                        packet = bytearray(packetsize) # returns all zeros
                        break
                    continue    
         
        int32col[i] = np.frombuffer(packet, dtype='>i4')

    end = time.process_time()
    cpu_seconds = end - start
    print(f"{NPDFs} PDFs cold-pressed in {cpu_seconds:.6f} CPU seconds")

    return int32col


def encode_from_binned(PDF, zvector, ini_quantiles=72, packetsize=80, tolerance=0.0002, validate=True, optimize=True, units='redshift'):
    """Encodes binned PDFs into compressed byte packets.

    This is a high-level wrapper that takes binned PDFs and encodes them
    by calling the internal `_batch_encode` function.

    Args:
        PDF (np.ndarray): A 2D array where each row is a binned PDF.
        zvector (np.ndarray): A 1D array of the redshift bin centers.
        ini_quantiles (int, optional): Initial number of quantiles to try.
            Defaults to 71.
        packetsize (int, optional): Target size of the output byte packet.
            Defaults to 80.
        tolerance (float, optional): Tolerance for validation.
        validate (bool, optional): Whether to perform validation.

    Returns:
        np.ndarray: A 2D array where each row is a compressed PDF packet.
    """
    # select valid PDFs for encoding
    Nsources = PDF.shape[0]
    encoded = np.zeros((Nsources,packetsize//4),dtype='>i4') 
    valid = (np.sum(PDF,axis=1) > 0) & (np.min(PDF,axis=1) >= 0)
    
    data = {'format': 'PDF_histogram', 'zvector': zvector, 'PDF': PDF[valid]}    
    encoded[valid] = _batch_encode(data, ini_quantiles=ini_quantiles, packetsize=packetsize, tolerance=tolerance, validate=validate, units=units)
    return encoded
    
def encode_from_density(PDF, zvector, ini_quantiles=72, packetsize=80, tolerance=0.0002, validate=True, optimize=True, units='redshift'):
    """Encodes Probability densities sampled in a grid into compressed byte packets.

    This is a high-level wrapper that takes PDFs and encodes them
    by calling the internal `_batch_encode` function.

    Args:
        PDF (np.ndarray): A 2D array where each row is a PDF sampled in a grid.
        zvector (np.ndarray): A 1D array of the redshift grid values.
        ini_quantiles (int, optional): Initial number of quantiles to try.
            Defaults to 71.
        packetsize (int, optional): Target size of the output byte packet.
            Defaults to 80.
        tolerance (float, optional): Tolerance for validation.
        validate (bool, optional): Whether to perform validation.

    Returns:
        np.ndarray: A 2D array where each row is a compressed PDF packet.
    """
    # select valid PDFs for encoding
    Nsources = PDF.shape[0]
    encoded = np.zeros((Nsources,packetsize//4),dtype='>i4') 
    valid = (np.sum(PDF,axis=1) > 0) & (np.min(PDF,axis=1) >= 0)
    
    data = {'format': 'PDF_density', 'zvector': zvector, 'PDF': PDF[valid]}    
    encoded[valid] = _batch_encode(data, ini_quantiles=ini_quantiles, packetsize=packetsize, tolerance=tolerance, validate=validate, units=units)
    return encoded
    
def encode_from_samples(samples, ini_quantiles=72, packetsize=80, tolerance=0.0002, validate=True, optimize=True, clip_fraction=0., clip_range=None, min_valid_samples=100, units='redshift'):
    """Encodes PDFs from random samples into compressed byte packets.

    This is a high-level function that takes random samples for each PDF,
    cleans them up, and encodes the valid ones by calling the internal 
    `_batch_encode` function.

    Args:
        samples (np.ndarray): A 2D array where each row contains random
            samples from a single PDF.
        ini_quantiles (int, optional): Initial number of quantiles to try.
            Defaults to 71.
        packetsize (int, optional): Target size of the output byte packet.
            Defaults to 80.
        tolerance (float, optional): Tolerance for validation.
        validate (bool, optional): Whether to perform validation.

    Returns:
        np.ndarray: A 2D array where each row is a compressed PDF packet.
    """
    
    # define default clip_range if not provided or incomplete
    if units == 'redshift':
        if clip_range is None:
            clip_range=[Q0_ZMIN,Q0_ZMAX]
        if clip_range[0] is None:
            clip_range[0] = Q0_ZMIN
        if clip_range[1] is None:
            clip_range[1] = Q0_ZMAX
    elif units == 'zeta':
        if clip_range is None:
            clip_range=[Q0_ZETAMIN,Q0_ZETAMAX]
        if clip_range[0] is None:
            clip_range[0] = Q0_ZETAMIN
        if clip_range[1] is None:
            clip_range[1] = Q0_ZETAMAX       
    
    # create array of integers to contain encoded PDFs
    Nsources = samples.shape[0]
    encoded = np.zeros((Nsources,packetsize//4),dtype='>i4') 
 
    # clean up samples for non-finite values or those outside the desired range
    valid_samples = np.isfinite(samples) & (samples >= clip_range[0]) & (samples <= clip_range[1])
    # count valid samples per source
    n_valid = np.sum(valid_samples, axis=1)

    # Reject sources with fewer valid samples than quantiles requested, or all samples with same value
    
    zmin = np.nanmin(samples, axis=1)
    zmax = np.nanmax(samples, axis=1)
    valid_source = (n_valid >= min_valid_samples) & (zmax-zmin > 0)
    n_valid_sources = len(valid_source[valid_source])
        
    # create array to contain clean samples
    clean_samples = np.full((n_valid_sources,np.max(n_valid)),np.nan,dtype=float)
    
    # sort samples and (optionally) remove extreme values
    i = -1 # counter for all sources
    j = -1 # counter for valid sources
    while i < Nsources-1:
        i += 1
        if not valid_source[i]:
            continue 
        j += 1     
        if clip_fraction > 0:
            zsorted = np.sort(samples[i,valid_samples[i]])
            nclip = int(np.floor(len(zsorted)*clip_fraction))
            if nclip > 0:
                zsorted = zsorted[nclip:-nclip]
        else:
            zsorted = np.sort(samples[i,valid_samples[i]])
            nclean = zsorted.shape[0]
        clean_samples[j,:nclean] = zsorted  
    
    data = {'format': 'samples', 'PDF': clean_samples}
    encoded[valid_source] = _batch_encode(data, ini_quantiles=ini_quantiles, packetsize=packetsize, tolerance=tolerance, validate=validate, optimize=optimize, units=units)
    return encoded