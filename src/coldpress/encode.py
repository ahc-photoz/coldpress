import numpy as np
import struct
from .constants import NEGATIVE_Z_OFFSET, LOG_DZ, EPSILON_MIN, EPSILON_BETA, Q0_ZMIN, Q0_ZMAX
import sys

def samples_to_quantiles(samples, Nquantiles=100, clip_fraction=0.0, clip_range=[Q0_ZMIN,20.]):
    """Calculates quantiles from a set of random samples of a PDF.

    This function takes random draws from a probability distribution and
    computes the values that correspond to evenly spaced quantiles
    (e.g., percentiles). It filters out non-finite values before computation.

    Args:
        samples (np.ndarray): A 1D array of random samples from the PDF.
        Nquantiles (int, optional): The number of quantiles to compute.
            Defaults to 100.
        clip_range (list, optional): minimum and maximum redshifts considered valid.    
        clip_fraction (float, optional): The fraction of outlier samples inside clip_range 
        to be clipped at each extreme of the distribution.

    Returns:
        np.ndarray: A 1D array of `Nquantiles` values representing the
            quantile locations.
    """
    valid = np.isfinite(samples) & (samples > clip_range[0]) & (samples < clip_range[1])
    Nvalid = len(valid[valid])
    if Nvalid < Nquantiles:
        raise ValueError('There are too few valid samples ({Nvalid}) to compute {Nquantiles} quantiles.')
        
    zsorted = np.sort(samples[valid])
    targets = np.linspace(0.0, 1.0, Nquantiles) # target probability for quantiles

    if clip_fraction > 0: # clip the most extreme values from the distribution
        nclip = int(np.floor(len(zsorted)*clip_fraction))
        if nclip > 0:
            zsorted = zsorted[nclip:-nclip]
            
    return np.quantile(zsorted, targets, method='linear')


def binned_to_quantiles(z_grid, Pz, Nquantiles=100):
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
    targets = np.linspace(0.0, 1.0, Nquantiles) # target probability for quantiles
    qs = np.interp(targets, cdf_edges, edges)

    return qs
    
def density_to_quantiles(zvector, pdf_density, Nquantiles=100, upsample_factor=10):
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
    target_quantiles = np.linspace(0, 1, Nquantiles)

    # 4. Interpolate the inverted CDF to find the redshift for each target quantile.
    quantile_redshifts = np.interp(target_quantiles, cdf, z_hires)

    return quantile_redshifts    


def encode_quantiles(quantiles, packetsize=80, validate=True, tolerance=0.001):
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
            Defaults to 0.001.

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

    logq = np.log(1+quantiles) # convert quantiles to log(1+z) scale 
    
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
    
    eps_target = np.max([EPSILON_MIN,eps_min2,eps_min3]) # target for epsilon
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
        qrecovered = decode_quantiles(packet)
        if len(qrecovered) != len(quantiles):
           print('Error: packet decodes to wrong number of quantiles.')
           print('packet: ',[int(x) for x in packet])
           print('recovered quantiles:')
           print(qrecovered)
           print('original quantiles:')
           print(quantiles)
           import code
           code.interact(local=locals())
           
           # raise ValueError('Error: packet decodes to wrong number of quantiles.')
        shift = quantiles[1:]-qrecovered[1:]
        if max(abs(shift)) > tolerance:
            raise ValueError('Error: shift in quantiles exceeds tolerance = {tolerance:.5f}.')
    
    return L, bytes(packet)

def _batch_encode(data, ini_quantiles=72, packetsize=80, tolerance=None, validate=None, clip_fraction=None):
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
            encoding. Defaults to 71.
        packetsize (int, optional): The target size of the output byte packet.
            Defaults to 80.
        tolerance (float, optional): The tolerance for validation, passed to
            `encode_quantiles`.
        validate (bool, optional): The validation flag, passed to
            `encode_quantiles`.

    Returns:
        np.ndarray: A 2D numpy array of type `>i4` (big-endian 4-byte
            integer), where each row is a compressed PDF packet.

    Raises:
        ValueError: If `packetsize` is not a multiple of 4 or if
            `ini_quantiles` is too large.
    """   
    if packetsize % 4 != 0:
        raise ValueError(f"Error: packetsize must be a multiple of 4, but got {packetsize}.")

    if packetsize - ini_quantiles < 2:
        raise ValueError('Error: ini_quantiles must be at most packetsize-2')

    if data['format'] == 'PDF_histogram':
        valid = np.sum(data['PDF'],axis=1) > 0
    if data['format'] == 'PDF_density':
        valid = np.sum(data['PDF'],axis=1) > 0
    if data['format'] == 'samples':
        # check that at least some samples are finite, within encodable z range, and different from each other
        zmin = np.nanmin(data['samples'], axis=1)
        zmax = np.nanmax(data['samples'], axis=1)
        valid = (zmin < Q0_ZMAX) & (zmax > Q0_ZMIN) & (zmax-zmin > 0)

    int32col = np.zeros((len(valid),packetsize//4),dtype='>i4') 

    for i in range(len(valid)):
        if not valid[i]:
            continue

        Nquantiles = ini_quantiles
        lastgood = None
        while True:
            if data['format'] == 'PDF_histogram':    
                quantiles = binned_to_quantiles(data['zvector'],data['PDF'][i],Nquantiles=Nquantiles)
            if data['format'] == 'PDF_density':    
                quantiles = density_to_quantiles(data['zvector'],data['PDF'][i],Nquantiles=Nquantiles)
            if data['format'] == 'samples':
                quantiles = samples_to_quantiles(data['samples'][i],Nquantiles=Nquantiles,clip_fraction=clip_fraction)

            try:
                payload_length, packet = encode_quantiles(quantiles,packetsize=packetsize,tolerance=tolerance,validate=validate)
            except ValueError as e:
                if 'packet decodes' in str(e):
                    print(e,file=sys.stderr)
                    sys.exit(1)
                
                if lastgood is not None:
                    packet = lastgood
                    break
                else:
                    if Nquantiles < 10:
                        print('Error: the quantile counts have decreased too much!')
                        import code
                        code.interact(local=locals())
                    Nquantiles -= 2
                    continue    

            if payload_length < packetsize-3:
                lastgood = packet
                Nquantiles += 2
                continue

            if payload_length == packetsize-3:
                break
         
        int32col[i] = np.frombuffer(packet, dtype='>i4')

    return int32col


def encode_from_binned(PDF, zvector, ini_quantiles=71, packetsize=80, tolerance=None, validate=None):
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
    data = {'format': 'PDF_histogram', 'zvector': zvector, 'PDF': PDF}    
    return _batch_encode(data, ini_quantiles=ini_quantiles, packetsize=packetsize, tolerance=tolerance, validate=validate)

def encode_from_density(PDF, zvector, ini_quantiles=71, packetsize=80, tolerance=None, validate=None):
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
    data = {'format': 'PDF_density', 'zvector': zvector, 'PDF': PDF}    
    return _batch_encode(data, ini_quantiles=ini_quantiles, packetsize=packetsize, tolerance=tolerance, validate=validate)

def encode_from_samples(samples, ini_quantiles=71, packetsize=80, tolerance=None, validate=None):
    """Encodes PDFs from random samples into compressed byte packets.

    This is a high-level wrapper that takes random samples for each PDF
    and encodes them by calling the internal `_batch_encode` function.

    Args:
        samples (np.ndarray): A 2D array where each row contains random
            samples from a single PDF.
        ini_quantiles (int, optional): Initial number of quantiles to try.
            Defaults to 71.
        packetsize (int, optional): Target size of the output byte packet.
            Defaults to 80.
        tolerance (float, optional): Tolerance for validation.
        validate (bool, optional): Whether to perform validation.
        debug (bool, optional): Unused debug flag.

    Returns:
        np.ndarray: A 2D array where each row is a compressed PDF packet.
    """
    data = {'format': 'samples', 'samples': samples}
    return _batch_encode(data, ini_quantiles=ini_quantiles, packetsize=packetsize, tolerance=tolerance, validate=validate)