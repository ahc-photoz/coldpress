import numpy as np
import struct

def samples_to_quantiles(samples, Nquantiles=100):
    """Calculates quantiles from a set of random samples of a PDF.

    This function takes random draws from a probability distribution and
    computes the values that correspond to evenly spaced quantiles
    (e.g., percentiles). It filters out non-finite values before computation.

    Args:
        samples (np.ndarray): A 1D array of random samples from the PDF.
        Nquantiles (int, optional): The number of quantiles to compute.
            Defaults to 100.

    Returns:
        np.ndarray: A 1D array of `Nquantiles` values representing the
            quantile locations.
    """
    valid = np.isfinite(samples)
    zsorted = np.sort(samples[valid])
    targets = np.linspace(0.0, 1.0, Nquantiles) # target probability for quantiles
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


def encode_quantiles(quantiles, packetsize=80, validate=True, tolerance=0.001):
    """Encodes an array of quantiles into a compact byte packet.

    Compresses an array of quantile locations into a fixed-size byte array.
    The encoding strategy uses a scaling factor (`epsilon`) and represents
    most gaps between quantiles with a single byte. Larger gaps are
    represented with three bytes.

    The packet structure is:
    - 1 byte: Epsilon scaling factor.
    - 2 bytes: zmin (encoded as uint16).
    - 2 bytes: zmax (encoded as uint16).
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
    if Nq > packetsize-3:
        raise ValueError(f'Error: cannot fit {Nq} quantiles in an {packetsize}-bytes packet')

    zmin, zmax = quantiles[0], quantiles[-1]

    # encode endpoints as uint16
    xmin_int = int(np.floor((zmin+0.01) / 0.0002))
    xmax_int = int(np.ceil((zmax+0.01) / 0.0002))

    # recompute true zmin/zmax & epsilon
    zmin_rec = xmin_int * 0.0002 - 0.01
    zmax_rec = xmax_int * 0.0002 - 0.01

    max_big_gaps = (packetsize-(Nq+3)) // 2 

    gaps = np.sort(quantiles[1:-1]-quantiles[:-2]) 
    gapthreshold = gaps[-max_big_gaps-1] 
    eps_min = 0.00001*np.ceil(100000*gapthreshold/254)

    if eps_min > 0.00255:
        raise ValueError('Error: minimum usable epsilon={eps_min} is too large. Increase packet length or decrease number of quantiles.')

    eps_min2 = 0.00001*np.ceil(100000*gaps[-1]/(256**2 -1))

    eps = np.max([0.00001,eps_min,eps_min2])    

    if int(np.round(eps*100000)) > 255:
        raise ValueError(f'Error: epsilon={eps} is too large for 1 byte encoding.')

    # build payload from the *interior* N-2 quantiles
    payload = bytearray()
    prev = zmin_rec
    for z in quantiles[1:-1]:
        d = int(round((z - prev) / eps))
        if 0 <= d <= 254:
            payload.append(d)
        else:
            payload.append(255)
            payload += struct.pack('>H', d)
        prev = prev + d * eps

    L = len(payload)
    if L > packetsize-5: 
        raise ValueError(f'Error: payload of length {L} does not fit in packet of size {packetsize}.')

    packet = bytearray(packetsize)
    packet[0] = int(np.round(eps*100000))
    packet[1:3] = struct.pack('<H', xmin_int)            
    packet[3:5] = struct.pack('<H', xmax_int)
    packet[5:5+L] = payload

    if validate: 
        qrecovered = decode_quantiles(packet)
        if len(qrecovered) != len(quantiles):
            raise ValueError('Error: packet decodes to wrong number of quantiles.')
        shift = quantiles[1:-1]-qrecovered[1:-1]
        if max(abs(shift)) > tolerance:
            raise ValueError('Error: shift in quantiles exceeds tolerance.')

    return L, bytes(packet)

def _batch_encode(data, ini_quantiles=71, packetsize=80, tolerance=None, validate=None):
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

    if packetsize - ini_quantiles < 3:
        raise ValueError('Error: ini_quantiles must be at most packetsize-3')

    if data['format'] == 'PDF_histogram':
        valid = np.sum(data['PDF'],axis=1) > 0
    if data['format'] == 'samples':
        valid = np.all(np.isfinite(data['samples']), axis=1)

    int32col = np.zeros((len(valid),packetsize//4),dtype='>i4') 

    for i in range(len(valid)):
        if not valid[i]:
            continue

        Nquantiles = ini_quantiles
        lastgood = None
        while True:
            if data['format'] == 'PDF_histogram':    
                quantiles = binned_to_quantiles(data['zvector'],data['PDF'][i],Nquantiles=Nquantiles)
            if data['format'] == 'samples':
                quantiles = samples_to_quantiles(data['samples'][i],Nquantiles=Nquantiles)

            try:
                payload_length, packet = encode_quantiles(quantiles,packetsize=packetsize,tolerance=tolerance,validate=validate)
            except ValueError as e:
                if lastgood is not None:
                    packet = lastgood
                    break
                else:
                    Nquantiles -= 2
                    continue    

            if payload_length < packetsize-5:
                lastgood = packet
                Nquantiles += 2
                continue

            if payload_length == packetsize-5:
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