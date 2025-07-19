import numpy as np

QUANTITY_DESCRIPTIONS = {
    'Z_MODE': 'Mode of the redshift PDF, defined as the redshift with maximum probability density.',
    'Z_MEAN': 'Mean of the redshift PDF, defined as the integral over z of z*P(z).',
    'Z_MEDIAN': 'Median of the redshift PDF (i.e., the redshift that has a 50/50 chance of the true redshift being on either side).',
    'Z_RANDOM': 'A random redshift value obtained with the PDF as the underlying probability distribution.',
    'Z_MODE_ERR': '1-sigma uncertainty in Z_MODE.',
    'Z_MEAN_ERR': '1-sigma uncertainty in Z_MEAN.',
    'ODDS_MODE': 'Odds parameter for Z_MODE.',
    'ODDS_MEAN': 'Odds parameter for Z_MEAN.',
    'Z_MIN_HPDCI68': 'Lower bound of the 68% highest posterior density credible interval.',
    'Z_MAX_HPDCI68': 'Upper bound of the 68% highest posterior density credible interval.',
    'Z_MIN_HPDCI95': 'Lower bound of the 95% highest posterior density credible interval.',
    'Z_MAX_HPDCI95': 'Upper bound of the 95% highest posterior density credible interval.',
    'ODDS_MODE': 'Probability that the true redshift lies within a specific interval around Z_MODE (default is ± 0.03 × (1 + Z_MODE).',
    'ODDS_MEAN': 'Probability that the true redshift lies within a specific interval around Z_MEAN (default is ± 0.03 × (1 + Z_MEAN).'
}

ALL_QUANTITIES = set(QUANTITY_DESCRIPTIONS.keys())

def measure_from_quantiles(quantiles, quantities_to_measure, odds_window=0.03):
    """Computes a set of statistical quantities from a PDF's quantiles.

    This function acts as a dispatcher, calculating various point estimates
    and other statistics based on a list of requested quantities. It handles
    dependencies between calculations (e.g., `Z_MODE` is needed for
    `ODDS_MODE`).

    Args:
        quantiles (np.ndarray): A 1D array of monotonic quantile locations for
            a single PDF.
        quantities_to_measure (list[str]): A list of strings specifying the
            desired quantities. The strings should match the keys in
            `QUANTITY_DESCRIPTIONS`. Use 'ALL' to compute all available
            quantities.
        odds_window (float, optional): The half-width of the integration
            window for odds calculations, as a fraction of (1+z).
            Defaults to 0.03.

    Returns:
        dict[str, float]: A dictionary mapping the name of each requested
            quantity to its calculated value.

    Raises:
        ValueError: If an unknown quantity is requested.
    """
    dependencies = {
        'Z_MODE': ['Z_MIN_HPDCI68','Z_MAX_HPDCI68'],
        'Z_MODE_ERR': ['Z_MODE'],
        'ODDS_MODE': ['Z_MODE'],
        'ODDS_MEAN': ['Z_MEAN']
    }

    # Determine which quantities to compute
    requested_q = {q.upper() for q in quantities_to_measure}

    if 'ALL' in requested_q:
        q_to_compute = ALL_QUANTITIES
    else:
        unknown_q = requested_q - ALL_QUANTITIES
        if unknown_q:
            raise ValueError(f"Unknown quantities specified: {', '.join(unknown_q)}")
        q_to_compute = requested_q

    # Resolve dependencies to determine all internal calculations needed
    internal_calcs = set(q_to_compute)
    for q in q_to_compute:
        if q in dependencies:
            internal_calcs.update(dependencies[q])

    # --- Perform all necessary calculations ---
    temp_results = {}
    if 'Z_MIN_HPDCI68' in internal_calcs or 'Z_MAX_HPDCI68' in internal_calcs:
        HPDCI68_zmin, HPDCI68_zmax = HPDCI_from_quantiles(quantiles, conf=0.68, zinside=None)
        temp_results['Z_MIN_HPDCI68'] = HPDCI68_zmin
        temp_results['Z_MAX_HPDCI68'] = HPDCI68_zmax
    if 'Z_MIN_HPDCI95' in internal_calcs or 'Z_MAX_HPDCI95' in internal_calcs:
        HPDCI95_zmin, HPDCI95_zmax = HPDCI_from_quantiles(quantiles, conf=0.95, zinside=None)
        temp_results['Z_MIN_HPDCI95'] = HPDCI95_zmin
        temp_results['Z_MAX_HPDCI95'] = HPDCI95_zmax
    if 'Z_MODE' in internal_calcs:
        temp_results['Z_MODE'] = zmode_from_quantiles(quantiles, hpdci68=(HPDCI68_zmin,HPDCI68_zmax))
    if 'Z_MEAN' in internal_calcs:
        temp_results['Z_MEAN'] = zmean_from_quantiles(quantiles)
    if 'Z_MEDIAN' in internal_calcs:
        temp_results['Z_MEDIAN'] = zmedian_from_quantiles(quantiles)
    if 'Z_RANDOM' in internal_calcs:
        temp_results['Z_RANDOM'] = zrandom_from_quantiles(quantiles)
    if 'Z_MEAN_ERR' in internal_calcs:
        temp_results['Z_MEAN_ERR'] = zmean_err_from_quantiles(quantiles)
    if 'ODDS_MODE' in internal_calcs:
        temp_results['ODDS_MODE'] = odds_from_quantiles(quantiles, temp_results['Z_MODE'], odds_window=odds_window)
    if 'ODDS_MEAN' in internal_calcs:
        temp_results['ODDS_MEAN'] = odds_from_quantiles(quantiles, temp_results['Z_MEAN'], odds_window=odds_window)
    if 'Z_MODE_ERR' in internal_calcs:
        temp_results['Z_MODE_ERR'] = 0.5 * (HPDCI68_zmax - HPDCI68_zmin)

    # Filter the results to return only the originally requested quantities
    final_results = {key: temp_results[key] for key in q_to_compute if key in temp_results}
    
    return final_results

# def zmode_from_quantiles(quantiles, width=0.005):
#     """Calculates the mode of the PDF from its quantiles.
# 
#     This approach finds the location of the highest probability density by
#     identifying the narrowest interval for a given change in the cumulative
#     probability. It inspects intervals centered on the quantiles to mitigate
#     quantization effects.
# 
#     Args:
#         quantiles (np.ndarray): A 1D array of monotonic quantile locations.
#         width (float, optional): The half-width of the sliding window used
#             to estimate local density. Defaults to 0.005.
# 
#     Returns:
#         float: The estimated modal redshift (Z_MODE).
#     """
#     Nq = len(quantiles)
#     knots = np.linspace(0,1.0,Nq)
#     dkplus = np.interp(quantiles+width,quantiles,knots,left=0,right=1)-knots
#     dkminus = knots-np.interp(quantiles-width,quantiles,knots,left=0,right=1)
#     if max(dkplus) > max(dkminus):
#         i = np.argmax(dkplus)
#         return quantiles[i]+width/2.
#     else:
#         i = np.argmax(dkminus)
#         return quantiles[i]-width/2.
   
def zmode_from_quantiles(quantiles, hpdci68=None):
    # Compute HPDCI68 if not provided
    if hpdci68 is None:
        hpdci68 = HPDCI_from_quantiles(quantiles, conf=0.68)
    
    # find quantiles inside HPDCI68
    inside = np.where((quantiles >= hpdci68[0]) & (quantiles <= hpdci68[1]))[0]
    if len(inside) == 0:
        print('No quantiles inside HPDCI68??')
        import code
        code.interact(local=locals())
    minq = np.max((np.min(inside)-1,0))
    maxq = np.min((np.max(inside)+2,len(quantiles)))    
    
    qclipped = quantiles[minq:maxq] # remove quantiles intervals that do not overlap with HPDCI68
        
    diff = qclipped[1:] - qclipped[:-1]
    u = np.argmin(diff)
    return 0.5*(qclipped[u]+qclipped[u+1])
   
def zmedian_from_quantiles(quantiles):
    """Calculates the median redshift from the PDF's quantiles.

    The median is the 50th percentile of the distribution, which is found by
    interpolating the quantiles to the cumulative probability value of 0.5.

    Args:
        quantiles (np.ndarray): A 1D array of monotonic quantile locations.

    Returns:
        float: The median redshift (Z_MEDIAN).
    """
    knots = np.linspace(0, 1, len(quantiles))
    return np.interp(0.5, knots, quantiles)

def zmean_from_quantiles(quantiles):
    """Calculates the mean redshift from the PDF's quantiles.

    The mean is computed by integrating z * P(z) dz. This is equivalent to
    integrating the quantile function (z as a function of cumulative
    probability F) from F=0 to F=1.

    Args:
        quantiles (np.ndarray): A 1D array of monotonic quantile locations.

    Returns:
        float: The mean redshift (Z_MEAN).
    """
    knots = np.linspace(0, 1, len(quantiles))
    return np.trapz(quantiles, knots)

def zmean_err_from_quantiles(quantiles):
    """Calculates the standard deviation (error) of the mean redshift.

    Computes the standard deviation of the PDF, which is the square root of
    the variance (E[z^2] - (E[z])^2).

    Args:
        quantiles (np.ndarray): A 1D array of monotonic quantile locations.

    Returns:
        float: The standard deviation of the redshift distribution.
    """
    knots = np.linspace(0, 1, len(quantiles))
    mean = np.trapz(quantiles, knots)
    ez2 = np.trapz(quantiles**2, knots)
    variance = ez2 - mean**2
    return np.sqrt(variance)

def zrandom_from_quantiles(quantiles):
    """Draws a single random redshift value from the PDF.

    This uses inverse transform sampling by drawing a uniform random number
    on [0, 1] (representing the cumulative probability) and finding the
    corresponding redshift by interpolating the quantiles.

    Args:
        quantiles (np.ndarray): A 1D array of monotonic quantile locations.

    Returns:
        float: A single random redshift draw.
    """
    u = np.random.uniform(0, 1)
    knots = np.linspace(0, 1, len(quantiles))
    return np.interp(u, knots, quantiles)

def odds_from_quantiles(quantiles, zcenter, odds_window=0.03):
    """Calculates the 'odds' parameter for a given redshift.

    The odds parameter is defined as the total probability contained within a
    window centered on `zcenter`. The window size is `± odds_window * (1+zcenter)`.

    Args:
        quantiles (np.ndarray): A 1D array of monotonic quantile locations.
        zcenter (float): The central redshift around which to calculate the odds.
        odds_window (float, optional): The half-width of the integration window
            as a fraction of (1+z). Defaults to 0.03.

    Returns:
        float: The integrated probability within the defined window.
    """
    knots = np.linspace(0, 1, len(quantiles))
    zbinmin = zcenter - odds_window*(1+zcenter)
    zbinmax = zcenter + odds_window*(1+zcenter)
    qz = np.interp([zbinmin,zbinmax],quantiles,knots,left=0,right=1)
    return qz[1]-qz[0]

def HPDCI_from_quantiles(quantiles, conf=0.68, zinside=None):
    """Calculates the Highest Probability Density Credible Interval (HPDCI).

    Finds the narrowest possible interval of redshift that contains a
    specified fraction (`conf`) of the total probability. If `zinside` is
    provided, it finds the interval containing that point; otherwise, it
    searches all possible intervals.

    Args:
        quantiles (np.ndarray): A 1D array of monotonic quantile locations.
        conf (float, optional): The confidence level (i.e., total probability)
            the interval should contain. Defaults to 0.68.
        zinside (float, optional): A specific redshift that the interval must
            contain. If None, the globally narrowest interval is found.
            Defaults to None.

    Returns:
        tuple[float, float]: A tuple containing the lower and upper bounds
            (zmin, zmax) of the calculated credible interval.
    """
    knots = np.linspace(0, 1, len(quantiles))
    knot_interval = knots[1]-knots[0]

    if zinside is not None:
        qin = np.interp(zinside,quantiles,knots,left=0,right=1)
        qmin = max([0,qin-conf])
        qmax = min([1,qin+conf])
    else:
        qmin = 0
        qmax = 1

    start = np.arange(qmin,qmax-conf,0.2*knot_interval)
    end = np.arange(qmin+conf,qmax,0.2*knot_interval)
    
    if len(start) == 0 or len(end) == 0:
        z_center = np.interp(0.5, knots, quantiles)
        return (z_center, z_center)
        
    zmin = np.interp(start,knots,quantiles)
    zmax = np.interp(end,knots,quantiles)
    dz = zmax-zmin
    best = np.argmin(dz)
    return (zmin[best], zmax[best])