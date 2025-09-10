"""ColdPress: Efficient compression for redshift probability density functions.

This module provides functions to encode and decode redshift probability density
functions (PDFs) into a compact, fixed-size byte representation suitable for
efficient storage in databases.

It works by computing the redshifts {z_i} that correspond to the quantiles {q_i}
of the cumulative distribution function (CDF) and encoding the differences
∆i = z_i - z_{i-1} using (mostly) a single byte.

Attributes:
    __author__ (str): The primary author of the package.
    __email__ (str): The author's contact email.
    __version__ (str): The current version of the package.
    __license__ (str): The license under which the package is distributed.
    __copyright__ (str): The package's copyright notice.

Citation:
    If you use ColdPress in your research, please cite:
    Hernán-Caballero, A. 2025, Research Notes of the AAS, 9, 7, 170.
    DOI:10.3847/2515-5172/adeca6
"""

__author__ = "Antonio Hernán Caballero"
__email__ = "ahernan@cefca.es"
__version__ = "1.0.1"
__license__ = "GPLv3"
__copyright__ = "Copyright 2025, Antonio Hernán Caballero"

from .encode import encode_from_binned, encode_from_samples, encode_from_density, binned_to_quantiles
from .decode import (
    decode_quantiles, 
    quantiles_to_binned, 
    quantiles_to_density, 
    quantiles_to_samples, 
    decode_to_binned, 
    decode_to_density, 
    decode_to_samples
)
from .stats import (
    measure_from_quantiles,
    zmode_from_quantiles,
    zmedian_from_quantiles,
    zmean_from_quantiles,
    zmean_err_from_quantiles,
    zrandom_from_quantiles,
    odds_from_quantiles,
    HPDCI_from_quantiles,
)
from .utils import step_pdf_from_quantiles, plot_from_quantiles

__all__ = [
    'binned_to_quantiles',
    'density_to_quantiles',
    'samples_to_quantiles',
    'encode_quantiles',
    'encode_from_binned',
    'encode_from_density',
    'encode_from_samples',
    'decode_quantiles',
    'decode_to_binned',
    'decode_to_density',
    'decode_to_samples',
    'quantiles_to_binned',
    'quantiles_to_density',
    'quantiles_to_samples',    
    'measure_from_quantiles',
    'zmode_from_quantiles',
    'zmedian_from_quantiles',
    'zmean_from_quantiles',
    'zmean_err_from_quantiles',
    'zrandom_from_quantiles',
    'odds_from_quantiles',
    'HPDCI_from_quantiles',
    'step_pdf_from_quantiles',
    'plot_from_quantiles'
]
