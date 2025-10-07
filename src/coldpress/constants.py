from math import exp

###### Common constants for encoding and decoding algorithms ###################

# Offset that allows encoding of log(1+z) as a positive integer for small z<0
NEGATIVE_Z_OFFSET = 0.020202708 # a_shift

# Resolution in log(1+z) at the redshift of the first quantile
LOG_DZ = 4.0e-5 # df

# Minimum epsilon value that can be encoded in the first byte of the header
EPSILON_MIN = 2.0e-8

# Exponential term for encoding epsilon (determines resolution and maximum value)
EPSILON_BETA = 0.0361

# Minimum redshift for first quantile that can be encoded
Q0_ZMIN = exp(-NEGATIVE_Z_OFFSET) - 1

# Minimum zeta for first quantile that can be encoded
Q0_ZETAMIN = -NEGATIVE_Z_OFFSET

# Maximum redshift for first quantile that can be encoded
Q0_ZMAX = exp(LOG_DZ*(256**2-1) - NEGATIVE_Z_OFFSET) - 1

# Maximum zeta for first quantile that can be encoded
Q0_ZETAMAX = LOG_DZ*(256**2-1) - NEGATIVE_Z_OFFSET