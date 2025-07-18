###### Common constants for encoding and decoding algorithms ###################

# Offset that allows encoding of log(1+z) as a positive integer for z > -0.02
NEGATIVE_Z_OFFSET = 0.020202707 # a_shift

# Resolution in log(1+z) at the redshift of the first quantile
LOG_DZ = 4.0e-5 # df

# Minimum epsilon value that can be encoded in the first byte of the header
EPSILON_MIN = 2.0e-8

# Exponential term for encoding epsilon (determines resolution and maximum value)
EPSILON_BETA = 0.0361