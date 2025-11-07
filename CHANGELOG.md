# Changelog

## [1.1.0] - 2025-11-08

### Added
- **Support for zeta units.** New keyword *units* in encoding, decoding, and plotting functions allows to
    specify if the PDF is sampled in units of redshift (*z*) or *ζ* = ln(1+*z*).
    
- **Auto-optimization of the number of quantiles.** Batch-encoding functions now accept a new keyword *optimize* that instructs *ColdPress* to find the optimal number of quantiles for encoding each PDF within the given packet length. It finds a compromise between the competing goals of minimizing *epsilon* and maximizing the quantile count.
    
### Changed

- **Reorganized CLI options.** Some arguments for the encode and decode modes of the CLI have been redefined to accomodate both redshift and zeta units.

- **Modified treatment of tolerance parameter.** Now *tolerance* refers to the maximum error in ln(1+*z*), instead of *z*, for the quantiles. The condition on *epsilon* has been updated to *epsilon* < 2 * *tolerance*.


### Fixed

- **Ambiguous encoding for zero-sized last jump.** In some very weird PDFs, the last two quantiles can be close enough for their difference being rounded to 0 during quantization. A final jump of 0 is imposible to distinguish from trailing zeros indicating empty bytes in the packet. This causes the last quantile to be missed in decoding. Fixed by turning a zero-sized last jump into an 1.

- **Fixed minor issues in the decode module.**
- **Added access via API to additional functions.** These were already implemented but not directly accesible from the API.


## [1.0.1] - 2025-09-06

### Fixed

- Make FITS column name handling case insensitive in all CLI commands.
- Fix decoding error afecting CLI commands `decode`, `measure`, and `plot`  if the encoded PDF column has variable-length format.

## [1.0.0] - 2025-09-02

### Added

- Implemented decode\_to\_density().

### Fixed

- Added encode\_from\_density() to API.
- Fixed insufficient tolerance in range check.
- Fixed error in decode\_quantiles() that resulted in unsorted quantiles.
- Prevent \_batch\_encode() from crashing on weird, un-encodable PDFs. Rejects them instead.
- encode\_from\_samples() excludes sources if more than 10% of their samples are NaN.

## [1.0.0-beta] - 2025-07-19

### Changed
- **BREAKING CHANGE:** The encoding algorithm now stores differences in `log(1+z)` instead of `z`. This improves accuracy and extends the effective redshift range. The size and structure of the header of compressed PDF packets has also changed. **PDFs encoded with previous versions are no longer compatible and must be re-encoded.**
- The decoding algorithm now detects and corrects the seesaw pattern caused by rounding of inter-quantile jumps to small integers in intervals with high probability density. A mechanism for
preventing zero inter-quantile separation is also implemented.
- The definition of `Z_MODE` is now more robust, calculated as the center of the narrowest inter-quantile interval within the 68% Highest Posterior Density Credible Interval.

### Added
- New `--interactive` option for the `plot`command allows interactive visualization of PDFs.
- New `--clip-fraction` option for the `encode` command to remove outliers from random samples.
- New `clip_fraction` keyword in the `samples_to_quantiles()` function.

### Fixed
- Improved accuracy of `density_to_quantiles()` by up-sampling the PDF grid, which is especially important for narrowly peaked PDFs.
- Tiny shift in quantiles that decode to same value prevents singularities when reconstructing the PDF.
- Encoding logic restructured for readability and performance.

## [0.2.0] - 2025-07-17

### Added
- New `--density` option for the `encode` command to support PDFs defined by their probability density on a regular grid.

### Changed
- Exposed the `binned_to_quantiles()` function in the public API.

## [0.1.0] - 2025-07-13

- Initial release.