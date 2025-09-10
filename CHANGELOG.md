# Changelog

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