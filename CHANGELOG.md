# Changelog



## [2.0.0] - 2025-07-18

### Changed
- **BREAKING CHANGE:** The core encoding algorithm now compresses `log(1+z)` instead of `z`. This improves accuracy and extends the effective redshift range. PDFs encoded with previous versions are no longer compatible and must be re-encoded.
- The definition of `Z_MODE` is now more robust, calculated as the center of the narrowest inter-quantile interval within the 68% Highest Posterior Density Credible Interval.

### Added
- New `--clip-fraction` option for the `encode` command to handle outliers in PDF samples.
- New `clip_fraction` keyword in the `samples_to_quantiles()` function.

### Fixed
- Improved accuracy of `density_to_quantiles()` by up-sampling the PDF grid, which is especially important for narrowly peaked PDFs.
- Added a small redshift shift to decoded quantiles that have identical values to prevent singularities when reconstructing the PDF.

## [1.1.0] - 2025-07-17

### Added
- New `--density` option for the `encode` command to support PDFs defined by their probability density on a regular grid.

### Changed
- Exposed the `binned_to_quantiles()` function in the public API.

## [1.0.0] - 2025-07-13

- Initial release.