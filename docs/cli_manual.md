# ColdPress CLI User Manual

The `coldpress` command-line interface provides tools for the compression, analysis, and visualization of redshift probability density functions (PDFs).

## Data Requirements
* **File Format:** The CLI strictly operates on FITS tables. PDFs in other formats must be converted to FITS tables prior to execution.
* **HDU Targeting:** In a standard FITS file, Header Data Unit 0 (HDU 0) contains the primary header and optional image arrays, while the binary table is typically stored in HDU 1. Currently, all commands except `info` strictly assume the target table is in HDU 1 and will fail otherwise. Global `--hdu` argument support is planned for future releases.

## Global Usage
`coldpress [-h] [-v] {info,encode,decode,combine,measure,plot,check} ...`

---

## `info`
Displays metadata about a specific FITS file HDU, including dimensions, types, and column details.

### Usage
`coldpress info [-h] [--hdu HDU] [--header] input.fits`

### Arguments
* `input.fits`: Name of the input FITS file.
* `--hdu HDU`: Index of the HDU to inspect (default: 1).
* `--header`: Print the full FITS header of the specified HDU.

---

## `encode`
Compresses redshift PDFs into the fixed-size ColdPress format. Accepts input as probability densities, binned probabilities, or random samples.

**Input Data Types:**
The CLI handles three mutually exclusive representations of PDFs. It is critical to select the correct one, as the code cannot automatically distinguish between densities and binned probabilities.

* **Probability Densities (`--density`):** The continuous PDF *P(z)* evaluated at discrete points on a uniform grid. All PDFs must have the same number of elements. NaNs are not allowed; PDFs that do not cover the entire grid range must be padded with zeros.
* **Binned Probabilities (`--binned`):** The integrated probability within specific redshift bins. All PDFs must have the same number of elements. NaNs are not allowed; pad missing data with zeros.
* **Random Samples (`--samples`):** Discrete random draws from the underlying distribution. Variable-length arrays are not natively supported; missing samples in fixed-length arrays must be represented with `NaN` (do not pad with zeros).

**General Constraints:**

* The input column containing the uncompressed PDFs must be a floating-point array column.
* The grid must be uniform in units of either redshift *z* or ζ = ln(1+*z*). Arbitrary or non-uniform grids are unsupported.
* The units for the input PDFs are assumed to be redshift *z* unless the `--units zeta` flag is explicitly specified.

> **Important:** The input format arguments (`--density`, `--binned`, and `--samples`) are mutually exclusive. You must provide exactly one.

### Usage
`coldpress encode [-h] (--density COL | --binned COL | --samples COL) [-o [COL]] [--zmin ZMIN] [--zmax ZMAX] [--zetamin ZETAMIN] [--zetamax ZETAMAX] [--length [LENGTH]] [--validate] [--tolerance [TOLERANCE]] [--keep-orig] [--clip-fraction [CLIP_FRACTION]] [--units [{redshift,zeta}]] input.fits output.fits`

### Positional Arguments
* `input.fits`: Input FITS catalog name.
* `output.fits`: Output FITS catalog name.

### Required Named Arguments (Mutually Exclusive)
* `--density COL`: Column containing probability densities sampled on a redshift grid.
* `--binned COL`: Column containing probabilities inside redshift bins.
* `--samples COL`: Column containing random redshift samples from the underlying distribution.

### Optional Arguments
* `-o`, `--out-encoded COL`: Name of the output column for cold-pressed PDFs (default: `COLDPRESS_PDF`).
* `--zmin ZMIN` / `--zmax ZMAX`: Limits for the grid/bins (required if `--units redshift`). For binned PDFs, these values represent the centers of the first and last bins.
* `--zetamin ZETAMIN` / `--zetamax ZETAMAX`: Limits for the grid/bins (required if `--units zeta`). For binned PDFs, these values represent the centers of the first and last bins.
* `--length LENGTH`: Length of compressed PDFs in bytes. Must be a multiple of 4 (default: 80).
* `--validate`: Verify accuracy of recovered quantiles.
* `--tolerance TOLERANCE`: Maximum shift tolerated for the recovered redshift in ζ space. There is a trade-off between the encoding length (number of quantiles) and precision. A value of 0.001 is reasonable for broad-band photo-*z*. Verify the results with the `plot` command; if the main peak exhibits a see-saw artifact, lower this tolerance (default: 0.001).
* `--keep-orig`: Retain the original input column in the output file.
* `--clip-fraction FRAC`: Fraction of extreme redshift samples to clip (only valid with `--samples`). Use this strictly to filter suspected artifacts; otherwise, it will artificially distort the PDF (default: 0).
* `--units [{redshift,zeta}]`: Independent axis representation: `redshift` (*z*) or `zeta` (ζ = ln(1+*z*)) (default: `redshift`).

---

## `decode`
Extracts PDFs previously encoded with ColdPress back into binned distributions, density grids, or random samples.

[[NOTE: clarify that, for the binned and density modes, the user needs to specify the mininum and maximum redshifts of the grid as well as the number of points. The resulting grid will have uniform spacing in z or ζ units given by ∆ = (zmax - zmin)/(npoints -1) or ∆ = (zetamax - zetamin)/(npoints -1). The same grid will be used for all the PDFs. If the range of the compressed PDF is smaller than the span of the grid, it will be padded with zeros. If it is larger, decoding will fail unless the --force-range flago is used to allow truncation. In the --samples mode the output is npoints random samples taken from the PDFs and no redshift range needs to be specified.]]

[[NOTE: some context about the interpolation method. PDFs compressed with ColdPress store a sequence of redshifts corresponding to specific quantiles of the cumulative distribution function (CDF). The information about the shape of the CDF between those anchor points is lost. The only physical constraint is that the CDF must increase monotonically. ColdPress provides two methods for reconstruction of the continuous CDF: linear interpolation and monotonic splines. Linear interpolation results in a CDF composed of straight-line segments. Therefore its derivative, P(z), is a step function. Spline interpolation results in a more natural looking (and often more realistic) shape. In the wings of the PDF spline interpolation is replaced with a powerlaw for a more natural cut off. The default interpolation method remains linear for backwards compatibility but we strongly recommend using spline.]]

> **Note:** If the extracted PDF range exceeds your specified grid bounds, decoding will fail. Use the `--force-range` flag to allow truncation.

### Usage
`coldpress decode [-h] [--encoded [COL]] (--density COL | --binned COL | --samples COL) --nvalues NVALUES [--zmin ZMIN] [--zmax ZMAX] [--zetamin ZETAMIN] [--zetamax ZETAMAX] [--force-range] [--method [{linear,spline}]] [--units [{redshift,zeta}]] input output`

### Positional Arguments
* `input`: Input FITS catalog name.
* `output`: Output FITS catalog name.

### Required Named Arguments
* `--nvalues NVALUES`: Number of bins, steps, or samples for the output PDF.
* Mutually Exclusive Output Target:
  * `--density COL`: Output column for probability densities.
  * `--binned COL`: Output column for probabilities inside bins.
  * `--samples COL`: Output column for random samples.

### Optional Arguments
* `--encoded COL`: Column containing cold-pressed PDFs (default: `COLDPRESS_PDF`).
* `--zmin ZMIN` / `--zmax ZMAX`: Output grid limits. For binned PDFs, these values represent the centers of the first and last bins (required if `--units redshift` and not outputting samples).
* `--zetamin ZETAMIN` / `--zetamax ZETAMAX`: Output grid limits. For binned PDFs, these values represent the centers of the first and last bins (required if `--units zeta` and not outputting samples).
* `--force-range`: Force the specified range even if PDFs are truncated.
* `--method [{linear,spline}]`: Interpolation method (default: `linear`).
* `--units [{redshift,zeta}]`: Independent axis representation (default: `redshift`).

---

## `combine`
Combines two coldpress-encoded PDFs into a single compressed PDF or calculates their correlation p-value.

[[NOTE: some context: the combine command is new in version 1.2.0. ColdPress provides three distinct operations for PDFs: conflation is used to combine two statistically independent PDFs to obtain a narrower one. In the context of photometric redshifts, statistical independence means that they are obtained from independent sets of observations. Conflation is also the correct operation to combine a likelihood with a redshift prior. Note that the conflated PDF will be overconfident if the redshift priors applied to the individual PDFs combined are not independent (see Hernán-Caballero et al. (2024) url: https://ui.adsabs.harvard.edu/abs/2024A%26A...684A..61H/abstract) for details.)
Average is the correct strategy to combine PDFs obtained from the same data but with different methods (e.g. two different photo-z codes). Gemini: do you agree? if not, clarify here.
The correlation of two PDFs is given by: P(∆z) = Integral of P1(z)*P2(z - ∆z) dz. It is useful to test the hypothesis that two PDFs correspond to redshift distributions for the same object. For this we compute a p-value defined as the integral of P(∆z) in regions that verify P(∆z) < P(∆z = 0). (Gemini please check the math).
If the p-value is low, it means that the PDFs correspond to different objects or one of them is unrealistic, and they should not be combined.]]

[[NOTE: the --tolerance keyword indicates the maximum shift in the redshift of the quantiles that is allowed. Please clarify.]]

### Usage
`coldpress combine [-h] (--conflate COL1 COL2 | --average COL1 COL2 | --correlate COL1 COL2) [-o OUT_COMBINED] [--length [LENGTH]] [--tolerance [TOLERANCE]] input.fits output.fits`

### Positional Arguments
* `input.fits`: Input FITS catalog name.
* `output.fits`: Output FITS catalog name.

### Required Named Arguments (Mutually Exclusive)
* `--conflate COL1 COL2`: Conflate (multiply and renormalize) two PDFs.
* `--average COL1 COL2`: Average two PDFs.
* `--correlate COL1 COL2`: Correlate two PDFs (outputs p-value).

### Optional Arguments
* `-o`, `--out-combined OUT_COMBINED`: Output column name. Defaults to `CONFLATED_PDF`, `AVERAGE_PDF`, or `CORR_PVALUE` based on method.
* `--length LENGTH`: Length of compressed output PDFs in bytes (default: 80).
* `--tolerance TOLERANCE`: Maximum shift tolerated during re-encoding (default: 0.001).

---

## `measure`
Computes point-estimate statistics (e.g., mean, mode, credible intervals) directly from compressed PDFs.

> **Tip:** You can execute `coldpress measure --list-quantities` to print all available statistical keys and their descriptions without processing a file.

### Usage
`coldpress measure [-h] [--encoded [COL]] [--quantities QUANTITY [QUANTITY ...]] [--odds-window ODDS_WINDOW] [--seed SEED] [--list-quantities] [input] [output]`

### Positional Arguments
* `input`: Input FITS table.
* `output`: Output FITS table.

### Optional Arguments
* `--encoded COL`: Column containing cold-pressed PDFs (default: `COLDPRESS_PDF`).
* `--quantities QUANTITY ...`: List of specific quantities