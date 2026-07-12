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

* **Probability Densities (`--density`):** The continuous PDF P(z) evaluated at discrete points on a uniform grid. All PDFs must have the same number of elements. NaNs are not allowed; PDFs that do not cover the entire grid range must be padded with zeros.
* **Binned Probabilities (`--binned`):** The integrated probability within specific redshift bins. All PDFs must have the same number of elements. NaNs are not allowed; pad missing data with zeros.
* **Random Samples (`--samples`):** Discrete random draws from the underlying distribution. Variable-length arrays are not natively supported; missing samples in fixed-length arrays must be represented with `NaN` (do not pad with zeros).

**General Constraints:**

* The input column containing the uncompressed PDFs must be a floating-point array column.
* The grid must be uniform in units of either redshift z or ζ = ln(1+z). Arbitrary or non-uniform grids are unsupported.
* The units for the input PDFs are assumed to be redshift z unless the `--units zeta` flag is explicitly specified.

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
* `--tolerance TOLERANCE`: Maximum shift tolerated for the recovered redshift in ζ space. There is a trade-off between the encoding length (number of quantiles) and precision. A value of 0.001 is reasonable for broad-band photo-z. Verify the results with the `plot` command; if the main peak exhibits a see-saw artifact, lower this tolerance (default: 0.001).
* `--keep-orig`: Retain the original input column in the output file.
* `--clip-fraction FRAC`: Fraction of extreme redshift samples to clip (only valid with `--samples`). Use this strictly to filter suspected artifacts; otherwise, it will artificially distort the PDF (default: 0).
* `--units [{redshift,zeta}]`: Independent axis representation: `redshift` (z) or `zeta` (ζ = ln(1+z)) (default: `redshift`).

---

## `decode`
Extracts PDFs previously encoded with ColdPress back into binned distributions, density grids, or random samples.

For the `--binned` and `--density` modes, you must specify the minimum and maximum values of the grid alongside the number of points (`--nvalues`). The resulting grid will have a uniform spacing calculated as Δ = (zmax - zmin) / (npoints - 1) or Δ = (zetamax - zetamin) / (npoints - 1). This identical grid is applied to all processed PDFs. If the extracted PDF range is smaller than the grid's span, it will be padded with zeros. If the range exceeds the specified grid bounds, decoding will fail unless the `--force-range` flag is used to permit truncation. 

In `--samples` mode, the output is simply `npoints` random samples taken from the PDF; no redshift range parameters are required.

**Interpolation Methods:**
PDFs compressed with ColdPress store a sequence of redshifts corresponding to specific quantiles of the cumulative distribution function (CDF). Information regarding the shape of the CDF between these anchor points is lost, leaving monotonicity as the only physical constraint. ColdPress provides two methods for reconstructing the continuous CDF (`--method`):

* `linear`: Interpolates the CDF using straight-line segments, resulting in a step-function derivative P(z). This remains the default solely for backward compatibility.
* `spline`: Interpolates using monotonic splines for a more natural and realistic shape. In the wings of the PDF, the spline interpolation transitions to a power law for a realistic cutoff. We strongly recommend using `spline`.

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
The `combine` command (new in version 1.2.0) performs mathematical operations on two coldpress-encoded PDFs, returning a single compressed PDF or a calculated correlation p-value.

**Operations:**

* **Conflate (`--conflate`):** Multiplies and renormalizes two PDFs. This operation is used to combine statistically independent PDFs (i.e., derived from independent sets of observations) to yield a narrower constraint. Conflation is also the correct operation for applying a redshift prior to a likelihood. Note that if the redshift priors applied to the individual PDFs are not completely independent, conflation will result in an overconfident PDF (see [Hernán-Caballero et al. 2024](https://ui.adsabs.harvard.edu/abs/2024A%26A...684A..61H/abstract)).
* **Average (`--average`):** Calculates the mean of the two PDFs. This is the correct strategy to combine PDFs derived from the same observational data but processed using different methods or codes (e.g., combining two photo-z code outputs).
* **Correlate (`--correlate`):** Computes the cross-correlation P(Δz) = ∫ P1(z) * P2(z + Δz) dz. This tests the hypothesis that the two PDFs correspond to the same object. The output is a p-value defined as the integral of P(Δz) over all regions where P(Δz) ≤ P(Δz = 0). A low p-value signifies that the PDFs likely belong to different objects or that one is unrealistic, indicating they should not be combined.

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
* `--tolerance TOLERANCE`: Maximum shift in the redshift of the quantiles that is tolerated during the re-encoding step (default: 0.001).

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
* `--quantities QUANTITY ...`: List of specific quantities to measure (default: `ALL`).
* `--odds-window ODDS_WINDOW`: Half-width of the integration window for odds calculation (default: 0.03).
* `--seed SEED`: Random seed for deterministic `Z_RANDOM` extraction.
* `--list-quantities`: Display all available quantities and descriptions, then exit.

---

## `plot`
Reconstructs and plots PDFs encoded with ColdPress. Supports batch saving or interactive viewing.

### Usage
`coldpress plot [-h] (--id ID [ID ...] | --first N | --plot-all) [--interactive] [--idcol [IDCOL]] [--encoded ENCODED [ENCODED ...]] [--outdir OUTDIR] [--format FORMAT] [--method {steps,spline,all}] [--quantities QUANTITIES [QUANTITIES ...]] [--units [{redshift,zeta}]] input`

### Positional Arguments
* `input`: Input FITS table.

### Required Named Arguments (Mutually Exclusive)
* `--id ID ...`: Specific source ID(s) to plot.
* `--first N`: Plot the first N sources.
* `--plot-all`: Plot all sources in the file.

### Optional Arguments
* `--interactive`: Display plots interactively instead of saving to disk.
* `--idcol IDCOL`: Column containing source IDs (default: `ID`).
* `--encoded ENCODED ...`: Column(s) containing cold-pressed PDFs (default: `COLDPRESS_PDF`).
* `--outdir OUTDIR`: Directory for saved plots (default: `.`).
* `--format FORMAT`: Output image format (default: `png`).
* `--method {steps,spline,all}`: PDF reconstruction method for visualization (default: `all`).
* `--quantities QUANTITIES ...`: FITS columns to overplot as vertical markers.
* `--units [{redshift,zeta}]`: Axis representation (default: `redshift`).

---

## `check`
Analyzes input PDFs (binned or sampled) for non-finite values, delta-function-like properties, or truncation, and flags them.

> **Important:** If you use the `--list` argument to print flagged issues to standard output, you must also provide the `--idcol` argument.

### Usage
`coldpress check [-h] (--binned COL | --samples COL) [--truncation-threshold THRESHOLD] [--list] [--idcol IDCOL] input [output]`

### Positional Arguments
* `input`: Input FITS catalog.
* `output`: (Optional) Output FITS catalog with appended flag columns.

### Required Named Arguments (Mutually Exclusive)
* `--binned COL`: Evaluate binned PDFs.
* `--samples COL`: Evaluate sampled PDFs.

### Optional Arguments
* `--truncation-threshold THRESHOLD`: Probability density threshold at grid edges to trigger truncation flag (default: 0.05).
* `--list`: Print flagged source IDs to standard output.
* `--idcol IDCOL`: Column containing source IDs.