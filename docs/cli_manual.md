# ColdPress CLI User Manual

The `coldpress` command-line interface provides tools for the compression, analysis, and visualization of redshift probability density functions (PDFs).

[[NOTE: the CLI is intended to work with FITS tables. If the user has PDFs in a different file format, it should be converted to FITS table first.]] 

## Global Usage
`coldpress [-h] [-v] {info,encode,decode,combine,measure,plot,check} ...`

---

## `info`
Displays metadata about a FITS file HDU, including dimensions, types, and column details.

[[NOTE: indicate what HDU stands for. Explain that in FITS files, tables are usually in HDU 1, while HDU 0 contains... what?. Indicate that all commands other than info assume that the table to read is on HDU 1 and they will fail if it is in a different one (in the near future we will implement an optional --hdu keyword for all coldpress commands.]]

### Usage
`coldpress info [-h] [--hdu HDU] [--header] input.fits`

### Arguments
* `input.fits`: Name of the input FITS file.
* `--hdu HDU`: HDU to inspect (default: 1).
* `--header`: Print the full FITS header.
[[NOTE: is it the header of the HDU, a general header of the file, or both?]]

---

## `encode`
Compresses redshift PDFs into the fixed-size ColdPress format. Accepts input as probability densities, binned probabilities, or random samples.

[[NOTE: ColdPress expects the input column containing the uncompressed PDFs to be an array column of floats (or however this is called in FITS jargon). The redshift grid must be uniform in units of redshift (z) or ζ=ln(1+z). Input PDFs samples in an arbitrary grid are not supported yet.]]

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
* `--zmin ZMIN` / `--zmax ZMAX`: Limits for the grid/bins (required if `--units redshift`).
* `--zetamin ZETAMIN` / `--zetamax ZETAMAX`: Limits for the grid/bins (required if `--units zeta`).
* `--length LENGTH`: Length of compressed PDFs in bytes. Must be a multiple of 4 (default: 80).
* `--validate`: Verify accuracy of recovered quantiles.
* `--tolerance TOLERANCE`: Maximum shift tolerated for the redshift of the quantiles (default: 0.001).
* `--keep-orig`: Retain the original input column in the output file.
* `--clip-fraction FRAC`: Fraction of extreme redshift samples to clip (only valid with `--samples`) (default: 0).
* `--units [{redshift,zeta}]`: Independent axis representation: `redshift` ($z$) or `zeta` ($\ln(1+z)$) (default: `redshift`).

---

## `decode`
Extracts PDFs previously encoded with ColdPress back into binned distributions, density grids, or random samples.

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
* `--zmin ZMIN` / `--zmax ZMAX`: Output grid limits (required if `--units redshift` and not outputting samples).
* `--zetamin ZETAMIN` / `--zetamax ZETAMAX`: Output grid limits (required if `--units zeta` and not outputting samples).
* `--force-range`: Force the specified range even if PDFs are truncated.
* `--method [{linear,spline}]`: Interpolation method (default: `linear`).
* `--units [{redshift,zeta}]`: Independent axis representation (default: `redshift`).

---

## `combine`
Combines two coldpress-encoded PDFs into a single compressed PDF or calculates their correlation p-value.

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
