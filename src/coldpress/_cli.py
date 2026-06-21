#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import numpy as np

from . import __version__
from .encode import encode_from_binned, encode_from_density, encode_from_samples, density_to_quantiles, encode_quantiles
from .decode import decode_to_binned, decode_to_samples, decode_to_density, decode_quantiles, quantiles_to_density
from .stats import measure_from_quantiles, ALL_QUANTITIES, QUANTITY_DESCRIPTIONS
from .utils import plot_from_quantiles, combine_pdfs
from .constants import (
    Q0_ZMIN, Q0_ZMAX, Q0_ZETAMIN, Q0_ZETAMAX,
    DEFAULT_ID_COL, DEFAULT_ENCODED_COL,
    DEFAULT_PACKET_LENGTH, DEFAULT_TOLERANCE, DEFAULT_ODDS_WINDOW
)
from .io import find_column_name, fix_encoded_column, process_fits_table, print_fits_info

# Compatible trapz function
trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# --- Logic for the 'info' command ---
def info_logic(args):
    """Displays metadata about a FITS file.

    Reads a FITS file and prints information about a specified HDU,
    including its type (Image or Table), dimensions, and column details.
    Optionally prints the full FITS header.

    Args:
        args (argparse.Namespace): Command-line arguments from argparse.
            Expected attributes:
            - input (str): Path to the input FITS file.
            - hdu (int): The HDU index to inspect.
            - header (bool): If True, print the full FITS header.
    """
    print_fits_info(args.input, hdu_index=args.hdu, print_header=args.header)

# --- Logic for the 'encode' command ---
def encode_logic(args):
    """Encodes redshift PDFs into the ColdPress format.

    Takes redshift PDFs from a FITS table, either as binned histograms
    or random samples, and compresses them into a fixed-size byte format.
    The resulting compressed data is written to a new FITS file.

    Args:
        args (argparse.Namespace): Command-line arguments from argparse.
            Expected attributes:
            - input (str): Path to the input FITS file.
            - output (str): Path for the output FITS file.
            - length (int): Packet length in bytes for the encoded PDF.
            - density (str, optional): Name of the column with density PDFs.
            - binned (str, optional): Name of the column with binned PDFs.
            - samples (str, optional): Name of the column with PDF samples.
            - zmin (float, optional): Min redshift.
            - zmax (float, optional): Max redshift.
            - zetamin (float, optional): Min zeta.
            - zetamax (float, optional): Max zeta.
            - out_encoded (str): Name for the output column with encoded data.
            - validate (bool): Whether to validate the encoding accuracy.
            - tolerance (float): Tolerance for validation.
            - keep_orig (bool): Whether to keep the original PDF column.
            - clip_fraction (float): Fraction of samples to be clipped out at the extremes of the redshift range.
            - units (str): Specifies the representation for the PDF\'s independent axis: "redshift" (z) or "zeta" (ln(1+z)) (default: redshift).
    """
    if args.length % 4 != 0:
        print(f"Error: Packet length (--length) must be a multiple of 4, but got {args.length}.", file=sys.stderr)
        sys.exit(1)

    orig_column = args.samples or args.binned or args.density        

    # define range of independent variable:                
    if args.units == 'redshift':
        xrange = [args.zmin, args.zmax]
    elif args.units == 'zeta':
        xrange = [args.zetamin, args.zetamax]

    def encode_callback(data_arrays, actual_names, **kwargs):
        actual_orig_column = actual_names[orig_column]
        data = data_arrays[orig_column]

        if args.samples is not None:
            print(f"Generating quantiles from random samples and compressing into {args.length}-byte packets...")
            coldpress_PDF = encode_from_samples(data, packetsize=args.length, ini_quantiles=args.length-8, 
                                                validate=args.validate, tolerance=args.tolerance, clip_fraction=args.clip_fraction, 
                                                clip_range=xrange, units=args.units) 
        else:
            zvector = np.linspace(xrange[0], xrange[1], data.shape[1])
            if args.binned is not None:
               coldpress_PDF = encode_from_binned(data, zvector, packetsize=args.length, ini_quantiles=args.length-8, 
                                                  validate=args.validate, tolerance=args.tolerance, units=args.units)
            elif args.density is not None:
               coldpress_PDF = encode_from_density(data, zvector, packetsize=args.length, ini_quantiles=args.length-8, 
                                                   validate=args.validate, tolerance=args.tolerance, units=args.units)
                      
        nints = args.length // 4
        return {args.out_encoded: (f'{nints}J', coldpress_PDF)}

    drop_cols = [args.out_encoded]
    if not args.keep_orig:
        drop_cols.append(orig_column)
        print(f"Excluding column '{orig_column}' from output FITS table.")
    else:
        print(f"Including column '{orig_column}' in output FITS table.")

    history = f"PDFs from column {orig_column} cold-pressed as {args.out_encoded}"
    process_fits_table(args.input, args.output, [orig_column], drop_cols, history, encode_callback)


# --- Logic for the 'decode' command ---
def decode_logic(args):
    """Decodes coldpress PDFs back to binned PDFs.

    Reads a FITS table with a column of coldpress-encoded data,
    decompresses it, and reconstructs the PDFs on a specified redshift grid.
    The resulting binned PDFs are written to a new FITS file.

    Args:
        args (argparse.Namespace): Command-line arguments from argparse.
            Expected attributes:
            - input (str): Path to the input FITS file.
            - output (str): Path for the output FITS file.
            - encoded (str): Name of the column with encoded PDFs.
            - density (str, optional): Name for the output column with PDF density.
            - binned (str, optional): Name for the output column with binned PDF.
            - samples (str, optional): Name for the output column with samples.
            - zmin (float): Minimum redshift for the output grid.
            - zmax (float): Maximum redshift for the output grid.
            - zetamin (float): Minimum zeta=ln(1+z) for the output grid.
            - zetamax (float): Maximum zeta=ln(1+z) for the output grid.
            - nvalues (int): number of bins/steps/samples in the output PDF.
            - force_range (bool): If True, truncates PDFs outside the grid when needed.
            - method (str): Interpolation method ('linear' or 'spline').
            - units (str): Specifies the representation for the PDF\'s independent axis: 
               "redshift" (z) or "zeta" (ln(1+z)) (default: redshift).
          
    """
    # define range of independent variable:                
    if args.units == 'redshift':
        xrange = [args.zmin, args.zmax]
    elif args.units == 'zeta':
        xrange = [args.zetamin, args.zetamax]

    out_column_name = args.binned or args.density or args.samples

    def decode_callback(data_arrays, actual_names, **kwargs):
        # ensure array is big-endian
        qcold = data_arrays[args.encoded].astype('>i4') 
        
        if args.binned or args.density:
            zvector = np.linspace(xrange[0], xrange[1], args.nvalues)
                
            print(f"Decompressing PDFs using {args.method} interpolation of the quantiles...")
            try:
                if args.binned:
                    decoded_PDF = decode_to_binned(qcold, zvector, force_range=args.force_range, method=args.method, units=args.units)
                if args.density:
                    decoded_PDF = decode_to_density(qcold, zvector, force_range=args.force_range, method=args.method, units=args.units)                
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                print("Hint: Use the --force-range flag to proceed with truncation at your own risk.", file=sys.stderr)
                sys.exit(1)

        elif args.samples:
            decoded_PDF = decode_to_samples(qcold, Nsamples=args.nvalues, method=args.method, units=args.units)

        return {out_column_name: (f'{args.nvalues}E', decoded_PDF)}

    history = f"PDFs in column {args.encoded} extracted as {out_column_name}"
    process_fits_table(args.input, args.output, [args.encoded], [out_column_name], history, decode_callback)


# --- Logic for the 'combine' command ---
def combine_logic(args):
    """Combines two coldpress-encoded PDFs into a single encoded PDF.

    Args:
        args (argparse.Namespace): Command-line arguments from argparse.
            Expected attributes:
            - input (str): Path to the input FITS file.
            - output (str): Path for the output FITS file.
            - conflate (list, optional): Two column names for conflation.
            - average (list, optional): Two column names for averaging.
            - correlate (list, optional): Two column names for correlation.
            - out_combined (str, optional): Output column name for the combined PDF.
            - length (int): Packet length in bytes for encoding.
            - tolerance (float): Tolerance for validation during encoding.
    """
    if args.conflate:
        method = 'conflate'
        encoded_cols = args.conflate
        default_out = 'CONFLATED_PDF'
    elif args.average:
        method = 'average'
        encoded_cols = args.average
        default_out = 'AVERAGE_PDF'
    elif args.correlate:
        method = 'correlate'
        encoded_cols = args.correlate
        default_out = 'CORR_PVALUE'
    else:
        print("Error: No combination method specified.", file=sys.stderr)
        sys.exit(1)

    out_combined_col = args.out_combined if args.out_combined else default_out

    if args.length % 4 != 0:
        print(f"Error: Packet length (--length) must be a multiple of 4, but got {args.length}.", file=sys.stderr)
        sys.exit(1)

    def combine_callback(data_arrays, actual_names, **kwargs):
        qcold1 = data_arrays[encoded_cols[0]].astype('>i4')
        qcold2 = data_arrays[encoded_cols[1]].astype('>i4')
        Nsources = qcold1.shape[0]

        # Initialize the correct FITS column data type based on the combination method
        if method == 'correlate':
            out_data = np.full(Nsources, np.nan, dtype=np.float32)
            fits_format = 'E'
        else:
            out_data = np.zeros((Nsources, args.length // 4), dtype='>i4')
            fits_format = f'{args.length // 4}J'

        valid = (np.any(qcold1 != 0, axis=1)) & (np.any(qcold2 != 0, axis=1))
        valid_indices = np.where(valid)[0]

        print(f"Combining PDFs ({method}) for {len(valid_indices)} valid sources...")
        
        start = time.process_time()

        for i in valid_indices:
            result = combine_pdfs(
                qcold1[i].tobytes(),
                qcold2[i].tobytes(),
                method=method,
                length=args.length,
                tolerance=args.tolerance
            )
            if result is not None:
                out_data[i] = result

        cpu_seconds = time.process_time() - start
        print(f"PDFs combined in {cpu_seconds:.6f} CPU seconds")

        return {out_combined_col: (fits_format, out_data)}

    history = f"Combined ({method}) {encoded_cols[0]} and {encoded_cols[1]} into {out_combined_col}"
    process_fits_table(args.input, args.output, encoded_cols, [out_combined_col], history, combine_callback)


# --- Logic for the 'measure' command ---
def measure_logic(args):
    """Computes point-estimate statistics from compressed PDFs.

    Reads coldpress-encoded PDFs, decodes their quantiles, and computes
    various statistical quantities (e.g., mean, mode, median). The
    results are appended as new columns to the FITS table.

    Args:
        args (argparse.Namespace): Command-line arguments from argparse.
            Expected attributes:
            - input (str): Path to the input FITS file.
            - output (str): Path for the output FITS file.
            - encoded (str): Name of the column with encoded PDFs.
            - quantities (list): List of strings specifying which
              quantities to measure (e.g., ['Z_MEAN', 'Z_MODE']). 'ALL'
              computes all available quantities.
            - odds_window (float): Half-width for the odds calculation.
            - seed (int, optional): Seed for random number generation.
    """
    q_to_compute = {q.upper() for q in args.quantities}
    if 'ALL' in q_to_compute:
        q_to_compute = ALL_QUANTITIES
    
    print(f"Will compute: {', '.join(sorted(list(q_to_compute)))}")

    def measure_callback(data_arrays, actual_names, **kwargs):
        # ensure array is big-endian
        qcold = data_arrays[args.encoded].astype('>i4') 
        Nsources = qcold.shape[0]
        
        d = {}
        for q_name in q_to_compute:
             d[q_name] = np.full(Nsources, np.nan, dtype=np.float32)

        valid = np.any(qcold != 0, axis=1)
        valid_indices = np.where(valid)[0]
        print(f"Calculating point estimates for {len(valid_indices)} valid sources...")
        
        rng = np.random.default_rng(args.seed)
        u_array = rng.uniform(0, 1, size=len(valid_indices))
        
        for idx, i in enumerate(valid_indices):
            quantiles = decode_quantiles(qcold[i].tobytes())
                            
            results = measure_from_quantiles(
                quantiles,
                quantities_to_measure=list(q_to_compute),
                odds_window=args.odds_window,
                u=float(u_array[idx])
            )
            for q_name, value in results.items():
                d[q_name][i] = value

        return {name: ('E' if array.dtype == np.float32 else 'I', array) for name, array in d.items()}

    history = f"Computed point estimates from column: {args.encoded}"
    process_fits_table(args.input, args.output, [args.encoded], list(q_to_compute), history, measure_callback)

# --- Logic for the 'plot' command ---
def plot_logic(args):
    """Generates plots of PDFs from compressed data.

    Reads coldpress-encoded data, reconstructs the PDF for specified
    sources, and saves them as image files. Can plot individual sources
    by ID, the first N sources, or all sources.

    Args:
        args (argparse.Namespace): Command-line arguments from argparse.
            Expected attributes:
            - input (str): Path to the input FITS file.
            - id (list, optional): List of source IDs to plot.
            - first (int, optional): Plot the first N sources.
            - plot_all (bool): Flag to plot all sources.
            - interactive (bool): Flag to show plots interactively.
            - idcol (str, optional): Column name containing source IDs.
            - encoded (list): List of column names with encoded PDFs.
            - outdir (str): Directory to save plot files.
            - format (str): Output image format (e.g., 'png', 'pdf').
            - method (str): PDF reconstruction method for plotting 
              ('linear' or 'spline').
            - quantities (list, optional): List of FITS columns to
              overplot as vertical lines.
            - units (str): Specifies the representation for the PDF\'s independent axis: 
              "redshift" (z) or "zeta" (ln(1+z)) (default: redshift).
  
    """
    try:
        import matplotlib
    except ImportError:
        print("Error: matplotlib is required for the plot command.", file=sys.stderr)
        sys.exit(1)

    # Consolidate all columns required for the read operation
    required_cols = list(args.encoded)
    if args.idcol:
        required_cols.append(args.idcol)
    if args.quantities:
        required_cols.extend(args.quantities)
        
    # Remove duplicates while preserving order
    required_cols = list(dict.fromkeys(required_cols))

    def plot_callback(data_arrays, actual_names, **kwargs):
        actual_encoded_names = [actual_names[col] for col in args.encoded]
        qcold_list = [data_arrays[col].astype('>i4') for col in args.encoded]
        
        actual_idcol = actual_names.get(args.idcol)
        Nrows = qcold_list[0].shape[0]

        # Determine which rows to plot based on the mutually exclusive arguments
        if args.plot_all:
            indices_to_plot = range(Nrows)
        elif args.first is not None:
            num_to_plot = min(args.first, Nrows)
            if args.first > Nrows:
                print(f"Warning: Requested first {args.first} PDFs, but file only contains {Nrows}. Plotting all sources.")
            indices_to_plot = range(num_to_plot)
        else:  # This handles the 'id' case, which is the default if not the others
            if actual_idcol is None:
                print(f"Error: --id specified, but no '{args.idcol}' column found in {args.input}", file=sys.stderr)
                sys.exit(1)
            
            source_ids = list(args.id)
            id_column_as_str = data_arrays[args.idcol].astype(str)
            source_ids_as_str = np.asarray(source_ids, dtype=str)
            indices_to_plot = np.where(np.isin(id_column_as_str, source_ids_as_str))[0]

            if len(indices_to_plot) != len(source_ids):
                print("Warning: Some specified IDs were not found in the file.", file=sys.stderr)
            
        print(f"Plotting {len(indices_to_plot)} source(s)...")

        if not args.interactive:
            os.makedirs(args.outdir, exist_ok=True)

        for i in indices_to_plot:
            source_id_val = data_arrays[args.idcol][i] if actual_idcol is not None else f"row_{i}"
            
            quantiles_list = []
            labels_list = []
            
            for j, qcold in enumerate(qcold_list):
                if np.any(qcold[i] != 0):
                    quantiles = decode_quantiles(qcold[i].tobytes(), units=args.units)
                    quantiles_list.append(quantiles)
                    labels_list.append(actual_encoded_names[j])
                    
            if not quantiles_list:
                print(f"Skipping source {source_id_val}: No valid PDF data.")
                continue

            markers_to_plot = {}
            if args.quantities:
                for q_col in args.quantities:
                    if q_col in data_arrays:
                        markers_to_plot[q_col] = data_arrays[q_col][i]
                            
            output_filename = None
            if not args.interactive:
                output_filename = os.path.join(args.outdir, f"pdf_{source_id_val}.{args.format.lower()}")
            
            # Enforce single interpolation method if multiple PDFs are plotted
            plot_method = args.method
            if len(quantiles_list) > 1 and args.method == 'all':
                plot_method = 'spline'
                
            try:
                plot_from_quantiles(
                    quantiles_list,
                    output_filename=output_filename,
                    interactive=args.interactive,
                    source_id=str(source_id_val),
                    method=plot_method,
                    markers=markers_to_plot,
                    units=args.units,
                    labels=labels_list
                )
                if not args.interactive:
                    print(f"Saved plot to {output_filename}")
            except ImportError as e:
                print(e, file=sys.stderr)
                sys.exit(1)

        # Returning None instructs process_fits_table to skip the FITS writing step
        return None

    # Pass output_path=None and history_msg=None for a read-only operation
    process_fits_table(args.input, None, required_cols, [], None, plot_callback)
    
# --- Logic for the 'check' command ---
def check_logic(args):
    """Checks PDFs for potential issues and creates flags.

    Analyzes binned or sampled PDFs for problems like non-finite values,
    unresolved (delta-function-like) distributions, or truncation at the
    edges of the redshift range. Can optionally list problematic sources
    or write flags to a new output FITS file.

    Args:
        args (argparse.Namespace): Command-line arguments from argparse.
            Expected attributes:
            - input (str): Path to the input FITS file.
            - output (str, optional): Path for the output FITS file with flags.
            - binned (str, optional): Name of the column with binned PDFs.
            - samples (str, optional): Name of the column with PDF samples.
            - truncation_threshold (float): Threshold for truncation detection.
            - list (bool): If True, list flagged source IDs to stdout.
            - idcol (str, optional): Column name of source IDs, required for --list.
    """
    orig_column = args.binned or args.samples
    required_cols = [orig_column]
    if args.list:
        required_cols.append(args.idcol)

    def check_callback(data_arrays, actual_names, **kwargs):
        data = data_arrays[orig_column]
        if args.list:
            ID = data_arrays[args.idcol]
            
        if args.binned is None:
            invalid = (np.sum(~np.isfinite(data), axis=1) > 0)
            v = ~invalid
            unresolved = (np.max(data[v], axis=1) - np.min(data[v], axis=1) == 0)
            print(f"Column {actual_names[orig_column]} contains {data.shape[0]} sampled PDFs, each containing {data.shape[1]} random redshift samples.")
        else:
            invalid = (np.sum(~np.isfinite(data), axis=1) > 0) | (np.nanmin(data, axis=1) < 0) | (np.nanmax(data, axis=1) == 0.)
            v = ~invalid
            unresolved = (np.sum(data[v], axis=1) - np.max(data[v], axis=1) == 0)
            threshold = args.truncation_threshold * np.max(data[v], axis=1)
            truncated = ((data[v, 0] > threshold) | (data[v, -1] > threshold))
            print(f"Column {actual_names[orig_column]} contains {data.shape[0]} binned PDFs, each containing {data.shape[1]} redshift bins.")
            
        print(f"{np.sum(invalid)} PDFs have been flagged as 'invalid'")
        print(f"{np.sum(unresolved)} PDFs have been flagged as 'unresolved'")
        if args.binned is not None:
            print(f"{np.sum(truncated)} PDFs have been flagged as 'truncated'")
            
        if args.list:
            print("List of source IDs with flagged issues in their PDFs:")
            for source in ID[invalid]:
                print(f"{source}  invalid")
            for i, source in enumerate(ID[v]):
                tag = ""
                if unresolved[i]: tag += " unresolved"
                if args.binned is not None and truncated[i]: tag += " truncated"
                if tag != "": print(f"ID = {source}:{tag}")

        if args.output:                                
            d = {}
            d['Z_FLAGS'] = np.zeros(len(invalid), dtype=np.int16)
            d['PDF_invalid'] = invalid
            d['PDF_unresolved'] = np.zeros(len(invalid), dtype=bool)
            d['PDF_unresolved'][v] = unresolved
            d['Z_FLAGS'][d['PDF_invalid']] = 1
            d['Z_FLAGS'][d['PDF_unresolved']] += 2
            if args.binned is not None:
                d['PDF_truncated'] = np.zeros(len(invalid), dtype=bool)
                d['PDF_truncated'][v] = truncated
                d['Z_FLAGS'][d['PDF_truncated']] += 4

            result = {}
            for name, array in d.items():
                if array.dtype == np.float32: format_str = 'E'
                elif array.dtype == np.int16: format_str = 'I'
                elif array.dtype == bool: format_str = 'L'    
                result[name] = (format_str, array)
            return result
        
        return None

    drop_cols = ['Z_FLAGS', 'PDF_invalid', 'PDF_unresolved', 'PDF_truncated'] if args.output else []
    hist_cols = ['Z_FLAGS', 'PDF_invalid', 'PDF_unresolved'] + (['PDF_truncated'] if args.binned else [])
    history = f"Added flags columns indicating issues in the PDFs: {hist_cols}" if args.output else None

    process_fits_table(args.input, args.output, required_cols, drop_cols, history, check_callback)
    
# --- Main Entry Point and Parser Configuration ---
def main():
    """Main entry point for the coldpress command-line interface.

    Parses command-line arguments and calls the appropriate logic function
    (e.g., info_logic, encode_logic) based on the specified subcommand.
    """
    parser = argparse.ArgumentParser(description='Compression, analysis, and visualization of redshift PDFs.')
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}')
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- Parser for the "info" command ---
    parser_info = subparsers.add_parser('info', help='Display information about a FITS file HDU.')
    parser_info.add_argument('input', metavar='input.fits', type=str, help='Name of the input FITS file.')
    parser_info.add_argument('--hdu', type=int, default=1, help='HDU to inspect (default: 1).')
    parser_info.add_argument('--header', action='store_true', help='Print the full FITS header.')
    parser_info.set_defaults(func=info_logic)

    # --- Parser for the "encode" command ---
    parser_encode = subparsers.add_parser('encode', help='Compress PDFs into coldpress format.')
    parser_encode.add_argument('input', metavar='input.fits', type=str, help='Name of input FITS catalog.')
    parser_encode.add_argument('output', metavar='output.fits', type=str, help='Name of output FITS catalog.')
    encode_group = parser_encode.add_mutually_exclusive_group(required=True)
    encode_group.add_argument('--density', type=str, help='Name of input column containing probability densities sampled in a grid of redshifts.')
    encode_group.add_argument('--binned', type=str, help='Name of input column containing probabilities inside redshift bins.')
    encode_group.add_argument('--samples', type=str, help='Name of input column containing a set of random redshift samples from the underlying probability distribution.')
    parser_encode.add_argument('-o', '--out-encoded', type=str, nargs='?', default=DEFAULT_ENCODED_COL, help='Name of output column containing the cold-pressed PDFs.')
    parser_encode.add_argument('--zmin', type=float, help='Lowest redshift in the grid/bins')
    parser_encode.add_argument('--zmax', type=float, help='Highest redshift in the grid/bins')
    parser_encode.add_argument('--zetamin', type=float, help='Lowest zeta=ln(1+z) in the grid/bins')
    parser_encode.add_argument('--zetamax', type=float, help='Highest zeta=ln(1+z) in the grid/bins')    
    parser_encode.add_argument('--length', type=int, nargs='?', default=DEFAULT_PACKET_LENGTH, help='Length of compressed PDFs in bytes (must be multiple of 4).')
    parser_encode.add_argument('--validate', action='store_true', default=False, help='Verify accuracy of recovered quantiles.')
    parser_encode.add_argument('--tolerance', type=float, nargs='?', default=DEFAULT_TOLERANCE, help='Maximum shift tolerated for the redshift of the quantiles.')
    parser_encode.add_argument('--keep-orig', action='store_true', help='Include the original input column with binned PDFs or samples in the output file.')
    parser_encode.add_argument('--clip-fraction', type=float, nargs='?', default=0, help='Fraction of samples to be clipped out at the extremes of the redshift range.')
    parser_encode.add_argument('--units', type=str, nargs='?', default='redshift', choices=['redshift','zeta'], help='Specifies the representation for the PDF\'s independent axis: "redshift" (z) or "zeta" (ln(1+z)) (default: redshift).')
    parser_encode.set_defaults(func=encode_logic)

    # --- Parser for the "decode" command ---
    parser_decode = subparsers.add_parser('decode', help='Extract PDFs previously encoded with ColdPress.')
    parser_decode.add_argument('input', type=str, help='Name of input FITS catalog')
    parser_decode.add_argument('output', type=str, help='Name of output FITS catalog')
    parser_decode.add_argument('--encoded', type=str, nargs='?', default=DEFAULT_ENCODED_COL, help='Name of column containing cold-pressed PDFs.')
    decode_group = parser_decode.add_mutually_exclusive_group(required=True)
    decode_group.add_argument('--density', type=str, help='Name of output column to contain probability densities sampled in a grid.')
    decode_group.add_argument('--binned', type=str, help='Name of output column to contain probabilities inside bins.')
    decode_group.add_argument('--samples', type=str, help='Name of output column to contain a set of random samples from the underlying probability distribution.')
    parser_decode.add_argument('--nvalues', type=int, required=True, help='Number of bins/steps/samples for the output PDF')
    parser_decode.add_argument('--zmin', type=float, help='Lowest redshift in the grid/bins/samples')
    parser_decode.add_argument('--zmax', type=float, help='Highest redshift in the grid/bins/samples')
    parser_decode.add_argument('--zetamin', type=float, help='Lowest zeta=ln(1+z) in the grid/bins/samples')
    parser_decode.add_argument('--zetamax', type=float, help='Highest zeta=ln(1+z) in the grid/bins/samples')    
    parser_decode.add_argument('--force-range', action='store_true', help='Force the redshift/zeta range even if PDFs are truncated.')
    parser_decode.add_argument('--method', type=str, nargs='?', default='linear', choices=['linear','spline'], help='Interpolation method for PDF reconstruction (default: linear).')
    parser_decode.add_argument('--units', type=str, nargs='?', default='redshift', choices=['redshift','zeta'], help='Specifies the representation for the PDF\'s independent axis: "redshift" (z) or "zeta" (ln(1+z)) (default: redshift).')
    parser_decode.set_defaults(func=decode_logic)

    # --- Parser for the "combine" command ---
    parser_combine = subparsers.add_parser('combine', help='Combine two coldpress-encoded PDFs.')
    parser_combine.add_argument('input', metavar='input.fits', type=str, help='Name of input FITS file.')
    parser_combine.add_argument('output', metavar='output.fits', type=str, help='Name of output FITS file.')
    
    combine_group = parser_combine.add_mutually_exclusive_group(required=True)
    combine_group.add_argument('--conflate', type=str, nargs=2, metavar=('COL1', 'COL2'), help='Conflate two PDFs.')
    combine_group.add_argument('--average', type=str, nargs=2, metavar=('COL1', 'COL2'), help='Average two PDFs.')
    combine_group.add_argument('--correlate', type=str, nargs=2, metavar=('COL1', 'COL2'), help='Correlate two PDFs.')
    
    parser_combine.add_argument('-o', '--out-combined', dest='out_combined', type=str, help='Name of output column containing the combined PDF or p-value (defaults depend on method).')
    parser_combine.add_argument('--length', type=int, nargs='?', default=DEFAULT_PACKET_LENGTH, help='Length of compressed PDFs in bytes (must be multiple of 4).')
    parser_combine.add_argument('--tolerance', type=float, nargs='?', default=DEFAULT_TOLERANCE, help='Maximum shift tolerated for the redshift of the quantiles.')
    parser_combine.set_defaults(func=combine_logic)

    # --- Parser for the "measure" command ---
    parser_measure = subparsers.add_parser('measure', help='Compute point estimates from compressed PDFs.')
    parser_measure.add_argument('input', type=str, nargs='?', help='Name of input FITS table containing cold-pressed PDFs.')
    parser_measure.add_argument('output', type=str, nargs='?', help='Name of output FITS table containing point estimates measured on the PDFs.')
    parser_measure.add_argument('--encoded', type=str, nargs='?', default=DEFAULT_ENCODED_COL, help='Name of column containing cold-pressed PDFs.')
    choices_list = sorted(list(ALL_QUANTITIES) + ['ALL'])
    parser_measure.add_argument('--quantities', type=str.upper, nargs='+', default=['ALL'], choices=choices_list, metavar='QUANTITY', help='List of quantities to measure from the PDFs (default: all).')
    parser_measure.add_argument('--odds-window', type=float, default=DEFAULT_ODDS_WINDOW, help='Half-width of the integration window for odds calculation.')
    parser_measure.add_argument('--seed', type=int, default=None, help='Seed for the random number generator used in Z_RANDOM.')
    parser_measure.add_argument('--list-quantities', action='store_true', help='List all available quantities and their descriptions.')
                              
    parser_measure.set_defaults(func=measure_logic)

    # --- Parser for the "plot" command ---
    parser_plot = subparsers.add_parser('plot', help='Reconstruct and plot PDFs encoded with ColdPress.')
    parser_plot.add_argument('input', type=str, help='Name of input FITS table containing cold-pressed PDFs.')
    plot_group = parser_plot.add_mutually_exclusive_group(required=True)
    plot_group.add_argument('--id', nargs='+', type=str, help='List of ID(s) of the source(s) to plot.')
    plot_group.add_argument('--first', metavar='N', type=int, help='plot PDFs for the first N sources in the file.')
    plot_group.add_argument('--plot-all', action='store_true', dest='plot_all', help='Plot PDFs for all the sources in the file.')
    parser_plot.add_argument('--interactive', action='store_true', help='Display plots in an interactive window instead of saving to file.')
    parser_plot.add_argument('--idcol', type=str, nargs='?', default=DEFAULT_ID_COL, help='Name of input column containing source IDs.')
    parser_plot.add_argument('--encoded', type=str, nargs='+', default=[DEFAULT_ENCODED_COL], help='Name(s) of input column(s) containing cold-pressed PDFs.')
    parser_plot.add_argument('--outdir', type=str, default='.', help='Output directory for plot files.')
    parser_plot.add_argument('--format', type=str, default='png', help='Output format for plots.')
    parser_plot.add_argument('--method', type=str, default='all', choices=['steps', 'spline', 'all'], help='PDF reconstruction method for plots.')
    parser_plot.add_argument('--quantities', nargs='+', type=str, help='List of FITS columns to overplot as vertical lines.')
    parser_plot.add_argument('--units', type=str, nargs='?', default='redshift', choices=['redshift','zeta'], help='Specifies the representation for the PDF\'s independent axis: "redshift" (z) or "zeta" (ln(1+z)) (default: redshift).')
    parser_plot.set_defaults(func=plot_logic)
    
    # --- Parser for the "check" command ---
    parser_check = subparsers.add_parser('check', help='Check the PDFs for issues and flag them.')
    parser_check.add_argument('input', type=str, help='Name of input FITS catalog.')
    parser_check.add_argument('output', type=str, nargs='?', help='(Optional) name of output FITS catalog.')
    check_group = parser_check.add_mutually_exclusive_group(required=True)
    check_group.add_argument('--binned', type=str, help='Name of input column containing binned PDFs.')
    check_group.add_argument('--samples', type=str, help='Name of input column containing redshift samples.')
    parser_check.add_argument('--truncation-threshold', type=float, default=0.05, help='Threshold value for PDF truncation detection.')
    parser_check.add_argument('--list', action='store_true', help='List ID and flags of all flagged PDFs.')
    parser_check.add_argument('--idcol', type=str, help='Name of input column containing source IDs (required with --list).')
    parser_check.set_defaults(func=check_logic)

    args = parser.parse_args()
    
    if args.command in ['encode','decode']:
        if args.units == 'redshift':
            if args.zetamin is not None or args.zetamax is not None:
                parser.error("Cannot use --zetamin/--zetamax with --units redshift. Use --zmin/--zmax instead.")
            if args.binned or args.density:
                if args.zmin is None or args.zmax is None:
                    parser.error("--zmin and --zmax are required for --units redshift.")
     
        elif args.units == 'zeta':
            if args.zmin is not None or args.zmax is not None:
                parser.error("Cannot use --zmin/--zmax with --units zeta. Use --zetamin/--zetamax instead.")
            if args.binned or args.density:
                if args.zetamin is None or args.zetamax is None:
                    parser.error("--zetamin and --zetamax are required for --units zeta.")

    if args.command == 'encode':
        if (args.binned or args.density) and args.clip_fraction != 0.:
            parser.error('--clip-fraction can only be used with PDFs by random samples (--samples)')                            
     
    if args.command == 'check' and args.list and args.idcol is None:
        parser.error('--idcol is required when listing sources with flagged issues (--list)')

    if args.command == 'measure' and args.list_quantities:
        print("Available quantities for the 'measure' command:")
        for name, desc in QUANTITY_DESCRIPTIONS.items():
            print(f"  {name:<15} {desc}")
        sys.exit(0) 

    if args.command == 'measure':
        if not args.input or not args.output:
            parser_measure.error("the following arguments are required: input, output")

    args.func(args)

if __name__ == '__main__':
    main()