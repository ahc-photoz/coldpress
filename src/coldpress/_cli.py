#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import numpy as np
from astropy.io import fits

from . import __version__
from .encode import _batch_encode
from .decode import decode_to_binned, decode_quantiles, quantiles_to_binned
from .stats import measure_from_quantiles, ALL_QUANTITIES
from .utils import step_pdf_from_quantiles, plot_from_quantiles

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
    try:
        with fits.open(args.input) as h:
            if args.hdu >= len(h):
                print(f"Error: HDU {args.hdu} not found. File has {len(h)} HDUs (0 to {len(h)-1}).", file=sys.stderr)
                sys.exit(1)
                
            hdu = h[args.hdu]
            
            print(f"Inspecting '{args.input}'...")
            
            if not hdu.is_image:
                print(f"HDU {args.hdu} (Name: '{hdu.name}')")
                print(f"  Rows: {hdu.header['NAXIS2']}")
                print(f"  Columns: {len(hdu.columns)}")
                print("  --- Column Details ---")
                for col in hdu.columns:
                    print(f"    - Name: {col.name:<20} Format: {col.format}")
            else:
                print(f"HDU {args.hdu} is an Image HDU (Name: '{hdu.name}')")
                print(f"  Dimensions: {hdu.shape}")
                print(f"  Data Type (BITPIX): {hdu.header['BITPIX']}")

            if args.header:
                print("\n--- FITS Header ---")
                print(repr(hdu.header))

    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


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
            - zmin (float, optional): Min redshift for binned PDFs.
            - zmax (float, optional): Max redshift for binned PDFs.
            - out_encoded (str): Name for the output column with encoded data.
            - validate (bool): Whether to validate the encoding accuracy.
            - tolerance (float): Tolerance for validation.
            - keep_orig (bool): Whether to keep the original PDF column.
    """
    import time
    if args.length % 4 != 0:
        print(f"Error: Packet length (--length) must be a multiple of 4, but got {args.length}.", file=sys.stderr)
        sys.exit(1)

    print(f"Opening input file: {args.input}")

    with fits.open(args.input) as h:
        header = h[1].header
        if args.samples is not None:
            orig_column = args.samples
            data = {'format': 'samples', 'samples': h[1].data[args.samples]}
            history = f'PDFs from samples in column {args.samples} cold-pressed as {args.out_encoded}'
            print(f"Generating quantiles from random redshift samples and compressing into {args.length}-byte packets...")
        else:
            if args.binned is not None:
                orig_column = args.binned
                data = {'format': 'PDF_histogram'}
                history = f'Binned PDFs in column {args.binned} cold-pressed as {args.out_encoded}'
            else:
                orig_column = args.density   
                data = {'format': 'PDF_density'}
                history = f'Probability density in column {args.density} cold-pressed as {args.out_encoded}'
                 
            data['PDF'] = h[1].data[orig_column]
            data['zvector'] = np.linspace(args.zmin, args.zmax, data['PDF'].shape[1])    
      
            cratio = data['PDF'].shape[1]*data['PDF'].itemsize/args.length
            print(f"Compressing PDFs into {args.length}-byte packets (compression ratio: {cratio:.2f})...")

        orig_cols = list(h[1].columns)

        start = time.process_time()

        coldpress_PDF = _batch_encode(data, packetsize=args.length, ini_quantiles=args.length-9, validate=args.validate, tolerance=args.tolerance) 
                                                                                               
        end = time.process_time()
        cpu_seconds = end - start
        print(f"{coldpress_PDF.shape[0]} PDFs cold-pressed in {cpu_seconds:.6f} CPU seconds")

        nints = args.length // 4
        new_col = fits.Column(name=args.out_encoded, format=f'{nints}J', array=coldpress_PDF)

        final_cols = [c for c in orig_cols if c.name != args.out_encoded]
              
        if args.keep_orig:
            print(f"Including column '{orig_column}' in output FITS table.")
        else:
            final_cols = [c for c in final_cols if c.name != orig_column]
            print(f"Excluding column '{orig_column}' from output FITS table.")

        final_cols.append(new_col)
        new_hdu = fits.BinTableHDU.from_columns(final_cols, header=header)

    new_hdu.header.add_history(history)
    print(f"Writing compressed data to: {args.output}")
    new_hdu.writeto(args.output, overwrite=True)
    print('Done.')


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
            - out_binned (str): Name for the output column with binned PDFs.
            - zmin (float): Minimum redshift for the output grid.
            - zmax (float): Maximum redshift for the output grid.
            - zstep (float): Step size for the output redshift grid.
            - force_range (bool): If True, truncates PDFs outside the grid.
            - method (str): Interpolation method ('linear' or 'spline').
    """
    
    print(f"Opening input file: {args.input}")

    with fits.open(args.input) as h:
        header = h[1].header
        coldpress_PDF = h[1].data[args.encoded]
        orig_cols = list(h[1].columns)

        zvector = np.arange(args.zmin, args.zmax + args.zstep/2, args.zstep)
        zvsize = len(zvector)

        print(f"Decompressing PDFs using {args.method} interpolation of the quantiles...")
        try:
            decoded_PDF = decode_to_binned(coldpress_PDF, zvector, force_range=args.force_range, method=args.method)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            print("Hint: Use the --force-range flag to proceed with truncation at your own risk.", file=sys.stderr)
            sys.exit(1)

        new_col = fits.Column(name=args.out_binned, format=f'{zvsize}E', array=decoded_PDF)
        final_cols = [c for c in orig_cols if c.name != args.out_binned]
        final_cols.append(new_col)
        new_hdu = fits.BinTableHDU.from_columns(final_cols, header=header)
    
    new_hdu.header.add_history(f'PDFs in column {args.encoded} extracted as {args.out_binned}')
    print(f"Writing decompressed data to: {args.output}")
    new_hdu.writeto(args.output, overwrite=True)
    print('Done.')

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
    """

    print(f"Opening input file: {args.input}")
    with fits.open(args.input) as h:
        data = h[1].data
        header = h[1].header
        original_columns = data.columns

    qcold = data[args.encoded]
    Nsources = qcold.shape[0]
    
    q_to_compute = {q.upper() for q in args.quantities}
    if 'ALL' in q_to_compute:
        q_to_compute = ALL_QUANTITIES
    
    print(f"Will compute: {', '.join(sorted(list(q_to_compute)))}")

    d = {}
    for q_name in q_to_compute:
         d[q_name] = np.full(Nsources, np.nan, dtype=np.float32)

    valid = np.any(qcold != 0, axis=1)
    valid_indices = np.where(valid)[0]
    print(f"Calculating point estimates for {len(valid_indices)} valid sources...")
    
    for i in valid_indices:
        quantiles = decode_quantiles(qcold[i].tobytes())
        results = measure_from_quantiles(
            quantiles,
            quantities_to_measure=list(q_to_compute),
            odds_window=args.odds_window
        )
        for q_name, value in results.items():
            d[q_name][i] = value

    final_cols = []
    for col in original_columns:
        if col.name not in q_to_compute:
            final_cols.append(col)

    for name, array in d.items():
        format_str = 'E' if array.dtype == np.float32 else 'I'
        final_cols.append(fits.Column(name=name, format=format_str, array=array))

    new_hdu = fits.BinTableHDU.from_columns(final_cols, header=header)
    new_hdu.header['HISTORY'] = f'Computed point estimates from column: {args.encoded}'
    print(f"Writing point estimates to: {args.output}")
    new_hdu.writeto(args.output, overwrite=True)
    print('Done.')


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
            - idcol (str, optional): Column name containing source IDs.
            - encoded (str): Name of the column with encoded PDFs.
            - outdir (str): Directory to save plot files.
            - format (str): Output image format (e.g., 'png', 'pdf').
            - method (str): PDF reconstruction method for plotting 
              ('linear' or 'spline').
            - quantities (list, optional): List of FITS columns to
              overplot as vertical lines.
    """
    print(f"Opening input file: {args.input}")
    with fits.open(args.input) as h:
        try:
            import matplotlib
        except ImportError:
            print("Error: matplotlib is required for the plot command.", file=sys.stderr)
            sys.exit(1)
        data = h[1].data

    qcold = data[args.encoded]

    if args.quantities:
        all_cols_upper = {c.upper() for c in data.columns.names}
        for q_col in args.quantities:
            if q_col.upper() not in all_cols_upper:
                print(f"Error: Quantity column '{q_col}' not found in FITS table.", file=sys.stderr)
                sys.exit(1)

    if args.plot_all:
        indices_to_plot = range(len(data))
        print(f"Plotting all {len(indices_to_plot)} sources...")
    elif args.first is not None:
        if args.first <= len(data):
            indices_to_plot = range(args.first)
            print(f"Plotting first {len(indices_to_plot)} sources...")
        else:
            print(f"Warning: first {args.first} PDFs were requested but file contais only {len(data)}. Will plot all of them.")
            indices_to_plot = range(len(data))
    else:    
        if args.idcol not in data.columns.names:
            print(f"Error: --id specified, but no '{args.idcol}' column found in {args.input}", file=sys.stderr)
            sys.exit(1)
        source_ids = list(args.id)
        id_column_as_str = data[args.idcol].astype(str)
        source_ids_as_str = np.asarray(source_ids, dtype=str)
        indices_to_plot = np.where(np.isin(id_column_as_str, source_ids_as_str))[0]

        if len(indices_to_plot) != len(source_ids):
            print("Warning: Some specified IDs were not found in the file.", file=sys.stderr)
        print(f"Found {len(indices_to_plot)} of {len(source_ids)} specified IDs to plot.")

    os.makedirs(args.outdir, exist_ok=True)

    for i in indices_to_plot:
        source_id_val = data[args.idcol][i] if (args.idcol is not None) and (args.idcol in data.columns.names) else f"row_{i}"
        
        if not np.any(qcold[i] != 0):
            print(f"Skipping source {source_id_val}: No valid PDF data.")
            continue

        quantiles = decode_quantiles(qcold[i].tobytes())
        
        markers_to_plot = {}
        if args.quantities:
            for q_col in args.quantities:
                actual_col_name = next((c for c in data.columns.names if c.upper() == q_col.upper()), None)
                if actual_col_name:
                    markers_to_plot[q_col] = data[actual_col_name][i]

        output_filename = os.path.join(args.outdir, f"pdf_{source_id_val}.{args.format.lower()}")
        
        plot_from_quantiles(
            quantiles,
            output_filename=output_filename,
            source_id=source_id_val,
            method=args.method,
            markers=markers_to_plot
        )
        print(f"Saved plot to {output_filename}")

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
    print(f"Opening input file: {args.input}")
    with fits.open(args.input) as h:
        data = h[1].data
        header = h[1].header
        original_columns = data.columns
        if args.list:
            ID = data[args.idcol]
        
        if args.binned is None:
            samples = data[args.samples]
            invalid = (np.sum(~np.isfinite(samples),axis=1) > 0)
            v = ~invalid
            unresolved = (np.max(samples[v],axis=1)-np.min(samples[v],axis=1) == 0)
        else:
            PDF = data[args.binned]
            invalid = (np.sum(~np.isfinite(PDF),axis=1) > 0) | (np.nanmin(PDF,axis=1) < 0) | (np.nanmax(PDF,axis=1) == 0.)
            v = ~invalid
            unresolved = (np.sum(PDF[v],axis=1)-np.max(PDF[v],axis=1) == 0)
            threshold = args.truncation_threshold * np.max(PDF[v],axis=1)
            truncated = ((PDF[v,0] > threshold) | (PDF[v,-1] > threshold))
    
    if args.binned is None:
        print(f'Column {args.samples} contains {samples.shape[0]} sampled PDFs, each containing {samples.shape[1]} random redshift samples.')
    else:    
        print(f'Column {args.binned} contains {PDF.shape[0]} binned PDFs, each containing {PDF.shape[1]} redshift bins.')
    
    print(f"{np.sum(invalid)} PDFs have been flagged as 'invalid'")
    print(f"{np.sum(unresolved)} PDFs have been flagged as 'unresolved'")
    if args.binned is not None:
        print(f"{np.sum(truncated)} PDFs have been flagged as 'truncated'")
        
    if args.list:
        print('List of source IDs with flagged issues in their PDFs:')
        for source in ID[invalid]:
            print(f"{source}  invalid")
        for i, source in enumerate(ID[v]):
            tag = ""
            if unresolved[i]:
                tag += " unresolved"
            if args.binned is not None and truncated[i]:
                tag += " truncated"
            if tag != "":
                print(f"ID = {source}:{tag}")
    
    if args.output:                                
        d = {}
        d['Z_FLAGS'] = np.zeros(len(invalid), dtype=np.int16)
        d['PDF_invalid'] = invalid
        d['PDF_unresolved'] = np.zeros(len(invalid),dtype=bool)
        d['PDF_unresolved'][v] = unresolved
        d['Z_FLAGS'][d['PDF_invalid']] = 1
        d['Z_FLAGS'][d['PDF_unresolved']] += 2
        if args.binned is not None:
            d['PDF_truncated'] = np.zeros(len(invalid),dtype=bool)
            d['PDF_truncated'][v] = truncated
            d['Z_FLAGS'][d['PDF_truncated']] += 4

        final_cols = []
        new_col_names = d.keys()
        for col in original_columns:
            if col.name not in new_col_names:
                final_cols.append(col)

        for name, array in d.items():
            if array.dtype == np.float32: format_str = 'E'
            elif array.dtype == np.int16: format_str = 'I'
            elif array.dtype == bool: format_str = 'L'    
            final_cols.append(fits.Column(name=name, format=format_str, array=array))

        new_hdu = fits.BinTableHDU.from_columns(final_cols, header=header)
        new_hdu.header['HISTORY'] = f'Added flags columns indicating issues in the PDFs: {list(new_col_names)}'
        print(f"Writing point estimates to: {args.output}")
        new_hdu.writeto(args.output, overwrite=True)
        print('Done.')

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
    format_group = parser_encode.add_mutually_exclusive_group(required=True)
    format_group.add_argument('--density', type=str, help='Name of input column containing probability densities sampled in a grid of redshifts.')
    format_group.add_argument('--binned', type=str, help='Name of input column containing cumulative probabilities inside redshift bins.')
    format_group.add_argument('--samples', type=str, help='Name of input column containing a set of random redshift samples from the underlying probability distribution.')
    parser_encode.add_argument('-o', '--out-encoded', type=str, nargs='?', default='coldpress_PDF', help='Name of output column containing the cold-pressed PDFs.')
    parser_encode.add_argument('--zmin', type=float, help='Lowest redshift in the grid (with --density) or bins (with --binned).')
    parser_encode.add_argument('--zmax', type=float, help='Highest redshift in the grid (with --density) or bins (with --binned).')
    parser_encode.add_argument('--length', type=int, nargs='?', default=80, help='Length of cold-pressed PDFs in bytes (must be multiple of 4).')
    parser_encode.add_argument('--validate', action='store_true', default=False, help='Verify accuracy of recovered quantiles.')
    parser_encode.add_argument('--tolerance', type=float, nargs='?', default=0.001, help='Maximum shift tolerated for the redshift of the quantiles.')
    parser_encode.add_argument('--keep-orig', action='store_true', help='Include the original input column with binned PDFs or samples in the output file.')
    parser_encode.set_defaults(func=encode_logic)

    # --- Parser for the "decode" command ---
    parser_decode = subparsers.add_parser('decode', help='Extract PDFs previously encoded with ColdPress.')
    parser_decode.add_argument('input', type=str, help='Name of input FITS catalog')
    parser_decode.add_argument('output', type=str, help='Name of output FITS catalog')
    parser_decode.add_argument('--encoded', type=str, nargs='?', default='coldpress_PDF', help='Name of column containing cold-pressed PDFs.')
    parser_decode.add_argument('-o', '--out-binned', type=str, nargs='?', default='PDF_decoded', help='Name of output column for extracted binned PDFs.')
    parser_decode.add_argument('--zmin', type=float, help='Redshift of the first bin.')
    parser_decode.add_argument('--zmax', type=float, help='Redshift of the last bin.')
    parser_decode.add_argument('--zstep', type=float, help='Width of the redshift bins.')
    parser_decode.add_argument('--force-range', action='store_true', help='Force binning to the range given by [zmin,zmax] even if some PDFs are truncated.')
    parser_decode.add_argument('--method', type=str, nargs='?', default='linear', choices=['linear','spline'], help='Interpolation method for reconstruction of the binned PDF (default: linear).')
    parser_decode.set_defaults(func=decode_logic)

    # --- Parser for the "measure" command ---
    parser_measure = subparsers.add_parser('measure', help='Compute point estimates from compressed PDFs.')
    parser_measure.add_argument('input', type=str, nargs='?', help='Name of input FITS table containing cold-pressed PDFs.')
    parser_measure.add_argument('output', type=str, nargs='?', help='Name of output FITS table containing point estimates measured on the PDFs.')
    parser_measure.add_argument('--encoded', type=str, nargs='?', default='coldpress_PDF', help='Name of column containing cold-pressed PDFs.')
    choices_list = sorted(list(ALL_QUANTITIES) + ['ALL'])
    parser_measure.add_argument('--quantities', type=str, nargs='+', default=['all'], choices=choices_list, metavar='QUANTITY', help='List of quantities to measure from the PDFs (default: all).')
    parser_measure.add_argument('--odds-window', type=float, default=0.03, help='Half-width of the integration window for odds calculation.')
    parser_measure.add_argument('--list-quantities', action='store_true', help='List all available quantities and their descriptions.')
                              
    parser_measure.set_defaults(func=measure_logic)

    # --- Parser for the "plot" command ---
    parser_plot = subparsers.add_parser('plot', help='Reconstruct and plot PDFs encoded with ColdPress.')
    parser_plot.add_argument('input', type=str, help='Name of input FITS table containing cold-pressed PDFs.')
    plot_group = parser_plot.add_mutually_exclusive_group(required=True)
    plot_group.add_argument('--id', nargs='+', type=str, help='List of ID(s) of the source(s) to plot.')
    plot_group.add_argument('--first', metavar='N', type=int, help='plot PDFs for the first N sources in the file.')
    plot_group.add_argument('--plot-all', action='store_true', dest='plot_all', help='Plot PDFs for all the sources in the file.')
    parser_plot.add_argument('--idcol', type=str, nargs='?', help='Name of input column containing source IDs.')
    parser_plot.add_argument('--encoded', type=str, nargs='?', default='coldpress_PDF', help='Name of input column containing cold-pressed PDFs.')
    parser_plot.add_argument('--outdir', type=str, default='.', help='Output directory for plot files.')
    parser_plot.add_argument('--format', type=str, default='png', help='Output format for plots.')
    parser_plot.add_argument('--method', type=str, default='all', choices=['steps', 'spline', 'all'], help='PDF reconstruction method for plots.')
    parser_plot.add_argument('--quantities', nargs='+', type=str, help='List of FITS columns to overplot as vertical lines.')
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
    
    if args.command == 'encode':
        if args.binned and (args.zmin is None or args.zmax is None):
            parser.error('--zmin and --zmax are required when encoding from binned PDFs (--binned)')
        if args.samples and (args.zmin is not None or args.zmax is not None):
            parser.error('--zmin and --zmax can only be used with binned PDFs (--binned), not random samples (--samples)')

    if args.command == 'check' and args.list and args.idcol is None:
        parser.error('--idcol is required when listing sources with flagged issues (--list)')

    if args.command == 'measure' and args.list_quantities:
        print("Available quantities for the 'measure' command:")
        from .stats import QUANTITY_DESCRIPTIONS
        for name, desc in QUANTITY_DESCRIPTIONS.items():
            print(f"  {name:<15} {desc}")
        sys.exit(0) 

    if args.command == 'measure':
        if not args.input or not args.output:
            parser_measure.error("the following arguments are required: input, output")

    args.func(args)

if __name__ == '__main__':
    main()