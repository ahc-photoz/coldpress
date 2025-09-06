#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import numpy as np
from astropy.io import fits

from . import __version__
from .encode import encode_from_binned, encode_from_density, encode_from_samples
from .decode import decode_to_binned, decode_quantiles
from .stats import measure_from_quantiles, ALL_QUANTITIES, QUANTITY_DESCRIPTIONS
from .utils import plot_from_quantiles

# --- Constants for Default Column Names ---
DEFAULT_ID_COL = 'ID'
DEFAULT_ENCODED_COL = 'coldpress_PDF'
DEFAULT_DECODED_COL = 'PDF_decoded'

def fix_encoded_column(hdu, colname):
    """
    Check if the column colname has variable length and if so recreate the entire HDU
    with a fixed length version of the column.
    """
    if colname not in hdu.columns.names:
        print(f"Error: column {colname} not found in input table. List of columns found:")
        print(hdu.columns.names)
        sys.exit(1)
        
    # astropy treats variable-length array columns as objects, not 2D arrays
    if hdu.data[colname].dtype == 'object':
        
        # Create the corrected NumPy array from the problematic column
        qcold = hdu.data[colname]
        lengths = np.array([len(x) for x in qcold])
        maxlength = np.max(lengths)
        if maxlength == 0:
            print(f"Error: All rows contain NULL values in column '{colname}'.")
            sys.exit(1)
    
        qcold_fixed = np.zeros((lengths.shape[0],maxlength),dtype='>i4') # ensure integers are big-endian
        valid_indices = np.where(lengths == maxlength)[0]
        for i in valid_indices:
            qcold_fixed[i] = qcold[i]
                       
        # Create a new, fixed-width FITS Column object
        new_format = f'{qcold_fixed.shape[1]}J'
        new_column = fits.Column(name=colname, format=new_format, array=qcold_fixed)
        
        print(f"Warning: column '{colname}' converted from variable-length to fixed-length format.")

        # Replace the old column definition in the list of columns
        original_columns = list(hdu.columns)
        col_idx = [i for i, col in enumerate(original_columns) if col.name == colname][0]
        original_columns[col_idx] = new_column
        
        # Return a new HDU built from the corrected column list
        return fits.BinTableHDU.from_columns(original_columns, header=hdu.header)
    
    # If no fix is needed, return the original HDU
    return hdu 
 
 
 
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
            - zmin (float, optional): Min redshift.
            - zmax (float, optional): Max redshift.
            - out_encoded (str): Name for the output column with encoded data.
            - validate (bool): Whether to validate the encoding accuracy.
            - tolerance (float): Tolerance for validation.
            - keep_orig (bool): Whether to keep the original PDF column.
            - clip_fraction (float): Fraction of samples to be clipped out at the extremes of the redshift range.
    """
    if args.length % 4 != 0:
        print(f"Error: Packet length (--length) must be a multiple of 4, but got {args.length}.", file=sys.stderr)
        sys.exit(1)

    print(f"Opening input file: {args.input}")

    with fits.open(args.input) as h:
        header = h[1].header
        orig_cols = list(h[1].columns)
        
        if args.samples is not None:
            orig_column = args.samples
            history = f'PDFs from samples in column {args.samples} cold-pressed as {args.out_encoded}'
            print(f"Generating quantiles from random redshift samples and compressing into {args.length}-byte packets...")
            if (args.zmin is not None) and (args.zmax) is not None:
                clip_range = [args.zmin, args.zmax]
            else:
                clip_range = None    
            samples = h[1].data[args.samples]
            coldpress_PDF = encode_from_samples(samples, packetsize=args.length, ini_quantiles=args.length-8, 
                                                validate=args.validate, tolerance=args.tolerance, clip_fraction=args.clip_fraction, 
                                                clip_range=clip_range) 
        elif args.binned is not None:
            orig_column = args.binned
            history = f'Binned PDFs in column {args.binned} cold-pressed as {args.out_encoded}'
            PDF = h[1].data[args.binned]
            zvector = np.linspace(args.zmin, args.zmax, PDF.shape[1])
            cratio = PDF.shape[1]*PDF.itemsize/args.length
            print(f"Compressing binned PDFs into {args.length}-byte packets (compression ratio: {cratio:.2f})...")
            coldpress_PDF = encode_from_binned(PDF, zvector, packetsize=args.length, ini_quantiles=args.length-8, 
                                                validate=args.validate, tolerance=args.tolerance)
        elif args.density is not None:
            orig_column = args.density
            history = f'Probability density in column {args.density} cold-pressed as {args.out_encoded}'
            PDF = h[1].data[args.density]
            zvector = np.linspace(args.zmin, args.zmax, PDF.shape[1])
            cratio = PDF.shape[1]*PDF.itemsize/args.length
            print(f"Compressing density PDFs into {args.length}-byte packets (compression ratio: {cratio:.2f})...")
            coldpress_PDF = encode_from_density(PDF, zvector, packetsize=args.length, ini_quantiles=args.length-8, 
                                                validate=args.validate, tolerance=args.tolerance)
            
                                                                                               
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
        # Fix the HDU in memory if necessary
        hdu = fix_encoded_column(h[1], args.encoded)
        qcold = hdu.data[args.encoded].astype('>i4') #ensure array is big-endian

        header = hdu.header
        orig_cols = list(hdu.columns)

        zvector = np.arange(args.zmin, args.zmax + args.zstep/2, args.zstep)
        zvsize = len(zvector)

        print(f"Decompressing PDFs using {args.method} interpolation of the quantiles...")
        try:
            decoded_PDF = decode_to_binned(qcold, zvector, force_range=args.force_range, method=args.method)
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
        # Fix the HDU in memory if necessary
        hdu = fix_encoded_column(h[1], args.encoded)
        qcold = hdu.data[args.encoded].astype('>i4') #ensure array is big-endian

        header = hdu.header
        orig_cols = list(hdu.columns)

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
    for col in orig_cols:
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
            - interactive (bool): Flag to show plots interactively.
            - idcol (str, optional): Column name containing source IDs.
            - encoded (str): Name of the column with encoded PDFs.
            - outdir (str): Directory to save plot files.
            - format (str): Output image format (e.g., 'png', 'pdf').
            - method (str): PDF reconstruction method for plotting 
              ('linear' or 'spline').
            - quantities (list, optional): List of FITS columns to
              overplot as vertical lines.
    """
    try:
        import matplotlib
    except ImportError:
        print("Error: matplotlib is required for the plot command.", file=sys.stderr)
        sys.exit(1)

    print(f"Opening input file: {args.input}")         
    with fits.open(args.input) as h:
        # Fix the HDU in memory if necessary
        hdu = fix_encoded_column(h[1], args.encoded)
        qcold = hdu.data[args.encoded].astype('>i4') #ensure array is big-endian

        data = hdu.data
        header = hdu.header
        orig_cols = list(hdu.columns)
    
    if args.quantities:
        all_cols_upper = {c.upper() for c in data.columns.names}
        for q_col in args.quantities:
            if q_col.upper() not in all_cols_upper:
                print(f"Error: Quantity column '{q_col}' not found in FITS table.", file=sys.stderr)
                sys.exit(1)

    # Determine which rows to plot based on the mutually exclusive arguments
    if args.plot_all:
        indices_to_plot = range(len(data))
    elif args.first is not None:
        num_to_plot = min(args.first, len(data))
        if args.first > len(data):
            print(f"Warning: Requested first {args.first} PDFs, but file only contains {len(data)}. Plotting all sources.")
        indices_to_plot = range(num_to_plot)
    else:  # This handles the 'id' case, which is the default if not the others
        if args.idcol not in data.columns.names:
            print(f"Error: --id specified, but no '{args.idcol}' column found in {args.input}", file=sys.stderr)
            sys.exit(1)
        
        source_ids = list(args.id)
        id_column_as_str = data[args.idcol].astype(str)
        source_ids_as_str = np.asarray(source_ids, dtype=str)
        indices_to_plot = np.where(np.isin(id_column_as_str, source_ids_as_str))[0]

        if len(indices_to_plot) != len(source_ids):
            print("Warning: Some specified IDs were not found in the file.", file=sys.stderr)
        
    print(f"Plotting {len(indices_to_plot)} source(s)...")

    if not args.interactive:
        os.makedirs(args.outdir, exist_ok=True)

    for i in indices_to_plot:
        # Corrected logic for determining the source ID string
        source_id_val = data[args.idcol][i] if args.idcol is not None and args.idcol in data.columns.names else f"row_{i}"
        
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

        output_filename = None
        if not args.interactive:
            output_filename = os.path.join(args.outdir, f"pdf_{source_id_val}.{args.format.lower()}")
        
        try:
            plot_from_quantiles(
                quantiles,
                output_filename=output_filename,
                interactive=args.interactive,
                source_id=source_id_val,
                method=args.method,
                markers=markers_to_plot
            )
            if not args.interactive:
                print(f"Saved plot to {output_filename}")
        except ImportError as e:
            print(e, file=sys.stderr)
            sys.exit(1)

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
    format_group.add_argument('--binned', type=str, help='Name of input column containing probabilities inside redshift bins.')
    format_group.add_argument('--samples', type=str, help='Name of input column containing a set of random redshift samples from the underlying probability distribution.')
    parser_encode.add_argument('-o', '--out-encoded', type=str, nargs='?', default=DEFAULT_ENCODED_COL, help='Name of output column containing the cold-pressed PDFs.')
    parser_encode.add_argument('--zmin', type=float, help='Lowest redshift in the grid/bins')
    parser_encode.add_argument('--zmax', type=float, help='Highest redshift in the grid/bins')
    parser_encode.add_argument('--length', type=int, nargs='?', default=80, help='Length of compressed PDFs in bytes (must be multiple of 4).')
    parser_encode.add_argument('--validate', action='store_true', default=False, help='Verify accuracy of recovered quantiles.')
    parser_encode.add_argument('--tolerance', type=float, nargs='?', default=0.001, help='Maximum shift tolerated for the redshift of the quantiles.')
    parser_encode.add_argument('--keep-orig', action='store_true', help='Include the original input column with binned PDFs or samples in the output file.')
    parser_encode.add_argument('--clip-fraction', type=float, nargs='?', default=0, help='Fraction of samples to clip out at the extremes of the redshift range.')
    parser_encode.set_defaults(func=encode_logic)

    # --- Parser for the "decode" command ---
    parser_decode = subparsers.add_parser('decode', help='Extract PDFs previously encoded with ColdPress.')
    parser_decode.add_argument('input', type=str, help='Name of input FITS catalog')
    parser_decode.add_argument('output', type=str, help='Name of output FITS catalog')
    parser_decode.add_argument('--encoded', type=str, nargs='?', default=DEFAULT_ENCODED_COL, help='Name of column containing cold-pressed PDFs.')
    parser_decode.add_argument('-o', '--out-binned', type=str, nargs='?', default=DEFAULT_DECODED_COL, help='Name of output column for extracted binned PDFs.')
    parser_decode.add_argument('--zmin', type=float, help='Redshift of the first bin.')
    parser_decode.add_argument('--zmax', type=float, help='Redshift of the last bin.')
    parser_decode.add_argument('--zstep', type=float, help='Width of the redshift bins.')
    parser_decode.add_argument('--force-range', action='store_true', help='Force binning to the range [zmin,zmax] even if PDFs are truncated.')
    parser_decode.add_argument('--method', type=str, nargs='?', default='linear', choices=['linear','spline'], help='Interpolation method for PDF reconstruction (default: linear).')
    parser_decode.set_defaults(func=decode_logic)

    # --- Parser for the "measure" command ---
    parser_measure = subparsers.add_parser('measure', help='Compute point estimates from compressed PDFs.')
    parser_measure.add_argument('input', type=str, nargs='?', help='Name of input FITS table containing cold-pressed PDFs.')
    parser_measure.add_argument('output', type=str, nargs='?', help='Name of output FITS table containing point estimates measured on the PDFs.')
    parser_measure.add_argument('--encoded', type=str, nargs='?', default=DEFAULT_ENCODED_COL, help='Name of column containing cold-pressed PDFs.')
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
    parser_plot.add_argument('--interactive', action='store_true', help='Display plots in an interactive window instead of saving to file.')
    parser_plot.add_argument('--idcol', type=str, nargs='?', help='Name of input column containing source IDs.')
    parser_plot.add_argument('--encoded', type=str, nargs='?', default=DEFAULT_ENCODED_COL, help='Name of input column containing cold-pressed PDFs.')
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
        if args.density and (args.zmin is None or args.zmax is None):
            parser.error('--zmin and --zmax are required when encoding from probability density (--density)')
        if args.samples is None and args.clip_fraction != 0.:
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