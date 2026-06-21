import sys
import numpy as np
from astropy.io import fits

def find_column_name(columns, name):
    """
    Finds the actual name of a column in a FITS table, case-insensitively.

    Args:
        columns (astropy.io.fits.column.ColDefs): The column definitions object.
        name (str): The case-insensitive name to find.

    Returns:
        str or None: The actual column name if found, otherwise None.
    """
    if name is None:
        return None
    names_upper = [c.name.upper() for c in columns]
    name_upper = name.upper()
    try:
        idx = names_upper.index(name_upper)
        return columns[idx].name
    except ValueError:
        return None


def fix_encoded_column(hdu, colname):
    """
    Check if the column colname has variable length and if so recreate the entire HDU
    with a fixed length version of the column.
    """
    actual_colname = find_column_name(hdu.columns, colname)
    if actual_colname is None:
        print(f"Error: column {colname} not found in input table. List of columns found:")
        print(hdu.columns.names)
        sys.exit(1)
        
    # astropy treats variable-length array columns as objects, not 2D arrays
    if hdu.data[actual_colname].dtype == 'object':
        
        # Create the corrected NumPy array from the problematic column
        qcold = hdu.data[actual_colname]
        lengths = np.array([len(x) for x in qcold])
        maxlength = np.max(lengths)
        if maxlength == 0:
            print(f"Error: All rows contain NULL values in column '{actual_colname}'.")
            sys.exit(1)
    
        qcold_fixed = np.zeros((lengths.shape[0],maxlength),dtype='>i4') # ensure integers are big-endian
        valid_indices = np.where(lengths == maxlength)[0]
        for i in valid_indices:
            qcold_fixed[i] = qcold[i]
                       
        # Create a new, fixed-width FITS Column object
        new_format = f'{qcold_fixed.shape[1]}J'
        new_column = fits.Column(name=actual_colname, format=new_format, array=qcold_fixed)
        
        print(f"Warning: column '{actual_colname}' converted from variable-length to fixed-length format.")

        # Replace the old column definition in the list of columns
        original_columns = list(hdu.columns)
        col_idx = [i for i, col in enumerate(original_columns) if col.name == actual_colname][0]
        original_columns[col_idx] = new_column
        
        # Return a new HDU built from the corrected column list
        return fits.BinTableHDU.from_columns(original_columns, header=hdu.header)
    
    # If no fix is needed, return the original HDU
    return hdu
    
def process_fits_table(input_path, output_path, required_cols, drop_cols, history_msg, callback, **kwargs):
    """
    Reads a FITS table, validates and extracts specific columns, applies a processing callback,
    and writes the updated table to a new file.

    Args:
        input_path (str): Path to the input FITS file.
        output_path (str): Path for the output FITS file.
        required_cols (list): List of column names (str) needed for processing.
        drop_cols (list): List of column names (str) to omit from the output file.
        history_msg (str): Message to append to the FITS header history.
        callback (callable): Function containing the core logic. 
                             Signature: callback(data_arrays, actual_names, **kwargs) -> dict
                             Returns: {new_col_name: (fits_format_string, data_array)}
    """
    print(f"Opening input file: {input_path}")
    try:
        with fits.open(input_path) as h:
            if len(h) < 2:
                print(f"Error: '{input_path}' does not contain table data at HDU 1.", file=sys.stderr)
                sys.exit(1)
                
            hdu = h[1]
            data_arrays = {}
            actual_names = {}

            for colname in required_cols:
                if colname is None:
                    continue
                
                hdu = fix_encoded_column(hdu, colname)
                actual_name = find_column_name(hdu.columns, colname)
                
                if actual_name is None:
                    print(f"Error: Required column '{colname}' not found in '{input_path}'.", file=sys.stderr)
                    sys.exit(1)

                actual_names[colname] = actual_name
                data_arrays[colname] = hdu.data[actual_name]

            header = hdu.header
            orig_cols = list(hdu.columns)

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'", file=sys.stderr)
        sys.exit(1)

    new_columns_data = callback(data_arrays, actual_names, **kwargs)

    if new_columns_data is None:
        return

    final_cols = []
    drop_upper = {c.upper() for c in drop_cols if c is not None}

    for col in orig_cols:
        if col.name.upper() not in drop_upper:
            final_cols.append(col)

    for name, (format_str, array) in new_columns_data.items():
        final_cols.append(fits.Column(name=name, format=format_str, array=array))

    new_hdu = fits.BinTableHDU.from_columns(final_cols, header=header)
    if history_msg:
        new_hdu.header.add_history(history_msg)

    print(f"Writing output to: {output_path}")
    new_hdu.writeto(output_path, overwrite=True)
    print('Done.')    