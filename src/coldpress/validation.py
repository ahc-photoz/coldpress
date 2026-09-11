"""Spectroscopic validation of ColdPress PDFs (monotone spline CDFs)."""
import os
import struct
import numpy as np

from .decode import decode_quantiles
from .utils import _monotone_natural_spline
from .stats import zmode_from_quantiles, zmean_from_quantiles, zmedian_from_quantiles

DIAGNOSTICS = ('qqplot', 'pit', 'outlier-rate')


def validate_quantiles(quantiles, zspec, estimator='mode', odds_window=0.03):
    """Return PIT, odds and outlier flag for one continuous redshift PDF.

    Outliers lie outside zphot +/- odds_window*(1+zphot).
    Odds integrate within +/- odds_window*(1+zphot), as in `measure`.
    zspec is always in redshift units, not ln(1+z).
    """
    q = np.asarray(quantiles, dtype=float)
    if q.ndim != 1 or q.size < 2 or not np.all(np.isfinite(q)) or np.any(np.diff(q) <= 0):
        raise ValueError('Quantiles must be finite and strictly increasing, with at least two values.')
    if not np.isfinite(zspec):
        raise ValueError('Spectroscopic redshift must be finite.')
    for name, value in [('odds_window', odds_window)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive.')
    estimators = {'mode': zmode_from_quantiles, 'mean': zmean_from_quantiles,
                  'median': zmedian_from_quantiles}
    if estimator not in estimators:
        raise ValueError('Unknown point estimator.')
    zphot = estimators[estimator](q)
    half_width = odds_window * (1 + zphot)
    lower, upper = zphot - half_width, zphot + half_width
    points = np.array([zspec, lower, upper])
    # Clamp outside support explicitly; do not extrapolate the spline wings.
    cdf = np.zeros(3)
    cdf[points >= q[-1]] = 1
    inside = (points > q[0]) & (points < q[-1])
    if np.any(inside):
        cdf[inside] = _monotone_natural_spline(
            points[inside], q, np.linspace(0, 1, q.size))
    if not np.all(np.isfinite(cdf)):
        raise ValueError('Spline produced a non-finite CDF.')
    cdf = np.clip(cdf, 0, 1)
    pit = cdf[0]
    odds = np.clip(cdf[2] - cdf[1], 0, 1)
    # Use the very same bounds as the odds integral, including both endpoints.
    outlier = zspec < lower or zspec > upper
    return float(pit), float(odds), bool(outlier)



def magnitude_groups(magnitudes):
    """Yield occupied, integer-aligned [lower, lower+1) bins and row masks."""
    values = np.asarray(magnitudes, dtype=float)
    finite = np.isfinite(values)
    floors = np.floor(values)
    for lower in np.unique(floors[finite]):
        yield int(lower), finite & (floors == lower)


def outlier_rate_bins(odds, outliers, bins=10):
    """Mean odds, outlier fraction, Wilson 68% bounds and count per occupied bin.

    Bins are [left, right), except that the last includes odds=1.
    """
    odds = np.asarray(odds, dtype=float)
    outliers = np.asarray(outliers, dtype=bool)
    if odds.ndim != 1 or odds.shape != outliers.shape or not np.all(np.isfinite(odds)) or np.any((odds < 0) | (odds > 1)):
        raise ValueError('Odds must be a finite 1D array in [0, 1], matching outliers.')
    if not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError('bins must be a positive integer.')
    indices = np.minimum((odds*bins).astype(int), bins-1)
    result = []
    for index in np.unique(indices):
        selected = indices == index
        n = int(selected.sum())
        if n < 4:
            continue
        rate = float(outliers[selected].mean())
        center = (rate + 0.5/n)/(1+1/n)
        half = np.sqrt(rate*(1-rate)/n + 0.25/n**2)/(1+1/n)
        result.append((odds[selected].mean(), rate, center-half, center+half, n))
    return np.asarray(result, dtype=float).reshape(-1, 5)


def plot_diagnostic(kind, groups, filename, bins=10,
                    estimator='mode', odds_window=0.03):
    """Overlay (label, PIT, odds, outliers) groups on one diagnostic's axes."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.pyplot import get_cmap
    fig = Figure(figsize=(8, 5.5), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    # Same ordered groups and colors across every diagnostic.
    palette = get_cmap('tab10' if len(groups) <= 10 else 'viridis')
    colors = palette(np.linspace(0, 1, max(2, len(groups))))
    for (label, pit, odds, outliers), color in zip(groups, colors):
        n = len(pit)
        label = f'{label} (N={n})'
        if kind == 'pit':
            ax.hist(pit, bins=np.linspace(0, 1, bins+1), density=True,
                    histtype='step', linewidth=1.8, color=color, label=label)
        elif kind == 'qqplot':
            ax.plot((np.arange(n)+0.5)/n, np.sort(pit), color=color,
                    linewidth=1.5, label=label)
        elif kind == 'outlier-rate':
            x, y, lo, hi, _ = outlier_rate_bins(odds, outliers, bins).T
            ax.errorbar(x, y, yerr=np.maximum(0, np.vstack((y-lo, hi-y))),
                        fmt='o-', capsize=3, color=color, label=label)
        else:
            raise ValueError(f'Unknown diagnostic: {kind}')
    if kind == 'pit':
        ax.axhline(1, color='black', ls='--', label='Uniform expectation')
        ax.set(xlabel='PIT = CDF(zspec)', ylabel='Probability density', xlim=(0, 1))
    elif kind == 'qqplot':
        ax.plot([0, 1], [0, 1], 'k--', label='Uniform expectation')
        ax.set(xlabel='Theoretical uniform quantile', ylabel='Observed PIT quantile',
               xlim=(0, 1), ylim=(0, 1))
    else:
        ax.plot([0, 1], [1, 0], 'k--', label='η = 1 − ⟨odds⟩')
        ax.set(xlabel=f'Mean odds about Z_{estimator.upper()} (window={odds_window:g})',
               ylabel='Outlier fraction η', xlim=(0, 1), ylim=(-0.03, 1.03))
    title = kind
    if kind == 'outlier-rate':
        title += f' (68% Wilson intervals)\n|zphot − zspec| / (1 + zphot) > {odds_window:g}'
    ax.set_title(title)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.grid(alpha=0.2)
    fig.savefig(filename, dpi=160)
    fig.clear()


def run_tests(args):
    """Read HDU 1 without modifying the input and save diagnostic plots."""
    from astropy.table import Table
    table = Table.read(args.input, hdu=1)
    names = {name.upper(): name for name in table.colnames}

    def column(name):
        if name.upper() not in names:
            raise ValueError(f"Required column '{name}' not found in input table.")
        return table[names[name.upper()]]

    def scalar_column(name):
        data = np.ma.asarray(column(name), dtype=float).filled(np.nan)
        if data.ndim != 1:
            raise ValueError(f"Column '{name}' must contain scalar values.")
        return data

    packets = column(args.encoded)
    zspec = scalar_column(args.zspec)
    mag = scalar_column(args.mag) if args.mag else None
    eligible = np.isfinite(zspec) & (zspec >= 0)
    if mag is not None:
        eligible &= np.isfinite(mag)
    accepted, results = [], []
    bad_packets = 0
    for i in np.flatnonzero(eligible):
        try:
            packet = np.ma.asarray(packets[i])
            if np.any(np.ma.getmaskarray(packet)) or packet.ndim != 1 or packet.size < 1 or not np.any(packet):
                raise ValueError('Missing packet')
            if packet.dtype.kind not in 'iu':
                raise ValueError('Encoded PDFs must contain integer packets')
            packet = np.asarray(packet, dtype='>i4')
            q = decode_quantiles(packet.tobytes(), units='redshift')
            result = validate_quantiles(q, zspec[i], args.estimator,
                                        args.odds_window)
        except (ValueError, IndexError, struct.error, OverflowError):
            bad_packets += 1
            continue
        accepted.append(i)
        results.append(result)
    print(f'Using {len(accepted)}/{len(table)} rows; skipped '
          f'{int((~eligible).sum())} invalid zspec/magnitude rows and {bad_packets} invalid PDFs.')
    if not accepted:
        raise ValueError('No valid PDFs with spectroscopic redshifts remain.')
    pit, odds, outliers = np.asarray(results).T
    masks = [(None, np.ones(len(pit), dtype=bool))] if mag is None else magnitude_groups(mag[accepted])
    groups = []
    for lower, mask in masks:
        label = 'All valid sources' if lower is None else f'{lower} ≤ {args.mag} < {lower+1}'
        count = int(mask.sum())
        if lower is not None and count < args.nmin:
            print(f'Excluding {label}: {count} valid sources < nmin={args.nmin}.')
            continue
        groups.append((label, pit[mask], odds[mask], outliers[mask]))
    if not groups:
        raise ValueError('No magnitude bins meet --nmin; no plots were generated.')
    selected_tests = DIAGNOSTICS if 'all' in args.tests else tuple(dict.fromkeys(args.tests))
    os.makedirs(args.outdir, exist_ok=True)
    suffix = 'all' if mag is None else 'by_mag'
    for kind in selected_tests:
        filename = os.path.join(args.outdir, f'{kind}_{suffix}.{args.format}')
        plot_diagnostic(kind, groups, filename, args.bins, args.estimator, args.odds_window)
        print(f'Saved {filename}')
