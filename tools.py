import numpy as np
import scipy.signal as ss
from scipy.stats import entropy


def psd(signal):
    f, P = ss.welch(signal, nperseg=256, noverlap=128, axis=1)
    return f, P.mean(axis=0)


def calculate_metrics(restored, signal, reference, artifact_mask):
    _res = restored[:, artifact_mask]
    _ref = reference[:, artifact_mask]
    _sig = signal[:, artifact_mask]

    # SNR
    snr_before = calculate_snr(_ref, _sig - _ref)
    snr_after = calculate_snr(_ref, _res - _ref)
    delta_snr = snr_after - snr_before

    # NMSE
    nmse = calculate_nmse(_ref, _res)

    # Correlation (on full signal)
    r_before = calculate_correlation(signal, reference)
    r_after = calculate_correlation(restored, reference)
    delta_r = r_after - r_before

    # Spectral features
    _, P_sig = psd(signal)
    _, P_ref = psd(reference)
    _, P_res = psd(restored)

    sp_snr_before = calculate_snr(P_sig, P_sig - P_ref)
    sp_snr_after = calculate_snr(P_ref, P_res - P_ref)
    sp_delta_snr = sp_snr_after - sp_snr_before

    # Mutual information
    mi = calculate_mutual_information(_ref, _res)

    return {
        'SNR_before': snr_before,
        'SNR_after': snr_after,
        'ΔSNR': delta_snr,
        'NMSE': nmse,
        'R_before': r_before,
        'R_after': r_after,
        'ΔR': delta_r,
        'Spectral ΔSNR': sp_delta_snr,
        'MI': mi,
    }


def calculate_snr(signal, noise):
    return 10 * np.log10(signal.var() / noise.var())


def signal_to_artifact_ratio(signal, artifact):
    snrs = [s.var() / a.var() for s, a in zip(signal, artifact)]
    return np.mean(10 * np.log10(snrs))


def calculate_nmse(reference, restored):
    nmse = ((reference - restored)**2).sum() / (reference**2).sum()
    return 10 * np.log10(nmse)


def calculate_mutual_information(reference, restored):
    x, y = reference.ravel(), restored.ravel()

    num_bins = 128, 128
    p_xy = np.histogram2d(x, y, bins=num_bins)[0]
    p_xy /= p_xy.sum()
    p_xy = p_xy.clip(np.finfo(float).eps)

    p_x = p_xy.sum(axis=1).reshape(-1, 1)
    p_y = p_xy.sum(axis=0).reshape(1, -1)

    mi = (p_xy * np.log(p_xy / (p_x * p_y))).sum()

    H_x = entropy(p_x.ravel())
    H_y = entropy(p_y.ravel())

    return mi / np.sqrt(H_x * H_y)


def calculate_correlation(signal, other):
    return np.mean([np.corrcoef(a, b)[1, 0] for a, b in zip(signal, other)])


def filter_bandpass(signal, low, high, fs, order=2):
    Wn = 2 * np.array([low, high]) / fs
    sos = ss.butter(order, Wn, btype='bandpass', output='sos')

    return ss.sosfiltfilt(sos, signal, axis=0)


def filter_highpass(signal, freq, fs, order=2):
    Wn = 2 * freq / fs
    sos = ss.butter(order, Wn, btype='highpass', output='sos')

    return ss.sosfiltfilt(sos, signal, axis=0)


def apply_artifact_removal(original_signals, corrupted_signals, func,
                           **kwargs):
    restored_signals = []
    for original, corrupted in zip(original_signals, corrupted_signals):
        restored = corrupted.copy()
        keep_len = original.shape[1] // 3
        start = keep_len
        end = start + keep_len

        for n in range(original.shape[0]):
            s = corrupted[n][start:end]
            refs = [corrupted[n][:start], corrupted[n][end:end + keep_len]]
            restored[n][start:end] = func(s, refs, **kwargs)

        restored_signals.append(restored)

    return restored_signals


def mask_to_intervals(mask, index=None):
    """Convert a boolean mask to a sequence of intervals.
    Caveat: when no index is given, the returned values correspond to the
    Python pure integer indexing (starting element included, ending element
    excluded). When an index is passed, pandas label indexing convention
    with strict inclusion is used.
    For example `mask_to_intervals([0, 1, 1, 0])` will return `[(1, 3)]`,
    but `mask_to_intervals([0, 1, 1, 0], ["a", "b", "c", "d"])` will return
    the value `[("b", "c")]`.
    Parameters
    ----------
    mask : numpy.ndarray
        A boolean array.
    index : Sequence, optional
        Elements to use as indices for determining interval start and end. If
        no index is given, integer array indices are used.
    Returns
    -------
    intervals : Sequence[Tuple[Any, Any]]
        A sequence of (start_index, end_index) tuples. Mindful of the caveat
        described above concerning the indexing convention.
    """
    if not np.any(mask):
        return []

    edges = np.flatnonzero(np.diff(np.pad(mask, 1)))
    intervals = edges.reshape((len(edges) // 2, 2))

    if index is not None:
        return [(index[i], index[j - 1]) for i, j in intervals]

    return [(i, j) for i, j in intervals]


def intervals_to_mask(intervals, size=None):
    mask = np.zeros(size, dtype=bool)
    for i, j in intervals:
        mask[i:j] = True

    return mask
