import pywt
import numpy as np

from tools import intervals_to_mask


class SingleChannelDenoiser:
    def run(self, signal, artifacts, fs=None):
        norm_signal, norm_params = self.normalize(signal)
        filtered = np.zeros_like(signal)

        for n in range(norm_signal.shape[0]):
            filtered[n] = self.run_single_channel(
                norm_signal[n], artifacts, fs)

        return self.denormalize(filtered, norm_params)

    def normalize(self, signal):
        s_mean = signal.mean(axis=-1).reshape(-1, 1)
        s_std = signal.std(axis=-1).reshape(-1, 1)

        return (signal - s_mean) / s_std, (s_mean, s_std)

    def denormalize(seld, signal, params):
        s_mean, s_std = params
        return signal * s_std + s_mean

    def run_single_channel(signal, artifacts):
        raise NotImplementedError()


class SWTDenoiser(SingleChannelDenoiser):
    def __init__(self, wavelet='sym4', level=5):
        self.wavelet = wavelet
        self.level = level

    def pad(self, data):
        min_div = 2**self.level
        remainder = len(data) % min_div
        pad_len = (min_div - remainder) % min_div

        return np.pad(data, (0, pad_len))


class WaveletThresholding(SWTDenoiser):
    def __init__(self, wavelet='sym4', level=5, mode='hard'):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

    def run_single_channel(self, signal, artifacts, fs=None):
        sig_ = self.pad(signal)
        coeffs = pywt.swt(sig_, self.wavelet, self.level, norm=True,
                          trim_approx=True)
        coeffs = np.array(coeffs)

        artifact_mask = intervals_to_mask(artifacts, coeffs.shape[1])

        k = np.sqrt(2 * np.log(coeffs.shape[1]))
        thresholds = k * np.median(np.abs(coeffs), axis=1) / 0.6745

        for ws, th in zip(coeffs, thresholds):
            ws[artifact_mask] = self.threshold(ws[artifact_mask], th)

        rec = pywt.iswt(coeffs, wavelet=self.wavelet, norm=True)

        return rec[:len(signal)]

    def threshold(self, coeffs, threshold):
        if self.mode == 'hard':
            return np.where(np.abs(coeffs) <= threshold, coeffs, 0.)
        elif self.mode == 'soft':
            return np.clip(coeffs, -threshold, threshold)

        raise RuntimeError(f'Invalid thresholding mode `{self.mode}`.')


class WaveletQuantileNormalization(SingleChannelDenoiser):
    def __init__(self, wavelet='sym4', mode='periodization', alpha=1, n=30):
        self.wavelet = wavelet
        self.alpha = alpha
        self.mode = mode
        self.n = n

    def run_single_channel(self, signal, artifacts, fs=None):
        restored = signal.copy()

        for n, (i, j) in enumerate(artifacts):
            min_a = 0
            max_b = signal.size

            if n > 0:
                min_a = artifacts[n - 1][1]
            if n + 1 < len(artifacts):
                max_b = artifacts[n + 1][0]

            size = j - i

            level = int(np.log2(size / self.n))

            if level < 1:
                continue

            # level = pywt.dwt_max_level(size, self.wavelet) - 1

            ref_size = max(self.n * 2**level, size)
            a = max(min_a, i - ref_size)
            b = min(max_b, j + ref_size)

            coeffs = pywt.wavedec(signal[a:b], self.wavelet,
                                  mode=self.mode, level=level)

            for cs in coeffs:
                k = int(np.round(np.log2(b - a) - np.log2(cs.size)))
                ik, jk = np.array([i - a, j - a]) // 2**k

                refs = [cs[:ik], cs[jk:]]
                if len(refs[0]) == 0 and len(refs[1]) == 0:
                    continue

                order = np.argsort(np.abs(cs[ik:jk]))
                inv_order = np.empty_like(order)
                inv_order[order] = np.arange(len(order))

                vals_ref = np.abs(np.concatenate(refs))
                ref_order = np.argsort(vals_ref)
                ref_sp = np.linspace(0, len(inv_order), len(ref_order))
                vals_norm = np.interp(inv_order, ref_sp, vals_ref[ref_order])

                r = vals_norm / np.abs(cs[ik:jk])

                cs[ik:jk] *= np.minimum(1, r)**self.alpha

            rec = pywt.waverec(coeffs, self.wavelet, mode=self.mode)
            restored[i:j] = rec[i - a:j - a]

        return restored


class MWFDenoiser:
    def __init__(self, matlab, delay=0):
        self.matlab = matlab
        self.delay = delay

    def run(self, signal, artifacts):
        mask = intervals_to_mask(artifacts, signal.shape[-1])
        restored = self.matlab.mwf_process(signal, mask, nargout=1).copy()
        restored[:, ~mask] = signal[:, ~mask]

        return restored


class OriginalSuBAR:
    def __init__(self, matlab, block_size=10, num_surrogates=1000):
        self.matlab = matlab
        self.block_size = block_size
        self.num_surrogates = num_surrogates

    def run(self, signal, artifacts, fs):
        block_len = int(self.block_size * fs)

        filtered = np.zeros_like(signal)
        for n in range(signal.shape[0]):
            filtered[n] = self.run_single_channel(signal[n], artifacts,
                                                  block_len)

        return filtered

    def run_single_channel(self, signal, artifacts, block_len):
        artifact_mask = intervals_to_mask(artifacts, signal.size)

        filtered = signal.copy()
        for i in range(0, signal.size, block_len):
            epoch = slice(i, i + block_len)

            # Run the denoising only if the block contains artifacts
            if artifact_mask[epoch].any():
                filtered[epoch] = self.run_block(signal[epoch])

        return filtered

    def run_block(self, signal):
        # Run the original Matlab code
        filtered = self.matlab.surrogateMODWTdespikingFiltering(
            signal, self.num_surrogates, 0)

        return filtered[0]
