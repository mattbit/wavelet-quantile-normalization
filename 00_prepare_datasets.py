"""Preprocess and prepare the datasets used for testing."""
import wfdb
import h5py
import scipy.io
import numpy as np
from pathlib import Path
import scipy.signal as ss
import scipy.ndimage as ndi
from tools import calculate_snr
from tools import mask_to_intervals, intervals_to_mask

from tools import filter_bandpass


FILTER_LOW = .1
FILTER_HIGH = 100.

###############################################################################
# %% PHYSIOBANK MOTION ARTIFACTS DATASET

excluded = {'eeg_21'}
partially_excluded = {'eeg_10': slice(0, 75000), 'eeg_20': slice(25000, None)}

dataset_path = Path('data/physiobank-motion-artifacts')
tot_art = 0
tot_tot = 0
corr = []
with h5py.File('data/01_physiobank.h5', 'w') as f:
    f.attrs['name'] = 'Physiobank Motion Artifacts'
    f.attrs['author'] = 'Kevin Sweeney et al.'

    for path in sorted(dataset_path.glob('*.hea'), key=lambda p: int(p.stem[4:])):
        record_name = path.stem

        if record_name in excluded:
            continue

        select = partially_excluded.get(record_name, slice(None, None))

        record_path = str(path.with_name(record_name))
        record = wfdb.rdrecord(record_path)
        annots = wfdb.rdann(record_path, 'trigger')

        fs = record.fs / 8
        eeg = record.p_signal[:, :2]

        reference = filter_bandpass(
            eeg[:, 0], FILTER_LOW, FILTER_HIGH, record.fs, 2)[::8].reshape(1, -1)
        signal = filter_bandpass(
            eeg[:, 1], FILTER_LOW, FILTER_HIGH, record.fs, 2)[::8].reshape(1, -1)

        reference = reference[:, select]
        signal = signal[:, select]

        # Add artifacts labels
        mask = record.p_signal[::8, 9] > .5
        mask = mask[select]

        dist = np.sqrt(np.abs((signal - reference)[0]**2))
        dist = ndi.gaussian_filter1d(dist, 2 * fs)
        high = np.quantile(dist, 0.80)

        trig = dist > high
        trig = ndi.binary_closing(trig, np.ones(int(5 * fs)))
        trig = ndi.binary_opening(trig, np.ones(int(5 * fs)))
        trig = ndi.binary_dilation(trig, np.ones(int(3 * fs)))

        trig = trig & ~mask
        tot_art += trig.sum()
        tot_tot += trig.size
        corr.append(np.corrcoef(reference[0], signal[0])[1, 0])

        intervals = mask_to_intervals(trig)[:4]
        artifacts = intervals_to_mask(intervals, trig.size)

        r = f.create_group(record_name)
        r['eeg_signal'] = signal
        r['eeg_reference'] = reference
        r['artifacts'] = artifacts
        r.attrs['freq'] = fs
        r.attrs['filtered'] = f'BANDPASS {FILTER_LOW}-{FILTER_HIGH} Hz'


###############################################################################
# %% EOG DATASET

excluded = {36}

data_signals = scipy.io.loadmat('data/eog-data/Pure_Data.mat')
data_artifact = scipy.io.loadmat('data/eog-data/Contaminated_Data.mat')

fs = 200

with h5py.File('data/02_semisimulated_eog.h5', 'w') as f:
    f.attrs['name'] = 'Semi-simulated EOG'
    f.attrs['author'] = 'Manousos A. Klados and Panagiotis D. Bamidis'

    for n in range(1, 55):
        if n in excluded:
            continue
        reference = data_signals[f'sim{n}_resampled']
        artifact = data_artifact[f'sim{n}_con'] - reference

        keep_len = reference.shape[1] // 3
        start = keep_len
        end = 2 * keep_len

        window = ss.general_gaussian(keep_len, 6, keep_len // 2)

        artifact[:, :start] = 0
        artifact[:, start:end] *= window
        artifact[:, end:] = 0

        signal = reference + artifact
        artifacts = np.zeros(signal.shape[1], dtype=bool)
        artifacts[start:end] = True

        noise = signal - reference
        snr = 10 * np.log10(
            reference[:, start:end].var() / noise[:, start:end].var())
        if snr >= 10:
            print(f'Skipping record {n}: SNR too high ({snr:.2f} dB).')
            continue

        record = f.create_group(f'sim{n}')
        record['eeg_signal'] = signal
        record['eeg_reference'] = reference
        record['artifacts'] = artifacts
        record.attrs['freq'] = fs
        record.attrs['filtered'] = ''


###############################################################################
# %% DENOISE-NET DATASET

fs = 256

eeg = np.load('data/eeg-denoise-net/EEG_all_epochs.npy')
eog = np.load('data/eeg-denoise-net/EOG_all_epochs.npy')
emg = np.load('data/eeg-denoise-net/EMG_all_epochs.npy')


artifacts = np.zeros(2 * fs, dtype=bool)
artifacts[fs:] = True

for snr in [-0.5, -1, -5, -10, -20]:
    with h5py.File(f'data/03_denoise-net_eog_{snr}dB.h5', 'w') as f:
        f.attrs['name'] = f'Denoise-Net EOG ({snr} dB)'
        f.attrs['author'] = ''

        for n in range(3400):
            start = fs + fs // 10
            end = 2 * fs - fs // 10
            signal = eeg[n].copy()
            signal /= signal[artifacts].std()

            reference = signal.copy()
            noise = np.zeros_like(reference)
            noise[start:end] = eog[n][start:end]
            noise /= noise[artifacts].std()
            noise *= np.sqrt(10**(-0.1 * snr))

            signal += noise

            _snr = calculate_snr(
                reference[artifacts], noise[artifacts])
            assert abs(_snr - snr) < 1e-3

            record = f.create_group(f'eog_{n}')
            record['eeg_signal'] = [signal]
            record['eeg_reference'] = [reference]
            record['artifacts'] = artifacts
            record.attrs['freq'] = fs
            record.attrs['filtered'] = ''

    with h5py.File(f'data/03_denoise-net_emg_{snr}dB.h5', 'w') as f:
        f.attrs['name'] = f'Denoise-Net EMG ({snr} dB)'
        f.attrs['author'] = ''

        for n in range(3400):
            start = fs + fs // 10
            end = 2 * fs - fs // 10
            signal = eeg[n].copy()
            signal /= signal[artifacts].std()

            reference = signal.copy()
            noise = np.zeros_like(reference)
            noise[start:end] = emg[n][start:end]
            noise /= noise[artifacts].std()
            noise *= np.sqrt(10**(-0.1 * snr))

            signal += noise

            _snr = calculate_snr(
                reference[artifacts], noise[artifacts])
            assert abs(_snr - snr) < 1e-3

            record = f.create_group(f'emg_{n}')
            record['eeg_signal'] = [signal]
            record['eeg_reference'] = [reference]
            record['artifacts'] = artifacts
            record.attrs['freq'] = fs
            record.attrs['filtered'] = ''

    with h5py.File(f'data/03_denoise-net_eog+emg_{snr}dB.h5', 'w') as f:
        f.attrs['name'] = f'Denoise-Net EOG+EMG ({snr} dB)'
        f.attrs['author'] = ''

        for n in range(3400):
            start = fs + fs // 10
            end = 2 * fs - fs // 10
            signal = eeg[n].copy()
            signal /= signal[artifacts].std()

            reference = signal.copy()
            noise = np.zeros_like(reference)
            noise[start:end] = emg[n][start:end] / emg[n][start:end].std() + \
                eog[n][start:end] / eog[n][start:end].std()
            noise /= noise[artifacts].std()
            noise *= np.sqrt(10**(-0.1 * snr))

            signal += noise

            _snr = calculate_snr(
                reference[artifacts], noise[artifacts])
            assert abs(_snr - snr) < 1e-3

            record = f.create_group(f'eog+emg_{n}')
            record['eeg_signal'] = [signal]
            record['eeg_reference'] = [reference]
            record['artifacts'] = artifacts
            record.attrs['freq'] = fs
            record.attrs['filtered'] = ''

# %%
