"""
Run artifact removal methods and collect metrics (SNR, R, …).
"""
import h5py
import transplant
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from methods import (
    WaveletThresholding,
    OriginalSuBAR,
    WaveletQuantileNormalization,
    EMDICA,
    EMDCCA,
)
from tools import calculate_metrics, mask_to_intervals

dataset_paths = [
    "data/01_physiobank.h5",
    "data/02_semisimulated_eog.h5",
    "data/03_denoise-net_emg_-20dB.h5",
    "data/03_denoise-net_eog_-20dB.h5",
    "data/03_denoise-net_eog+emg_-20dB.h5",
]

# %%

matlab = transplant.Matlab()
matlab.addpath(matlab.genpath("extras/surrogates"))

methods = {
    "WT-hard (sym5,L5)": WaveletThresholding("sym5", level=5, mode="hard"),
    "WT-soft (sym5,L5)": WaveletThresholding("sym5", level=5, mode="soft"),
    "SuBAR (bs=30)": OriginalSuBAR(matlab, block_size=30, num_surrogates=1000),
    "WQN (sym5)": WaveletQuantileNormalization("sym5", n=20),
    "EMD-ICA": EMDICA(),
    "EMD-CCA": EMDCCA(),
}

# %%

for method_name, method in methods.items():
    for path in dataset_paths:
        with h5py.File(path, "r") as dataset:
            dataset_name = dataset.attrs["name"]
            desc = method_name + ": " + dataset.attrs["name"]

            with h5py.File(
                f"results/restored__{dataset_name}__{method_name}.h5", "a"
            ) as output:
                for i, record in tqdm(
                    enumerate(dataset), total=len(dataset), desc=desc
                ):
                    if record in output:
                        continue

                    # Run the restoration algorithm
                    signal = dataset[record]["eeg_signal"][:]
                    reference = dataset[record]["eeg_reference"][:]
                    artifact_mask = dataset[record]["artifacts"][:]
                    artifacts = mask_to_intervals(artifact_mask)
                    freq = dataset[record].attrs["freq"]

                    restored = method.run(signal, artifacts, freq, reference)
                    output[record] = restored

# %%

for method_name, method in methods.items():
    _metrics = []
    for path in dataset_paths:
        with h5py.File(path, "r") as dataset:
            dataset_name = dataset.attrs["name"]
            desc = method_name + ": " + dataset.attrs["name"]

            with h5py.File(
                f"results/restored__{dataset_name}__{method_name}.h5", "a"
            ) as output:
                for i, record in tqdm(
                    enumerate(dataset), total=len(dataset), desc=desc
                ):
                    # Run the restoration algorithm
                    signal = dataset[record]["eeg_signal"][:]
                    reference = dataset[record]["eeg_reference"][:]
                    artifact_mask = dataset[record]["artifacts"][:]
                    freq = dataset[record].attrs["freq"]
                    restored = output[record][:]

                    # Calculate metrics
                    _m = calculate_metrics(
                        restored, signal, reference, artifact_mask, freq=freq
                    )

                    _metrics.append(
                        {
                            "dataset": dataset_name,
                            "method": method_name,
                            "record": record,
                            **_m,
                        }
                    )

        pd.DataFrame(_metrics).set_index(["record"]).to_parquet(
            f"results/metrics_{method_name}.parquet", compression="zstd"
        )

# %%

metrics = pd.concat(
    [pd.read_parquet(p) for p in Path("results").glob("metrics_*.parquet")]
)

metrics.to_parquet("results/metrics.parquet")

# %%

datasets_ = [
    "Physiobank Motion Artifacts",
    "Semi-simulated EOG",
    "Denoise-Net EMG (-20 dB)",
    "Denoise-Net EOG (-20 dB)",
    "Denoise-Net EOG+EMG (-20 dB)",
]

methods_ = [
    "WQN (sym5)",
    "WT-hard (sym5,L5)",
    "WT-soft (sym5,L5)",
    "SuBAR (bs=30)",
    "EMD-ICA",
    "EMD-CCA",
]

agg = (
    metrics.groupby(["dataset", "method"])
    .mean()
    .loc[:, ("ΔSNR", "NMSE", "ΔR", "ΔCoh_normalized")]
)
agg = agg.reindex(datasets_, level=0).reindex(methods_, level=1)
print(agg.round(2))
