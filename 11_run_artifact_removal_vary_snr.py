"""
Run artifact removal for various SNRs.
"""
import h5py
import transplant
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from methods import WaveletThresholding, OriginalSuBAR, WaveletQuantileNormalization
from tools import calculate_metrics, mask_to_intervals

datasets = ["data/03_denoise-net_emg_-1dB.h5", "data/03_denoise-net_eog_-1dB.h5"]

# %%

matlab = transplant.Matlab()
matlab.addpath(matlab.genpath("extras/surrogates"))

methods = {
    "WT-hard (sym5,L5)": WaveletThresholding("sym5", level=5, mode="hard"),
    "WT-soft (sym5,L5)": WaveletThresholding("sym5", level=5, mode="soft"),
    "SuBAR": OriginalSuBAR(matlab, block_size=2, num_surrogates=1000),
    "WQN (sym5)": WaveletQuantileNormalization("sym5", n=20),
}

snrs = np.arange(-20, 5 + 1, 1)

for method_name, method in methods.items():
    if Path(f"results/vary_snr_metrics_{method_name}.parquet").exists():
        continue

    _metrics = []
    for dataset_path in datasets:
        with h5py.File(dataset_path) as dataset:
            for snr in snrs:
                dataset_name = dataset.attrs["name"].split("(")[0].strip()
                desc = method_name + ": " + dataset_name + f" ({snr} dB)"

                for i, record in tqdm(
                    enumerate(dataset), total=len(dataset), desc=desc
                ):
                    reference = dataset[record]["eeg_reference"][:]
                    artifact = dataset[record]["eeg_signal"][:] - reference

                    # Rescale the artifact to the obtain the desired SNR
                    ref_var = reference[:, 256:].var()
                    art_var = artifact[:, 256:].var()
                    artifact *= np.sqrt((ref_var * 10 ** (-0.1 * snr)) / art_var)

                    signal = reference + artifact

                    artifact_mask = dataset[record]["artifacts"][:]
                    artifacts = mask_to_intervals(artifact_mask)
                    freq = dataset[record].attrs["freq"]
                    restored = method.run(signal, artifacts, freq)

                    # Calculate metrics
                    _m = calculate_metrics(
                        restored, signal, reference, artifact_mask, freq=freq
                    )

                    _metrics.append(
                        {
                            "dataset": dataset_name,
                            "snr": snr,
                            "method": method_name,
                            "record": record,
                            **_m,
                        }
                    )

        pd.DataFrame(_metrics).set_index(["record"]).to_parquet(
            f"results/vary_snr_metrics_{method_name}.parquet"
        )

# %%

metrics = pd.concat(
    [pd.read_parquet(p) for p in Path("results").glob("vary_snr_metrics_*.parquet")]
)

metrics.to_parquet("results/vary_snr_metrics.parquet", compression="zstd")
