"""
Calculate computational cost of artifact removal methods.
"""
import h5py
import timeit
import transplant
import pandas as pd
import matplotlib.pyplot as plt

from methods import (
    WaveletThresholding,
    OriginalSuBAR,
    WaveletQuantileNormalization,
    EMDICA,
    EMDCCA,
)

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

dataset_path = "data/01_physiobank.h5"

with h5py.File(dataset_path) as dataset:
    record = "eeg_1"

    freq = int(dataset[record].attrs["freq"])
    signal = dataset[record]["eeg_signal"][:, : 30 * freq]
    reference = dataset[record]["eeg_reference"][:, : 30 * freq]

artifacts = [(15 * freq, 30 * freq)]

# %%
# Calculate time for Python to Matlab communication,
# so that we can subtract it from the benchmark.

matlab_lag = max(
    timeit.repeat(
        "matlab.surrogateNOOP(signal)",
        globals={"signal": signal, "matlab": matlab},
        repeat=10,
        number=1,
    )
)
print(f"MATLAB noop lag: {matlab_lag}")

# %%

_metrics = []
for method_name, method in methods.items():
    context = {
        "method": method,
        "signal": signal,
        "artifacts": artifacts,
        "freq": freq,
        "reference": reference,
    }

    ex_time = min(
        timeit.repeat(
            "method.run(signal, artifacts, freq, reference)",
            globals=context,
            repeat=10,
            number=1,
        )
    )

    _metrics.append({"method": method_name, "time": ex_time, "repeat": 10})

metrics = pd.DataFrame(_metrics)
metrics.to_csv("results/comp_time_fixed.csv", index=False)
