[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4783450.svg)](https://doi.org/10.5281/zenodo.4783450)

# Wavelet quantile normalization for EEG artifact removal

This repository contains the implementation of the WQN (wavelet quantile normalization) method presented in:
> [M. Dora and D. Holcman, "Adaptive single-channel EEG artifact removal for real-time clinical monitoring," IEEE Transactions on Neural Systems and Rehabilitation Engineering (2022), doi: 10.1109/TNSRE.2022.3147072](https://ieeexplore.ieee.org/abstract/document/9694664).


## Contents

The code allow to reproduce the benchmarks reported in the article:
- `00_prepare_datasets.py` contains the code to pre-process the datasets and format them for the benchmarking script;
- `10_run_artifact_removal.py` contains the main code which applies various artifact removal algorithms and computes the metrics (see Table I in the article);
- `11_run_artifact_removal_vary_snr.py` computes the benchmarks for varying SNR (Figure 3 in the article)
- `20_computation_cost.py` was used to compute the performances of the algorithms (Table II in the article).
- `methods.py` contains the implementations of the 
various algorithms, including WQN.

The datasets are publicly available as referenced in the
article:
- [Physiobank motion artifacts](https://physionet.org/content/motion-artifact/1.0.0/)
- [EEG Denoise-Net](https://github.com/ncclabsustech/EEGdenoiseNet)
- [Semi-simulated EOG](https://data.mendeley.com/datasets/wb6yvr725d/4)

Dependencies are managed with [poetry](https://python-poetry.org/) (see `pyproject.toml`).

If you have troubles running the code, you have doubts or need any help, don't hesitate to reach out at matteo.dora@ieee.org!


## Authors

The code was written by Matteo Dora (matteo.dora@ieee.org) from the _applied mathematics and computational biology group_ at École Normale Supérieure in Paris, France.


## License

Please note that the WQN technique is protected by the patent “Computer-implemented method for assisting a general anesthesia of a subject” (File No. EP21306053).
While you are free to experiment with it for purely experimental use (e.g. to test its effectiveness and verify the results reported in the published article), any other purpose (for example, part of industrial activity) may constitute patent infringement.

In general, we are glad to allow its usage for research purposes, if you plan to use the algorithm for your research
please contact David Holcman (david.holcman@ens.psl.eu) to get an exemption.
