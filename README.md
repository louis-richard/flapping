# Observations of Short-Period Ion-Scale Current Sheet Flapping
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![LASP](https://img.shields.io/badge/datasets-MMS_SDC-orange.svg)](https://lasp.colorado.edu/mms/sdc/) [![Twitter
](https://img.shields.io/twitter/follow/IRF_Space?style=social)](https://twitter.com/irf_space)

## Abstract

 Kink-like flapping motions of current sheets are commonly observed in the magnetotail. Such
  oscillations have periods of a few minutes down to a few seconds and they propagate toward the flanks of the plasma sheet. Here, we report a short-period (T≈25 s) flapping event of a thin current sheet observed by the Magnetospheric Multiscale (MMS) spacecraft in the dusk-side plasma sheet following a fast earthward plasma flow. We characterize the flapping structure using the multi-spacecraft spatiotemporal derivative and timing methods, and we find that the wave-like structure is propagating along the average current direction with a phase velocity comparable to the ion velocity. We show that the wavelength of the oscillating current sheet scales with its thickness as expected for a drift-kink mode. The decoupling of the ion motion from the electron motion suggests that the current sheet is thin. We observe that in such a thin current sheet, the ion motion becomes chaotic. We discuss the presence of the lower hybrid waves associated with gradients of density as a broadening process of the thin current sheet.

## Reproducing our results
- Instructions for reproduction are given within each section folder, in the associated README.md
 file.

## Requirements
- A [`requirements.txt`](./requirements.txt) file is available at the root of this repository, specifying the
 required
 packages for our analysis.

- Routines specific to this study [`ShortPeriodFlapping`](./ShortPeriodFlapping) is pip-installable: from the [`ShortPeriodFlapping`](./ShortPeriodFlapping) folder run `pip install .`


## Acknowledgement
We thank the entire MMS team and instrument PIs for data access and support. All of the data used
 in this paper are publicly available from the MMS Science Data Center https://lasp.colorado.edu
 /mms/sdc/. Data analysis was performed using the pyrfu analysis package available at https://github.com/louis-richard/irfu-python. This work is supported by the SNSA grant 139/18.
