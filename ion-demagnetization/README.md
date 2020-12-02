# Ion demagnetization

## Organisation
- [`ion-demagnetization.py`](./ion-demagnetization.py) contains the code to load data, compute the
 adiabacity parameter and the current density and reproduce
 the Figure 2. The data
 rates, levels
 and figure parameters are stored in the [`./config/ion-demagnetization.yml`](./config/ion-demagnetization.yml) file.   


## Datasets used
- The magnetic field measured by the Flux Gate Magnetometer (FGM) ([Russell et al. 2014](https://link.springer.com/article/10.1007/s11214-014-0057-3))
 
|             |   Data rate   | level |
|-------------|:-------------:|------:|
| $`B`$ (GSE) | srvy          | l2    |

> **_NOTE:_**  An offset in $`B_z`$ (GSE) is removed. The offset are computed between ['2019-09-14T09:17:24.000', '2019-09-14T09:18:22.000']. The offsets are stored in [bz_offsets.csv](../data/bz_offsets.csv).

- The ion and electron moments are computed using the partial moments of the velocity
 distribution functions measured by the Fast Plasma Investigation (FPI) ([Pollock et al. 2016](https://link.springer.com/article/10.1007/s11214-016-0245-4)) removing
 the background low-energy noise for ions and photoelectrons.

|                |   Data rate   | level | Split Energy Level |
|:---------------|:-------------:|:------|-------------------:|
| $`V_i`$ (GSE)  | fast          | l2    |        19          |
| $`n_i`$        | fast          | l2    |        19          |
| $`V_e`$ (GSE)  | fast          | l2    |         7          |
| $`n_e`$        | fast          | l2    |         7          |

> **_NOTE:_** The spintone is removed from the bulk velocity

## Reproducibility
```bash
python3.8 ion-demagnetization.py -v --config ./config/ion-demagnetization.yml
```

[![Figure 2](../figures/figure_2.png)](../figures/figure_2.png)