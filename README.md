# Missing Data and Imputation in Craniofacial Measurements

The missingness in this dataset is notscattered noise. **237 of the 238 missing cells (99.6%) are explained by 15anatomical landmarks that were never digitised in particular subjects.** Every decision downstream follows from that.

\---

## 1\. Requirements

* Python **3.10+**
* Packages (see `requirements.txt`):

```
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
openpyxl>=3.1
```

Install:

```bash
python -m venv .venv \\\&\\\& source .venv/bin/activate   # Windows: .venv\\\\Scripts\\\\activate
pip install -r requirements.txt
```

No internet access, database, or GPU is required. A full run takes about
**12–15 minutes** on a laptop, almost all of it in the validation step
(missForest is the slow component).

\---

## 2\. Input files

|File|Description|
|-|-|
|`data/raw/Craniofacial\\\_Data.xlsx`|The supplied workbook, sheet `control`, unmodified|

Expected columns: `Patient ID`, `gender`, `age`, then 152 measurement columns
named after the landmarks they connect (`ANS-PNS`, `N-S-BA`, `ZMs-CP`, …).

The `age` column arrives as free text in four different formats
(`8.3m`, `1.75y`, `4y 11m`, `17m 13d`) and is parsed to decimal years by
`utils.parse\\\_age\\\_to\\\_years`. This is the only edit made to the source data.

\---

## 3\. How to reproduce

```bash
bash run\\\_all.sh
```

or step by step, from the repository root:

```bash
python src/01\\\_diagnostics.py    # Task 1 — missingness diagnosis     
python src/02\\\_impute.py         # Task 2 — fit and write imputations  
python src/03\\\_validate.py       # Task 3 — masking validation         ```

`03\\\_validate.py` caches every replicate under `outputs/tables/\\\_cache/`, so it can
be interrupted and resumed: re-running it picks up where it stopped.

Scripts must be run \*\*in order\*\*: `03` reads the completed dataset written by
`02`. Every stochastic step is seeded from `utils.RANDOM\\\_SEED = 20250809`, so
reruns reproduce the numbers in `WRITEUP.md` exactly.

\\---

## 4\\. Repository layout

```

.
├── README.md                     you are here
├── WRITEUP.md                    the report (findings + recommendation)
├── WRITEUP.docx                  same report, Word format
├── WRITEUP_1page.md / .docx      one-page version (the submission copy)
├── requirements.txt
├── run_all.sh
├── data/
│   ├── raw/Craniofacial_Data.xlsx
│   └── processed/craniofacial_clean.csv        parsed age, tidy column order
├── src/
│   ├── utils.py                  paths, age parsing, variable taxonomy
│   ├── imputers.py               all imputation engines (shared by 02 and 03)
│   ├── 01_diagnostics.py         Task 1
│   ├── 02_impute.py              Task 2
│   └── 03_validate.py            Task 3
└── outputs/
    ├── craniofacial_imputed.csv               ← primary deliverable
    ├── craniofacial_imputed_long_m20.csv      all 20 imputations, long format
    ├── imputation_uncertainty.csv             per-cell SD, 95% interval, rule used
    ├── figures/                               fig01 … fig13
    └── tables/                                every number quoted in the write-up


```

\\---

## 5\\. Output files

### Primary

|File|Contents|
|-|-|
|`outputs/craniofacial\\\_imputed.csv`|24 × 152 completed dataset. Observed values are byte-identical to the source; only missing cells are filled. Includes `age\\\_years` and `sex\\\_M`.|
|`outputs/craniofacial\\\_imputed\\\_long\\\_m20.csv`|The 20 individual imputations stacked, with an `imputation` column, for anyone who wants to pool with Rubin's rules rather than use the point estimate.|
|`outputs/imputation\\\_uncertainty.csv`|One row per imputed cell: value, between-imputation SD, 95% interval, SD as a fraction of that variable's observed SD, which rule produced it, and whether it is flagged for review.|

### Diagnostics (Task 1)

`missing\\\_by\\\_variable.csv`, `missing\\\_by\\\_subject.csv`, `missing\\\_patterns.csv`,
`landmark\\\_missingness.csv`, `unexplained\\\_cells.csv`,
`bilateral\\\_comissingness.csv`, `mechanism\\\_tests.csv`, `variable\\\_metadata.csv`

### Validation (Task 3)

`validation\\\_method\\\_comparison.csv`, `validation\\\_matched\\\_replicates.csv`,
`validation\\\_by\\\_variable.csv`,
`validation\\\_by\\\_type.csv`, `validation\\\_by\\\_region.csv`,
`validation\\\_coverage.csv`, `validation\\\_cells\\\_\\\*.csv`,
`validation\\\_corr\\\_shift.csv`, `validation\\\_asymmetry\\\_index.csv`

### Figures

|Figure|Shows|
|-|-|
|`fig01`|missingness per variable, coloured by anatomical region|
|`fig02`|missingness per subject|
|`fig03`|missingness matrix — the block structure at a glance|
|`fig04`|Jaccard co-missingness heatmap, hierarchically ordered|
|`fig05`|landmark-level missingness (the actual mechanism)|
|`fig06`|missingness vs acquisition order and vs age|
|`fig07`|block indicators vs craniofacial size (MCAR check)|
|`fig08`|method comparison across all three masking designs|
|`fig09`|error by measurement type and anatomical region|
|`fig10`|imputed vs held-out true values|
|`fig11`|observed vs imputed distributions|
|`fig12`|correlation and bilateral-asymmetry preservation|
|`fig13`|95% predictive-interval coverage|

\\---

## 6\\. Method in one paragraph

Missingness is diagnosed at the \*\*landmark\*\* level rather than the variable
level, which reduces 238 scattered missing cells to 15 undigitised landmarks
and one isolated stray value. Imputation is a two-stage hybrid: a
\*\*bilateral-symmetry regression\*\* wherever the contralateral homologue survives,
then \*\*multiple imputation by regression with predictive mean matching\*\*
(m = 20, PMM donor pool 3) using a `quickpred`-style restricted predictor set,
each target is regressed on the five most strongly (Spearman-)associated
variables that are observed in \*every\* subject needing imputation, plus age and
sex. Linear distances are modelled on the log scale, angles on the raw scale.
Validation uses three masking designs (MCAR cells, block hold-out,
burden-matched blocks) and reports normalised RMSE, bias, correlation
preservation and interval coverage \*\*per variable, per measurement type and per
anatomical region\*\*. Full reasoning, results and the production recommendation
are in `WRITEUP.md`.

\\---

## 7\\. Known limitations

\* n = 24 with p = 152: every regression is small-sample, and the validation
metrics themselves carry appreciable Monte-Carlo error.
\* Whether a landmark was undigitised for a \*technical\* reason (field of view,
protocol change) or an \*anatomical\* one (a structure not yet ossified) cannot
be determined from the data supplied. This distinction decides between MAR and
MNAR for one subject and is flagged, not resolved, see `WRITEUP.md` §1.
\* Angular measurements are imputed but should not be trusted; see the flag
column in `imputation\\\_uncertainty.csv` and the recommendation in `WRITEUP.md` §4.
\* missForest is evaluated on 3 replicates per design against 20+ for the other
methods, because a single fit costs \\\~95 s. `validation\\\_matched\\\_replicates.csv`
re-scores every method on the identical replicates for a fair comparison, and
`WRITEUP.md` §3 uses that table rather than the headline one.
\* 95% predictive intervals achieve \\\~80% empirical coverage. This is reported,
not corrected.


