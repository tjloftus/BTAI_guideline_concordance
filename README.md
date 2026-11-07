# Guideline-concordant management of blunt thoracic aortic injury

This repository contains code for the analysis of guideline-concordant management and in-hospital outcomes of blunt thoracic aortic injury (BTAI) in the American College of Surgeons National Trauma Data Bank (NTDB), 2017–2024, applying the 2011 Society for Vascular Surgery (SVS) clinical practice guidelines and the 2026 SVS focused update.

> Citation: Pending.
> This work will be presented at the Southern Surgical Association 138th Annual Meeting on December 8th, 2026, in Palm Beach, Florida.

## Data

NTDB participant use files are licensed by the American College of Surgeons and cannot be redistributed. This repository contains code only. Researchers can request the data from the ACS: https://www.facs.org/quality-programs/trauma/quality/national-trauma-data-bank/

## Contents

| File | Purpose |
|---|---|
| `src/process_aortic_injuries_2017_2024_step_1.py` | Identifies aortic injuries; assigns SVS grade; flags TEVAR and open repair; defines the outcome, transfusion volumes, and hemodynamic instability |
| `src/analyze_aortic_injuries_2017_2024.py` | Cohort exclusions, guideline concordance (2011 and 2026), Kaplan-Meier and Cox models, bootstrapped risk-adjusted survival curves, time-varying Cox models for immortal time bias, sensitivity analyses, temporal trends, descriptive comparisons |
| `src/stats_utils.py` | Fisher's exact and Kruskal-Wallis comparison tables |
| `src/config.py` | Input and output paths |

## Data preparation

The analysis runs on a patient-level file built in three stages:

1. **Patient-level vascular injury file** (`VASC_DF_PATH`): one row per patient with a vascular injury, built from the NTDB participant use files by the authors' shared NTDB processing pipeline (maintained in a separate repository). Fields used here include `AISPREDOT` (AIS predot codes), `dx` (ICD-10-CM codes), `procs` (ICD-10-PCS codes), `highest_vasc_inj_severity`, `time_to_first_vasc_proc` (minutes from arrival), `los_d` (length of stay, days), discharge and emergency department disposition, mechanism of injury, admission vital signs, and blood products within 4 hours.
2. **Step 1** (this repository): `process_aortic_injuries_2017_2024_step_1.py` writes `STEP_1_PATH`.
3. **Merge** (shared pipeline): adds `max_head_ais` and `max_solid_organ_ais` (highest head and solid organ AIS severity) and `iss_nonvasc` (Injury Severity Score recomputed without vascular AIS codes) by `inc_key`, producing `DATA_PATH`.

## Key definitions

| Element | Definition |
|---|---|
| Aortic injury | ICD-10-CM S25.0 (thoracic) or S35.0 (abdominal), or AIS 4202xx / 5202xx. Primary cohort: any thoracic component |
| SVS grade | Rupture codes (420210, 420212, 420216, 420218, 520208): grade IV. Laceration or NFS codes with highest vascular AIS ≥4: grade III; ≤3: grade II. Intimal tear codes (420202, 420204, 520202): grade I |
| TEVAR | ICD-10-PCS 02UW3[DJK], 02UW4[DJK], 02VW3[DJK], 02VW4[DJK] with an aortic injury |
| Open repair (excluded) | Open thoracic or abdominal aortic replacement, bypass, patch, primary repair, ligation, restriction, or dilation (see step 1) |
| Outcome | In-hospital death, withdrawal of life-sustaining therapy, or discharge to hospice |
| Hemodynamic instability | SBP <90 mmHg, shock index >1, or ≥2 units of red blood cells within 4 hours (300 mL per unit) |
| Severe TBI / solid organ injury | Highest head / solid organ AIS ≥4 |
| 2011 concordance | Grade III: TEVAR ≤24 h (or later with severe TBI). Grade IV: TEVAR ≤6 h (or later with severe TBI) |
| 2026 concordance | Grade III: TEVAR ≤24 h if unstable; >24 h if stable; any time with severe TBI or solid organ injury. Grade IV: TEVAR ≤6 h |

## Setup

```
pip install -r requirements.txt
```

Edit the paths in `src/config.py`, or set the environment variables named there (`BTAI_VASC_DF_PATH`, `BTAI_STEP_1_PATH`, `BTAI_DATA_PATH`, `BTAI_RESULTS_ROOT`).

## Running

From `src/`:

```
python process_aortic_injuries_2017_2024_step_1.py
python analyze_aortic_injuries_2017_2024.py
```

The analysis runs four configurations in parallel: {2011, 2026} SVS guidelines × {thoracic only (primary), total cohort including isolated abdominal aortic injury codes (sensitivity)}. To run one configuration:

```
python analyze_aortic_injuries_2017_2024.py --single --svs2026 0 --thoracic 1
```

The 2,000-resample bootstrap is the slowest step (about 20 minutes for all four configurations in parallel on a workstation). All stochastic steps use seed 42; the seed and package versions are written to the `repro` sheet of `adjusted_hazard_ratios_<date>.xlsx`.

## Outputs

Written to `<RESULTS_ROOT>/{2011,2026} guidelines/{thoracic_only,total_cohort}/analysis_performed_<date>/`:

| Output | Content |
|---|---|
| `cohort_exclusions_<date>.txt` | Exclusions at each step |
| `dataset_description_<date>.txt` | Counts by grade and management |
| `cat_var_`, `cont_var_`, `grade_3_*`, `grade_4_*` `<date>.xlsx` | Characteristics by guideline concordance (grade III–IV, grade III, grade IV) |
| `tevar_vs_no_tevar_*`, `tevar_early_mortality_*` `<date>.xlsx` | TEVAR vs. no TEVAR; death within 7 days of TEVAR vs. survival |
| `missing_var_raw_<date>.xlsx` | Missingness by variable |
| `concordance_trend_by_year_<date>.xlsx` | Logistic regression of concordance on admission year |
| `survival for BAI <year> by mgmt <date>.jpg` | Kaplan-Meier curves and log-rank tests |
| `adj survival BAI <year> by mgmt <date>.jpg` | Risk-adjusted survival with bootstrapped 95% confidence bands, and adjusted hazard ratios |
| `adjusted_hazard_ratios_<date>.xlsx` | Cox models: primary, follow-up to day 90, days 0–7, after day 7, non-vascular ISS as B-splines; ISS functional-form test |
| `immortal_time_bias_hr_<date>.xlsx` | Time-varying Cox models with the same specifications |
| `tevar_timing_g3_<date>.xlsx` | TEVAR within vs. after 24 hours among stable grade III injuries (2011 runs only) |
| `proportional_hazards_check_<date>.txt` | Schoenfeld residual tests |
| `model_fit_log_<date>.txt` | Penalizer used for every model fit |
| `quality_check_by_year_<date>.xlsx` | Variable means by admission year |

## License

BSD 3. See `LICENSE`.
