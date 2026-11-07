# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 07:03:19 2024

@author: tloftus
"""

import contextlib
import io
import os as _os
import shutil as _shutil
import sys
import time as _time
import warnings
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as _mticker
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, CoxTimeVaryingFitter, KaplanMeierFitter
from lifelines.fitters.coxph_fitter import SemiParametricPHFitter as _SemiParametricPHFitter
from lifelines.statistics import logrank_test
from matplotlib.offsetbox import AnchoredOffsetbox, VPacker, HPacker, TextArea
from patsy import dmatrix, build_design_matrices
from scipy.stats import chi2, norm as _norm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from config import DATA_PATH, RESULTS_ROOT, LOCAL_CACHE
from stats_utils import raw_kruskal_wallace, raw_fisher_crosstab


def read_csv_robust(path, retries=3, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except (FileNotFoundError, OSError) as e:
        print(f"[read_csv_robust] direct read of {path} failed ({e}); copying to {LOCAL_CACHE}")
    _os.makedirs(LOCAL_CACHE, exist_ok=True)
    parent = _os.path.basename(_os.path.dirname(path))
    local = _os.path.join(LOCAL_CACHE, f"{parent}_{_os.path.basename(path)}")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            _shutil.copyfile(path, local)
            return pd.read_csv(local, **kwargs)
        except (FileNotFoundError, OSError) as e:
            last_err = e
            print(f"  [read_csv_robust] attempt {attempt}/{retries} failed: {e}")
            _time.sleep(10)
    raise last_err

# single seed used for every stochastic step to facilitate reproducibility
RANDOM_SEED = 42

import lifelines as _lifelines, scipy as _scipy, sklearn as _sklearn, patsy as _patsy
REPRO_STAMP = (
    f"seed={RANDOM_SEED} | "
    f"numpy={np.__version__} pandas={pd.__version__} "
    f"lifelines={_lifelines.__version__} scipy={_scipy.__version__} "
    
    f"sklearn={_sklearn.__version__} patsy={_patsy.__version__}"
)
print(f"[repro] {REPRO_STAMP}")


def get_export_path_string(base_path_str):
    today = date.today().strftime("%m-%d-%Y")
    folder_name = f"analysis_performed_{today}"
    full_path_obj = Path(base_path_str) / folder_name
    # create today's results path if it doesn't exist
    full_path_obj.mkdir(parents=True, exist_ok=True)
    return str(full_path_obj)

def percent_missing_series(df, columns):
    missing_percent = {}
    for column in columns:
        if column in df.columns:  
            num = df[column].isnull().sum()
            den = len(df[column])
            missing_percent[column] = 100 * (num / den)
        else:
            missing_percent[column] = None  
    return missing_percent


# shared helpers 
def assign_concordance(df):
    """Single source of truth for SVS guideline concordance, used by the
    descriptive tables, Kaplan-Meier/Cox models, and time-varying (immortal time)
    models. Thresholds:
      severe TBI / solid organ injury (SOI): max head / solid organ AIS >= 4
      hemodynamic instability: SBP < 90, shock index > 1, or >= 2 u PRBC within 4 h
    2011 SVS: grade 3 TEVAR <= 24 h; grade 4 TEVAR <= 6 h; delayed TEVAR concordant
              when severe TBI is present (repair deferred for serious concomitant injury)
    2026 SVS: grade 3 + severe TBI/SOI: TEVAR at any time
              grade 3 + unstable: TEVAR <= 24 h
              grade 3 + stable (no severe TBI/SOI): TEVAR > 24 h
              grade 4: TEVAR <= 6 h (emergent)"""
    df = df.copy()
    g3 = df['svs_grade_final'] == 3
    g4 = df['svs_grade_final'] == 4
    tevar = df['traumatic_endo_repair'] == 1
    t = df['time_to_first_vasc_proc']
    severe_tbi = df['max_head_ais'].fillna(0) >= 4
    severe_soi = df['max_solid_organ_ais'].fillna(0) >= 4
    df['tbi_or_soi'] = (severe_tbi | severe_soi).astype(int)
    df['unstable_or_2_plus_rbc'] = ((df['shock_per_sbp_or_shock_index'] == 1) |
                                    (df['units_rbc_4h'] >= 2)).astype(int)
    unstable = df['unstable_or_2_plus_rbc'] == 1

    # 2011 component flags
    df['gr_3_tevar_within_24h'] = (g3 & tevar & (t <= 1440)).astype(int)
    df['gr_3_tevar_after_24h'] = (g3 & tevar & (t > 1440)).astype(int)
    df['gr_3_severe_tbi_tevar_after_24h'] = (g3 & tevar & severe_tbi & (t > 1440)).astype(int)
    df['gr_3_no_tevar'] = (g3 & ~tevar).astype(int)
    df['gr_4_tevar_within_6h'] = (g4 & tevar & (t <= 360)).astype(int)
    df['gr_4_tevar_after_6h'] = (g4 & tevar & (t > 360)).astype(int)
    df['gr_4_severe_tbi_tevar_after_6h'] = (g4 & tevar & severe_tbi & (t > 360)).astype(int)
    df['gr_4_no_tevar'] = (g4 & ~tevar).astype(int)
    df['guideline_concordant_care_2011'] = (
        (df['gr_3_tevar_within_24h'] == 1) | (df['gr_3_severe_tbi_tevar_after_24h'] == 1) |
        (df['gr_4_tevar_within_6h'] == 1) | (df['gr_4_severe_tbi_tevar_after_6h'] == 1)).astype(int)

    # 2026 component flags (stability now required for delayed grade 3 repair)
    df['gr_3_tevar_plus_TBI_or_SOI_2026'] = (g3 & tevar & (df['tbi_or_soi'] == 1)).astype(int)
    df['gr_3_tevar_within_24h_unstable_2026'] = (g3 & tevar & unstable & (t <= 1440)).astype(int)
    df['gr_3_tevar_stable_after_24h_2026'] = (g3 & tevar & ~unstable & (t > 1440)).astype(int)
    df['guideline_concordant_care_2026'] = (
        (df['gr_3_tevar_plus_TBI_or_SOI_2026'] == 1) | (df['gr_3_tevar_within_24h_unstable_2026'] == 1) |
        (df['gr_3_tevar_stable_after_24h_2026'] == 1) | (df['gr_4_tevar_within_6h'] == 1)).astype(int)
    return df


def add_bh_q(tbl):
    """Append a Benjamini-Hochberg q-value column (within one table)."""
    tbl = tbl.copy()
    p = pd.to_numeric(tbl['p value'], errors='coerce')
    q = pd.Series(np.nan, index=tbl.index)
    ok = p.notna()
    if ok.any():
        q[ok] = multipletests(p[ok], method='fdr_bh')[1]
    tbl['q value (BH)'] = q
    return tbl


# primary models are unpenalized; a small L2 penalty is used only when an unpenalized fit does not converge
COX_PENALIZER = 0.0
ITB_PENALIZER_LADDER = [0.0, 1e-4, 1e-3, 1e-2]   # smallest penalizer that converges is used
COX_PENALIZER_LADDER = [COX_PENALIZER, 1e-4, 1e-3, 1e-2]
_BAD_FIT_MSGS = ('norm(delta) is still high', 'failed to converge', 'suspiciously close to 0')
FIT_LOG = []


class _BootstrapCoxPH(_SemiParametricPHFitter):
    """Cox fitter for bootstrap resamples: lifelines' underlying Breslow-baseline fitter (what
    CoxPHFitter delegates to), minus the pandas describe() of the training data that lifelines
    keeps only as plotting defaults. Coefficients, baseline hazard and predictions are unchanged."""
    def _compute_central_values_of_raw_training_data(self, df, strata=None, name="baseline"):
        return pd.DataFrame(index=[name])


def fit_with_ladder(make_fitter, data, ladder, label='', **fit_kw):
    """Fit with the smallest penalizer in `ladder` that converges cleanly.
    Returns (fitted model, penalizer used); raises the last error if none converge."""
    last = None
    for pen in ladder:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                m = make_fitter(pen)
                m.fit(data, **fit_kw)
            except Exception as e:
                last = e
                FIT_LOG.append(f"{label} | penalizer {pen}: {type(e).__name__} - {str(e)[:200]}")
                continue
        bad = [str(w.message).strip() for w in caught
               if any(k in str(w.message) for k in _BAD_FIT_MSGS)]
        if bad or not np.all(np.isfinite(m.params_)):
            msg = bad[0][:200] if bad else 'non-finite coefficients'
            last = RuntimeError(msg)
            FIT_LOG.append(f"{label} | penalizer {pen}: rejected - {msg}")
            continue
        FIT_LOG.append(f"{label} | penalizer {pen}: converged")
        return m, pen
    raise last
ISS_SPLINE_LABEL = 'non-vascular ISS as B-splines (4 df)'   
EARLY_WINDOW_DAYS = 7                            

# early vs. delayed TEVAR among stable grade III injury sensitivity analysis
EVD_CUTOFF_DAYS = 1.0          # 24h, the 2011 vs. 2026 boundary for grade III
EVD_STATES = ['repaired_early', 'repaired_delayed']   # reference: not yet repaired or never repaired


def build_early_vs_delayed_tv(d, duration_col='los_d', landmark=True):
    """Start-stop data with time-varying repair state. One or two rows per patient.
    landmark=True: patients alive and in hospital at 24 h; clock starts at 24 h."""
    d = d.copy()
    d['patient_id'] = np.arange(len(d))
    t_end = d[duration_col].astype(float)
    tevar = (d['traumatic_endo_repair'] == 1) & d['time_to_first_vasc_proc'].notna()
    t_rep = (d['time_to_first_vasc_proc'] / 1440.0).where(tevar)
    t_rep = t_rep.clip(lower=0.001)
    event = d['dc_dispo_death_or_hospice'].astype(int)
    t0 = EVD_CUTOFF_DAYS if landmark else 0.0
    keep = t_end > t0
    # a repair recorded at or after the end of stay cannot change the exposure
    repaired = tevar & (t_rep < t_end)
    early = repaired & (t_rep <= EVD_CUTOFF_DAYS)
    late = repaired & (t_rep > EVD_CUTOFF_DAYS)
    cov = ['AGEYEARS', 'SBP', 'ISS_nonvasc']
    base = d[cov].assign(patient_id=d['patient_id'])
    rows = []
    # never repaired during the stay: unrepaired throughout
    m = keep & ~repaired
    rows.append(base[m].assign(start=t0, stop=t_end[m], state='unrepaired', event=event[m]))
    # repaired early: with a landmark they enter already repaired; otherwise they wait first
    m = keep & early
    if landmark:
        rows.append(base[m].assign(start=t0, stop=t_end[m], state='repaired_early', event=event[m]))
    else:
        rows.append(base[m].assign(start=t0, stop=t_rep[m], state='unrepaired', event=0))
        rows.append(base[m].assign(start=t_rep[m], stop=t_end[m], state='repaired_early', event=event[m]))
    # repaired late: unrepaired until repair, then repaired_delayed
    m = keep & late
    rows.append(base[m].assign(start=t0, stop=t_rep[m], state='unrepaired', event=0))
    rows.append(base[m].assign(start=t_rep[m], stop=t_end[m], state='repaired_delayed', event=event[m]))
    out = pd.concat(rows, ignore_index=True)
    out = out[out['stop'] > out['start']].copy()
    for s in EVD_STATES:
        out[s] = (out['state'] == s).astype(int)
    out = out.sort_values(['patient_id', 'start'], kind='stable').reset_index(drop=True)
    counts = {'N patients': int(keep.sum()),
              'N TEVAR <=24 h': int((keep & early).sum()),
              'N TEVAR >24 h': int((keep & late).sum()),
              'N no TEVAR': int((keep & ~repaired).sum()),
              'Median days to delayed TEVAR': (round(float(t_rep[keep & late].median()), 2)
                                               if (keep & late).any() else np.nan)}
    for s in ['unrepaired'] + EVD_STATES:
        counts[f'N deaths while {s}'] = int(out.loc[out['state'] == s, 'event'].sum())
    return out, counts


def fit_early_vs_delayed(tv, label=''):
    """Cox time-varying fit; returns HRs for delayed vs. early TEVAR and each vs. unrepaired."""
    tv = tv.replace([np.inf, -np.inf], np.nan).dropna(subset=['AGEYEARS', 'SBP', 'ISS_nonvasc']).copy()
    for c in ['AGEYEARS', 'SBP', 'ISS_nonvasc']:
        sd = tv[c].std()
        tv[c] = (tv[c] - tv[c].mean()) / (sd if sd > 0 else 1.0)
    formula = ' + '.join(EVD_STATES + ['bs(AGEYEARS, df=4)', 'bs(SBP, df=4)', 'ISS_nonvasc'])
    data = tv[['patient_id', 'start', 'stop', 'event', 'AGEYEARS', 'SBP', 'ISS_nonvasc'] + EVD_STATES]
    m, pen = fit_with_ladder(lambda p: CoxTimeVaryingFitter(penalizer=p), data, ITB_PENALIZER_LADDER,
                             label=f"EVD {label}", id_col='patient_id', event_col='event',
                             start_col='start', stop_col='stop', formula=formula)
    b = m.params_
    V = pd.DataFrame(np.asarray(m.variance_matrix_), index=b.index, columns=b.index)  # CTV matrix is unlabeled
    z = 1.959963984540054

    def _hr(vec):
        est = float(sum(w * b[k] for k, w in vec.items()))
        var = float(sum(w1 * w2 * V.loc[k1, k2] for k1, w1 in vec.items() for k2, w2 in vec.items()))
        se = np.sqrt(var)
        return (np.exp(est), np.exp(est - z * se), np.exp(est + z * se), float(2 * _norm.sf(abs(est / se))))
    res = {'delayed vs early': _hr({'repaired_delayed': 1, 'repaired_early': -1}),
           'early vs unrepaired': _hr({'repaired_early': 1}),
           'delayed vs unrepaired': _hr({'repaired_delayed': 1})}
    return res, pen, float(m.log_likelihood_)

# run configuration:
# by default this module runs all four analyses: {2011, 2026} SVS guidelines x
# {thoracic-only (primary), total cohort (sensitivity)}. 
#   * Run a single configuration: set RUN_ALL = False and edit svs_2026 /
#     exclude_isolated_abd_aortic below, or from a terminal:
#         python analyze_aortic_injuries_2017_2024.py --single --svs2026 1 --thoracic 1
#   * Run a subset: edit RUN_CONFIGS.
#   * RUN_PARALLEL runs the configurations simultaneously (about 4x faster; the
#     bootstrap is the slow step). Set False to run them one after another.
#
# exclude_isolated_abd_aortic: when True, drop patients whose only aortic injury is
# abdominal (S35.0 / AIS 5202*). Patients with a thoracic component — with or without a
# concurrent abdominal component — are kept, since the SVS BAI guidelines are defined
# for thoracic injury. Those outputs land in the "thoracic_only" subfolder; total cohort
# outputs land in "total_cohort".
RUN_ALL = True
RUN_PARALLEL = True
RUN_CONFIGS = [  # (svs_2026, exclude_isolated_abd_aortic)
    (False, True),   # 2011 guidelines, thoracic-only (primary)
    (True, True),    # 2026 guidelines, thoracic-only (primary)
    (False, False),  # 2011 guidelines, total cohort
    (True, False),   # 2026 guidelines, total cohort
]
svs_2026 = False                    # used only when RUN_ALL = False
exclude_isolated_abd_aortic = True  # used only when RUN_ALL = False

_CHILD_ENV = 'BTAI_ANALYSIS_CONFIG'


def _config_label(svs, thor):
    return f"{'2026' if svs else '2011'} SVS, {'thoracic-only' if thor else 'total cohort'}"


def _run_all_configs(configs, parallel):
    """Launch one child process per configuration and stream its output with a prefix."""
    import subprocess, threading

    script = _os.path.abspath(__file__)
    procs = []
    lock = threading.Lock()

    def _pump(proc, tag, log):
        for line in proc.stdout:
            with lock:
                print(f"[{tag}] {line}", end='', flush=True)
            log.append(line)

    def _launch(svs, thor):
        env = dict(_os.environ)
        env[_CHILD_ENV] = f"{int(svs)},{int(thor)}"
        env['BTAI_DATA_PATH'] = _data_for_children
        env['MPLBACKEND'] = 'Agg'          
        env['PYTHONUNBUFFERED'] = '1'
        if parallel:                       
            env.setdefault('OMP_NUM_THREADS', str(max(1, (_os.cpu_count() or 4) // len(configs))))
        tag = _config_label(svs, thor)
        proc = subprocess.Popen([sys.executable, '-u', script], env=env, cwd=_os.path.dirname(script),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace')
        log = []
        t = threading.Thread(target=_pump, args=(proc, tag, log), daemon=True)
        t.start()
        return tag, proc, t, log

    try:
        _os.makedirs(LOCAL_CACHE, exist_ok=True)
        _local = _os.path.join(LOCAL_CACHE, _os.path.basename(DATA_PATH))
        _src = _os.stat(DATA_PATH)
        if (not _os.path.exists(_local) or _os.stat(_local).st_size != _src.st_size
                or _os.stat(_local).st_mtime < _src.st_mtime):
            print(f"Copying {DATA_PATH} to {_local} ...")
            _shutil.copy2(DATA_PATH, _local)
        _data_for_children = _local
    except OSError as e:
        print(f"Local copy failed ({e}); children will read from the share")
        _data_for_children = DATA_PATH
    t0 = _time.time()
    for svs, thor in configs:
        procs.append(_launch(svs, thor))
        if not parallel:
            procs[-1][1].wait(); procs[-1][2].join()
    for tag, proc, t, log in procs:
        proc.wait(); t.join()
    print(f"\n==== Finished {len(configs)} analyses in {(_time.time() - t0) / 60:.1f} min ====")
    for tag, proc, t, log in procs:
        status = 'OK' if proc.returncode == 0 else f'FAILED (exit code {proc.returncode})'
        print(f"  {tag}: {status}")
        if proc.returncode != 0:
            print('    last output lines:\n      ' + '      '.join(log[-15:]))


if _os.environ.get(_CHILD_ENV):
    _svs, _thor = (bool(int(x)) for x in _os.environ[_CHILD_ENV].split(','))
    svs_2026, exclude_isolated_abd_aortic = _svs, _thor
else:
    if '--single' in sys.argv:
        RUN_ALL = False
        _a = sys.argv
        if '--svs2026' in _a:
            svs_2026 = bool(int(_a[_a.index('--svs2026') + 1]))
        if '--thoracic' in _a:
            exclude_isolated_abd_aortic = bool(int(_a[_a.index('--thoracic') + 1]))
    if RUN_ALL:
        _run_all_configs(RUN_CONFIGS, RUN_PARALLEL)
        sys.exit(0)
print(f"Running analysis: {_config_label(svs_2026, exclude_isolated_abd_aortic)}")

df = read_csv_robust(DATA_PATH)

today = date.today().strftime("%m-%d-%Y")

exclusion_log = [f"Aortic injury records (step 2 file): {len(df)}"]
def log_exclusion(label, n_before, n_after):
    exclusion_log.append(f"{label}: excluded {n_before - n_after}; {n_after} remain")
    print(exclusion_log[-1])

# exclude deaths in the ED  
_n = len(df); df = df[df['ed_dispo_death'] != 1]; log_exclusion('Died in emergency department', _n, len(df))

# exclude pediatric trauma
_n = len(df); df = df[df['age_16_plus'] == 1]; log_exclusion('Age < 16 years', _n, len(df))


base_results_path = _os.path.join(RESULTS_ROOT, '2026 guidelines' if svs_2026 else '2011 guidelines',
                                  'thoracic_only' if exclude_isolated_abd_aortic else 'total_cohort')

results_path = get_export_path_string(base_results_path)

_n = len(df); df = df[df['moi_blunt'] == 1]; log_exclusion('Non-blunt mechanism', _n, len(df))

# sensitivity analysis: exclude patients whose only aortic injury is abdominal
if exclude_isolated_abd_aortic == True:
    n_before = len(df)
    df = df[df['has_thoracic_aortic_injury'] == 1]
    log_exclusion('Isolated abdominal aortic injury codes', n_before, len(df))

# drop rows with missing data for variables that will be used for Cox modeling later
_n = len(df)
df = df.dropna(subset=['los_d', 'dc_dispo_death_or_hospice',
              'AGEYEARS', 'SBP'])

log_exclusion('Missing age, SBP, length of stay, or discharge disposition', _n, len(df))

# exclude patients who underwent open repair
_n = len(df); df = df[df['traumatic_open_repair'] != 1]; log_exclusion('Open aortic repair', _n, len(df))
with open(_os.path.join(results_path, f"cohort_exclusions_{today}.txt"), 'w') as _f:
    _f.write('\n'.join(exclusion_log))

# add a variable for critical SOI
df['max_soi_ais_4_plus'] = (df['max_solid_organ_ais'] >= 4).astype(int)

# identify concordance between highest SVS grade and management strategy
# grades 1 and 2 typically managed with anti-impulse therapy
# grade 3 requires stent graft ideally within 24 hours
# grade 4 requires immediate intervention ideally by stent graft ideally within 6 hours

# unified concordance flags for BOTH guideline versions (see assign_concordance)
df = assign_concordance(df)
conc_col = 'guideline_concordant_care_2026' if svs_2026 else 'guideline_concordant_care_2011'

# non-vascular ISS: true ISS recomputed without vascular AIS codes
# (find_and_add_nonvasc_iss.py -> merged in step 2)
df['ISS_nonvasc'] = df['iss_nonvasc']

# follow-up variant for sensitivity analysis: patients discharged alive remain at
# risk (event-free) through day 90 instead of being censored at discharge
df['los_d_dc90'] = np.where(df['dc_dispo_death_or_hospice'] == 0,
                            np.maximum(df['los_d'], 90), df['los_d'])

# temporal trend in concordance (logistic regression on admission year, within grade)
trend_rows = []
for _gc in ['guideline_concordant_care_2011', 'guideline_concordant_care_2026']:
    for _g in [3, 4]:
        _d = df[df['svs_grade_final'] == _g][[_gc, 'YoAdmit']].dropna()
        try:
            _m = smf.logit(f"{_gc} ~ YoAdmit", data=_d).fit(disp=0)
            trend_rows.append({'Guideline': _gc[-4:], 'Grade': _g, 'N': len(_d),
                               'OR per year': np.exp(_m.params['YoAdmit']),
                               'Lower 95% CI': np.exp(_m.conf_int().loc['YoAdmit', 0]),
                               'Upper 95% CI': np.exp(_m.conf_int().loc['YoAdmit', 1]),
                               'p-value': _m.pvalues['YoAdmit'],
                               'First year %': 100 * _d[_d['YoAdmit'] == _d['YoAdmit'].min()][_gc].mean(),
                               'Last year %': 100 * _d[_d['YoAdmit'] == _d['YoAdmit'].max()][_gc].mean()})
        except Exception as e:
            print(f"trend test failed for {_gc} grade {_g}: {e}")
pd.DataFrame(trend_rows).to_excel(_os.path.join(results_path, f"concordance_trend_by_year_{today}.xlsx"), index=False)

# survival analysis
# 4 cohorts: guideline-concordant TEVAR vs. no TEVAR, within grade 3 and grade 4
# non-concordant TEVAR (e.g., mistimed) is 'Other' and excluded from KM/Cox (but is
# retained as unexposed time in the time-varying models)
def categorize_cohort(row):
    if row['svs_grade_final'] == 3:
        if row[conc_col] == 1:
            return 'Grade III + timely TEVAR'
        elif row['traumatic_endo_repair'] == 0:
            return 'Grade III No TEVAR'
    if row['svs_grade_final'] == 4:
        if row[conc_col] == 1:
            return 'Grade IV + timely TEVAR'
        elif row['traumatic_endo_repair'] == 0:
            return 'Grade IV No TEVAR'
    return 'Other'

df['survival_cohort'] = df.apply(categorize_cohort, axis=1)
df_clean = df[df['survival_cohort'] != 'Other'].copy()

# ISS_nonvasc, tbi_or_soi, and concordance flags are created on df above (assign_concordance)

comparisons = [('Grade III + timely TEVAR', 'Grade III No TEVAR', 'p_g3'),
               ('Grade IV + timely TEVAR', 'Grade IV No TEVAR', 'p_g4'),
               ('Grade III No TEVAR', 'Grade IV + timely TEVAR', 'p_cross')]

p_values = {}

for groupA, groupB, label in comparisons:
    data_A = df_clean[df_clean['survival_cohort'] == groupA]
    data_B = df_clean[df_clean['survival_cohort'] == groupB]
    
    if len(data_A) > 0 and len(data_B) > 0:
        result = logrank_test(
            data_A['los_d'], data_B['los_d'], 
            event_observed_A=data_A['dc_dispo_death_or_hospice'], 
            event_observed_B=data_B['dc_dispo_death_or_hospice']
        )
        p_values[label] = result.p_value
    else:
        p_values[label] = 1.0

def format_p(p):
    if p is None: return "p = N/A"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

balanced_colors = ['#377eb8', '#ff7f00', '#4daf4a', '#e41a1c']


# unadjusted survival
fig, ax = plt.subplots(figsize=(12, 7))
kmf = KaplanMeierFitter()

cohorts = ['Grade III + timely TEVAR', 'Grade III No TEVAR',
           'Grade IV + timely TEVAR', 'Grade IV No TEVAR']

for i, cohort in enumerate(cohorts):
    mask = df_clean['survival_cohort'] == cohort
    if mask.sum() > 0:
        kmf.fit(durations=df_clean.loc[mask, 'los_d'], 
                event_observed=df_clean.loc[mask, 'dc_dispo_death_or_hospice'], 
                label=f"{cohort} (n={mask.sum()})")
        kmf.plot_survival_function(ax=ax, color=balanced_colors[i], alpha=1.0, linewidth=2)
if svs_2026 == True: 
    plt.title('In-hospital survival by blunt aortic injury grade and management\nper 2026 Society for Vascular Surgery guidelines', 
              fontsize=17, fontweight='bold')
else:    
    plt.title('In-hospital survival by blunt aortic injury grade and management\nper 2011 Society for Vascular Surgery guidelines', 
              fontsize=17, fontweight='bold')
plt.xlabel('In-hospital survival (days)', fontsize=15)
plt.ylabel('Survival probability', fontsize=15)
plt.ylim(0, 1.05)
plt.xlim(-1, 90)
plt.grid(True, linestyle='--', alpha=0.6)

legend_fontsize = 13
leg = ax.legend(facecolor='white', loc='lower left', fontsize=legend_fontsize)

for line, text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
    
plt.draw() 

bbox = leg.get_window_extent()
inv = ax.transData.inverted()
_, legend_top_data = inv.transform((0, bbox.p1[1]))

lines = ax.get_lines()

c_g3_repair = lines[0].get_color()
c_g3_no     = lines[1].get_color()
c_g4_repair = lines[2].get_color()
c_g4_no     = lines[3].get_color()

def create_colored_line(label_start, repair_color, no_repair_color, p_val):
    return HPacker(children=[
        TextArea(f"{label_start} ", textprops=dict(size=legend_fontsize)),
        TextArea("Repair", textprops=dict(color=repair_color, size=legend_fontsize)),
        TextArea(" vs ", textprops=dict(size=legend_fontsize)),
        TextArea("No Repair", textprops=dict(color=no_repair_color, size=legend_fontsize)),
        TextArea(f": {p_val}", textprops=dict(size=legend_fontsize))
    ], align="baseline", pad=0, sep=2)

line1 = create_colored_line("Grade III", c_g3_repair, c_g3_no, format_p(p_values.get('p_g3')))

line2 = create_colored_line("Grade IV", c_g4_repair, c_g4_no, format_p(p_values.get('p_g4')))

line3 = HPacker(children=[
    TextArea("Grade III ", textprops=dict(size=legend_fontsize)),
    TextArea("No Repair", textprops=dict(color=c_g3_no, size=legend_fontsize)),
    TextArea(" vs Grade IV ", textprops=dict(size=legend_fontsize)),
    TextArea("Repair", textprops=dict(color=c_g4_repair, size=legend_fontsize)),
    TextArea(f": {format_p(p_values.get('p_cross'))}", textprops=dict(size=legend_fontsize))
], align="baseline", pad=0, sep=2)

box_content = VPacker(children=[line1, line2, line3], align="left", pad=0, sep=5)

stat_box = AnchoredOffsetbox(child=box_content, loc='lower left',
                            bbox_to_anchor=(40, 0.016),
                            bbox_transform=ax.transData,
                            frameon=True)

stat_box.patch.set_boxstyle("round,pad=0.4")
stat_box.patch.set_edgecolor("#CCCCCC")
stat_box.patch.set_facecolor("white")
stat_box.patch.set_alpha(0.8)

ax.add_artist(stat_box)
plt.savefig(_os.path.join(results_path, f"survival for BAI {'2026' if svs_2026 else '2011'} by mgmt {today}.jpg"), dpi=1200, bbox_inches='tight')
plt.show() 


# adjusted survival
# plot survival with risk adjustment for age, non-vascular ISS, and SBP
# drop highest_vasc_inj_severity: redundant with survival_cohort (which encodes grade)
# replace ISS_05 with ISS_nonvasc to remove chest-AIS double-counting that scales
# with grade
# use patsy splines on AGEYEARS and SBP to capture J-shaped /
# accelerating risk that linear terms miss
model_cols = ['los_d', 'dc_dispo_death_or_hospice', 'survival_cohort',
              'AGEYEARS', 'ISS_nonvasc', 'SBP']
df_adj = df_clean[model_cols].replace([np.inf, -np.inf], np.nan).dropna()

# cohort ordering — locked before bootstrap so dummy columns are stable
df_adj['survival_cohort'] = pd.Categorical(df_adj['survival_cohort'], categories=cohorts)

# patsy formula: splines on age and SBP, linear on ISS_nonvasc
# reference cohort is fixed so coefficient names are stable across runs
ref_cohort = cohorts[0]
formula = (f"C(survival_cohort, Treatment(reference='{ref_cohort}'))"
           " + bs(AGEYEARS, df=4) + ISS_nonvasc + bs(SBP, df=4)")

cph, PRIMARY_PEN = fit_with_ladder(lambda p: CoxPHFitter(penalizer=p), df_adj,
                                   COX_PENALIZER_LADDER, label='Cox primary',
                                   duration_col='los_d', event_col='dc_dispo_death_or_hospice',
                                   formula=formula)

def cohort_term(cohort):
    return f"C(survival_cohort, Treatment(reference='{ref_cohort}'))[T.{cohort}]"

# export hazard ratio table
hr_table = cph.summary[['exp(coef)', 'exp(coef) lower 95%',
                        'exp(coef) upper 95%', 'p']].copy()
hr_table.columns = ['Hazard Ratio', 'Lower 95% CI', 'Upper 95% CI', 'p-value']
# additional specifications and within-/cross-grade contrasts
df_adj_x = df_adj.join(df_clean.loc[df_adj.index, ['los_d_dc90']])
# time-split columns: early window = events through day EARLY_WINDOW_DAYS (administrative
# censoring at that day); late window = patients still hospitalized at that day, entering
# the risk set at that day (left truncation)
_W = EARLY_WINDOW_DAYS
df_adj_x['los_d_early'] = df_adj_x['los_d'].clip(upper=_W)
df_adj_x['event_early'] = ((df_adj_x['dc_dispo_death_or_hospice'] == 1) & (df_adj_x['los_d'] <= _W)).astype(int)
df_adj_x['entry_late'] = float(_W)

def fit_spec(duration, ref, window=None, iss_spline=False):
    d = df_adj_x.copy()
    event = 'dc_dispo_death_or_hospice'
    kw = {}
    if window == 'early':
        d = d.drop(columns=['dc_dispo_death_or_hospice']).rename(columns={'event_early': 'dc_dispo_death_or_hospice'})
        duration = 'los_d_early'
    elif window == 'late':
        d = d[d['los_d'] > _W]
        kw['entry_col'] = 'entry_late'
    keep = [c for c in d.columns if c not in ['los_d', 'los_d_dc90', 'los_d_early', 'event_early', 'entry_late']]
    keep += [duration] + (['entry_late'] if window == 'late' else [])
    d = d[keep]
    f = formula.replace(f"reference='{ref_cohort}'", f"reference='{ref}'")
    if iss_spline:
        f = f.replace(' + ISS_nonvasc', ' + bs(ISS_nonvasc, df=4)')
    kw.update(dict(duration_col=duration, event_col=event, formula=f))
    return fit_with_ladder(lambda p: CoxPHFitter(penalizer=p), d, COX_PENALIZER_LADDER,
                           label=f"Cox {window or 'full'} {duration} ref={ref}", **kw)

contrast_defs = [('Grade III: no TEVAR vs. timely TEVAR', 'Grade III No TEVAR', 'Grade III + timely TEVAR'),
                 ('Grade IV: no TEVAR vs. timely TEVAR', 'Grade IV No TEVAR', 'Grade IV + timely TEVAR'),
                 ('Grade III no TEVAR vs. Grade IV timely TEVAR', 'Grade III No TEVAR', 'Grade IV + timely TEVAR')]
PRIMARY_LABEL = 'primary (unpenalized)'
spec_defs = [(PRIMARY_LABEL, 'los_d', None),
             ('discharged alive followed to day 90', 'los_d_dc90', None),
             (f'days 0-{EARLY_WINDOW_DAYS}', 'los_d', 'early'),
             (f'after day {EARLY_WINDOW_DAYS}', 'los_d', 'late'),
             (ISS_SPLINE_LABEL, 'los_d', None)]
spec_summaries, contrast_rows = {}, []
for _label, _dur, _win in spec_defs:
    for _clabel, _a, _b in contrast_defs:
        try:
            _m, _pu = fit_spec(_dur, _b, window=_win, iss_spline=(_label == ISS_SPLINE_LABEL))
            _term = f"C(survival_cohort, Treatment(reference='{_b}'))[T.{_a}]"
            _r = _m.summary.loc[_term]
            contrast_rows.append({'Specification': _label, 'Contrast': _clabel,
                                  'Hazard Ratio': _r['exp(coef)'], 'Lower 95% CI': _r['exp(coef) lower 95%'],
                                  'Upper 95% CI': _r['exp(coef) upper 95%'], 'p-value': _r['p'],
                                  'Penalizer used': _pu,
                                  'N patients': int(_m._n_examples),
                                  'N deaths': int(_m.event_observed.sum())})
            if _b == ref_cohort and _label != PRIMARY_LABEL:
                _t = _m.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].copy()
                _t.columns = ['Hazard Ratio', 'Lower 95% CI', 'Upper 95% CI', 'p-value']
                spec_summaries[_label] = _t
        except Exception as e:
            print(f"Cox specification '{_label}' / '{_clabel}' failed: {type(e).__name__} - {e}")
            FIT_LOG.append(f"FAILED Cox '{_label}' / '{_clabel}': {type(e).__name__} - {e}")
            contrast_rows.append({'Specification': _label, 'Contrast': _clabel,
                                  'Hazard Ratio': np.nan, 'Lower 95% CI': np.nan,
                                  'Upper 95% CI': np.nan, 'p-value': np.nan,
                                  'Penalizer used': 'failed'})

# functional form of non-vascular ISS: linear (primary) vs. B-splines, same data and penalizer
# linear ISS is nested in the spline basis, so a likelihood ratio test applies when both are unpenalized
_iss_rows = []
try:
    _m_lin, _p_lin = fit_spec('los_d', ref_cohort)
    _m_spl, _p_spl = fit_spec('los_d', ref_cohort, iss_spline=True)
    _lr = 2 * (_m_spl.log_likelihood_ - _m_lin.log_likelihood_)
    _ddf = len(_m_spl.params_) - len(_m_lin.params_)
    for _lab, _m, _pp in [('linear (primary)', _m_lin, _p_lin), ('B-splines, 4 df', _m_spl, _p_spl)]:
        _iss_rows.append({'Model': 'Cox', 'Non-vascular ISS': _lab, 'Partial log-likelihood': _m.log_likelihood_,
                          'Parameters': len(_m.params_), 'AIC (partial)': _m.AIC_partial_, 'Penalizer used': _pp})
    _iss_rows.append({'Model': 'Cox', 'Non-vascular ISS': 'likelihood ratio test, splines vs. linear',
                      'Partial log-likelihood': _lr, 'Parameters': _ddf,
                      'AIC (partial)': np.nan,
                      'Penalizer used': f"p={chi2.sf(_lr, _ddf):.4g}" + ('' if _p_lin == _p_spl == 0 else ' (penalized fit; LR test approximate)')})
except Exception as e:
    FIT_LOG.append(f"FAILED ISS functional-form comparison: {type(e).__name__} - {e}")
_iss_form_df = pd.DataFrame(_iss_rows)

# proportional hazards check (Schoenfeld residual tests), primary model
_buf = io.StringIO()
try:
    with contextlib.redirect_stdout(_buf):
        cph.check_assumptions(df_adj, p_value_threshold=0.05, show_plots=False)
except Exception as e:
    _buf.write(f"check_assumptions failed: {type(e).__name__} - {e}")
with open(_os.path.join(results_path, f"proportional_hazards_check_{today}.txt"), 'w') as _f:
    _f.write(_buf.getvalue())

# write HR tables + a 'repro' sheet so seed/package versions travel with the artifact
with pd.ExcelWriter(_os.path.join(results_path, f"adjusted_hazard_ratios_{today}.xlsx")) as _xw:
    hr_table.to_excel(_xw, sheet_name='hazard_ratios', index=True)
    for _label, _t in spec_summaries.items():
        _t.to_excel(_xw, sheet_name=_label[:31], index=True)
    pd.DataFrame(contrast_rows).to_excel(_xw, sheet_name='contrasts_all_specs', index=False)
    _iss_form_df.to_excel(_xw, sheet_name='iss_functional_form', index=False)
    pd.DataFrame({'key': REPRO_STAMP.replace(' | ', '|').split('|'),
                  }).to_excel(_xw, sheet_name='repro', index=False)

def build_avg_patient(cohort_name):
    """Average-patient row at population means; lifelines re-applies the formula."""
    return pd.DataFrame([{
        'survival_cohort': cohort_name,
        'AGEYEARS': df_adj['AGEYEARS'].mean(),
        'ISS_nonvasc': df_adj['ISS_nonvasc'].mean(),
        'SBP': df_adj['SBP'].mean()
    }])

# bootstrap CIs for adjusted survival curves 
n_boot = 2000           
ci_alpha = 0.05        
common_times = np.linspace(0, 90, 91)   # daily grid to 90 days, matches xlim below

print(f"Bootstrapping adjusted survival CIs ({n_boot} resamples; {REPRO_STAMP})...")
rng = np.random.default_rng(RANDOM_SEED)
boot_curves = {c: [] for c in cohorts}
n_failed = 0

_X_full = cph.regressors.transform_df(df_adj)['beta_']
_X_full = _X_full[list(cph.params_.index)]
_X_full['los_d'] = df_adj['los_d'].values
_X_full['dc_dispo_death_or_hospice'] = df_adj['dc_dispo_death_or_hospice'].values
_X_avg = pd.concat([cph.regressors.transform_df(build_avg_patient(c))['beta_'] for c in cohorts],
                   ignore_index=True)[list(cph.params_.index)]
_boot_pens = []
for b in range(n_boot):
    if (b + 1) % 100 == 0:
        print(f"  resample {b + 1}/{n_boot}")

    # resample patients (rows) with replacement
    idx = rng.integers(0, len(df_adj), size=len(df_adj))
    df_b = _X_full.iloc[idx]

    sf = None
    for _pen in [p for p in COX_PENALIZER_LADDER if p >= PRIMARY_PEN]:
        with warnings.catch_warnings(record=True) as _caught:
            warnings.simplefilter('always')
            try:
                cph_b = _BootstrapCoxPH(penalizer=_pen)
                cph_b.fit(df_b, duration_col='los_d', event_col='dc_dispo_death_or_hospice')
            except Exception:
                continue
        if any(k in str(w.message) for w in _caught for k in _BAD_FIT_MSGS) \
                or not np.all(np.isfinite(cph_b.params_.values)):
            continue
        sf = cph_b.predict_survival_function(_X_avg, times=common_times)
        _boot_pens.append(_pen)
        break
    if sf is None:
        n_failed += 1
        continue
    for _j, cohort in enumerate(cohorts):
        boot_curves[cohort].append(sf.iloc[:, _j].values)

_bp = pd.Series(_boot_pens).value_counts().sort_index()
_boot_msg = (f"Bootstrap: {len(_boot_pens)} of {n_boot} resamples used; {n_failed} failed at every "
             f"penalizer; penalizer used: " + ", ".join(f"{k}: {v}" for k, v in _bp.items()))
print("  " + _boot_msg)
FIT_LOG.append(_boot_msg)
for cohort in cohorts:
    _arr = np.array(boot_curves[cohort])
    if len(_arr):
        FIT_LOG.append(f"Bootstrap {cohort}: day-90 survival median {np.median(_arr[:, -1]):.3f}, "
                       f"2.5th-97.5th percentile {np.percentile(_arr[:, -1], 2.5):.3f}-"
                       f"{np.percentile(_arr[:, -1], 97.5):.3f}; point estimate "
                       f"{cph.predict_survival_function(build_avg_patient(cohort), times=common_times).iloc[-1, 0]:.3f}")

# pointwise percentile CIs
lo_q, hi_q = 100 * ci_alpha / 2, 100 * (1 - ci_alpha / 2)
boot_ci = {}
for cohort, curves in boot_curves.items():
    if curves:
        arr = np.array(curves)
        boot_ci[cohort] = {'lower': np.percentile(arr, lo_q, axis=0),
                           'upper': np.percentile(arr, hi_q, axis=0)}
    else:
        boot_ci[cohort] = {'lower': None, 'upper': None}

fig = plt.figure(figsize=(18, 8))
gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.5])
ax = fig.add_subplot(gs[0])
ax_fp = fig.add_subplot(gs[1])

balanced_colors = ['#377eb8', '#ff7f00', '#4daf4a', '#e41a1c']
legend_labels = []

legend_fontsize = 13
for i, cohort in enumerate(cohorts):
    n_count = (df_adj['survival_cohort'] == cohort).sum()
    legend_labels.append(f"{cohort} (n={n_count})")
    
    central = cph.predict_survival_function(build_avg_patient(cohort),
                                            times=common_times)
    ax.plot(common_times, central.iloc[:, 0], color=balanced_colors[i], linewidth=2.5)
    
    if boot_ci[cohort]['lower'] is not None:
        ax.fill_between(common_times,
                        boot_ci[cohort]['lower'],
                        boot_ci[cohort]['upper'],
                        color=balanced_colors[i], alpha=0.15)


ax.set_title('Risk-adjusted in-hospital survival\nby injury grade and TEVAR status\n', fontsize=18, fontweight='bold')
ax.set_xlabel('In-hospital survival (days)', fontsize=18)
ax.set_ylabel('Risk-adjusted survival', fontsize=18)

set_ymin = 0.5
ax.set_ylim(set_ymin, 1.05); ax.set_xlim(-1, 90); ax.grid(True, linestyle='--', alpha=0.4)
all_lines = ax.get_lines()
leg = ax.legend(handles=all_lines, labels=legend_labels, loc='lower left', facecolor='white',
                fontsize=legend_fontsize)
for line, text in zip(all_lines, leg.get_texts()):
    text.set_color(line.get_color())

# forest plot: show only the cohort contrasts 
cohort_term_names = [cohort_term(c) for c in cohorts[1:]
                     if cohort_term(c) in cph.params_.index]

design_info_full = dmatrix(formula, df_adj, return_type='dataframe').design_info

def covariate_contrast_hr(var, low_value, high_value, label):
    """HR + 95% CI for changing `var` from low to high, holding other
    covariates at their means and a fixed reference cohort. Works for both
    linear and spline-modeled variables because we operate on the design
    matrix, not on individual coefficients."""
    template = build_avg_patient(ref_cohort)
    low_row = template.copy();  low_row[var] = low_value
    high_row = template.copy(); high_row[var] = high_value

    dm_low = pd.DataFrame(build_design_matrices([design_info_full], low_row)[0],
                          columns=design_info_full.column_names)
    dm_high = pd.DataFrame(build_design_matrices([design_info_full], high_row)[0],
                           columns=design_info_full.column_names)

    fitted_cols = [c for c in cph.params_.index if c in dm_low.columns]
    contrast = (dm_high[fitted_cols].iloc[0] - dm_low[fitted_cols].iloc[0]).values
    beta = cph.params_.loc[fitted_cols].values
    cov = cph.variance_matrix_.loc[fitted_cols, fitted_cols].values

    log_hr = float(contrast @ beta)
    se = float(np.sqrt(contrast @ cov @ contrast))
    return {'label': label,
            'hr': np.exp(log_hr),
            'lower': np.exp(log_hr - 1.96 * se),
            'upper': np.exp(log_hr + 1.96 * se)}

covariate_contrasts = [
    covariate_contrast_hr('AGEYEARS', 45, 75,   'Age: 75 vs. 45 yrs'),
    covariate_contrast_hr('SBP',      120, 80,  'Admission SBP: 80 vs. 120 mmHg'),
    covariate_contrast_hr('ISS_nonvasc', 9, 25, 'ISS (non-vascular): 25 vs. 9')
]

rows = []
for cohort in cohorts[1:]:
    term = cohort_term(cohort)
    if term in cph.params_.index:
        hr = float(np.exp(cph.params_[term]))
        ci = cph.confidence_intervals_.loc[term]
        rows.append({'label': cohort, 'hr': hr,
                     'lower': float(np.exp(ci.iloc[0])),
                     'upper': float(np.exp(ci.iloc[1])),
                     'group': 'cohort'})
for c in covariate_contrasts:
    rows.append({**c, 'group': 'covariate'})

# manual forest plot
ax_fp.clear()
cohort_color = '#7b3294'  # purple
covar_color  = '#c51b7d'  # fuchsia
for i, r in enumerate(rows):
    color = cohort_color if r['group'] == 'cohort' else covar_color
    ax_fp.errorbar(r['hr'], i,
                   xerr=[[r['hr'] - r['lower']], [r['upper'] - r['hr']]],
                   fmt='o', color=color, capsize=4, markersize=7, elinewidth=1.5)
    ax_fp.text(max(r['upper'], 1.0) * 1.05, i,
               f"{r['hr']:.2f} ({r['lower']:.2f}–{r['upper']:.2f})",
               va='center', fontsize=10, color=color)

n_cohort = sum(1 for r in rows if r['group'] == 'cohort')
ax_fp.axhline(n_cohort - 0.5, color='gray', linestyle='-', alpha=0.4)

ax_fp.set_yticks(range(len(rows)))
ax_fp.set_yticklabels([r['label'] for r in rows])
ax_fp.invert_yaxis()
ax_fp.axvline(1, color='black', linestyle='--', alpha=0.5)
ax_fp.set_xscale('log')
# plain-number tick labels (default log ticks render as 2x10^0)
ax_fp.xaxis.set_major_locator(_mticker.FixedLocator([0.5, 0.75, 1, 1.5, 2, 3, 4, 6]))
ax_fp.xaxis.set_major_formatter(_mticker.FormatStrFormatter('%g'))
ax_fp.xaxis.set_minor_formatter(_mticker.NullFormatter())
ax_fp.set_xlabel('Hazard Ratio (95% CI)\nReference: Grade III + timely TEVAR', fontsize=15)
ax_fp.set_title('Adjusted hazard of in-hospital mortality by\ninjury grade, TEVAR exposure, and clinical covariates\n', fontsize=18, fontweight='bold')

ylimits = ax_fp.get_ylim()
ax_fp.text(-0.02, ylimits[1],
           'Cohort\n(vs. Grade III + timely TEVAR)',
           transform=ax_fp.get_yaxis_transform(),
           rotation=0, va='center', ha='right',
           fontsize=10, fontweight='bold', color=cohort_color)
ax_fp.text(-0.02, n_cohort - 0.5,
           'Covariate\n(binary comparisons)',
           transform=ax_fp.get_yaxis_transform(),
           rotation=0, va='center', ha='right',
           fontsize=10, fontweight='bold', color=covar_color)

# apply Wald test logic
def manual_wald_p(cph, col1, col2):
    diff = cph.params_[col1] - cph.params_[col2]
    cov = cph.variance_matrix_.loc[col1, col2]
    var_diff = cph.variance_matrix_.loc[col1, col1] + cph.variance_matrix_.loc[col2, col2] - 2*cov
    return chi2.sf((diff**2) / var_diff, 1)

p_summary = cph.summary['p']
p_g3 = p_summary.get(cohort_term('Grade III No TEVAR'), 1.0)
# within-grade IV contrast (previously the Grade IV No TEVAR vs. Grade III reference p-value)
p_g4 = manual_wald_p(cph, cohort_term('Grade IV No TEVAR'), cohort_term('Grade IV + timely TEVAR'))
p_cross_adj = manual_wald_p(cph,
    cohort_term('Grade III No TEVAR'),
    cohort_term('Grade IV + timely TEVAR'))

# format_p is defined earlier in this block and reused here

c_g3_rep, c_g3_no = all_lines[0].get_color(), all_lines[1].get_color()
c_g4_rep, c_g4_no = all_lines[2].get_color(), all_lines[3].get_color()

line1 = create_colored_line("Grade III", c_g3_rep, c_g3_no, format_p(p_g3))
line2 = create_colored_line("Grade IV", c_g4_rep, c_g4_no, format_p(p_g4))
line3 = HPacker(children=[
    TextArea("Grade III ", textprops=dict(size=legend_fontsize)),
    TextArea("No Repair", textprops=dict(color=c_g3_no, size=legend_fontsize)),
    TextArea(" vs Grade IV ", textprops=dict(size=legend_fontsize)),
    TextArea("Repair", textprops=dict(color=c_g4_rep, size=legend_fontsize)),
    TextArea(f": {format_p(p_cross_adj)}", textprops=dict(size=legend_fontsize))
], align="baseline", pad=0, sep=2)

box_content = VPacker(children=[line1, line2, line3], align="left", pad=0, sep=5)
stat_box = AnchoredOffsetbox(child=box_content, loc='lower left', bbox_to_anchor=(40, set_ymin),
                             bbox_transform=ax.transData, frameon=True)
stat_box.patch.set_boxstyle("round,pad=0.4"); stat_box.patch.set_facecolor("white"); stat_box.patch.set_edgecolor("#CCCCCC"); stat_box.patch.set_alpha(0.8)
ax.add_artist(stat_box)

plt.tight_layout()
plt.savefig(_os.path.join(results_path, f"adj survival BAI {'2026' if svs_2026 else '2011'} by mgmt {today}.jpg"), dpi=1200, bbox_inches='tight')
plt.show()


# immortal time bias analysis that addresses the "guaranteed alive at the time of procedure" bias
# inherent to having a cohort that undergoes delayed intervention
def transform_to_tv_guidelines(df_input, svs_version_2026=True, duration_col='los_d'):
    """Build start-stop long-format dataframe with `is_concordant` as the
    time-varying exposure. Pre-repair time is unexposed; post-repair time is
    exposed only if the repair meets guideline-concordance criteria."""
    d = df_input.reset_index().rename(columns={'index': 'patient_id'})
    covariates = ['AGEYEARS', 'ISS_nonvasc', 'SBP']
    epsilon = 0.001
    conc_flag = 'guideline_concordant_care_2026' if svs_version_2026 else 'guideline_concordant_care_2011'
    t_end = d[duration_col].clip(lower=epsilon)
    t_proc_min = d['time_to_first_vasc_proc']
    t_proc = t_proc_min / 1440.0  # minutes -> days
    event = d['dc_dispo_death_or_hospice']
    # concordance from the shared definition (assign_concordance); exposure starts at TEVAR
    conc = ((d[conc_flag] == 1) & (d['traumatic_endo_repair'] == 1) & t_proc_min.notna()
            & (t_proc >= 0) & (t_proc < t_end))
    base = d[covariates].assign(patient_id=d['patient_id'], grade=d['svs_grade_final'])
    safe_t_proc = t_proc.clip(lower=epsilon)
    # non-concordant repair or no repair: unexposed throughout
    unexposed = base[~conc].assign(start=0.0, stop=t_end[~conc], is_concordant=0, event=event[~conc])
    # concordant repair: waiting interval (unexposed), then post-repair interval (exposed)
    waiting = base[conc].assign(start=0.0, stop=safe_t_proc[conc], is_concordant=0, event=0)
    post = base[conc].assign(start=safe_t_proc[conc], stop=t_end[conc], is_concordant=1, event=event[conc])
    out = pd.concat([unexposed, waiting, post])
    out['_order'] = out.index
    out = out.sort_values(['_order', 'start'], kind='stable').drop(columns='_order')
    return out.reset_index(drop=True)

def fit_itb_model(df_grade, scaler, penalizer=0.1, _ladder=None, label='', iss_spline=False):
    """Fit a CoxTimeVaryingFitter on a single-grade subset; return summary
    dict for is_concordant. Splines on age and SBP, linear on the rest."""
    df_grade = df_grade.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df_grade['patient_id'].unique()) < 5:
        return None
    # scale linear continuous covariates so the L2 penalty hits coefficients
    # on a comparable scale to the binary is_concordant indicator
    cols_to_scale = ['AGEYEARS', 'ISS_nonvasc', 'SBP']
    df_grade[cols_to_scale] = scaler.fit_transform(df_grade[cols_to_scale])
    varying = [c for c in cols_to_scale if df_grade[c].nunique() > 1]
    # splines on age and SBP, linear on ISS_nonvasc
    formula_parts = ['is_concordant']
    for c in varying:
        if c in ('AGEYEARS', 'SBP') or (iss_spline and c == 'ISS_nonvasc'):
            formula_parts.append(f'bs({c}, df=4)')
        else:
            formula_parts.append(c)
    ctv_formula = ' + '.join(formula_parts)
    ladder = [penalizer] if penalizer != 'auto' else ITB_PENALIZER_LADDER
    ctv, used = fit_with_ladder(lambda p: CoxTimeVaryingFitter(penalizer=p), df_grade, ladder,
                                label=f"ITB {label}", id_col='patient_id', event_col='event',
                                start_col='start', stop_col='stop', formula=ctv_formula)
    summary_row = ctv.summary.loc['is_concordant']
    ci_row = ctv.confidence_intervals_.loc['is_concordant']
    return {
        'hr': float(np.exp(summary_row['coef'])),
        'lower': float(np.exp(ci_row.iloc[0])),
        'upper': float(np.exp(ci_row.iloc[1])),
        'p': float(summary_row['p']),
        'n_patients': int(df_grade['patient_id'].nunique()),
        'n_concordant_rows': int((df_grade['is_concordant'] == 1).sum()),
        'n_events': int(df_grade['event'].sum()),
        'n_events_concordant': int(df_grade.loc[df_grade['is_concordant'] == 1, 'event'].sum()),
        'll': float(ctv.log_likelihood_), 'k': int(len(ctv.params_)),
        'penalizer': used,
    }

results_summary = []
scaler = StandardScaler()

df_for_itb = df[df['svs_grade_final'].isin([3, 4])].copy()

def window_tv(d, window):
    """Restrict start-stop data to a follow-up window (days)."""
    if window is None:
        return d
    W = float(EARLY_WINDOW_DAYS)
    d = d.copy()
    if window == 'early':
        d = d[d['start'] < W]
        d.loc[d['stop'] > W, 'event'] = 0
        d['stop'] = d['stop'].clip(upper=W)
    else:
        d = d[d['stop'] > W]
        d['start'] = d['start'].clip(lower=W)
    return d[d['stop'] > d['start']]

itb_specs = [('primary', 'los_d', 'auto', None),
             ('discharged alive followed to day 90', 'los_d_dc90', 'auto', None),
             (f'days 0-{EARLY_WINDOW_DAYS}', 'los_d', 'auto', 'early'),
             (f'after day {EARLY_WINDOW_DAYS}', 'los_d', 'auto', 'late'),
             (ISS_SPLINE_LABEL, 'los_d', 'auto', None)]

svs_label = "2026 SVS" if svs_2026 else "2011 SVS"
_tv_cache = {}
for spec_label, dur_col, pen, win in itb_specs:
    if dur_col not in _tv_cache:
        _tv_cache[dur_col] = transform_to_tv_guidelines(
            df_for_itb, svs_version_2026=svs_2026, duration_col=dur_col)
    df_tv_all = window_tv(_tv_cache[dur_col], win)

    for grade in [3, 4]:
        df_grade = df_tv_all[df_tv_all['grade'] == grade].copy()
        _itb_label = f"{spec_label} {svs_label} grade {grade}"
        try:
            fit = fit_itb_model(df_grade, scaler, penalizer=pen, label=_itb_label,
                                iss_spline=(spec_label == ISS_SPLINE_LABEL))
            if fit is None:
                continue
            results_summary.append({
                'Specification': spec_label,
                'Guideline': svs_label,
                'Injury Grade': f"Grade {grade}",
                'N patients': fit['n_patients'],
                'N concordant rows': fit['n_concordant_rows'],
                'N deaths': fit['n_events'],
                'N deaths while concordant': fit['n_events_concordant'],
                'Hazard Ratio': round(fit['hr'], 3),
                '95% CI': f"{fit['lower']:.2f}-{fit['upper']:.2f}",
                'p-value': round(fit['p'], 4),
                'Penalizer used': fit['penalizer'],
                'Log-likelihood': round(fit['ll'], 2), 'N parameters': fit['k'],
                'Converged': 'Yes'})
        except Exception as e:
            print(f"Error in {spec_label} {svs_label} Grade {grade}: {type(e).__name__} - {e}")
            FIT_LOG.append(f"FAILED ITB {_itb_label}: {type(e).__name__} - {e}")
            results_summary.append({
                'Specification': spec_label,
                'Guideline': svs_label, 'Injury Grade': f"Grade {grade}",
                'N patients': df_grade['patient_id'].nunique(),
                'N concordant rows': int((df_grade['is_concordant'] == 1).sum()),
                'Hazard Ratio': np.nan, '95% CI': 'N/A',
                'p-value': np.nan, 'Converged': 'No'})

itb_results_df = pd.DataFrame(results_summary)
itb_results_df.to_excel(_os.path.join(results_path, f"immortal_time_bias_hr_{today}.xlsx"), index=False)

# early vs. delayed TEVAR, stable grade III, guideline-independent, so it runs once per 
# cohort (in the 2011 runs only)
if not svs_2026:
    _g3 = df[df['svs_grade_final'] == 3]
    _stable = _g3['unstable_or_2_plus_rbc'] == 0
    _no_tbi_soi = _g3['tbi_or_soi'] == 0
    evd_specs = [
        ('landmark 24 h (primary)', _stable & _no_tbi_soi, 'los_d', True),
        ('from admission, no landmark', _stable & _no_tbi_soi, 'los_d', False),
        ('landmark 24 h, including severe TBI/SOI', _stable, 'los_d', True),
        ('landmark 24 h, discharged alive followed to day 90', _stable & _no_tbi_soi, 'los_d_dc90', True)]
    evd_rows = []
    for _lab, _mask, _dur, _lm in evd_specs:
        _row = {'Specification': _lab,
                'Population': 'stable grade III' + ('' if 'including' in _lab else ', no severe TBI/SOI')}
        try:
            _tv, _counts = build_early_vs_delayed_tv(_g3[_mask], duration_col=_dur, landmark=_lm)
            _row.update(_counts)
            _res, _pen, _ll = fit_early_vs_delayed(_tv, label=_lab)
            for _k, (_h, _lo, _hi, _p) in _res.items():
                _row[f'HR {_k}'] = round(_h, 3)
                _row[f'95% CI {_k}'] = f"{_lo:.2f}-{_hi:.2f}"
                _row[f'p {_k}'] = round(_p, 4)
            _row.update({'Penalizer used': _pen, 'Log-likelihood': round(_ll, 2), 'Converged': 'Yes'})
        except Exception as e:
            print(f"Early vs delayed TEVAR, {_lab}: {type(e).__name__} - {e}")
            FIT_LOG.append(f"FAILED EVD {_lab}: {type(e).__name__} - {e}")
            _row['Converged'] = 'No'
        evd_rows.append(_row)
    pd.DataFrame(evd_rows).to_excel(
        _os.path.join(results_path, f"tevar_timing_g3_{today}.xlsx"), index=False)

with open(_os.path.join(results_path, f"model_fit_log_{today}.txt"), 'w') as _f:
    _f.write("\n".join(FIT_LOG))


# main statistical analysis

def categorize_columns(df):
    cat_cols = []
    cont_cols = []

    for column in df.columns:
        if df[column].nunique() == 2:
            cat_cols.append(column)
        elif pd.api.types.is_numeric_dtype(df[column]) and df[column].nunique() > 2:
            cont_cols.append(column)

    return cat_cols, cont_cols

# patient-level pulmonary complication composite: unplanned intubation, VAP, or ARDS
_pulm = [c for c in ['HC_INTUBATION', 'HC_VAPNEUMONIA', 'HC_RESPIRATORY'] if c in df.columns]
if _pulm:
    df['pulmonary_complication'] = (df[_pulm].fillna(0).max(axis=1) > 0).astype(int)

cat_cols, cont_cols = categorize_columns(df)

def remove_strings(original_list, strings_to_remove):
    # Use list comprehension to filter out the strings to remove
    return [item for item in original_list if item not in strings_to_remove]

# remove columns that are not needed or did not pass the quality check
cols_to_remove = ['num_unnamed_vessel_or_unk_inj_per_proc', 'num_endo_repair_proc',
                     'num_fluoro_proc', 'num_xenograft_proc', 'num_synthetic_proc',
                     'num_autogenous_proc', 'num_stent_proc', 'num_embo', 
                     'any_num_unnamed_vessel_or_unk_inj_per_proc_indicator',
                     'any_num_open_proc_indicator', 'any_num_endo_repair_proc_indicator', 
                     'any_num_fluoro_proc_indicator', 'any_num_xenograft_proc_indicator', 
                     'any_num_synthetic_proc_indicator', 'any_num_autogenous_proc_indicator', 
                     'any_num_stent_proc_indicator', 'any_num_embo_indicator',
                     'HC_RETURNOR', 'num_gi_procs', 'minutes_to_first_gi_or', 
                     'num_stomach_inj_per_proc', 'num_duo_inj_per_proc', 
                     'num_small_bowel_inj_per_proc', 'num_large_bowel_inj_per_proc', 
                     'num_rectum_inj_per_proc', 'any_num_stomach_inj_per_proc_indicator', 
                     'any_num_duo_inj_per_proc_indicator', 'any_num_small_bowel_inj_per_proc_indicator', 
                     'any_num_large_bowel_inj_per_proc_indicator', 'any_num_rectum_inj_per_proc_indicator', 
                     'num_or_gi_trips', 'minutes_to_last_gi_or', 'any_num_gi_inj_per_proc',
                     'num_disartic_per_proc', 'any_num_disartic_per_proc_indicator',
                     'num_adrenal_inj_per_proc', 'any_num_adrenal_inj_per_proc_indicator',
                     'replace_desc_thoracic_autologous', 'bypass_aorta_to_mes_art_autologous', 
                     'bypass_aorta_to_cia_eci_autologous', 'bypass_aorta_to_fem_autologous', 
                     'bypass_aorta_to_renal_autologous', 'bypass_aorta_to_low_ext_autologous', 
                     'bypass_aorta_to_cia_eci_synth', 'bypass_aorta_to_mes_art_synth', 
                     'bypass_aorta_to_low_ext_synth', 'patch_abd_aorta_synth', 
                     'replace_desc_thoracic_xeno', 'patch_desc_thoracic_non_autologous', 
                     'bypass_aorta_to_low_ext_non_autologous', 'open_aorta_primary_repair', 
                     'open_aorta_ligation', 'open_aorta_restriction', 'open_aorta_dilation',
                     'any_open_aorta_proc', 'traumatic_open_repair', 'traumatic_hybrid_open_plus_endo_repair']
df = df.drop(columns=cols_to_remove, errors='ignore')
# IVC procedure flags and the grade-overlap check written by earlier versions of step 1 are not
# part of this analysis
df = df.drop(columns=[c for c in df.columns if c.startswith(('ivc_', 'ivcf_')) or c == 'grade_count'])

cat_cols, cont_cols = categorize_columns(df)

# remove strings that are not needed or did not pass the quality check
strings_to_remove = ['any_art_inj_per_dx', 'any_vein_inj_per_dx', 'any_unnamed_vessel_or_unk_inj_per_dx',
                     'art_plus_vein_inj_per_dx', 'only_art_inj_per_dx', 'only_vein_inj_per_dx',
                     'single_art_per_dx', 'any_num_art_inj_per_proc_indicator',
                     'any_num_vein_inj_per_proc_indicator', 'any_num_iliac_art_inj_per_proc_indicator',
                     'any_num_iliac_vein_inj_per_proc_indicator', 'any_num_ivc_inj_per_proc_indicator',
                     'no_angio_per_angio_field' , 'had_angio_without_embo_per_angio_field',
                     'svs_grade_final',
                     'any_tear', 'any_minor', 'any_major', 'any_rupture', 'aortic_inj_thoracic_intimal_tear', 
                     'aortic_inj_thoracic_lac_minor', 'aortic_inj_thoracic_rupture', 
                     'aortic_inj_thoracic_NFS', 'aortic_inj_abd_intimal_tear', 
                     'aortic_inj_abd_lac_minor', 'aortic_inj_abd_rupture', 'aortic_inj_abd_NFS']
cat_cols = remove_strings(cat_cols, strings_to_remove)

strings_to_remove = ['YoAdmit', 'inc_key', 'svs_aortic_grade_final', 'added_vasc_inj_severity',
                     'los_d_dc90', 'iss_nonvasc', 'iss_recomputed', 
                     'highest_aortic_inj_severity', 'highest_vasc_inj_severity',
                     'num_art_inj_per_dx', 'num_art_inj_per_proc', 'num_dx',
                     'num_iliac_art_inj_per_dx', 'num_iliac_art_inj_per_proc', 'num_iliac_vein_inj_per_dx', 
                     'num_iliac_vein_inj_per_proc', 'num_ivc_inj_per_dx', 'num_ivc_inj_per_proc', 
                     'num_knee_amp_per_proc', 'num_liver_inj_per_proc', 'num_open_proc', 
                     'num_or_amp_trips', 'num_or_soi_trips', 'num_procs', 'num_soi_procs', 
                     'num_unnamed_vessel_or_unk_inj_per_dx', 'num_vasc_inj_per_ais', 
                     'num_vein_inj_per_dx', 'num_vein_inj_per_proc']
cont_cols = remove_strings(cont_cols, strings_to_remove)

# check vairable variance across years as a quality check
numeric_cols = df.select_dtypes('number').columns
summary_table = df[numeric_cols].groupby('YoAdmit').agg(['mean'])
summary_table.to_excel(_os.path.join(results_path, f"quality_check_by_year_{today}.xlsx"))

missing_df = percent_missing_series(df, list(df.columns[df.isnull().any()]))
missing_df = pd.DataFrame(list(missing_df.items()), columns=['Attribute', 'Value'])
missing_df.rename(columns={'Attribute': 'Variable', 'Value': '% Missing'}, inplace=True)
missing_df['% Missing'] = missing_df['% Missing'].apply(lambda x: f'{x:.2f}')
missing_df.to_excel(_os.path.join(results_path, f"missing_var_raw_{today}.xlsx"), index=False)

# cohort description


if svs_2026 == True:
    cohort_desc = f'''
    {df.shape[0]} patients had a BAI Grade 1, 2, 3, or 4
    {df.svs_grade_3.value_counts().get(1, 0)} patients had grade 3 BAI
    {df.gr_3_tevar_plus_TBI_or_SOI_2026.value_counts().get(1, 0)} patients had grade 3 BAI plus severe TBI or SOI and had TEVAR during admission
    {df.gr_3_tevar_within_24h_unstable_2026.value_counts().get(1, 0)} patients had grade 3 BAI and hemodynamically unstable or 2+ units PRBC and had TEVAR within 24h
    {df.gr_3_tevar_stable_after_24h_2026.value_counts().get(1, 0)} patients had grade 3 BAI plus TEVAR after 24h (standard care per 2026 guidelines)
    {df.gr_3_no_tevar.value_counts().get(1, 0)} patients had grade 3 BAI with no TEVAR
    {df.svs_grade_4.value_counts().get(1, 0)} patients had grade 4 BAI
    {df.gr_4_tevar_within_6h.value_counts().get(1, 0)} patients had grade 4 BAI with TEVAR within 6h
    {df.gr_4_tevar_after_6h.value_counts().get(1, 0)} patients had grade 4 BAI with TEVAR after 6h
    {df.gr_4_no_tevar.value_counts().get(1, 0)} patients had grade 4 BAI with no TEVAR
    {df.traumatic_endo_repair.value_counts().get(1, 0)} patients underwent endovascular repair of a BAI
    '''
else:
    cohort_desc = f'''
    {df.shape[0]} patients had a BAI Grade 1, 2, 3, or 4
    {df.svs_grade_3.value_counts().get(1, 0)} patients had grade 3 BAI
    {df.gr_3_tevar_within_24h.value_counts().get(1, 0)} patients had grade 3 BAI with TEVAR within 24h
    {df.gr_3_tevar_after_24h.value_counts().get(1, 0)} patients had grade 3 BAI with TEVAR after 24h
    {df.gr_3_severe_tbi_tevar_after_24h.value_counts().get(1, 0)} patients had grade 3 BAI plus severe TBI with TEVAR after 24h
    {df.gr_3_no_tevar.value_counts().get(1, 0)} patients had grade 3 BAI with no TEVAR
    {df.svs_grade_4.value_counts().get(1, 0)} patients had grade 4 BAI
    {df.gr_4_tevar_within_6h.value_counts().get(1, 0)} patients had grade 4 BAI with TEVAR within 6h
    {df.gr_4_tevar_after_6h.value_counts().get(1, 0)} patients had grade 4 BAI with TEVAR after 6h
    {df.gr_4_severe_tbi_tevar_after_6h.value_counts().get(1, 0)} patients had grade 4 BAI plus severe TBI with TEVAR after 6h
    {df.gr_4_no_tevar.value_counts().get(1, 0)} patients had grade 4 BAI with no TEVAR
    {df.traumatic_endo_repair.value_counts().get(1, 0)} patients underwent endovascular repair of a BAI
    '''

print(cohort_desc)

txt_path = _os.path.join(results_path, f"dataset_description_{today}.txt")

with open(txt_path, 'w') as file:
    file.write(cohort_desc)

# figures

# overall analysis


if svs_2026 == True:
    level_column = 'guideline_concordant_care_2026'
else:
    level_column = 'guideline_concordant_care_2011'

# whole-cohort comparisons restricted to grade III-IV injuries: grade I/II patients
# (for whom nonoperative management is recommended) are not part of the comparator
df_34 = df[df['svs_grade_final'].isin([3, 4])]

# cat var
cat_var = raw_fisher_crosstab(input_df=df_34.copy(), level_column=level_column, column_list=cat_cols, abbrev_delimeter='_')
cat_var.rename(columns={0: f'{level_column} = no, N={df_34[level_column].value_counts()[0]}', 1: f'{level_column} = yes, N={df_34[level_column].value_counts()[1]}', cat_var.columns[3]: 'p value'}, inplace=True)
new_order_indices = [0, 2, 1, 3]
cat_var = cat_var.iloc[:, new_order_indices]
add_bh_q(cat_var).to_excel(_os.path.join(results_path, f"cat_var_{today}.xlsx"), index=False)

# grade 3 only
df_3 = df[df['svs_grade_3'] == 1]

cat_var = raw_fisher_crosstab(input_df=df_3.copy(), level_column=level_column, column_list=cat_cols, abbrev_delimeter='_')
cat_var.rename(columns={0: f'{level_column} = no, N={df_3[level_column].value_counts()[0]}', 1: f'{level_column} = yes, N={df_3[level_column].value_counts()[1]}', cat_var.columns[3]: 'p value'}, inplace=True)
new_order_indices = [0, 2, 1, 3]
cat_var = cat_var.iloc[:, new_order_indices]
add_bh_q(cat_var).to_excel(_os.path.join(results_path, f"grade_3_cat_var_{today}.xlsx"), index=False)

# grade 4 only
df_4 = df[df['svs_grade_4'] == 1]

cat_var = raw_fisher_crosstab(input_df=df_4.copy(), level_column=level_column, column_list=cat_cols, abbrev_delimeter='_')
cat_var.rename(columns={0: f'{level_column} = no, N={df_4[level_column].value_counts()[0]}', 1: f'{level_column} = yes, N={df_4[level_column].value_counts()[1]}', cat_var.columns[3]: 'p value'}, inplace=True)
new_order_indices = [0, 2, 1, 3]
cat_var = cat_var.iloc[:, new_order_indices]
add_bh_q(cat_var).to_excel(_os.path.join(results_path, f"grade_4_cat_var_{today}.xlsx"), index=False)

# cont var
cont_var = raw_kruskal_wallace(input_df=df_34.copy(), column_list=cont_cols, level_column=level_column, abbrev_delimeter='_')
cont_var.rename(columns={0: f'{level_column} = no, N={df_34[level_column].value_counts()[0]}', 1: f'{level_column} = yes, N={df_34[level_column].value_counts()[1]}', cont_var.columns[3]: 'p value'}, inplace=True)
new_order_indices = [0, 2, 1, 3]
cont_var = cont_var.iloc[:, new_order_indices]
add_bh_q(cont_var).to_excel(_os.path.join(results_path, f"cont_var_{today}.xlsx"), index=False)

# grade 3 only
cont_var = raw_kruskal_wallace(input_df=df_3.copy(), column_list=cont_cols, level_column=level_column, abbrev_delimeter='_')
cont_var.rename(columns={0: f'{level_column} = no, N={df_3[level_column].value_counts()[0]}', 1: f'{level_column} = yes, N={df_3[level_column].value_counts()[1]}', cont_var.columns[3]: 'p value'}, inplace=True)
new_order_indices = [0, 2, 1, 3]
cont_var = cont_var.iloc[:, new_order_indices]
add_bh_q(cont_var).to_excel(_os.path.join(results_path, f"grade_3_cont_var_{today}.xlsx"), index=False)

# grade 4 only
cont_var = raw_kruskal_wallace(input_df=df_4.copy(), column_list=cont_cols, level_column=level_column, abbrev_delimeter='_')
cont_var.rename(columns={0: f'{level_column} = no, N={df_4[level_column].value_counts()[0]}', 1: f'{level_column} = yes, N={df_4[level_column].value_counts()[1]}', cont_var.columns[3]: 'p value'}, inplace=True)
new_order_indices = [0, 2, 1, 3]
cont_var = cont_var.iloc[:, new_order_indices]
add_bh_q(cont_var).to_excel(_os.path.join(results_path, f"grade_4_cont_var_{today}.xlsx"), index=False)

# subgroup analysis: TEVAR vs. no TEVAR (all blunt subjects)

tevar_lc = 'traumatic_endo_repair'

cat_var = raw_fisher_crosstab(input_df=df.copy(), level_column=tevar_lc,
                              column_list=cat_cols, abbrev_delimeter='_')
cat_var.rename(columns={0: f'No TEVAR, N={df[tevar_lc].value_counts().get(0, 0)}',
                        1: f'TEVAR, N={df[tevar_lc].value_counts().get(1, 0)}',
                        cat_var.columns[3]: 'p value'}, inplace=True)
new_order_indices = [0, 2, 1, 3]
cat_var = cat_var.iloc[:, new_order_indices]
add_bh_q(cat_var).to_excel(_os.path.join(results_path, f"tevar_vs_no_tevar_cat_var_{today}.xlsx"), index=False)

cont_var = raw_kruskal_wallace(input_df=df.copy(), column_list=cont_cols,
                               level_column=tevar_lc, abbrev_delimeter='_')
cont_var.rename(columns={0: f'No TEVAR, N={df[tevar_lc].value_counts().get(0, 0)}',
                         1: f'TEVAR, N={df[tevar_lc].value_counts().get(1, 0)}',
                         cont_var.columns[3]: 'p value'}, inplace=True)
cont_var = cont_var.iloc[:, new_order_indices]
add_bh_q(cont_var).to_excel(_os.path.join(results_path, f"tevar_vs_no_tevar_cont_var_{today}.xlsx"), index=False)


# subgroup analysis: TEVAR + death within 7 days of TEVAR vs. TEVAR + survived >= 7 days post-TEVAR

df_tevar = df[df['traumatic_endo_repair'] == 1].copy()

# time from TEVAR to end of stay (days)
df_tevar['days_tevar_to_dc'] = df_tevar['los_d'] - (df_tevar['time_to_first_vasc_proc'] / 1440.0)

# drop rows where we can't compute the gap (missing TEVAR timing or missing LOS)
n_before = len(df_tevar)
df_tevar = df_tevar.dropna(subset=['days_tevar_to_dc'])
n_dropped = n_before - len(df_tevar)
if n_dropped:
    print(f"TEVAR subgroup: dropped {n_dropped} patients with missing TEVAR timing/LOS")

# early mortality = died/hospice AND end of stay within 7 days of TEVAR
# comparator = everyone else with TEVAR (alive at day 7 post-TEVAR, regardless of later events)
df_tevar['mortality_within_7d_of_tevar'] = (
    (df_tevar['dc_dispo_death_or_hospice'] == 1) &
    (df_tevar['days_tevar_to_dc'] <= 7)
).astype(int)

early_mort_lc = 'mortality_within_7d_of_tevar'

cat_var = raw_fisher_crosstab(input_df=df_tevar.copy(), level_column=early_mort_lc,
                              column_list=cat_cols, abbrev_delimeter='_')
cat_var.rename(columns={0: f'TEVAR, survived 7+ days, N={df_tevar[early_mort_lc].value_counts().get(0, 0)}',
                        1: f'TEVAR, death within 7 days, N={df_tevar[early_mort_lc].value_counts().get(1, 0)}',
                        cat_var.columns[3]: 'p value'}, inplace=True)
cat_var = cat_var.iloc[:, new_order_indices]
add_bh_q(cat_var).to_excel(_os.path.join(results_path, f"tevar_early_mortality_cat_var_{today}.xlsx"), index=False)

cont_var = raw_kruskal_wallace(input_df=df_tevar.copy(), column_list=cont_cols,
                               level_column=early_mort_lc, abbrev_delimeter='_')
cont_var.rename(columns={0: f'TEVAR, survived 7+ days, N={df_tevar[early_mort_lc].value_counts().get(0, 0)}',
                         1: f'TEVAR, death within 7 days, N={df_tevar[early_mort_lc].value_counts().get(1, 0)}',
                         cont_var.columns[3]: 'p value'}, inplace=True)
cont_var = cont_var.iloc[:, new_order_indices]
add_bh_q(cont_var).to_excel(_os.path.join(results_path, f"tevar_early_mortality_cont_var_{today}.xlsx"), index=False)