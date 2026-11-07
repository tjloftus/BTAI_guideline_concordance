# -*- coding: utf-8 -*-
"""
Local paths. Edit these, or set the environment variables, before running.

NTDB participant use files are licensed by the American College of Surgeons and
are not distributed with this repository.
"""

import os

# step 1 input: one row per patient with an arterial or venous injury, built from the
# NTDB participant use files (2017-2024) by the shared NTDB processing pipeline
VASC_DF_PATH = os.environ.get('BTAI_VASC_DF_PATH', r"path\to\vasc_df_all_2017_2024.csv")

# step 1 output
STEP_1_PATH = os.environ.get('BTAI_STEP_1_PATH', r"path\to\aorta_df_2017_2024_step_1_complete.csv")

# analysis input: the step 1 file after merging head and solid organ AIS and
# non-vascular ISS (see README, "Data preparation")
DATA_PATH = os.environ.get('BTAI_DATA_PATH', r"path\to\aorta_df_2017_2024_step_2_complete.csv")

# results are written to <RESULTS_ROOT>\{2011,2026} guidelines\[sensitivity\]analysis_performed_<date>
RESULTS_ROOT = os.environ.get('BTAI_RESULTS_ROOT', r"path\to\results")

# a local copy of DATA_PATH is cached here when reading from a network share
LOCAL_CACHE = os.environ.get('BTAI_LOCAL_CACHE', r"C:\NTDB temp\cache")
