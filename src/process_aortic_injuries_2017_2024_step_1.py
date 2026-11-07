# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 07:03:19 2024

@author: tloftus
"""

import os
import re
import shutil
import time

import numpy as np
import pandas as pd

from config import VASC_DF_PATH, STEP_1_PATH, LOCAL_CACHE
from stats_utils import fill_null_with_zero


def read_csv_robust(path, retries=3, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except (FileNotFoundError, OSError) as e:
        print(f"[read_csv_robust] direct read of {path} failed ({e}); copying to {LOCAL_CACHE}")
    os.makedirs(LOCAL_CACHE, exist_ok=True)
    parent = os.path.basename(os.path.dirname(path))
    local = os.path.join(LOCAL_CACHE, f"{parent}_{os.path.basename(path)}")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            shutil.copyfile(path, local)
            return pd.read_csv(local, **kwargs)
        except (FileNotFoundError, OSError) as e:
            last_err = e
            print(f"  [read_csv_robust] attempt {attempt}/{retries} failed: {e}")
            time.sleep(10)
    raise last_err


def universal_aortic_extractor(val):
    val_clean = str(val).replace('[', '').replace(']', '').replace("'", "").replace('"', '')
    matches = re.findall(r'(4202\d{2}|5202\d{2})', val_clean)

    if not matches:
        return [0, 0, 0, 0, 0]

    max_sev = 0
    tear, minor, major, rupture = 0, 0, 0, 0

    for code in matches:
        if code in ['420202', '420204', '520202']:
            s, tear = 2, 1
        elif code in ['420206', '520204', '420299', '520299']:
            s, minor = 3, 1
        elif code in ['420208', '520206']:
            s, major = 4, 1
        elif code in ['420210', '420212', '420216', '420218', '520208']:
            s, rupture = 5, 1
        else:
            s = 0

        if s > max_sev:
            max_sev = s

    return [max_sev, tear, minor, major, rupture]


def main():
    df = read_csv_robust(VASC_DF_PATH)

    df['teaching_status_non_university'] = (df['teaching_status_university'] == 0).astype(int)
    del df['teaching_status_nonteaching']
    del df['teaching_status_other_or_unknown']

    columns_to_check = ['first_surg_type_exlap', 'first_surg_type_extremity',
                        'first_surg_type_neck', 'first_surg_type_preperitoneal_pelvic_pack',
                        'first_surg_type_skin_soft_tiss', 'first_surg_type_sternotomy',
                        'first_surg_type_thoracotomy']

    all_zero_mask = df[columns_to_check].eq(0).all(axis=1)
    df.loc[all_zero_mask, 'first_surg_type_unk'] = 1

    cols_to_remove = ['first_surg_type_none', 'LOSMins', 'LOSDays',
                      'WITHDRAWALLSTDays', 'WITHDRAWALLSTMins', 'VTEPROPHYLAXISTYPE',
                      'acs_verification_peds_level_one', 'acs_verification_peds_level_two',
                      'state_verification_peds_level_unk', 'acs_verification_peds_level_unk']
    df = df.drop(columns=cols_to_remove)

    df['any_iliac_artery_injury'] = ((df['any_iliac_art_inj_per_dx'] == 1) | (df['any_num_iliac_art_inj_per_proc_indicator'] == 1)).astype(int)
    df['any_iliac_vein_injury'] = ((df['any_iliac_vein_inj_per_dx'] == 1) | (df['any_num_iliac_vein_inj_per_proc_indicator'] == 1)).astype(int)
    df['any_ivc_injury'] = ((df['any_ivc_inj_per_dx'] == 1) | (df['any_num_ivc_inj_per_proc_indicator'] == 1)).astype(int)
    df['any_iliac_vein_or_ivc_injury'] = ((df['any_iliac_vein_injury'] == 1) | (df['any_ivc_injury'] == 1)).astype(int)

    del df['any_iliac_art_inj_per_dx']
    del df['any_iliac_vein_inj_per_dx']
    del df['any_ivc_inj_per_dx']

    df['age_16_plus'] = (df['AGEYEARS'] >= 16).astype(int)
    df['age_18_plus'] = (df['AGEYEARS'] >= 18).astype(int)

    # transfusion outcomes
    df['transfusion_total_vol_4h'] = df['BLOOD4HOURS'] + df['PLASMA4HOURS'] + df['PLATELETS4HOURS'] + df['CRYOPRECIPITATE4HOURS']
    # assume that there are avg 300 ml per 1 U, so mtp is 3L blood products (or 10 units)
    df['mtp_vol_within_4h'] = np.where(df['transfusion_total_vol_4h'] >= 3000, 1, 0)

    # make a dispo == death or hospice outcome
    df['dc_dispo_death_or_hospice'] = (
        (df[['withdrawal', 'dc_dispo_death', 'dc_dispo_hospice']] == 1)
        .any(axis=1)
        .astype(int))

    # ml per unit accoring to:
    # Postpartum Hemorrhage.
    # Bienstock JL, Eke AC, Hueppchen NA. NEJM. 2021;384(17):1635-1645. doi:10.1056/NEJMra1513247.
    # Society of Interventional Radiology Consensus Guidelines for the Periprocedural Management of Thrombotic and Bleeding Risk in Patients Undergoing Percutaneous Image-Guided Interventions-Part II: Recommendations: Endorsed by the Canadian Association for Interventional Radiology and the Cardiovascular and Interventional Radiological Society of Europe.
    # Patel IJ, Rahim S, Davidson JC, et al. Journal of Vascular and Interventional Radiology : JVIR. 2019;30(8):1168-1184.e1. doi:10.1016/j.jvir.2019.04.017.
    df['units_rbc_4h'] = df['BLOOD4HOURS'] / 300
    df['units_plasma_4h'] = df['PLASMA4HOURS'] / 250
    df['units_platelets_4h'] = df['PLATELETS4HOURS'] / 275
    df['units_cryo_4h'] = df['CRYOPRECIPITATE4HOURS'] / 150

    df['total_units_4h'] = df['units_rbc_4h'] + df['units_plasma_4h'] + df['units_platelets_4h'] + df['units_cryo_4h']
    df['mtp_unit_count_within_4h'] = np.where(df['total_units_4h'] >= 10, 1, 0)
    df['mtp_by_vol_or_unit_count'] = np.where((df['mtp_vol_within_4h'] == 1) | (df['mtp_unit_count_within_4h'] == 1), 1, 0)

    # aorta procs
    results_list = df['AISPREDOT'].apply(universal_aortic_extractor).tolist()

    cols = ['highest_aortic_inj_severity', 'any_tear', 'any_minor', 'any_major', 'any_rupture']
    extracted_df = pd.DataFrame(results_list, columns=cols, index=df.index)

    df = df.drop(columns=cols, errors='ignore')
    df = pd.concat([df, extracted_df], axis=1)

    print(df['highest_aortic_inj_severity'].value_counts())

    # thoracic aortic injuries (AIS 4202xx)
    df['aortic_inj_thoracic_intimal_tear'] = df['AISPREDOT'].str.contains('420202|420204', na=False).astype(int)
    df['aortic_inj_thoracic_lac_minor'] = df['AISPREDOT'].str.contains('420206|420208', na=False).astype(int)
    df['aortic_inj_thoracic_rupture'] = df['AISPREDOT'].str.contains('420210|420212|420216|420218', na=False).astype(int)
    df['aortic_inj_thoracic_NFS'] = df['AISPREDOT'].str.contains('420299', na=False).astype(int)

    # abdominal aortic injuries (AIS 5202xx)
    df['aortic_inj_abd_intimal_tear'] = df['AISPREDOT'].str.contains('520202', na=False).astype(int)
    df['aortic_inj_abd_lac_minor'] = df['AISPREDOT'].str.contains('520204|520206', na=False).astype(int)
    df['aortic_inj_abd_rupture'] = df['AISPREDOT'].str.contains('520208', na=False).astype(int)
    df['aortic_inj_abd_NFS'] = df['AISPREDOT'].str.contains('520299', na=False).astype(int)

    df['has_aortic_injury'] = (
        df['dx'].str.contains('S25.0|S35.0', na=False) |
        df['AISPREDOT'].str.contains('4202|5202', na=False)
    ).astype(int)

    # thoracic-only flag for the isolated-abdominal exclusion sensitivity analysis
    # downstream, `has_aortic_injury` includes abdominal (S35.0 / AIS 5202*), which
    # the SVS BAI guidelines were not designed to govern
    df['has_thoracic_aortic_injury'] = (
        df['dx'].str.contains('S25.0', na=False) |
        df['AISPREDOT'].str.contains('4202', na=False)
    ).astype(int)

    # open

    # autologous (excluding arch 'X', focusing on descending 'W')
    df['replace_desc_thoracic_autologous'] = df['procs'].str.contains('02RW07Z', na=False).astype(int)
    df['bypass_aorta_to_mes_art_autologous'] = df['procs'].str.contains('0410092|04100A2', na=False).astype(int)
    df['bypass_aorta_to_cia_eci_autologous'] = df['procs'].str.contains('0410096|041009D|04100AC|0410097|041009E|04100AE', na=False).astype(int)
    df['bypass_aorta_to_fem_autologous'] = df['procs'].str.contains('041009[HJK]|04100A[HJK]', na=False).astype(int)
    df['bypass_aorta_to_renal_autologous'] = df['procs'].str.contains('041009[345]|04100A[345]', na=False).astype(int)
    df['bypass_aorta_to_low_ext_autologous'] = df['procs'].str.contains('041009[QR]|04100A[QR]|021W09V|021W0AV', na=False).astype(int)

    # synthetic / non-autologous / xeno
    df['bypass_aorta_to_cia_eci_synth'] = df['procs'].str.contains('04100J9|04100JF|04100J7|04100J[68CDE]', na=False).astype(int)
    df['bypass_aorta_to_mes_art_synth'] = df['procs'].str.contains('04100J2', na=False).astype(int)
    df['bypass_aorta_to_low_ext_synth'] = df['procs'].str.contains('04100J[QR]|021W0JV', na=False).astype(int)
    df['patch_abd_aorta_synth'] = df['procs'].str.contains('04U00JZ', na=False).astype(int)
    df['replace_desc_thoracic_xeno'] = df['procs'].str.contains('02RW08Z', na=False).astype(int)
    df['patch_desc_thoracic_non_autologous'] = df['procs'].str.contains('02UW0KZ|021W0KR', na=False).astype(int)
    df['bypass_aorta_to_low_ext_non_autologous'] = df['procs'].str.contains('04100K[QR]|021W0KV', na=False).astype(int)

    # primary repair, ligation, restriction, dilation (strictly descending/abdominal)
    df['open_aorta_primary_repair'] = df['procs'].str.contains('02QW0ZZ|04Q00ZZ', na=False).astype(int)
    df['open_aorta_ligation'] = df['procs'].str.contains('02LW0[ZC]Z|04L00[ZC]Z', na=False).astype(int)
    df['open_aorta_restriction'] = df['procs'].str.contains('02VW0ZZ|04V00ZZ', na=False).astype(int)
    df['open_aorta_dilation'] = df['procs'].str.contains('027W0ZZ|04700ZZ', na=False).astype(int)

    # endo

    # character 6 [DJK] ensures a graft/stent was actually placed
    df['any_endo_aorta_proc'] = df['procs'].str.contains(r'(02[UV]W[34][DJK])|(04[UV]0[34][DJK])', na=False).astype(int)
    # TEVAR = endograft in the descending thoracic aorta only (02[UV]W...), abdominal aortic
    # endografts (04[UV]0...) are kept in any_endo_aorta_proc but are NOT counted as TEVAR
    df['thoracic_endo_aorta_proc'] = df['procs'].str.contains(r'02[UV]W[34][DJK]', na=False).astype(int)

    open_flags = [
        'replace_desc_thoracic_autologous', 'bypass_aorta_to_mes_art_autologous',
        'bypass_aorta_to_cia_eci_autologous', 'bypass_aorta_to_fem_autologous',
        'bypass_aorta_to_renal_autologous', 'bypass_aorta_to_low_ext_autologous',
        'bypass_aorta_to_cia_eci_synth', 'bypass_aorta_to_mes_art_synth',
        'bypass_aorta_to_low_ext_synth', 'patch_abd_aorta_synth',
        'replace_desc_thoracic_xeno', 'patch_desc_thoracic_non_autologous',
        'bypass_aorta_to_low_ext_non_autologous', 'open_aorta_primary_repair',
        'open_aorta_ligation', 'open_aorta_restriction', 'open_aorta_dilation'
    ]

    df['any_open_aorta_proc'] = df[open_flags].any(axis=1).astype(int)

    df['traumatic_open_repair'] = ((df['any_open_aorta_proc'] == 1) & (df['has_aortic_injury'] == 1)).astype(int)
    df['traumatic_endo_repair'] = ((df['thoracic_endo_aorta_proc'] == 1) & (df['has_aortic_injury'] == 1)).astype(int)
    df['traumatic_hybrid_open_plus_endo_repair'] = ((df['traumatic_open_repair'] == 1) & (df['traumatic_endo_repair'] == 1)).astype(int)
    df['aorta_inj_no_procs'] = np.where((df['traumatic_open_repair'] == 0) & (df['traumatic_endo_repair'] == 0) & (df['has_aortic_injury'] == 1), 1, 0)

    df = df[df['has_aortic_injury'] == 1]

    # add SVS grades (1: intimal tear, 2: intramural hematoma, 3: pseudoaneyrusm, 4: free rupture)
    any_tear = (df['aortic_inj_thoracic_intimal_tear'] == 1) | (df['aortic_inj_abd_intimal_tear'] == 1)
    any_minor = (df['aortic_inj_thoracic_lac_minor'] == 1) | (df['aortic_inj_abd_lac_minor'] == 1)
    any_nfs = (df['aortic_inj_thoracic_NFS'] == 1) | (df['aortic_inj_abd_NFS'] == 1)
    any_rupture = (df['aortic_inj_thoracic_rupture'] == 1) | (df['aortic_inj_abd_rupture'] == 1)

    # logic for differentiating SVS grade 2 vs 3
    is_minor_or_nfs = (any_minor | any_nfs)
    # AIS severity 1=minor, 2=moderate, 3=serious, 4=severe, 5=critical, 6=untreatable or fatal
    ais_score = df['highest_vasc_inj_severity']

    # an AIS 3 (Serious) vascular injury is often still hemodynamically stable and non-operative
    # an AIS 4 (Severe) usually implies a structural defect that requires intervention
    # placing AIS 3 into SVS grade 2 aligns better with the clinical management threshold
    conditions = [
        (any_rupture),                                # grade 4
        (is_minor_or_nfs & (ais_score >= 4)),         # grade 3 (AIS severe)
        (is_minor_or_nfs & (ais_score <= 3)),         # grade 2 (AIS less than severe)
        (any_tear)                                    # grade 1
    ]

    # the SVS Grade value (0 if no conditions met)
    df['svs_grade_final'] = np.select(conditions, [4, 3, 2, 1], default=0)

    # expand into binary columns, which are mutually exclusive
    for i in range(1, 5):
        df[f'svs_grade_{i}'] = (df['svs_grade_final'] == i).astype(int)

    # add shock index
    df['sbp_less_than_90'] = np.where(df['SBP'] < 90, 1, 0)
    df['shock_index'] = df['PULSERATE'] / df['SBP']
    # shock index > 1 is consistent with hemorrhagic shock and associated with outcomes similar to SBP < 90
    # is included in “Red Criteria" in the National Field Triage Guidelines, indicating high risk of serious injury and need for trauma center care, similar to SBP <90 mmHg
    df['shock_index_greater_than_1'] = np.where(df['shock_index'] > 1, 1, 0)
    df['shock_per_sbp_or_shock_index'] = np.where((df['sbp_less_than_90'] == 1) | (df['shock_index_greater_than_1'] == 1), 1, 0)

    # fill null values with 0
    fill_null_cols = ['TOTALICULOS', 'TOTALVENTDAYS', 'BLOOD4HOURS',
                      'PLASMA4HOURS', 'PLATELETS4HOURS', 'CRYOPRECIPITATE4HOURS',
                      'num_iliac_vein_inj_per_dx', 'num_ivc_inj_per_dx',
                      'num_or_vasc_trips', 'transfusion_total_vol_4h', 'units_rbc_4h',
                      'units_plasma_4h', 'units_platelets_4h', 'units_cryo_4h', 'total_units_4h']

    df = fill_null_with_zero(input_df=df, input_col_list=fill_null_cols)

    df.to_csv(STEP_1_PATH, index=False)


if __name__ == '__main__':
    main()
