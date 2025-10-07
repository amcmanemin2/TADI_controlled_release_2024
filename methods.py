# Script methods for analysis
# Author: Audrey McManemin
# Date Created: 2024-12-16
# Date Last Modified: 2024-12-16

# Any general function used throughout in any of the notebooks. However, the following are explicitly NOT included:  
# > Any file that specifically outputs a figure will be saved in plot_methods
# > Methods for specifically cleaning the operator reports 

# Imports
import pathlib
import pandas as pd
import numpy as np
import datetime
import math
from scipy.stats import circmean, circstd
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta, time

from mobile_solutions.methods_source import make_logistic_regression

OPERATOR_LIST = ['Aeromon', 
                 'GSMA', 
                 'Flylogix',
                 'SeekOps',
                 'DTU',
                 'UHEI',
                 'UU_LSCE_CYI_RHUL',
                 'EMPA',
                 'FAAM',
                 'GHGSat',
                 'Sensirion',
                 'SLB',
                 'Sensia'
                 ]
OPERATOR_WEEK = {'aeromon': 1, 
                 'gsma': 1,
                 'dtu': 1,
                 'seekops': 2,
                 'uhei': 2, 
                 'flylogix': 4,
                 'ghgsat': 1234,
                 'empa': 4,
                 'uu_lsce_cyi_rhul': 4,
                 'faam': 3,
                 'sensirion': 1234,
                 'slb': 34,
                 'sensia': 34
                 }

OPERATOR_FOLDER = {'aeromon': 'mobile_solutions', 
                 'gsma': 'mobile_solutions',
                 'dtu': 'mobile_solutions',
                 'seekops': 'mobile_solutions',
                 'uhei': 'mobile_solutions', 
                 'flylogix': 'mobile_solutions',
                 'ghgsat': 'mobile_solutions',
                 'empa': 'mobile_solutions',
                 'uu_lsce_cyi_rhul': 'mobile_solutions',
                 'sensirion': 'continuous_monitors',
                 'slb': 'continuous_monitors',
                 'sensia': 'continuous_monitors',
                 'faam': 'mobile_solutions',
                 }
OPERATOR_TYPE = {'Drone': ['aeromon', 'gsma', 'seekops', 'flylogix'],
                 'Aircraft and Satellite': ['faam', 'empa', 'ghgsat'],
                 'Vehicle': ['uu_lsce_cyi_rhul', 'uhei', 'dtu'],
                 'Continuous Monitor': ['sensirion', 'slb', 'sensia']
                 }

COMMERCIAL_OPERATOR_LIST = ['Aeromon',
                            'GSMA', 
                            'Flylogix',
                            'SeekOps',
                            'GHGSat',
                            'Sensirion',
                            'SLB',
                            'Sensia'
                            ]
ACADEMIC_OPERATOR_LIST = [ 'DTU',
                            'UHEI',
                            'UU_LSCE_CYI_RHUL',
                            'EMPA',
                            'FAAM',
]

RELEASE_NUMBER_LIST = {
    1: 40,
    2: 35,
    3: 37,
    4: 40
}

# %% Load release data for a given week 
def load_release_data(week):
    path = pathlib.PurePath('release_data/clean_release_data', f'W{week}_clean.csv')
    df = pd.read_csv(path, index_col=0)
    return df 

# %% Load release report for an operator

def load_release_summary(operator):
    op_ab = operator.lower()
    folder = OPERATOR_FOLDER[op_ab]
    path = pathlib.PurePath(f'{folder}', '03_results', 'release_summary', f'{operator}_releases.csv')
    return pd.read_csv(path, index_col=0)


# %% Load clean data for an operator

def load_clean_operator_report(operator, discard_level=None):
    op_ab = operator.lower()
    folder = OPERATOR_FOLDER[op_ab]
    if discard_level is None:
        path = pathlib.PurePath(f'{folder}', '01_clean_reports', f'{operator}_clean.csv')
    else: 
        path = pathlib.PurePath(f'{folder}', '01_clean_reports', f'{operator}_clean_{discard_level}.csv')
    return pd.read_csv(path, index_col=0)

# %% Generate release summary for each operator

def generate_release_summary(operator, discard_level=None):
    
    """Generate clean dataframe for all releases with columns indicating QC status.

    Inputs:
      - operator: operator name, spelled out in full

    Columns are:
        - week_id: ID for week of experiments (1, 2, 3, 4)
        - release_id: this is the ID for the specific release during that week, and matches ReleaseID number in the clean operator data
        - release_date: date of release, in local (UTC+2) time. 
        - zero_release: True if the release by Stanford is 0 kgh, False if greater than 0 kgh
        - non_zero_release: True if release by Stanford is greater than 0 kgh, False if equal to 0 kgh
        - operator_kept: True if this release passed the operator QC criteria
        - stanford_kept: True if this release passed Stanford's QC criteria
        - operator_detected: True if operator detected a release. False if they did not
        - operator_quantification: operator's quantification estimate as reported in operator report
        - operator_lower: lower bound on operator's lower bound quantification estimate
        - operator_upper: upper bound on operator's quantification estimate
        - pass_all_qc: True if passed both operator and Stanford QC
        - fail_all_qc: True if this overpass failed both operator and Stanford QC
        - qc_summary: summarizes results of both operator and Stanford QC. Must be one of the following: 'pass_all', 'fail_stanford', 'fail_operator', 'fail_all'
        - release_rate_kgh: TADI reported release rate in kg/h
        - operator_detected: True if operator detected a release. False if they did not
        - windspeed_20m_avg: average windspeed for each release in m/s from the TADI wind lidar at 20m 
        - windspeed_20m_std: standard deviation of each windspeed measurement for each release in m/s of the TADI wind lidar at 20m height
        - winddir_20m_avg: average windspeed for each release in m/s from the TADI wind lidar at 20m 
        - winddir_20m_std: standard deviation of each windspeed measurement for each release in m/s of the TADI wind lidar at 20m height
        - winddir_20m_cv: coefficient of variation for each release from the TADI wind lidar at 20m 
        - operator_windspeed: operator reported windspeed (not required)
        - true_positive: 1 if the operator quantification and true release rate are both > 0 
        - false_positive: 1 if the operator quantification > 0 and true release rate = 0
        - true_negative: 1 if the operator quantification and true release rate are both = 0 
        - false_negative: 1 if the operator quantification = 0 and true release rate > 0
        - qc_flag: operator reported QC flag for each release
        """
    op_ab = operator.lower()

    # Load operator report
    operator_report = load_clean_operator_report(op_ab, discard_level)

    # Load release data
    week = OPERATOR_WEEK[op_ab]
    release_data = load_release_data(week)

    # Combine operator report and meter data
    combined_df = operator_report.merge(release_data, on=['ReleaseID', 'Week'])
    
    # Rename columns to be machine-readable
    combined_df.rename(columns={'Flowrate (kg/h)': 'release_rate_kgh'}, inplace=True)

    # Make dataframe with all relevant info
    release_summary = pd.DataFrame()
    release_summary['week'] = combined_df.Week
    release_summary['release_id'] = combined_df.ReleaseID
    release_summary['release_date'] = combined_df.Date

    # Release info
    release_summary['zero_release'] = combined_df.release_rate_kgh == 0
    release_summary['non_zero_release'] = combined_df.release_rate_kgh != 0  # True if we conducted a release
    release_summary['operator_kept'] = combined_df.OperatorKeep
    release_summary['stanford_kept'] = combined_df.StanfordKeep
    release_summary['strict_qc_kept'] = combined_df.StrictQCKeep
    release_summary['pass_all_qc'] = release_summary.stanford_kept & release_summary.operator_kept
    release_summary['fail_all_qc'] = release_summary['operator_kept'] & release_summary['stanford_kept']

    # Meter data
    release_summary['release_rate_kgh'] = combined_df.release_rate_kgh

    # Load operator data
    release_summary['operator_quantification'] = combined_df.EstimatedEmissionRate
    release_summary['operator_lower'] = combined_df.EstimatedEmissionRateLower
    release_summary['operator_upper'] = combined_df.EstimatedEmissionRateUpper
    release_summary['operator_detected'] = combined_df.EstimatedEmissionRate > 0

    # Summarize QC results
    # Here is the list of different QC options based on the current QC boolean columns
    qc_conditions = [
        release_summary['pass_all_qc'] == 1,  # pass all
        release_summary['fail_all_qc'] == 1,  # fail all
        (release_summary['stanford_kept'] == 1) & (release_summary['operator_kept'] == 0),  # fail_operator
        (release_summary['stanford_kept'] == 0) & (release_summary['operator_kept'] == 1)  # stanford_fail
    ]

    # Based on the above conditions, the final QC evalation will be one of the following:
    qc_choices = [
        'pass_all',
        'fail_all',
        'fail_operator',
        'fail_stanford',
    ]

    # Apply the conditions to generate a new column for 'qc_summary'
    release_summary['qc_summary'] = np.select(qc_conditions, qc_choices, 'not_measured')
    
    # Wind statistics for each release
    release_summary['windspeed_20m_avg'] = combined_df.Windspeed20m_Avg
    release_summary['windspeed_20m_std'] = combined_df.Windspeed20m_Std
    release_summary['winddir_20m_avg'] = combined_df.WindDir20m_Avg
    release_summary['winddir_20m_std'] = combined_df.WindDir20m_Std
    release_summary['winddir_20m_cv'] = combined_df.WindDir20m_CV
    release_summary['operator_windspeed'] = combined_df.OperatorWindspeed
    
    # TP, FP, TN, FN characterization
    #     - true_positive: 1 if the operator quantification and true release rate are both > 0 
    #     - false_positive: 1 if the operator quantification > 0 and true release rate = 0
    #     - true_negative: 1 if the operator quantification and true release rate are both = 0 
    #     - false_negative: 1 if the operator quantification = 0 and true release rate > 0
    
    release_summary['true_positive'] = (release_summary.operator_quantification > 0) & release_summary.non_zero_release
    release_summary['false_positive'] = (release_summary.operator_quantification > 0) & release_summary.zero_release
    release_summary['true_negative'] = (release_summary.operator_quantification == 0) & release_summary.zero_release
    release_summary['false_negative'] = (release_summary.operator_quantification == 0) & release_summary.non_zero_release
    
    release_summary['qc_flag'] = combined_df.QCFLag
    ############# Save Data #############
    folder = OPERATOR_FOLDER[op_ab]
    save_path = pathlib.PurePath(f'{folder}', '03_results', 'release_summary', f'{op_ab}_releases.csv')
    release_summary.to_csv(save_path)

    return release_summary  