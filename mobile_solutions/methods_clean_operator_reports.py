# Script for cleaning each of the operator data
# Note: all scripts in this file are meant to be run directly on the operator loaded data
# Author: Audrey McManemin
# Edited from code writted by Sahar H. El Abbadi
# Date Created: 2024-09-09
# Date Last Modified: 2024-10-03

import pathlib

# Methods in this file:
# > clean_aeromon: Clean Aeromon data reports
# > clean_gsma: Clean GSMA data reports
# > clean_seekops: Clean SeekOps data reports
# > clean_uhei: Clean Heidelberg University data reports
# > clean_dtu: Clean DTU data reports
# > clean_flylogix: Clean Flylogix data reports
# > clean_empa: Clean EMPA data reports
# > clean_uu_lsce_cyi_rhul: CLEAN UU/LSCE/CYI/RHUL team data reports 

# Imports
import numpy as np
import pandas as pd

from methods_source import RELEASE_NUMBER_LIST, convert_utc_to_local_france

# %% Operator QC analysis: only counted if quantification was completed
def operator_qc(measurement_taken, quantification_status):
    if measurement_taken == 'no':
        return False
    elif (measurement_taken == 'yes') and (quantification_status == 'failed'):
        return False
    elif (measurement_taken == 'yes') and (quantification_status == 'completed'):
        return True 

    return np.nan

# %% Stanford QC analysis: depends on the explanation
def stanford_qc(release, schedule):

    measurement_taken = schedule.loc[release-1, "Measurement Taken"].lower()
    quantification_status = schedule.loc[release-1, "Quantification Status"].lower()
    if measurement_taken == 'yes' and quantification_status == 'completed':
        return True
    elif measurement_taken == 'yes' and quantification_status == 'failed':
        # EMPA 
        if (schedule.loc[release-1, "Explanation"] == 'no plume visible'):
            return True
        return False
    
    return False 

# %% Strict QC: anything that failed quantification is counted as a zero 
def strict_qc(measurement_taken, quantification_status):
    if measurement_taken == 'no':
        return False
    elif measurement_taken == 'yes':
        return True 
    
    return np.nan
# %% Aeromon Data Cleaning

def clean_aeromon(aeromon_schedule, aeromon_results):
    num_releases = range(1, len(aeromon_schedule)+1) # for loop index
    release_list = [] # generating all new rows

    # fill na with 'None' for quantification status
    aeromon_schedule['Quantification Status'] = aeromon_schedule['Quantification Status'].fillna('None')
    
    for release in num_releases:
        if aeromon_schedule.loc[release-1, "Measurement Taken"] == "YES" and aeromon_schedule.loc[release-1, "Quantification Status"] == 'Completed':
            quantified = True
            uncertainty_type = 'ci_95'
            windspeed = aeromon_results.loc[release-1, "WindSpeed"].split(",")[0].split(" ")[2]
            emission_rate = aeromon_results.loc[release-1, 'EstimatedEmissionRate']
        elif aeromon_schedule.loc[release-1, "Measurement Taken"] == "YES" and aeromon_schedule.loc[release-1, "Quantification Status"] == 'Failed':
            quantified = False
            uncertainty_type = np.nan
            windspeed = np.nan
            emission_rate = 0
        else:
            quantified = False
            uncertainty_type = np.nan
            windspeed = np.nan
            emission_rate = np.nan
        
        ## QC analysis
        measurement_taken = aeromon_schedule.loc[release-1, "Measurement Taken"].lower()
        quantification_status = aeromon_schedule.loc[release-1, "Quantification Status"].lower()
        
        operator_keep = operator_qc(measurement_taken, quantification_status)
        stanford_keep = stanford_qc(release, aeromon_schedule)
        strict_qc_keep = strict_qc(measurement_taken, quantification_status)
        
        new_row = {
            'Operator': 'Aeromon',
            'Week': 1,
            'DateOfSurvey': aeromon_schedule.loc[release-1, "Date"],
            'ReleaseID': release, 
            'SurveyStartTime': aeromon_results.loc[release-1, "StartTime"],
            'SurveyEndTime': aeromon_results.loc[release-1, "EndTime"],
            'QuantifiedPlume': quantified,
            'EstimatedEmissionRate': emission_rate,
            'EstimatedEmissionRateUpper': aeromon_results.loc[release-1, 'EstimatedEmissionRateUpper'],
            'EstimatedEmissionRateLower': aeromon_results.loc[release-1, 'EstimatedEmissionRateLower'],
            'UncertaintyType': uncertainty_type,
            'OperatorWindspeed': windspeed,
            'QCFLag': aeromon_schedule.loc[release-1, "Explanation"],
            'OperatorKeep': operator_keep,
            'StanfordKeep': stanford_keep,
            'StrictQCKeep': strict_qc_keep,
        }
        
        release_list.append(new_row)

    aeromon_clean = pd.DataFrame(release_list)
    return aeromon_clean

# %% GSMA Data Cleaning
def clean_gsma(gsma_schedule, gsma_results):
    num_releases = range(1, len(gsma_schedule)+1) # for loop index
    release_estimate_index = 0 # index for release estimate 
    release_list = [] # generating all new rows

    # fill na with 'None' for quantification status
    gsma_schedule['Quantification Status'] = gsma_schedule['Quantification Status'].fillna('None')
    
    for release in num_releases:
        
        if gsma_schedule.loc[release-1, "Quantification Status"] == 'Completed':
            quantified = True
            start_time = gsma_results.loc[release_estimate_index, "StartTime"]
            end_time = gsma_results.loc[release_estimate_index, "EndTime"]
            emission_rate = gsma_results.loc[release_estimate_index, 'EstimatedEmissionRate [kg/h]']
            emission_upper = gsma_results.loc[release_estimate_index, 'EstimatedEmissionRateUpper']
            emission_lower = gsma_results.loc[release_estimate_index, 'EstimatedEmissionRateLower']
            uncertainty_type ='percent_' + str(gsma_results.loc[release_estimate_index, 'UncertaintyType'])
            windspeed = gsma_results.loc[release_estimate_index, "WindSpeed [m/s]"]
            release_estimate_index += 1
        elif gsma_schedule.loc[release-1, "Measurement Taken"] == 'YES' and gsma_schedule.loc[release-1, "Quantification Status"] == 'Failed':
            quantified = False
            start_time = gsma_results.loc[release_estimate_index, "StartTime"]
            end_time = gsma_results.loc[release_estimate_index, "EndTime"]
            emission_rate = 0
            emission_upper = 0
            emission_lower = 0
            uncertainty_type = np.nan
            windspeed = np.nan
        else:
            quantified = False
            start_time = np.nan
            end_time = np.nan
            emission_rate = np.nan 
            emission_upper = np.nan
            emission_lower = np.nan 
            uncertainty_type = np.nan 
            windspeed = np.nan 
        
        ## QC analysis
        measurement_taken = gsma_schedule.loc[release-1, "Measurement Taken"].lower()
        quantification_status = gsma_schedule.loc[release-1, "Quantification Status"].lower()
        
        operator_keep = operator_qc(measurement_taken, quantification_status)
        stanford_keep = stanford_qc(release, gsma_schedule)
        strict_qc_keep = strict_qc(measurement_taken, quantification_status)
        
        new_row = {
            'Operator': 'GSMA',
            'Week': 1,
            'DateOfSurvey': gsma_schedule.loc[release-1, "Date"],
            'ReleaseID': release, 
            'SurveyStartTime': start_time,
            'SurveyEndTime': end_time,
            'QuantifiedPlume': quantified,
            'EstimatedEmissionRate': emission_rate,
            'EstimatedEmissionRateUpper': emission_upper,
            'EstimatedEmissionRateLower': emission_lower,
            'UncertaintyType': uncertainty_type,
            'OperatorWindspeed': windspeed,
            'QCFLag': gsma_schedule.loc[release-1, "Explanation"],
            'OperatorKeep': operator_keep,
            'StanfordKeep': stanford_keep,
            'StrictQCKeep': strict_qc_keep,
        }
            
        release_list.append(new_row)

    gsma_clean = pd.DataFrame(release_list)
    return gsma_clean

# %% SeekOps Data Cleaning
def clean_seekops(seekops_schedule, seekops_results):
    # Sometimes the pd.read_csv command reads an extra line in the code so drop this
    seekops_schedule = seekops_schedule[seekops_schedule['Date'].notna()]
    num_releases = range(1, len(seekops_schedule)+1) # for loop index
    release_list = [] # generating all new rows
    
    schedule = seekops_schedule
    results = seekops_results
    
    # fill na with 'None' for quantification status
    seekops_schedule.loc[:, 'Quantification Status'] = seekops_schedule['Quantification Status'].fillna('None')

    for release in num_releases:
        
        # Using start and end time of release schedule 
        start_time = seekops_schedule.loc[release-1, "Start Time"]
        end_time = seekops_schedule.loc[release-1, "End Time"]
        # filter out extra row where timing was written incorrectly
        if start_time > end_time:
            continue
            
        if seekops_schedule.loc[release-1, "Quantification Status"] == 'Completed':
            quantified = True
            # totaled emission rate reported in "Explanation" column in schedule
            emission_rate = float(seekops_schedule.loc[release-1, 'Explanation'].split(" ")[-1])
            # SeekOps did not report upper and lower bounds
            # SeekOps reported a standard 30% uncertainty 
            emission_upper = emission_rate * (1 + 0.3)
            emission_lower = emission_rate * (1 - 0.3)
            uncertainty_type ='percent_0.30'
            # SeekOps did not report average wind estimates for each release (only for each survey they took)
            windspeed = np.nan
        elif schedule.loc[release-1, "Measurement Taken"] == 'YES' and schedule.loc[release-1, "Quantification Status"] == 'Failed':
            quantified = False
            emission_rate = 0
            emission_lower = 0
            emission_upper = 0
            uncertainty_type = np.nan
            windspeed = np.nan
        else:
            quantified = False
            emission_rate = np.nan 
            emission_upper = np.nan
            emission_lower = np.nan 
            uncertainty_type = np.nan 
            windspeed = np.nan 
 
        ## QC analysis
        measurement_taken = schedule.loc[release-1, "Measurement Taken"].lower()
        quantification_status = schedule.loc[release-1, "Quantification Status"].lower()
        
        operator_keep = operator_qc(measurement_taken, quantification_status)
        stanford_keep = stanford_qc(release, schedule)
        strict_qc_keep = strict_qc(measurement_taken, quantification_status)
                       
        new_row = {
            'Operator': 'SeekOps',
            'Week': 2,
            'DateOfSurvey': seekops_schedule.loc[release-1, "Date"],
            'ReleaseID': release, 
            'SurveyStartTime': start_time,
            'SurveyEndTime': end_time,
            'QuantifiedPlume': quantified,
            'EstimatedEmissionRate': emission_rate,
            'EstimatedEmissionRateUpper': emission_upper,
            'EstimatedEmissionRateLower': emission_lower,
            'UncertaintyType': uncertainty_type,
            'OperatorWindspeed': windspeed,
            'QCFLag': seekops_schedule.loc[release-1, "Explanation"],
            'OperatorKeep': operator_keep,
            'StanfordKeep': stanford_keep,
            'StrictQCKeep': strict_qc_keep,
        }
                
        release_list.append(new_row)

    seekops_clean = pd.DataFrame(release_list)
    return seekops_clean

# %% UHEI data cleaning

def clean_uhei(schedule, results):
    week = 2
    num_releases = range(1, RELEASE_NUMBER_LIST[week] + 1) # for loop index
    release_list = [] # generating all new rows

    # fill na with 'None' for quantification status
    schedule['Quantification Status'] = schedule['Quantification Status'].fillna('None')
    
    for release in num_releases:
        if schedule.loc[release-1, "Quantification Status"] == 'Completed':
            quantified = True
            start_time = convert_utc_to_local_france(results.loc[release-1, "StartTime\n(UTC)"])
            end_time = convert_utc_to_local_france(results.loc[release-1, "EndTime\n(UTC)"])
            emission_rate = results.loc[release-1, 'EstimatedEmissionRate']
            delta_upper = float(results.loc[release-1, 'EstimatedEmissionRateUpper'].split("+")[1])
            emission_upper = emission_rate + delta_upper
            delta_lower = float(results.loc[release-1, 'EstimatedEmissionRateLower'].split("-")[1])
            emission_lower = emission_rate - delta_lower
        elif schedule.loc[release-1, "Measurement Taken"] == 'YES' and schedule.loc[release-1, "Quantification Status"] == 'Failed':
            quantified = False
            start_time = schedule.loc[release-1, "Start Time"]
            end_time = schedule.loc[release-1, "End Time"]
            emission_rate = 0
            emission_upper = 0
            emission_lower = 0
        else:
            quantified = False
            start_time = np.nan
            end_time = np.nan
            emission_rate = np.nan
            emission_upper = np.nan
            emission_lower = np.nan
        
        ## QC analysis
        measurement_taken = schedule.loc[release-1, "Measurement Taken"].lower()
        quantification_status = schedule.loc[release-1, "Quantification Status"].lower()
        
        operator_keep = operator_qc(measurement_taken, quantification_status)
        stanford_keep = stanford_qc(release, schedule)
        strict_qc_keep = strict_qc(measurement_taken, quantification_status)
        
        new_row = {
            'Operator': 'UHEI',
            'Week': week,
            'DateOfSurvey': schedule.loc[release-1, "Date"],
            'ReleaseID': release, 
            'SurveyStartTime': start_time,
            'SurveyEndTime': end_time,
            'QuantifiedPlume': quantified,
            'EstimatedEmissionRate': emission_rate,
            'EstimatedEmissionRateUpper': emission_upper,
            'EstimatedEmissionRateLower': emission_lower,
            'UncertaintyType': results.loc[release-1, 'UncertaintyType'],
            'OperatorWindspeed': results.loc[release-1, "WindSpeed"],
            'QCFLag': schedule.loc[release-1, "Explanation"],
            'OperatorKeep': operator_keep,
            'StanfordKeep': stanford_keep,
            'StrictQCKeep': strict_qc_keep,
        }
        
        release_list.append(new_row)

    uhei_clean = pd.DataFrame(release_list)
    
    return uhei_clean

# %% DTU data cleaning

def clean_dtu(results):
    week = 1
    num_releases = range(1, RELEASE_NUMBER_LIST[week] + 1) # for loop index
    release_list = [] # generating all new rows

    for release in num_releases:
        if results.loc[release-1, "EstimatedEmissionRate"] != 'BDL':
            quantified = True
            start_time = results.loc[release-1, "StartTime"]
            end_time = results.loc[release-1, "EndTime"]
            emission_rate = results.loc[release-1, 'EstimatedEmissionRate']
            emission_upper = results.loc[release-1, 'EstimatedEmissionRateUpper']
            emission_lower = results.loc[release-1, 'EstimatedEmissionRateLower']
            QCflag = ''
        elif results.loc[release-1, "EstimatedEmissionRate"] == 'BDL':
            quantified = True
            start_time = results.loc[release-1, "StartTime"]
            end_time = results.loc[release-1, "EndTime"]
            emission_rate = 0
            emission_upper = 0
            emission_lower = 0
            QCflag = 'Below detection limit'

        operator_keep = True
        stanford_keep = True
        strict_qc_keep = True
          
        new_row = {
            'Operator': 'DTU',
            'Week': week,
            'DateOfSurvey': results.loc[release-1, "DateOfSurvey"],
            'ReleaseID': release, 
            'SurveyStartTime': start_time,
            'SurveyEndTime': end_time,
            'QuantifiedPlume': quantified,
            'EstimatedEmissionRate': emission_rate,
            'EstimatedEmissionRateUpper': emission_upper,
            'EstimatedEmissionRateLower': emission_lower,
            'UncertaintyType': results.loc[release-1, 'UncertaintyType'],
            'OperatorWindspeed': results.loc[release-1, "WindSpeed"],
            'QCFLag': QCflag,
            'OperatorKeep': operator_keep,
            'StanfordKeep': stanford_keep,
            'StrictQCKeep': strict_qc_keep,
        }
        
        release_list.append(new_row)

    dtu_clean = pd.DataFrame(release_list)
    
    return dtu_clean

# %% Flylogix Data Cleaning

def clean_flylogix(schedule, results):
    num_releases = range(1, len(schedule)+1) # for loop index
    release_estimate_index = 0 # index for release estimate 
    release_list = [] # generating all new rows

    # fill na with 'None' for quantification status
    schedule['Quantification Status'] = schedule['Quantification Status'].fillna('None')
    
    for release in num_releases:
        
        if schedule.loc[release-1, "Quantification Status"] == 'Completed':
            quantified = True
            start_time = results.loc[release_estimate_index, "StartTime"]
            end_time = results.loc[release_estimate_index, "EndTime"]
            emission_rate = results.loc[release_estimate_index, 'EstimatedEmissionRate']
            emission_upper = results.loc[release_estimate_index, 'EstimatedEmissionRateUpper']
            emission_lower = results.loc[release_estimate_index, 'EstimatedEmissionRateLower']
            uncertainty_type =results.loc[release_estimate_index, 'UncertaintyType']
            windspeed = results.loc[release_estimate_index, "WindSpeed"]
            release_estimate_index += 1
        elif schedule.loc[release-1, "Measurement Taken"] == 'YES' and schedule.loc[release-1, "Quantification Status"] == 'Failed':
            quantified = False
            start_time = schedule.loc[release-1, "Start Time"]
            end_time = schedule.loc[release-1, "End Time"]
            emission_rate = 0
            emission_upper = 0
            emission_lower = 0
        else:
            quantified = False
            start_time = np.nan
            end_time = np.nan
            emission_rate = np.nan 
            emission_upper = np.nan
            emission_lower = np.nan 
            uncertainty_type = np.nan 
            windspeed = np.nan 
        
        ## QC analysis
        measurement_taken = schedule.loc[release-1, "Measurement Taken"].lower()
        quantification_status = schedule.loc[release-1, "Quantification Status"].lower()
        operator_keep = operator_qc(measurement_taken, quantification_status)
        stanford_keep = stanford_qc(release, schedule)
        strict_qc_keep = strict_qc(measurement_taken, quantification_status)
        
        new_row = {
            'Operator': 'Flylogix',
            'Week': 4,
            'DateOfSurvey': schedule.loc[release-1, "Date"],
            'ReleaseID': release, 
            'SurveyStartTime': start_time,
            'SurveyEndTime': end_time,
            'QuantifiedPlume': quantified,
            'EstimatedEmissionRate': emission_rate,
            'EstimatedEmissionRateUpper': emission_upper,
            'EstimatedEmissionRateLower': emission_lower,
            'UncertaintyType': uncertainty_type,
            'OperatorWindspeed': windspeed,
            'QCFLag': schedule.loc[release-1, "Explanation"],
            'OperatorKeep': operator_keep,
            'StanfordKeep': stanford_keep,
            'StrictQCKeep': strict_qc_keep,
        }
            
        release_list.append(new_row)

    clean_df = pd.DataFrame(release_list)
    return clean_df

# %% EMPA Data Cleaning

def clean_empa(schedule, results):
    num_releases = range(1, len(schedule)+1) # for loop index
    release_estimate_index = 0 # index for release estimate 
    release_list = [] # generating all new rows

    # fill na with 'None' for quantification status
    schedule['Quantification Status'] = schedule['Quantification Status'].fillna('None')
    
    for release in num_releases:
        
        if schedule.loc[release-1, "Quantification Status"] == "Completed":
            quantified = True
            start_time = results.loc[release_estimate_index, "StartTime"]
            end_time = results.loc[release_estimate_index, "EndTime"]
            emission_rate = results.loc[release_estimate_index, 'EstimatedEmissionRate']
            emission_upper = results.loc[release_estimate_index, 'EstimatedEmissionRateUpper']
            emission_lower = results.loc[release_estimate_index, 'EstimatedEmissionRateLower']
            uncertainty_type =results.loc[release_estimate_index, 'UncertaintyType']
            windspeed = results.loc[release_estimate_index, "WindSpeed"]
            release_estimate_index += 1
            
        # took measurement but no plume visible
        elif (schedule.loc[release-1, "Measurement Taken"] == "YES") & (schedule.loc[release-1, "Quantification Status"] == "Failed"):
            quantified = False
            start_time = schedule.loc[release-1, "Start Time"]
            end_time = schedule.loc[release-1, "End Time"]
            emission_rate = 0
            emission_upper = 0
            emission_lower = 0 
            uncertainty_type = np.nan 
            windspeed = np.nan 
            
        else:
            quantified = False
            start_time = schedule.loc[release-1, "Start Time"]
            end_time = schedule.loc[release-1, "End Time"]
            emission_rate = np.nan 
            emission_upper = np.nan
            emission_lower = np.nan 
            uncertainty_type = np.nan 
            windspeed = np.nan 

        ## QC analysis
        measurement_taken = schedule.loc[release-1, "Measurement Taken"].lower()
        quantification_status = schedule.loc[release-1, "Quantification Status"].lower()
        operator_keep = operator_qc(measurement_taken, quantification_status)
        stanford_keep = stanford_qc(release, schedule)
        strict_qc_keep = strict_qc(measurement_taken, quantification_status)
                 
        new_row = {
            'Operator': 'EMPA',
            'Week': 4,
            'DateOfSurvey': schedule.loc[release-1, "Date"],
            'ReleaseID': release, 
            'SurveyStartTime': start_time,
            'SurveyEndTime': end_time,
            'QuantifiedPlume': quantified,
            'EstimatedEmissionRate': emission_rate,
            'EstimatedEmissionRateUpper': emission_upper,
            'EstimatedEmissionRateLower': emission_lower,
            'UncertaintyType': uncertainty_type,
            'OperatorWindspeed': windspeed,
            'QCFLag': schedule.loc[release-1, "Explanation"],
            'OperatorKeep': operator_keep,
            'StanfordKeep': stanford_keep,
            'StrictQCKeep': strict_qc_keep,
        }
            
        release_list.append(new_row)

    clean_df = pd.DataFrame(release_list)
    return clean_df


# %% UU/LSCE/CYI/RHUL Data cleaning

def clean_uu_lsce_cyi_rhul(schedule, results):
    num_releases = range(1, len(schedule)+1) # for loop index
    release_list = [] # generating all new rows

    # fill na with 'None' for quantification status
    schedule['Quantification Status'] = schedule['Quantification Status'].fillna('None')
    
    for release in num_releases:
        if schedule.loc[release-1, "Quantification Status"] == 'Completed':
            quantified = True
            uncertainty_type = 'ci_95'
            windspeed = results.loc[release-1, "WindSpeed"]
            emission_rate = results.loc[release-1, 'EstimatedEmissionRate']
        elif schedule.loc[release-1, "Measurement Taken"] == 'YES' and schedule.loc[release-1, "Quantification Status"] == 'Failed':
            quantified = False
            uncertainty_type = np.nan
            windspeed = np.nan
            emission_rate = 0
        else:
            quantified = False
            uncertainty_type = np.nan
            windspeed = np.nan
            emission_rate = results.loc[release-1, 'EstimatedEmissionRate']

        ## QC analysis
        measurement_taken = schedule.loc[release-1, "Measurement Taken"].lower()
        quantification_status = schedule.loc[release-1, "Quantification Status"].lower()
        operator_keep = operator_qc(measurement_taken, quantification_status)
        stanford_keep = stanford_qc(release, schedule)
        strict_qc_keep = strict_qc(measurement_taken, quantification_status)
             
        new_row = {
            'Operator': 'UU/LSCE/CYI/RHUL',
            'Week': 4,
            'DateOfSurvey': schedule.loc[release-1, "Date"],
            'ReleaseID': release, 
            'SurveyStartTime': results.loc[release-1, "StartTime"],
            'SurveyEndTime': results.loc[release-1, "EndTime"],
            'QuantifiedPlume': quantified,
            'EstimatedEmissionRate': emission_rate,
            'EstimatedEmissionRateUpper': results.loc[release-1, 'EstimatedEmissionRateUpper'],
            'EstimatedEmissionRateLower': results.loc[release-1, 'EstimatedEmissionRateLower'],
            'UncertaintyType': uncertainty_type,
            'OperatorWindspeed': windspeed,
            'QCFLag': schedule.loc[release-1, "Explanation"],
            'OperatorKeep': operator_keep,
            'StanfordKeep': stanford_keep,
            'StrictQCKeep': strict_qc_keep,
        }
        
        release_list.append(new_row)

    clean_df = pd.DataFrame(release_list)
    return clean_df
