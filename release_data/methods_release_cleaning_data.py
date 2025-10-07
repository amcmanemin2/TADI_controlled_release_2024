# Script for extracting the relevant columns from the release data
# Note: all scripts in this file are meant to be run directly on the release data in the format given by TADI
# Author: Audrey McManemin
# Date Created: 2024-09-09
# Date Last Modified: 2025-01-14

# Imports
# imports
import pandas as pd 
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import PurePath
from datetime import datetime, time
from windrose import WindroseAxes

# %% Clean the release data from TADI 

def clean_TADI_release_data(path, sheet_name, cols='B:Z', engine='openpyxl'):
    
    data = pd.read_excel(path, sheet_name=sheet_name, engine=engine, usecols=cols, skiprows=1)
    
    # rename columns
    data.columns = ['Week', 'Date', 'BT = BlIND TEST', 'Test N°', 'Test Number', 'ReleaseStart', 'Release Start Stabilized', 'ReleaseEnd', 'Duration (hh:min)', 'Gas', 'Localisation', 'Leak Equipment', 'Leak Type / Location', 'Type of emission', 'Leak localisation - X(m)', 'Leak localisation - Y(m)', 'Leak localisation - Z(m)', 'Flowrate (g/s)', 'Flowrate (kg/h)', 'OP Valve (%)', 'Valve', 'Nominal size of orifice', 'Internal diameter (mm)', 'pressure (bar)', 'Comments']
    
    # drop empty rows
    data = data[(data['ReleaseEnd'].notna() & data['Flowrate (kg/h)'].notna())]
    
    # for W4 - drop the 15th release (no teams measured)
    if 'W38' in data['Week'].values:
        data = data.drop(21)
        print("DROPPING W4 EXTRA RELEASE")
    
    # update format of Week column
    data['Week'] = data['Week'].map({'W25': 1, 'W26': 2, 'W37': 3, 'W38': 4}).fillna(0).astype(int)

    # extract columns for cleaned data
    clean_data = data[['Week', 'Date', 'ReleaseStart', 'ReleaseEnd', 'Flowrate (kg/h)', 'Comments']]

    clean_data = clean_data.reset_index(drop=True)
    clean_data['ReleaseID'] = clean_data.index + 1
    
    # ensure the datetime columns are in datetime format
    clean_data['Date'] = pd.to_datetime(clean_data['Date'])
    clean_data['ReleaseStart'] = clean_data['ReleaseStart'].apply(lambda x: x if isinstance(x, time) else pd.to_datetime(x).time())
    clean_data['ReleaseEnd'] = clean_data['ReleaseEnd'].apply(lambda x: x if isinstance(x, time) else pd.to_datetime(x).time())
    
    # Combine Date with ReleaseStart and ReleaseEnd times
    clean_data['ReleaseStartDateTime'] = clean_data.apply(lambda row: pd.Timestamp.combine(row['Date'], row['ReleaseStart']), axis=1)
    clean_data['ReleaseEndDateTime'] = clean_data.apply(lambda row: pd.Timestamp.combine(row['Date'], row['ReleaseEnd']), axis=1)

    return clean_data

# %% Function to calculate wind speed and direction averages and standard deviations

def calculate_wind_stats(row, wind, windspeeds_dict, winddirs_dict):
    mask = (wind['datetime_local'] >= row['ReleaseStartDateTime']) & (wind['datetime_local'] <= row['ReleaseEndDateTime'])
    wind_speeds = wind.loc[mask, 'windspeed_h_20m']
    wind_dirs = wind.loc[mask, 'winddir_20m']
    wind_shears = wind.loc[mask, 'windshear_std']
    
    # Compute wind direction CV using resultant vector length
    wind_dirs_rad = np.radians(wind_dirs)
    R_x = np.mean(np.cos(wind_dirs_rad))
    R_y = np.mean(np.sin(wind_dirs_rad))
    R = np.sqrt(R_x**2 + R_y**2)
    wind_dir_std = np.sqrt(-2 * np.log(R)) * (180 / np.pi)  # Circular std deviation in degrees
    wind_dir_cv = wind_dir_std / 360  # Coefficient of variation for angular data
    
    windspeeds_dict[row.ReleaseID] = wind_speeds.to_list()
    winddirs_dict[row.ReleaseID] = wind_dirs.to_list()
    
    return (
        wind_speeds.mean(), 
        wind_speeds.std(), 
        wind_dirs.mean(), 
        wind_dir_std, 
        wind_dir_cv, 
        wind_shears.mean()
    )

# %% function to add wind statistics to the cleaned data files

def add_wind_stats(week):
    clean_data = load_clean_release_data(week)
    wind = pd.read_csv(f'clean_wind_data/W{week}_wind_data.csv', index_col=0)

    windspeeds_dict = {}
    winddirs_dict = {}
    
    # apply the function and create a new column with the averages (m/s)
    stats = clean_data.apply(lambda row: calculate_wind_stats(row, wind, windspeeds_dict, winddirs_dict), axis=1)
    clean_data['Windspeed20m_Avg'] = stats.apply(lambda x: x[0])  # windspeed mean
    clean_data['Windspeed20m_Std'] = stats.apply(lambda x: x[1])  # windspeed standard deviation
    clean_data['WindDir20m_Avg'] = stats.apply(lambda x: x[2])  # Wind direction mean
    clean_data['WindDir20m_Std'] = stats.apply(lambda x: x[3])  # Wind direction std dev
    clean_data['WindDir20m_CV'] = stats.apply(lambda x: x[4])   # Wind direction CV
    clean_data['WindShear_AvgStd'] = stats.apply(lambda x: x[5]) # Wind shear mean (std dev averaged across heights)
    
    # save data
    clean_data.to_csv(pathlib.PurePath('clean_release_data', f'W{week}_clean.csv'))
    print(f"Updating W{week}_clean.csv with wind statistics")
    
    # plot wind roses
    fig = plt.figure(figsize=(20, 34))
    for k in windspeeds_dict.keys():

        windrose_ax = fig.add_subplot(8, 5, k, projection='windrose')
        windrose_ax.bar(
            winddirs_dict[k],  # Wind directions for release k
            windspeeds_dict[k],  # Wind speeds for release k
            bins=np.arange(0, 8, 2),  # Define bins for wind speed
            normed=True  # Normalize frequencies
        )
        windrose_ax.set_title(f'Release {k}')
        windrose_ax.set_yticklabels([])
        if k % 5 == 0:
            windrose_ax.set_legend(loc='upper left', bbox_to_anchor=(1.1, 1))
    
    fig.suptitle(f'Week {week} Wind Plots by Release ID', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f'../plots/wind/W{week}_windrose_plots_by_release.png')
    # plt.show()
      
    return clean_data

# %% function to load the clean release data
def load_clean_release_data(week):
    return pd.read_csv(f'clean_release_data/W{week}_clean.csv', index_col=0)