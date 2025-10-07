# Script for methods for generating figures
# Author: Audrey McManemin
# Date Created: 2024-12-16
# Date Last Modified: 2024-12-16

# List of methods in this file:
# > get_parity_data
# > make_parity_plot
# > plot_parity

# Imports
import numpy as np
import pandas as pd
import pathlib
import datetime
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
from sklearn.linear_model import LogisticRegression
from methods import load_release_summary, OPERATOR_TYPE, OPERATOR_LIST, COMMERCIAL_OPERATOR_LIST, ACADEMIC_OPERATOR_LIST
from writing_analysis import classify_histogram_data

# from mobile_solutions.methods_source import make_logistic_regression

# from writing_analysis import calculate_residuals_and_error

WINDSPEED_BIN_COLOR_MAP = {
    '[0.0-2.0)': 'blue',
    '[2.0-4.0)': 'green',
    '[4.0-6.0)': 'orange',
    '>6.0': 'red'
}

WINDSPEED_VARIATION_COEFF_COLOR_MAP = {
    '[0-25%)': 'blue', 
    '[25-50%)': 'green', 
    '[50-75%)': 'orange', 
    '[75-100%)': 'gold',
    '>100%': 'red'
}

WINDDIR_VARIATION_COEFF_COLOR_MAP = {
        '[0-10%)': 'blue',
        '[10-20%)': 'green',
        '[20-30%)': 'orange',
        '[30-40%)': 'gold',
        '>40%': 'red'
}

OPERATOR_PLOT_TITLES = {
    'Sensia': 'Sensia Mileva 33',
    'SLB': 'SLB Methane Lidar Camera',
    'Sensirion': 'Sensirion Nubo Sphere',
    'Aeromon': 'Aeromon BH-12',
    'SeekOps': 'SeekOps SeekIR',
    'GSMA': 'GSMA AUSEA',
    'EMPA': 'EMPA AVIRIS-4',
    'GHGSat': 'GHGSat-C',
    'all': 'All Solutions',
    'commercial': 'Commercial Operators',
}

# %% Generate the title for each operator
def gen_operator_title(operator):
    if operator in OPERATOR_PLOT_TITLES.keys():
        op_title = OPERATOR_PLOT_TITLES[operator]
    else: 
        op_title = operator
    return op_title

# %% Functions for making parity plots

def get_parity_data(operator, error_type='operator_reported', wind_analysis=False, qc_type='stanford_kept'):
    """

    :param operator: name of operator
    :param error_type: indicate type of error. Default is 95% CI, alternative is "operator_reported"

    :return save_parity_data: dataframe with columns release_rate, operator_report, and operator_sigma [lower, upper]
    """

    # Load release summary csv file
    if operator in OPERATOR_TYPE.keys():
        operator_plot = pd.DataFrame()
        for op in OPERATOR_TYPE[operator]:
            operator_plot = pd.concat([operator_plot, load_release_summary(operator=op)])
    elif operator == 'all':
        operator_plot = pd.DataFrame()
        for op in OPERATOR_LIST:
            operator_plot = pd.concat([operator_plot, load_release_summary(operator=op)])
    elif operator == 'commercial':
        operator_plot = pd.DataFrame()
        for op in COMMERCIAL_OPERATOR_LIST:
            operator_plot = pd.concat([operator_plot, load_release_summary(operator=op)])
    elif operator == 'academic':
        operator_plot = pd.DataFrame()
        for op in ACADEMIC_OPERATOR_LIST:
            operator_plot = pd.concat([operator_plot, load_release_summary(operator=op)])
    elif operator == 'academic_no_faam':
        operator_plot = pd.DataFrame()
        for op in ACADEMIC_OPERATOR_LIST:
            if op != 'FAAM':
                operator_plot = pd.concat([operator_plot, load_release_summary(operator=op)])
            
    else:
        operator_plot = load_release_summary(operator=operator)
    
    # Apply the following filters to release data :
    # Must pass all QC filters:
    operator_plot = operator_plot[(operator_plot[f'{qc_type}'] == True)]

    # For parity plots:
    # All data entries must be a non-zero release
    operator_plot = operator_plot.query('non_zero_release == True')

    # Operator must have quantified the release as non-zero:
    operator_plot = operator_plot.query('operator_quantification > 0')

    # Select x data
    x_data = operator_plot['release_rate_kgh']
    y_data = operator_plot['operator_quantification']
    
    # Assuming no error for TADI release rate 
    x_error = 0 

    ######### Select error bars for operator quantification #########

    if error_type == 'operator_reported':
        operator_multiplier = 1
        legend_error = 'Operator Reported Error Bars'
    else:
        operator_multiplier = 1
        legend_error = 'Operator Reported Error Bars'
        print(f'Please correct input error type. Currently using operator reported error bars')

    y_error_lower = (operator_plot['operator_quantification'] - operator_plot['operator_lower']) * operator_multiplier
    y_error_upper = (operator_plot['operator_upper'] - operator_plot['operator_quantification']) * operator_multiplier

    # Save data used to make figure
    save_parity_data = pd.DataFrame()
    save_parity_data['release_rate'] = x_data
    save_parity_data['release_sigma'] = x_error
    save_parity_data['operator_report'] = y_data
    save_parity_data['operator_sigma_lower'] = y_error_lower
    save_parity_data['operator_sigma_upper'] = y_error_upper

    # Save data description
    data_description = {
        'operator': operator,
        'legend_error': legend_error,
    }
    
    # Add in windspeed bucketing
    if wind_analysis: 
        
        # average windspeed
        # Define the bins and labels
        bins = [0.0, 2.0, 4.0, 6.0, float('inf')]
        labels = ['[0.0-2.0)', '[2.0-4.0)', '[4.0-6.0)', '>6.0']

        # Add a new column with the categorized wind speeds
        save_parity_data['windspeed_bin'] = pd.cut(operator_plot['windspeed_20m_avg'], bins=bins, labels=labels, right=False)       
        
        # windspeed variation (steadiness)
        # Define the bins and labels
        bins = [0.0, 0.25, 0.5, 0.75, 1.0, float('inf')]
        labels = ['[0-25%)', '[25-50%)', '[50-75%)', '[75-100%)', '>100%']

        # Add a new column with the categorized windspeed variation
        save_parity_data['windspeed_variation_coeff_bin'] = pd.cut(operator_plot['windspeed_20m_std']/operator_plot['windspeed_20m_avg'], bins=bins, labels=labels, right=False) 
        
        # Wind direction variation (steadiness)
        # Define the bins and labels
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, float('inf')]
        labels = ['[0-10%)', '[10-20%)', '[20-30%)', '[30-40%)', '>40%']

        # Add a new column with the categorized wind direction variation
        save_parity_data['winddir_variation_coeff_bin'] = pd.cut(operator_plot['winddir_20m_cv'], bins=bins, labels=labels, right=False) 
        
    return save_parity_data, data_description


def make_parity_plot(data, data_description, ax, plot_lim='largest_kgh', wind_analysis=False, plot_axis='normal'):
    """
    :param data: processed data to be plotted
    :param data_description: dictionary with descriptions of data used for plot annotations
    :param ax: subplot ax to plot on
    :param plot_lim: limit of x and y axes
    :return: ax: is the plotted parity chart
    """

    ############ Data Preparation and Linear Regression ############

    operator = data_description['operator']
    legend_error = data_description['legend_error']

    # Set x and y data and error values
    x_data = data.release_rate
    y_data = data.operator_report
    x_error = data.release_sigma * 1.96  # value is sigma, multiply by 1.96 for 95% CI - assumed to be 0 for these experiments
    y_error_lower = data.operator_sigma_lower  # error bars are determined in get_parity_data function
    y_error_upper = data.operator_sigma_upper  # error bars are determined in get_parity_data function

    # Set x and y max values
    # Manually set largest x and y value by changing largest_kgh here to desired value:
    # largest_kgh = max(plot_lim)

    if plot_lim == 'largest_kgh':
        # Filter out NA because operations with NA returns NA
        if np.isnan(max(y_error_upper)) == 1:
            y_error_upper.iloc[:] = 0
            
        if np.isnan(max(y_error_lower)) == 1:
            y_error_lower.iloc[:] = 0

        largest_kgh = max(max(x_data), max(y_data)) + max(y_error_upper)
        largest_kgh = math.ceil(largest_kgh / 100) * 100

        # set plot_lim:
        plot_lim = [0, largest_kgh]
        smallest_kgh = 0
    else:
        largest_kgh = max(plot_lim)
        smallest_kgh = min(plot_lim)
    
    # Create a mask for entries where y_data <= largest_kgh
    mask = (x_data >= smallest_kgh) & (x_data <= largest_kgh)

    # Apply the mask to filter x_data, y_data, and their corresponding errors

    y_data = y_data[mask]
    x_data = x_data[mask]
    y_error_lower = y_error_lower[mask]
    y_error_upper = y_error_upper[mask]
    x_error = x_error[mask]
    
    sample_size = len(x_data)
    
    # Fit linear regression via least squares with numpy.polyfit
    # m is slope, intercept is b
    if len(y_data) > 1:
        m, b = np.polyfit(x_data, y_data, deg=1)

        # Calculate R^2 value
        # (using method described here: https://www.askpython.com/python/coefficient-of-determination)
        correlation_matrix = np.corrcoef(x_data, y_data)
        correlation = correlation_matrix[0, 1]
        r2 = correlation ** 2

    # Create sequence of numbers for plotting linear fit (x)
    x_seq = np.linspace(0, largest_kgh, num=100)

    ############ Generate Figure  ############

    # Add linear regression to in put ax
    if (sample_size > 1) & (plot_axis == 'normal'):
        ax.plot(x_seq, m * x_seq + b, color='k', lw=2,
                label=f'Best Fit, $R^2 =$ {r2:0.2f}\n$y = {m:0.2f}x+{b:0.2f}$')

    # Add parity line
    # With label:
    # ax.plot(x_seq, x_seq, color='k', lw=2, linestyle='--',
    #          label='Parity Line')
    # Without label:
        # update axes
    ax.plot(x_seq, x_seq, color='k', lw=2, linestyle='--')
    if plot_axis == 'loglog':
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(x_seq, x_seq * 10, color='gray', lw=1, linestyle='dotted')
        ax.plot(x_seq, x_seq / 10, color='gray', lw=1, linestyle='dotted')

    
    if wind_analysis == 'speed_avg': 
        # Add scatter plots with error bars and windspeed_bin color
        # Map windspeed_bin to colors

        for bin, color in WINDSPEED_BIN_COLOR_MAP.items():
            bin_data = data[data.windspeed_bin == bin]
            if not bin_data.empty:  # Only plot if there is data for the bin
                ax.errorbar(bin_data.release_rate, bin_data.operator_report,
                            xerr=bin_data.release_sigma * 1.96,
                            yerr=[bin_data.operator_sigma_lower, bin_data.operator_sigma_upper],
                            linestyle='none',
                            mfc='white',
                            mec=color,
                            ecolor=color,
                            label=f'Windspeed {bin}',
                            fmt='o',
                            markersize=5)
                
    elif wind_analysis == 'speed_var_coeff':
        for bin, color in WINDSPEED_VARIATION_COEFF_COLOR_MAP.items():
            bin_data = data[data.windspeed_variation_coeff_bin == bin]
            if not bin_data.empty:  # Only plot if there is data for the bin
                ax.errorbar(bin_data.release_rate, bin_data.operator_report,
                            xerr=bin_data.release_sigma * 1.96,
                            yerr=[bin_data.operator_sigma_lower, bin_data.operator_sigma_upper],
                            linestyle='none',
                            mfc='white',
                            mec=color,
                            ecolor=color,
                            label=f'Windspeed CoV {bin}',
                            fmt='o',
                            markersize=5)
                
    # wind direction coefficient of variation
    elif wind_analysis == 'dir_var_coeff':
        for bin, color in WINDDIR_VARIATION_COEFF_COLOR_MAP.items():
            bin_data = data[data.winddir_variation_coeff_bin == bin]
            if not bin_data.empty:  # Only plot if there is data for the bin
                ax.errorbar(bin_data.release_rate, bin_data.operator_report,
                            xerr=bin_data.release_sigma * 1.96,
                            yerr=[bin_data.operator_sigma_lower, bin_data.operator_sigma_upper],
                            linestyle='none',
                            mfc='white',
                            mec=color,
                            ecolor=color,
                            label=f'Wind Dir CoV {bin}',
                            # label=f'Wind Direction CoV {bin} (n = {sample_size})',
                            fmt='o',
                            markersize=5)
    else:
        # Add scatter plots with error bars  
        ax.errorbar(x_data, y_data,
                    xerr=x_error,
                    yerr=[y_error_lower, y_error_upper],
                    linestyle='none',
                    mfc='white',
                    # label=f'n = {sample_size}\n({legend_error})',
                    label=f'n = {sample_size}',
                    fmt='o',
                    markersize=5)
    
    # Set title
    operator_title = OPERATOR_PLOT_TITLES.get(operator, operator) if operator in OPERATOR_PLOT_TITLES else operator
    ax.set_title(f'{operator_title}')

    # Annotation box
    # text = f'{operator}'
    # ob = offsetbox.AnchoredText(text, loc='upper left')
    # ob.set(alpha=0.8)
    # ax.add_artist(ob)

    # Set axes
    ax.set(xlim=plot_lim,
           ylim=plot_lim,
           alpha=0.8)

    # Equalize Axes
    ax.set_aspect('equal', adjustable='box')

    # Set axes and background color to white
    ax.set_facecolor('white')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')

    # Axes labels
    ax.set_xlabel('Methane Release Rate (kgh)', fontsize=14)
    ax.set_ylabel('Reported Release Rate (kgh)', fontsize=14)
    ax.tick_params(direction='in', right=True, top=True)
    ax.tick_params(labelsize=16)
    ax.minorticks_on()
    ax.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    ax.tick_params(direction='in', which='minor', length=3, bottom=True, top=True, left=True, right=True)
    ax.tick_params(direction='in', which='major', length=6, bottom=True, top=True, left=True, right=True)
    ax.grid(False)  # remove grid lines

    # Customize the tick labels with commas at the thousands place
    ax.tick_params(labelsize=16)

    # Define a formatter function to add commas
    def comma_formatter(x, pos):
        return '{:,.0f}'.format(x)  # Add commas to the thousands place

    # Apply the formatter to the tick labels
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(comma_formatter))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(comma_formatter))

    # Legend
    ax.legend(facecolor='white', loc='upper left', fontsize=11)

    return ax


def plot_parity(operator, plot_lim='largest_kgh', save_parity_data=False, save_parity_plot=False):

    # Generate parity data
    parity_data, parity_notes = get_parity_data(operator=operator)
    
    # Initialize figure
    fig, ax = plt.subplots(1, figsize=(6, 6))

    # Make figure
    ax = make_parity_plot(parity_data, parity_notes, ax, plot_lim=plot_lim)

    # Axes labels
    ax.set_xlabel('Methane Release Rate (kgh)', fontsize=14)
    ax.set_ylabel('Reported Release Rate (kgh)', fontsize=14)

    now = datetime.datetime.now()
    op_ab = operator.lower()
    # save_time = now.strftime("%Y%m%d")
    # fig_name = f'{op_ab}_parity_{save_time}'
    # fig_path = pathlib.PurePath('04_figures', 'parity_plots', fig_name)
    if isinstance(save_parity_plot, str):
        plt.savefig(save_parity_plot)
    plt.show()

    # Save data used to make figure
    if save_parity_data:
        save_path = pathlib.PurePath('03_results', 'parity_plot_data', f'{op_ab}_parity_{save_time}.csv')
        parity_data.to_csv(save_path)

    return

# %% detection probability plot methods

def make_detection_limit_df(operator, n_bins, threshold, qc_type='stanford_kept'):
    
    # Load release summary for operator
    operator_df = load_release_summary(operator)

    # Apply QC filter
    operator_df = operator_df[(operator_df[f'{qc_type}'] == True)]

    # Must be non-zero values
    operator_df = operator_df.query('non_zero_release == True')

    # Select release under threshold value
    operator_df = operator_df.query('release_rate_kgh <= @threshold')

    # Create bins for plot
    bins = np.linspace(0, threshold, n_bins + 1)
    detection_probability = np.zeros(n_bins)

    # These variables are for keeping track of values as I iterate through the bins in the for loop below:
    bin_size, bin_num_detected = np.zeros(n_bins).astype('int'), np.zeros(n_bins).astype('int')
    bin_median = np.zeros(n_bins)
    bin_two_sigma = np.zeros(n_bins)
    two_sigma_upper, two_sigma_lower = np.zeros(n_bins), np.zeros(n_bins)

    # For each bin, find number of data points and detection probability
    for i in range(n_bins):

        # Set boundary of bin
        bin_min = bins[i]
        bin_max = bins[i + 1]
        bin_median[i] = (bin_min + bin_max) / 2

        # Select data within the bin range
        binned_data = operator_df.loc[operator_df.release_rate_kgh < bin_max].loc[
            operator_df.release_rate_kgh >= bin_min]

        # Count the total number of overpasses detected within each bin
        bin_num_detected[i] = binned_data.operator_detected.sum()

        n = len(binned_data)
        bin_size[i] = n  # this is the y-value for the bin in the plot
        p = binned_data.operator_detected.sum() / binned_data.shape[0] if binned_data.shape[0] > 0 else 0 
        detection_probability[i] = p

        # Standard Deviation of a binomial distribution
        sigma = np.sqrt(p * (1 - p) / n) if n > 0 else 0
        bin_two_sigma[i] = 2 * sigma

        # Find the lower and upper bound defined by two sigma
        two_sigma_lower[i] = 2 * sigma
        two_sigma_upper[i] = 2 * sigma
        if 2 * sigma + p > 1:
            two_sigma_upper[i] = 1 - p  # probability cannot exceed 1
        if p - 2 * sigma < 0:
            two_sigma_lower[i] = p  # if error bar includes zero, set lower bound to p?

    detection_prob = pd.DataFrame({
        "bin_median": bin_median,
        "detection_prob_mean": detection_probability,
        "detection_prob_two_sigma_upper": two_sigma_upper,
        "detection_prob_two_sigma_lower": two_sigma_lower,
        "n_data_points": bin_size,
        "n_detected": bin_num_detected})

    return detection_prob, operator_df

def make_detection_limit_plot(ax, operator, n_bins, threshold):
    """
      :param ax: subplot to plot
      :param operator: name of operator
      :param threshold: max value for plotting (we are looking at all releases under threshold value)
      :param n_bins: number of bins for plot
      """

    detection_plot, operator_df = make_detection_limit_df(operator, n_bins, threshold)

    # Set bin width:
    w = threshold / n_bins / 2.5

    # Use n_bins set above
    # Annotate probability of detectoin
    for i in range(n_bins):
        ax.annotate(f'{detection_plot.n_detected[i]}/{detection_plot.n_data_points[i]}',
                    [detection_plot.bin_median[i] - w / 1.8, -0.06], fontsize=9)

    # for plotting purpose, we don't want a small hyphen indicating zero uncertainty interval
    detection_plot.loc[detection_plot.detection_prob_two_sigma_lower == 0, 'detection_prob_two_sigma_lower'] = np.nan
    detection_plot.loc[detection_plot.detection_prob_two_sigma_upper == 0, 'detection_prob_two_sigma_upper'] = np.nan
    detection_plot.loc[detection_plot.detection_prob_mean == 0, 'detection_prob_mean'] = np.nan

    # To avoid RuntimeWarning: All-NaN axis encountered, set yerr to None if all values are np.nan in sigma values
    # (this is the case for Carbon Mapper)

    sigma_lower = detection_plot.detection_prob_two_sigma_lower
    sigma_upper = detection_plot.detection_prob_two_sigma_upper

    if sigma_lower.isnull().all() or sigma_upper.isnull().all():
        y_error = None
    else:
        y_error = [sigma_lower, sigma_upper]

    # Plot bars and detection points
    ax.bar(detection_plot.bin_median,
           detection_plot.detection_prob_mean,
           yerr=y_error,
           error_kw=dict(lw=2, capsize=3, capthick=1, alpha=0.3),
           width=threshold / n_bins - 0.5, alpha=0.6, color='#9ecae1', ecolor='black', capsize=2)

    x_data = operator_df.release_rate_kgh
    ax.scatter(x_data, np.multiply(operator_df.operator_detected, 1),
               facecolors='black',
               marker='|')

    # Add more room on top and bottom
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([0, threshold + 0.5])

    # Axes formatting and labels
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=11)

    # x-tick labels
    x_range = np.arange(0, threshold + threshold/n_bins, threshold/n_bins)
    ax.set_xticks(x_range)

    # Format the tick labels with the desired number of decimal places
    tick_labels = ["{:.0f}".format(tick) for tick in x_range] # set so no decimal points are shown
    ax.set_xticklabels(tick_labels, fontsize=11)

    ax.set_xlabel('Methane Release Rate (kg / hr)', fontsize=14)
    ax.set_ylabel('Proportion detected', fontsize=14)
    ax.tick_params(direction='in', right=True, top=True)
    ax.tick_params(labelsize=12)
    ax.minorticks_on()
    ax.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)
    ax.tick_params(direction='in', which='minor', length=3, bottom=False, top=False, left=True, right=True)
    ax.tick_params(direction='in', which='major', length=6, bottom=True, top=False, left=True, right=True)

    # Set minor tick marks below y=1
    minor_ticks = np.arange(0.1, 1.0, 0.05)  # Minor tick positions below y=1
    ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))

    # Set axes and background color to white
    ax.set_facecolor('white')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')

    # Set more room on top for annotation
    ax.set_ylim([-0.10, 1.22])
    ax.set_xlim([0, threshold])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=11)

    text = f'{gen_operator_title(operator)}'

    ax.annotate(text, xy=(0.03, 0.89), xycoords='axes fraction', fontsize=13)

    return ax

def plot_logistic_regression(ax, threshold, operator):
    
    operator_model = make_logistic_regression(operator=operator, threshold=threshold)

    # input values for plotting
    x_plot = np.linspace(0, threshold, 500)

    # calculate the probability from the model
    y_plot = operator_model.predict_proba(x_plot.reshape(-1, 1))[:, 1]
    ax.plot(x_plot, y_plot, label='Logistic best fit')
    return ax

def plot_detection_limit(operator, n_bins, threshold, save_plot=False, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax = make_detection_limit_plot(ax=ax, operator=operator, n_bins=n_bins,threshold=threshold)
    else:
        ax = make_detection_limit_plot(ax=ax, operator=operator, n_bins=n_bins,threshold=threshold)
        
    try:
        ax = plot_logistic_regression(ax, threshold=threshold, operator=operator)
    except ValueError as e:
        if 'at least 2 classes' in str(e):
            print(f'{operator} detected all non-zero releases.')

    if save_plot:
        if isinstance(save_plot, str):
            plt.savefig(save_plot)
        else:
            save_pod_plot(operator)

    if ax is None:
        plt.show()
        
    return
    
def save_pod_plot(operator):

    now = datetime.datetime.now()
    save_time = now.strftime("%Y%m%d")
    op_ab = operator.lower()
    fig_name = f'detect_limit_{op_ab}_{save_time}.png'
    fig_path = pathlib.PurePath('04_figures', 'detection_limit', fig_name)
    plt.savefig(fig_path)


def remove_wind_label(ax):
    handles, labels = ax.get_legend_handles_labels()

    # Filter out the items that contain the word 'Windspeed' in the label
    filtered_handles_labels = [(h, l) for h, l in zip(handles, labels) if 'Wind' not in l and 'Windspeed' not in l]
    handles, labels = zip(*filtered_handles_labels) if filtered_handles_labels else ([], [])

    # Update the legend with the filtered handles and labels
    ax.legend(handles, labels)
    
    return ax



# %% Make logistic regression for detection probability 

def make_logistic_regression(operator, threshold=30, qc_type='stanford_kept'):
    operator_df = load_release_summary(operator=operator)
    
    # QC check
    operator_df = operator_df[(operator_df[f'{qc_type}'] == True)]
    
    # filter out zero releases
    operator_df = operator_df[operator_df.non_zero_release == True]

    # Include all variables used in Conrad et al., 2023 even if I won't ultimately use them
    op_POD_variables = pd.DataFrame()
    op_POD_variables['release_id'] = operator_df['release_id']
    op_POD_variables['release_date'] = operator_df['release_date']
    op_POD_variables['Q'] = operator_df['release_rate_kgh'] # release rate in kg / hr
    op_POD_variables['D'] = operator_df['operator_detected'].apply(int) # operator detected as 0 or 1

    # Filter op_POD_variables to only include data below a threshold max kgh
    op_POD_variables = op_POD_variables.loc[op_POD_variables.Q <= threshold]

    # Reshape required for Logistic function
    x = op_POD_variables['Q'].values.reshape(-1, 1)
    y = op_POD_variables['D'].values.reshape(-1, 1).ravel()
    model = LogisticRegression(solver='liblinear', random_state=0)
   
    return model.fit(x, y)

def make_releases_histogram(operator, save_plot=False, fig=None, ax=None, qc_type='stanford_kept'):
    ############## Setup Data ##############

    # Create bins for middle histogram plot
    threshold_lower = 0
    threshold_upper = 50
    n_bins = 10
    op_histogram_low = classify_histogram_data(operator=operator, 
                                               threshold_lower=threshold_lower, threshold_upper=threshold_upper,
                                               n_bins=n_bins,
                                               qc_type=qc_type,
                                             )

    # Create bins for right histogram plot
    threshold_lower = 50
    threshold_upper = 350
    n_bins = 12
    op_histogram_high = classify_histogram_data(operator=operator, 
                                                threshold_lower=threshold_lower, threshold_upper=threshold_upper,
                                                n_bins=n_bins,
                                                qc_type=qc_type,
                                               )

    ############## Figure ##############
    if ax is None:
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3,
                                            figsize=(10, 3),
                                            gridspec_kw={'width_ratios': [0.6, 3, 4]})
        plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.05,
                        hspace=0.05)
    else:
        ax1=ax[0]
        ax2=ax[1]
        ax3=ax[2]

    # Determine max value for the y-axis
    low_height = op_histogram_low.bin_height.max()
    high_height = op_histogram_high.bin_height.max()
    y_height = max(low_height, high_height)
    # y_height = math.ceil(y_height / 5) * 5
    y_height = y_height + 1

    # Ram's colors:
    seshadri = ['#c3121e', '#0348a1', '#ffb01c', '#027608', '#0193b0', '#9c5300', '#949c01', '#7104b5']
    #           0sangre,    1neptune,  2pumpkin,  3clover,  4denim,     5cocoa,     6cumin  7berry

    # Color scheme
    tp_color = seshadri[3]
    tn_color = seshadri[1]
    fp_color = seshadri[2]
    fn_color = seshadri[0]
    su_color = seshadri[4]
    op_color = seshadri[5]

    ####### Left histogram #######
    bar_width = 0.2
    # add true negatives
    ax1.bar(0, op_histogram_low.true_negative, width=bar_width, edgecolor='black', color=tn_color)

    # # Zero release discarded by SU
    su_filter_height0 = op_histogram_low.true_negative
    ax1.bar(0, op_histogram_low.zero_filter_su, bottom=su_filter_height0, width=bar_width, label='Stanford Filtered',
            edgecolor='black', color=su_color)

    # Zero release discarded by operator
    op_filter_height0 = np.add(su_filter_height0, op_histogram_low.zero_filter_su).tolist()
    ax1.bar(0, op_histogram_low.zero_filter_op, bottom=op_filter_height0, width=bar_width, label='Operator Filtered',
            edgecolor='black', color=op_color)

    ####### Middle histogram #######
    bar_width = 4.2
    # Middle plot

    # Add True Positives
    ax2.bar(op_histogram_low.bin_median, op_histogram_low.true_positive, width=bar_width,
            label='True positive', edgecolor='black', color=tp_color)

    # Add False Positives
    ax2.bar(op_histogram_low.bin_median, op_histogram_low.false_positive, bottom=op_histogram_low.true_positive,
            width=bar_width, label='False positive', edgecolor='black', color=fp_color)

    # Add False Negatives
    fn_height = np.add(op_histogram_low.true_positive, op_histogram_low.false_positive).tolist()
    ax2.bar(op_histogram_low.bin_median, op_histogram_low.false_negative, bottom=fn_height,
            width=bar_width, label='False Negative', edgecolor='black', color=fn_color)

    # # Add Stanford QC
    su_filter_height = np.add(fn_height, op_histogram_low.false_negative).tolist()
    ax2.bar(op_histogram_low.bin_median, op_histogram_low.filter_stanford, bottom=su_filter_height, width=bar_width,
            label='Stanford Filtered', edgecolor='black', color=su_color)

    # Add Operator QC
    op_filter_height = np.add(su_filter_height, op_histogram_low.filter_stanford).tolist()
    ax2.bar(op_histogram_low.bin_median, op_histogram_low.filter_operator, bottom=op_filter_height, width=bar_width,
            label='Operator Filtered', edgecolor='black', color=op_color)


    ####### Right plot #######

    # reset bin width
    bar_width = 22.5
    # Add True Positives
    ax3.bar(op_histogram_high.bin_median, op_histogram_high.true_positive, width=bar_width, label='True positive',
            edgecolor='black', color=tp_color)

    # Add False Positives
    ax3.bar(op_histogram_high.bin_median, op_histogram_high.false_positive, bottom=op_histogram_high.true_positive,
            width=bar_width, label='False positive', edgecolor='black', color=fp_color)

    # Add False Negatives
    fn_height = np.add(op_histogram_high.true_positive, op_histogram_high.false_positive).tolist()
    ax3.bar(op_histogram_high.bin_median, op_histogram_high.false_negative, bottom=fn_height,
            width=bar_width, label='False Negative', edgecolor='black', color=fn_color)

    # # Add Stanford QC
    su_filter_height = np.add(fn_height, op_histogram_high.false_negative).tolist()
    ax3.bar(op_histogram_high.bin_median, op_histogram_high.filter_stanford, bottom=su_filter_height, width=bar_width,
            label='Stanford Filtered', edgecolor='black', color=su_color)

    # Add Operator QC
    op_filter_height = np.add(su_filter_height, op_histogram_high.filter_stanford).tolist()
    ax3.bar(op_histogram_high.bin_median, op_histogram_high.filter_operator, bottom=op_filter_height, width=bar_width,
            label='Operator Filtered', edgecolor='black', color=op_color)


    ############ Plot formatting ############
    # Set height of x and y axis limits
    # Left plot only shows zero
    ax1.set_ylim(bottom=0, top=y_height)
    ax1.set_xlim([-0.25, 0.25])

    # Middle plot shows >0 to 50 kgh
    ax2.set_ylim(bottom=0, top=y_height)
    ax2.set_xlim(left=-0.5, right=51)

    # Right plot shows 50 to 350
    ax3.set_ylim(bottom=0, top=y_height)
    ax3.set_xlim(left=30, right=350)

    if ax is None:
        # Common label for x-axis on all suplots
        txt_x_label = fig.text(0.5, -0.08, 'Release Rate (kgh)', ha='center', va='bottom', fontsize=14)

        # Plot title
        txt_title = fig.text(0.5, 1, f'{operator} Results Classification', ha='center', va='top', fontsize=15)

    # Axes formatting and labels
    ax1.set_xticks([0])  # only have a tick at 0
    ax1.set_ylabel('Number of Releases', fontsize=14)
    ax1.tick_params(labelsize=12)
    ax1.minorticks_on()
    ax1.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)  # only label left & bottom axis
    ax1.tick_params(direction='in', which='major', axis='y', length=4, left=True, right=True)  # y-axis major
    ax1.tick_params(direction='in', which='minor', length=2, left=True, right=True)  # y-axis minor
    ax1.tick_params(direction='out', axis='x', which='major', length=4, bottom=True, top=False)  # x-axis major

    # Format axes on middle plot
    ax2.tick_params(labelsize=12)
    ax2.minorticks_on()
    ax2.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=False)  # only label bottom axis
    ax2.tick_params(direction='in', which='major', axis='y', length=4, left=True, right=True)  # y-axis major
    ax2.tick_params(direction='in', which='minor', length=2, left=True, right=True)  # y-axis minor
    ax2.tick_params(direction='out', axis='x', which='major', length=4, bottom=True, top=False)  # x-axis major
    ax2.tick_params(which='minor', axis='x', bottom=False, top=False)
    x_ticks = ax2.xaxis.get_major_ticks()
    x_ticks[1].label1.set_visible(False)  # remove label on x=0
    x_ticks[1].set_visible(False)

    # Format axes on right plot
    ax3.tick_params(labelsize=12)
    ax3.minorticks_on()
    ax3.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=False)  # only label on bottom
    ax3.tick_params(axis='y', which='major', direction='in', length=4, left=True, right=True)  # y-axis major
    ax3.tick_params(axis='y', which='minor', direction='in', length=2, left=True, right=True)  # y-axis minor
    ax3.tick_params(direction='out', axis='x', which='major', length=4, bottom=True, top=False)  # x-axis major
    ax3.tick_params(which='minor', axis='x', bottom=False, top=False)

    # Set axes and background color to white
    ax1.set_facecolor('white')
    ax1.spines['top'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['bottom'].set_color('black')

    # Add legend

    histogram_legend = {
        'True Positive': tp_color,
        'True Negative': tn_color,
        'False Positive': fp_color,
        'False Negative': fn_color,
        'Stanford Filtered': su_color,
        'Operator Filtered': op_color,
    }

    legend_elements = [Patch(facecolor=v, edgecolor='black', label=k) for k, v in histogram_legend.items()]
    lgd = ax3.legend(title='Release Key', handles=legend_elements, loc='upper right')
    
    if save_plot:
        plt.savefig(f'report/plots/histograms/{operator}_histogram.png')
        
    if ax is None:
        plt.show()


def make_combined_operator_histogram_plots(OPERATOR_LIST):
    fig, axes = plt.subplots(len(OPERATOR_LIST), 3, figsize=(14, 2.5 * len(OPERATOR_LIST)), gridspec_kw={'width_ratios': [0.6, 3, 4]})

    for i, operator in enumerate(OPERATOR_LIST):
        make_releases_histogram(operator, fig=fig, ax=[axes[i, 0], axes[i, 1], axes[i, 2]])
        axes[i, 0].set_ylabel(f'{operator} \n Number of Releases', fontsize=14)
        
    plt.subplots_adjust(left=0.05,
                        bottom=0.02,
                        right=0.98,
                        top=0.99,
                        wspace=0.05,
                        hspace=0.15)
    
    # Common label for x-axis on all suplots
    txt_x_label = fig.text(0.5, 0, 'Release Rate (kgh)', ha='center', va='bottom', fontsize=14)
    # Plot title
    txt_title = fig.text(0.5, 1, f'Operator Results Classification', ha='center', va='top', fontsize=15)