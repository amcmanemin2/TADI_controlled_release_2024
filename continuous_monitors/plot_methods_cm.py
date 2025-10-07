# Script for methods for generating figures
# Author: Audrey McManemin
# Modified from code written by Sahar H. El Abbadi
# Date Created: 2024-09-10
# Date Last Modified: 2024-09-12

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
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.patches import Patch
import matplotlib.offsetbox as offsetbox
import matplotlib.ticker as ticker

from methods_source_cm import load_release_summary, make_logistic_regression
# from writing_analysis import calculate_residuals_and_error

# %% Functions for making parity plots

def get_parity_data(operator, error_type='operator_reported', operator_list=[]):
    """

    :param operator: name of operator
    :param error_type: indicate type of error. Default is 95% CI, alternative is "operator_reported"

    :return save_parity_data: dataframe with columns release_rate, operator_report, and operator_sigma [lower, upper]
    """

    # Load release summary csv file
    if operator == 'all':
        df_list = []
        for op in operator_list:
            df_list.append(load_release_summary(op))
        operator_plot = pd.concat(df_list)
    else: 
        operator_plot = load_release_summary(operator=operator)
    
    # Apply the following filters to release data :
    # Must pass all QC filters:
    operator_plot = operator_plot[(operator_plot.qc_summary == 'pass_all')]

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

    return save_parity_data, data_description


def make_parity_plot(data, data_description, ax, plot_lim='largest_kgh', operator_list=[]):
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

    # Fit linear regression via least squares with numpy.polyfit
    # m is slope, intercept is b
    m, b = np.polyfit(x_data, y_data, deg=1)

    # Calculate R^2 value
    # (using method described here: https://www.askpython.com/python/coefficient-of-determination)
    correlation_matrix = np.corrcoef(x_data, y_data)
    correlation = correlation_matrix[0, 1]
    r2 = correlation ** 2

    # Number of valid overpasses:
    sample_size = len(y_data)

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
    else:
        largest_kgh = max(plot_lim)

    # Create sequence of numbers for plotting linear fit (x)
    x_seq = np.linspace(0, largest_kgh, num=100)

    ############ Generate Figure  ############

    # Add linear regression to in put ax
    ax.plot(x_seq, m * x_seq + b, color='k', lw=2,
            label=f'Best Fit, $R^2 =$ {r2:0.2f}\n$y = {m:0.2f}x+{b:0.2f}$')

    # Add parity line
    # With label:
    # ax.plot(x_seq, x_seq, color='k', lw=2, linestyle='--',
    #          label='Parity Line')
    # Without label:
    ax.plot(x_seq, x_seq, color='k', lw=2, linestyle='--')

    # Add scatter plots with error bars
    ax.errorbar(x_data, y_data,
                xerr=x_error,
                yerr=[y_error_lower, y_error_upper],
                linestyle='none',
                mfc='white',
                label=f'n = {sample_size}\n({legend_error})',
                fmt='o',
                markersize=5)

    # Set title
    ax.set_title(f'{operator}')

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
    # ax.set_xlabel('Methane Release Rate (kgh)', fontsize=14)
    # ax.set_ylabel('Reported Release Rate (kgh)', fontsize=14)
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


def plot_parity(operator, plot_lim='largest_kgh', save_parity_data=False):

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
    save_time = now.strftime("%Y%m%d")
    fig_name = f'{op_ab}_parity_{save_time}'
    fig_path = pathlib.PurePath('04_figures', 'parity_plots', fig_name)
    plt.savefig(fig_path)
    plt.show()

    # Save data used to make figure
    if save_parity_data:
        save_path = pathlib.PurePath('03_results', 'parity_plot_data', f'{op_ab}_parity_{save_time}.csv')
        parity_data.to_csv(save_path)

    return

# %% detection probability plot methods

def make_detection_limit_df(operator, n_bins, threshold):
    
    # Load release summary for operator
    operator_df = load_release_summary(operator)

    # Apply QC filter
    operator_df = operator_df[(operator_df.qc_summary == 'pass_all')]

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
        bin_num_detected[i] = binned_data.detected.sum()

        n = len(binned_data)
        bin_size[i] = n  # this is the y-value for the bin in the plot
        p = binned_data.detected.sum() / binned_data.shape[0];  # df.shape[0] gives number of rows
        detection_probability[i] = p

        # Standard Deviation of a binomial distribution
        sigma = np.sqrt(p * (1 - p) / n)
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
    ax.scatter(x_data, np.multiply(operator_df.detected, 1),
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

    text = f'{operator}'

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

def plot_detection_limit(operator, n_bins, threshold, save_plot=False):
    
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax = make_detection_limit_plot(ax, operator=operator, n_bins=n_bins,threshold=threshold)

    try:
        ax = plot_logistic_regression(ax, threshold=threshold, operator=operator)
    except ValueError as e:
        if 'at least 2 classes' in str(e):
            print(f'{operator} detected all non-zero releases.')

    if save_plot:
        save_pod_plot(operator)

    plt.show()
    return
    
def save_pod_plot(operator):

    now = datetime.datetime.now()
    save_time = now.strftime("%Y%m%d")
    op_ab = operator.lower()
    fig_name = f'detect_limit_{op_ab}_{save_time}.png'
    fig_path = pathlib.PurePath('04_figures', 'detection_limit', fig_name)
    plt.savefig(fig_path)