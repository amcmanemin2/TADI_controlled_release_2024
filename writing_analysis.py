from methods import load_release_summary, OPERATOR_TYPE
import pandas as pd
import numpy as np

def classify_confusion_categories(release_summary):
    """Takes input that is an releases summary dataframe and outputs counts of true positive, false positive, true negative, false negative"""

    true_positives = release_summary.query('non_zero_release == True & operator_detected == True')
    false_positives = release_summary.query('non_zero_release == False & operator_detected == True')
    false_negatives = release_summary.query('non_zero_release == True & operator_detected == False')
    true_negatives = release_summary.query('non_zero_release == False & operator_detected == False')

    # Filtered zeros:

    return true_positives, false_positives, true_negatives, false_negatives


# %%
def classify_histogram_data(operator, threshold_lower, threshold_upper, n_bins, qc_type='stanford_kept'):
    # Load operator release data
    if operator in OPERATOR_TYPE.keys():
        op_reported = pd.DataFrame()
        for op in OPERATOR_TYPE[operator]:
            op_reported = pd.concat([op_reported, load_release_summary(operator=op)])
    else:
        op_reported = load_release_summary(operator=operator)

    # Pass all QC filter
    op_qc_pass = op_reported[(op_reported[f'{qc_type}'] == True)]

    # Select non-zero releases detected by operator
    tp, fp, tn, fn = classify_confusion_categories(op_qc_pass)

    bin_median = make_histogram_bins(tp, threshold_lower, threshold_upper, n_bins).bin_median
    count_tp = make_histogram_bins(tp, threshold_lower, threshold_upper, n_bins).n_data_points
    count_fp = make_histogram_bins(fp, threshold_lower, threshold_upper, n_bins).n_data_points
    count_fn = make_histogram_bins(fn, threshold_lower, threshold_upper, n_bins).n_data_points
    count_tn = make_histogram_bins(tn, threshold_lower, threshold_upper, n_bins).n_data_points

    # Filtered by Stanford
    # Non-zero SU QC fails
    su_qc_fail = op_reported.query('stanford_kept == False & strict_qc_kept == True & non_zero_release == True')
    count_su_fail = make_histogram_bins(su_qc_fail, threshold_lower, threshold_upper, n_bins).n_data_points

    # Zero SU QC fails
    zero_su_qc_fail = op_reported.query('stanford_kept == False & non_zero_release == False & strict_qc_kept == True')
    count_zero_su_fail = make_histogram_bins(zero_su_qc_fail, threshold_lower, threshold_upper, n_bins).n_data_points

    # Filtered by Operator
    # Non-zero
    op_qc_fail = op_reported.query('stanford_kept == True & operator_kept == False & non_zero_release == True')
    count_op_fail = make_histogram_bins(op_qc_fail, threshold_lower, threshold_upper, n_bins).n_data_points

    # Zero
    zero_op_qc_fail = op_reported.query('stanford_kept == True & operator_kept == False & non_zero_release == False')
    count_zero_op_fail = make_histogram_bins(zero_op_qc_fail, threshold_lower, threshold_upper, n_bins).n_data_points

    # Count total measurements reported by operator
    total_reported = op_reported.shape[0]

    # Count total releases
    total_releases = total_reported

    ################## store data #########################

    summary = pd.DataFrame({
        'bin_median': bin_median,
        'true_positive': count_tp,
        'false_positive': count_fp,
        'true_negative': count_tn,
        'false_negative': count_fn,
        'filter_stanford': count_su_fail,
        'filter_operator': count_op_fail,
        'zero_filter_su': count_zero_su_fail,
        'zero_filter_op': count_zero_op_fail,
        'total_releases': total_releases,
        'total_reported': total_reported,
    })

    # Determine max bin height for plotting:
    # exclude zeros, zero releases were targeted at 10% of all other releases
    col_for_summing = ['true_positive',
                       'false_positive',
                       'false_negative',
                       'filter_stanford',
                       'filter_operator',
                   ]

    summary['bin_height'] = summary[col_for_summing].sum(axis=1)

    return summary


def make_histogram_bins(df, threshold_lower, threshold_upper, n_bins):
    bins = np.linspace(threshold_lower, threshold_upper, n_bins + 1)

    # These variables are for keeping track of values as I iterate through the bins in the for loop below:
    bin_count, bin_num_detected = np.zeros(n_bins).astype('int'), np.zeros(n_bins).astype('int')
    bin_median = np.zeros(n_bins)

    # For each bin, find number of data points and detection probability

    for i in range(n_bins):
        # Set boundary of bin
        bin_min = bins[i]
        bin_max = bins[i + 1]
        bin_median[i] = (bin_min + bin_max) / 2

        binned_data = df.query('release_rate_kgh < @bin_max & release_rate_kgh >= @bin_min')
        bin_count[i] = len(binned_data)

    detection_prob = pd.DataFrame({
        "bin_median": bin_median,
        "n_data_points": bin_count,
    })

    return detection_prob


def test_parity(x_value, y_upper, y_lower):
    """Test if a given y-value and associated error pass the parity line. Returns boolean True or False"""

    # Test if x_value is in between y_upper and y_lower
    if (x_value <= y_upper) and (x_value >= y_lower):
        return True
    else:
        return False


def calc_parity_intersection(operator):
    """Determine the percent of quantification estimates that cross the parity line"""

    all_releases = load_release_summary(operator=operator)

    # Only consider points that pass all QC
    releases = all_releases.loc[all_releases.pass_all_qc == True].copy()

    # Only consider releases where operator quantification estimate is a real number
    releases = releases[releases.operator_quantification.notnull()]

    # Only consider non-zero releases
    releases = releases[releases.non_zero_release == True]

    # Operator quantification > 0
    releases = releases[releases.operator_quantification > 0]


    # Multiplier for concerting uncertainty
    uncertainty_multiplier = {
        '1-sigma': 1.96,
        '95_CI': 1,
    }

    op_ab = operator.lower()
    
    releases['intersect_parity_line'] = releases.apply(lambda x: test_parity(x['release_rate_kgh'],
                                                                                 x['operator_upper'],
                                                                                 x['operator_lower']),
                                                           axis=1)

    cross_parity = len(releases.loc[releases.intersect_parity_line == True])
    percent_cross_parity = cross_parity / len(releases)
    print(
        f'Fraction of {operator} with operator reported error bars that encompasses parity line: {percent_cross_parity * 100:.0f}%')

    return releases


def calc_residual(x, y, m, b):
    y_fit = m * x + b
    residual = y - y_fit
    return residual

def calc_residual_percent(x, y, m, b):
    y_fit = m * x + b
    residual_percent = (y - y_fit) / y_fit * 100
    return residual_percent

def calc_error_absolute(expected, observed):
    return observed - expected

def calc_error_percent(expected, observed):
    """Calculate perfect error between an observation and the expected value. Returns value as percent.  """

    # Remove zeros, don't divide by zero
    if expected == 0:
        # True zeros where expected and observed values are both zero
        if observed == 0:
            return 0
        else:
            return np.nan

    #
    # if observed == 0:
    #     return np.nan

    # keep releases that aren't quantified in series so it can be aligned later
    if pd.isnull(observed):
        return np.nan
    else:
        return (observed - expected) / expected * 100

def calculate_residuals_and_error(operator, qc_status):
    """ Calculate the measurement residuals for operator
    qc_status can be: 'pass_all', 'all_points', 'pass_operator'
    """
    release_summary = load_release_summary(operator)

    # Remove rows where operator did not quantify
    release_summary = release_summary.dropna(subset='operator_quantification')

    # Select which QC we want
    if qc_status == 'pass_all':
        # Pass SU QC
        # Pass operator QC
        # Must be a non-zero release
        # Operator quantification estimate must have been > 0

        qc_mask = (release_summary['qc_summary'] == 'pass_all') & \
                  (release_summary['non_zero_release'] == True) & \
                  (release_summary['operator_quantification'] > 0)
    elif qc_status == 'pass_operator':
        qc_mask = (release_summary['operator_kept'] == True)
    elif qc_status == 'all_points':
        qc_mask = release_summary['operator_quantification'].notna() # generic mask to select all points in dataset

    data = release_summary.loc[qc_mask].copy()

    # Set x and y data
    data['meter_data'] = release_summary.release_rate_kgh
    data['operator_data'] = release_summary.operator_quantification
    data['qc'] = release_summary.qc_summary

    # Fit linear regression via least squares with numpy.polyfit
    # m is slope, intercept is b
    m, b = np.polyfit(data.meter_data, data.operator_data, deg=1)
    
    correlation_matrix = np.corrcoef(data.meter_data, data.operator_data)
    correlation = correlation_matrix[0, 1]
    r2 = correlation ** 2

    # Calculate the residual for each row
    data['residual'] = data.apply(lambda dataset:
                                  calc_residual(dataset['meter_data'],
                                                dataset['operator_data'], m, b), axis=1)

    data['residual_percent_error'] = data.apply(lambda dataset:
                                                calc_residual_percent(dataset['meter_data'],
                                                                      dataset['operator_data'], m, b), axis=1)

    data['quant_error_absolute'] = data.apply(lambda dataset:
                                              calc_error_absolute(dataset['meter_data'],
                                                                            dataset['operator_data']), axis=1)

    data['quant_error_percent'] = data.apply(lambda dataset: calc_error_percent(dataset['meter_data'],
                                                                                dataset['operator_data']), axis=1)
    
    data['slope'] = m
    data['intercept'] = b
    data['r2'] = r2
    
    data['release_rate_0.50x'] = data['release_rate_kgh'] * 0.50
    data['release_rate_1.50x'] = data['release_rate_kgh'] * 1.50
    
    data['operator_within_0.50x'] = data.apply(lambda x: test_parity(x['operator_data'], x['release_rate_1.50x'], x['release_rate_0.50x']), axis=1)

    return data

def determine_relevant_error_ranges(operator, qc_status):
    """ Determine the relevant ranges """
    op_stage = calculate_residuals_and_error(operator, qc_status)
    max_residual = op_stage.residual.max()
    min_residual = op_stage.residual.min()
    max_error_percent = op_stage.quant_error_percent.max()
    min_error_percent = op_stage.quant_error_percent.min()
    max_error_absolute = op_stage.quant_error_absolute.max()
    min_error_absolute = op_stage.quant_error_absolute.min()
    slope = op_stage.slope.mean()
    percent_within_50pct = op_stage['operator_within_0.50x'].sum() / len(op_stage) * 100

    relevant_ranges = {
        'operator': operator,
        'max_residual': max_residual,
        'min_residual': min_residual,
        'max_error_percent': max_error_percent,
        'min_error_percent': min_error_percent,
        'max_error_absolute': max_error_absolute,
        'min_error_absolute': min_error_absolute,
        'slope': slope,
        'r2': op_stage.r2.mean(),
        'pct_op_estimates_within_50pct_flowrate': percent_within_50pct
    }

    return relevant_ranges