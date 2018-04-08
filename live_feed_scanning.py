"""
This script is aimed at identifying whether a Dryer activity is present in the live feed.It identifies power jumps over
threshold, groups them in blocks based on proximity in time, computes statistics
for each block and, based on them to make a decision on presence of the Dryer.

Unfiltered livefeed has been taken for 7 houses: 4 with Dryers, 2 without Dryers and 1 with Washer Dryer.
 Analysis was done on power deltas: jump were identified as change over 1000 watts. These jumps were grouped into
 blocks, blocks were considered separate if a gap between consecutive jumps is over 15 mins. Low/high power threshold
 for the house was identified as house median power + 500 watts. Then for each block 3 statistics were calculated:
 Median of length of high power periods; Median of length of low power periods; Standard deviation of length of high
 power periods. Dryer has been identified where at least in one block all 3 conditions were met: There are more than 10
  towers; Median of length of low power periods < 200 seconds; Heating towers appear every 5 mins on average;
  Standard deviation of length of high power periods is no greater than the median of length of high power periods.
"""

from datetime import datetime
import pandas as pd
from trial_constants import FOLDER_PARAMS, SUBM_FOLDER, FLAC_PATH, OUTPUT_PATH
import numpy as np
from matplotlib import pyplot as plt
# from utils.plotting import save_fig
import logging
# from processing.submeters.submeter_analysis import get_scores
# from utils.constants import Ws_PER_kWh
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('find_event_times')

DRYER = 'Dryer'
Ws_PER_kWh = 1000*3600
MIN_GAP_TO_SEPARATE_BLOCKS = 1200 # time gap between cycles
ADD_TO_MEDIAN = 500  # watts to add to median to determine low/high power threshold
SECONDS_PER_TOWER = 300  # expecting a tower every 5 minutes
LOW_POWER_LEN_THRESH = 200  # threshold for low power period length median
JUMP_THRESHOLD = 1000
MIN_TOWERS = 10 # minimum number of towers for a dryer block
EXTEND_CYCLE = 600 # extend ground truth cycle to allow non-overlapping blocks to be included in fp
MIN_BLOCKS_LEN_CONSIDERED = 1800
BLOCK_STATS_COLUMNS = ['trial_folder', 'device_name', 'many_towers', 'short_low_power_periods', 'num_towers',
                       'low_pow_t_median', 'high_pow_t_median', 'high_pow_t_std', 'low_power_period_std',
                       'start_utc', 'end_utc', 'start_time', 'end_time', 'energy']
STATISTICS_COLUMNS = BLOCK_STATS_COLUMNS + ['is_dryer', 'day', 'in_ground_truth_cycle']


def make_gt_start_end_times(trial_folder, device_name):
    """
    This function takes start-end times of the cycles from ground truth, groups cycles based on selected time gap
    and outputs a new data frame with separate columns for start and end of the cycle
    :param str trial_folder: location of submeter file
    :param str device_name: name of device
    :return: dataframe with ground truth start-end times for cycles
    :rtype: pd.DataFrame
    """
    st_end_fpath = os.path.join(
        OUTPUT_PATH, 'start_end_times_' + trial_folder + '_' + device_name + '.csv')
    start_end_times = pd.read_csv(st_end_fpath, names=['index', 'time'])
    start_end_times['blocks'] = (start_end_times['time'].diff() >= MIN_GAP_TO_SEPARATE_BLOCKS).astype(int).cumsum()
    grouped = start_end_times.groupby('blocks')['time']
    gt_start_end_times = grouped.agg(['min', 'max']).rename(columns={'min': 'start_time', 'max': 'end_time'})

    return gt_start_end_times


def load_live_feed(trial_folder, device_name, filtered=False):
    """
    Loads livefeed from one of submetered folders and filters it for device active periods if needed
    :param str trial_folder: location of submeter file
    :param bool filtered: flag for whether live feed is filtered for device activity periods
    :param str device_name: name of device
    :return: live feed
    :rtype: pd.DataFrame
    """
    live_path = FLAC_PATH
    live_feed = pd.read_hdf(live_path, 'data')[['Power RMS']]
    live_feed.index /= 100
    live_feed['timestamp'] = live_feed.index
    live_feed['power_deltas'] = live_feed['Power RMS'].diff()

    if filtered:
        st_end_fpath = os.path.join(
            OUTPUT_PATH, 'start_end_times/start_end_times_' + trial_folder + '_' + device_name + '.csv')
        start_end_times = pd.read_csv(st_end_fpath, names=['index', 'time'])
        live_feed = live_feed.loc[start_end_times['time']]
    if trial_folder == 'Lionel_2017_08_31':
        live_feed = live_feed[live_feed.index <= 1504052787]

    return live_feed


def identify_jumps_and_group_in_blocks(live_feed):
    """
    This function loads the data and identifies blocks of activity based on given time gap between large deltas.
    All blocks within 15 mins of each other are grouped into 1 block. Intention is to get something close to high power
    part of each dryer cycle. However, different Dryer cycles close to each other in time will be grouped into one.
    :param str trial_folder: name of the folder in DataDisk/submeter_data
    :param str device_name: name of device
    :param bool filtered: flag for whether live feed is filtered for device activity periods
    :return: live feed and lists of start and end times for blocks
    :rtype: (pd.DataFrame, list, list)
    """

    all_jumps = abs(live_feed['power_deltas']) > JUMP_THRESHOLD
    live_jumps = live_feed[all_jumps]
    live_jumps['jumps_t_deltas'] = live_jumps['timestamp'].diff()

    live_jumps['blocks'] = (abs(live_jumps['jumps_t_deltas']) >= MIN_GAP_TO_SEPARATE_BLOCKS).astype(int).cumsum()

    block_jumps_start_end = live_jumps["timestamp"].groupby(live_jumps['blocks']).agg([np.min, np.max])
    block_start_time = block_jumps_start_end["amin"]
    block_end_time = block_jumps_start_end["amax"]

    return block_start_time, block_end_time


def stats_for_periods_after_step(trial_folder, device_name, block_live_feed, median_power, block_start, block_end,
                                 plot=False):
    """
    This function computes stats relevant for Dryer identification for the given block and optionally saves the plot of
    the live feed in Dryer folder on DataDisk
    :param str trial_folder: name of the folder in DataDisk/submeter_data
    :param str device_name: name of device
    :param block_live_feed: portion of the live feed corresponding to a block
    :param float median_power: median of the live feed values
    :param int block_start: time when the block begins
    :param int block_end: time when the block ends
    :param bool plot: flag for whether to plot and save the block of the live feed when more than one observation is
    present
    :return: (folder name, device name and statistics for the block: flag for multiple towers, flag for whether
    low power periods are short, number of towers, median of the low power periods' length, median of the high power
    periods' length, std of the high power periods' length, time of the block start and block end)
    :rtype: list[Any]
    """

    cycle_len = block_end - block_start
    number_of_towers = sum(block_live_feed['power_deltas'] > JUMP_THRESHOLD)
    block_live_feed['low_power'] = block_live_feed['Power RMS'] < median_power + ADD_TO_MEDIAN
    block_live_feed['power_blocks'] = (abs(block_live_feed['power_deltas']) >= JUMP_THRESHOLD).astype(int).cumsum()
    block_live_feed['cum_block_len'] = block_live_feed.groupby('power_blocks').cumcount() + 1
    grouped_by_low_power = block_live_feed.groupby(['power_blocks', 'low_power']).max().reset_index(level=[1]).\
                            groupby('low_power')
    medians = pd.DataFrame(grouped_by_low_power.median())
    standard_deviations = pd.DataFrame(grouped_by_low_power.std())
    if True in block_live_feed['low_power'].unique():
        # Median
        high_power_period_median = round(medians.loc[False, 'cum_block_len'], 0)
        low_power_period_median = round(medians.loc[True, 'cum_block_len'], 0)
        # Standard deviation
        high_power_period_std = round(standard_deviations.loc[False, 'cum_block_len'], 0)
        low_power_period_std = round(standard_deviations.loc[True, 'cum_block_len'], 0)
    else:
        high_power_period_median, low_power_period_median, high_power_period_std, low_power_period_std = \
            (None, None, None, None)

    many_towers = (number_of_towers >= cycle_len / SECONDS_PER_TOWER) and (number_of_towers >= MIN_TOWERS)
    short_low_power_periods = low_power_period_median <= LOW_POWER_LEN_THRESH
    energy = block_live_feed['Power RMS'].mean()*(block_end - block_start)
    start_utc = datetime.utcfromtimestamp(int(block_start))
    end_utc = datetime.utcfromtimestamp(int(block_end))

    if plot and len(block_live_feed['Power RMS']) > 1 and many_towers:
        fig = plt.figure()
        subplot1 = fig.add_subplot(111)
        block_live_feed['Power RMS'].plot(ax=subplot1)
        plt.title(device_name + ', ' + trial_folder + ', ' + str(start_utc) + ', ' + str(end_utc), fontsize=10)
        img_full_path = os.path.join(OUTPUT_PATH, device_name + ', ' + trial_folder + ', ' + str(start_utc))
        save_fig(img_full_path, ext='png')
        plt.close(fig)

    return [trial_folder, device_name, many_towers, short_low_power_periods, number_of_towers,
            low_power_period_median, high_power_period_median, high_power_period_std, low_power_period_std,
            start_utc, end_utc, block_start, block_end, energy]


def split_live_feed_in_days(trial_folder, device_name, filtered=False):
    """
    This function splits live feed in days and outputs list of days, dictionary of sections of live feed and
    live feed median power
    :param str trial_folder: name of the folder in DataDisk/submeter_data
    :param str device_name: name of device
    :param bool filtered: flag for whether live feed is filtered for device activity periods
    :param bool plot: flag for whether to plot and save the block of the live feed
    :return: data frame with statistics for each block, a list with a flag for whether a dryer was identified
    :rtype: (pd.DataFrame, list[Any])
    """
    live_feed = load_live_feed(trial_folder, device_name, filtered)
    live_feed['date'] = live_feed['timestamp'].apply(lambda x: datetime.utcfromtimestamp(x)).dt.date
    median_power = live_feed['Power RMS'].median()

    days = live_feed['date'].unique()
    live_feeds = {}
    for i, day in enumerate(days):
        live_feeds[day] = live_feed[live_feed['date'] == day]

    return days, live_feeds, median_power


def identify_dryer(trial_folder, device_name, median_power, live_feed_day, plot=False):
    """
    This function combines the stats for all blocks in the live feed and outputs decision regarding the Dryer presence
    :param str trial_folder: name of the folder in DataDisk/submeter_data
    :param str device_name: name of device
    :param float median_power: median power for the whole of live feed
    :param pd.DataFrame live_feed_day: live feed filtered on a single days
    :param bool plot: flag for whether to plot and save the block of the live feed
    :return: a flag for whether a dryer was identified and data frame with statistics for each block, and for potential
    Dryer blocks
    :rtype: (bool, pd.DataFrame, pd.DataFrame)
    """

    # will only identify cycles that are completely within one day
    block_start_time, block_end_time = identify_jumps_and_group_in_blocks(live_feed_day)

    blocks_stats = pd.DataFrame(columns=BLOCK_STATS_COLUMNS)
    for i in range(len(block_start_time)):
        block_start = block_start_time[i]
        block_end = block_end_time[i]
        this_cycle = (live_feed_day.index >= block_start) & (live_feed_day.index < block_end)
        block_live_feed = live_feed_day[this_cycle]
        if len(block_live_feed) > 1:
            blocks_stats.loc[len(blocks_stats)] = stats_for_periods_after_step(trial_folder, device_name,
                                                block_live_feed, median_power, block_start, block_end, plot)
    # TODO: not working with Bill's Miel Dryer: his heating towers are very thin after the 1st one
    blocks_stats["small_high_power_std"] = (blocks_stats["high_pow_t_std"] <= blocks_stats["high_pow_t_median"])
    blocks_stats["is_dryer"] = blocks_stats["small_high_power_std"] & blocks_stats["short_low_power_periods"] & \
                               blocks_stats["many_towers"] &\
                               (blocks_stats['end_time'] - blocks_stats['start_time'] >= MIN_BLOCKS_LEN_CONSIDERED)
    many_towers_blocks_stats = blocks_stats[blocks_stats["num_towers"] >= MIN_TOWERS]
    if len(many_towers_blocks_stats) >= 1:
        is_dryer = any(many_towers_blocks_stats["is_dryer"] == 1)
    else:
        is_dryer = False

    return is_dryer, many_towers_blocks_stats, blocks_stats


def output_f1_and_plot(trial_folder, device_name, submeter_file_name=None, filtered=False, plot=False):
    """
    This function loops through days in data, calls identify_dryer function and outputs f1 score in a table and a plot
    if a Dryer has been identified.
    :param str trial_folder: name of the folder in DataDisk/submeter_data
    :param str device_name: name of device
    :param str submeter_file_name: file with submetered data
    :param bool filtered: flag for whether live feed is filtered for device activity periods
    :param bool plot: flag for whether to plot and save the block of the live feed
    """

    days, live_feeds, median_power = split_live_feed_in_days(trial_folder, device_name, filtered)

    all_potential_dryer_cycles = pd.DataFrame(columns=STATISTICS_COLUMNS)
    all_gt_start_end_times = pd.DataFrame()
    decision_matrix = pd.DataFrame(columns=['trial_folder', 'device_name', 'day', 'is_dryer', 'f1', 'precision', 'recall'])
    blocks_stats = pd.DataFrame(columns=BLOCK_STATS_COLUMNS + ['is_dryer', "small_high_power_std"])

    # bounds to filter on ground truth to make sure f1 calculation is accurate
    if device_name == DRYER:
        gt_start_end_times_0 = make_gt_start_end_times(trial_folder, device_name)
        start_gt = gt_start_end_times_0["start_time"].min()
        end_gt = gt_start_end_times_0["end_time"].max()
    else:
        (start_gt, end_gt) = (0, 0)

    for day in days:

        live_feed_day = live_feeds[day]
        live_feed_day = live_feed_day[(live_feed_day['timestamp'] >= start_gt) & (live_feed_day['timestamp'] <= end_gt)]

        is_dryer, many_towers_blocks_stats, device_blocks_stats = \
            identify_dryer(trial_folder, device_name, median_power, live_feed_day, plot)

        blocks_stats = pd.concat([blocks_stats, device_blocks_stats], ignore_index=True)

        if is_dryer:
            start_live_feed = live_feed_day["timestamp"].min()
            end_live_feed = live_feed_day["timestamp"].max()

            gt_start_end_times = gt_start_end_times_0[(gt_start_end_times_0["start_time"] >= start_live_feed - 3600) &
                                                    (gt_start_end_times_0["end_time"] <= end_live_feed + 3600)]
            def check_block_in_any_gt_cycle(start, end, df):
                result = ((df['start_time'] - EXTEND_CYCLE <= start) & (df['end_time'] + EXTEND_CYCLE >= end)).any()
                return result

            def check_any_block_in_gt_cycle(start, end, df):
                result = ((df['start_time'] >= start - EXTEND_CYCLE) & (df['end_time'] <= EXTEND_CYCLE + end)).any()
                return result

            potential_dryer_cycles = many_towers_blocks_stats[many_towers_blocks_stats["is_dryer"]]
            # making sure gt and live feed have common timestamp
            cycles_detected = len(potential_dryer_cycles) >= 1
            some_gt_events_present = len(gt_start_end_times) >= 1

            if cycles_detected and some_gt_events_present:
                potential_dryer_cycles["in_ground_truth_cycle"] = potential_dryer_cycles.apply(
                    lambda row: check_block_in_any_gt_cycle(row['start_time'], row['end_time'], gt_start_end_times), axis=1)
                gt_start_end_times["block_in_cycle"] = gt_start_end_times.apply(
                    lambda row: check_any_block_in_gt_cycle(row['start_time'], row['end_time'], potential_dryer_cycles), axis=1)
                gt_start_end_times["trial_folder"] = trial_folder
                gt_start_end_times["start_utc"] = pd.to_datetime(gt_start_end_times["start_time"], unit='s')
                gt_start_end_times["end_utc"] = pd.to_datetime(gt_start_end_times["end_time"], unit='s')

                true_positive = len(potential_dryer_cycles[potential_dryer_cycles["in_ground_truth_cycle"] == True])
                false_positive = len(potential_dryer_cycles[potential_dryer_cycles["in_ground_truth_cycle"] == False])
                false_negative = len(gt_start_end_times[gt_start_end_times["block_in_cycle"] == False])
                f1 = get_scores(true_positive, false_positive, false_negative)
                decision = [trial_folder, device_name, day, is_dryer, f1[0], f1[1], f1[2]]
                all_gt_start_end_times = pd.concat([all_gt_start_end_times, gt_start_end_times], ignore_index=True)
                all_potential_dryer_cycles = pd.concat([all_potential_dryer_cycles, potential_dryer_cycles], ignore_index=True)
                decision_matrix.loc[len(decision_matrix)] = decision

    # f1 output and plotting is done when both Dryer is detected and there is a ground truth for the dryer
    if sum(decision_matrix["is_dryer"]) >= 1 and cycles_detected and device_name == DRYER:
        f1, precision, recall = decision_matrix.loc[:, "f1":"recall"].mean()
        # Parameters to pass for plotting
        total_count = len(gt_start_end_times_0)
        total_energy = calculate_total_energy(trial_folder, submeter_file_name)
        summary = [trial_folder, device_name, f1, precision, recall]
        metrics = [f1, precision, recall]
        concatenate_results_plot_f1(metrics, total_count, total_energy, trial_folder)
    else:
        summary = []

    return [all_potential_dryer_cycles, all_gt_start_end_times, summary, decision_matrix, blocks_stats]


def save_df(df, filename):
    """
    saves pandas DataFrame into specific folder
    :param pd.DataFrame df: dataframe to save
    :param filename: name of the file to save df into
    """
    file_path = os.path.join(OUTPUT_PATH, filename + '.csv')
    df.to_csv(file_path)


def calculate_total_energy(trial_folder, submeter_file_name):
    """
    Calculates total energy from submeter data
    :param str trial_folder: name of the folder in DataDisk/submeter_data
    :param str submeter_file_name: file with submetered data
    :return: total energy from the submetered data
    :rtype: float
    """
    trial_path = os.path.join(SUBM_FOLDER, trial_folder)
    subm_path = os.path.join(trial_path, 'Data')
    file_path = os.path.join(subm_path, submeter_file_name)
    submeter_data = pd.read_csv(file_path, sep=' ', names=['timestamp', 'power'])
    start = submeter_data['timestamp'].min()
    end = submeter_data['timestamp'].max()
    # energy = mean power multiplied by length of submetering in seconds
    total_energy = submeter_data['power'].mean()*(end - start)
    # return energy in kWh
    return total_energy/Ws_PER_kWh


def plot_f1(scores_file):
    """
    Plot F1 / precision / recall per appliance
    :param str scores_file: path to file containing f1 scores  for the house
    :return: a flag for whether a produced plot is not empty
    :rtype: bool """

    df = pd.read_csv(scores_file, index_col=0)
    # character index for pretty printing
    df.index = ['%s\n%.1f kWh' % (idx, row['total_energy']) for idx, row in df.iterrows()]
    cmap = plt.get_cmap('magma')
    sub_df = df[['count_f1', 'count_precision', 'count_recall']]
    sub_df.columns = [col.split('_')[-1].title() for col in sub_df.columns]
    sub_df.plot(kind='bar', color=[cmap.colors[int(float(i+1)/5 * 256)] for i in range(4)], rot=0, ylim=(0, 1))
    plt.savefig(scores_file.replace('.csv', '_count.png'))
    generated_figure = len(plt.gcf().axes) > 0
    plt.close()
    return generated_figure


def concatenate_results_plot_f1(metrics, total_count, total_energy, trial_folder):
    """
    Summarises f1 stats for a house and plots the result
    :param list f1: a list containing f1 score, precison and recall for the house
    :param int total_count: count of cycles based on ground truth data
    :param float total_energy: total energy calculated based on submetered data
    :param  str trial_folder: name of the folder in DataDisk/submeter_data
    :return: a flag for whether a produced plot is not empty
    :rtype: bool
    """

    file_path = os.path.join(SUBM_FOLDER, trial_folder + '/analysis_output/scores.csv')
    column_names = ['count_f1', 'count_precision', 'count_recall', 'energy_f1', 'energy_precision',
                    'energy_recall', 'total_energy', 'total_count']
    scores = pd.read_csv(file_path, index_col=0)
    scores.columns = column_names
    dryer_scores = pd.DataFrame({'count_f1': [metrics[0]], 'count_precision': [metrics[1]],
                                'count_recall': [metrics[2]], 'energy_f1': [None], 'energy_precision': [None],
                    'energy_recall': [None], 'total_energy': [total_energy], 'total_count': [total_count]})
    dryer_scores.index = [DRYER]
    scores = pd.concat([scores, dryer_scores], ignore_index=False)

    new_file_path = os.path.join(SUBM_FOLDER, trial_folder, 'analysis_output', 'scores_wDryer.csv')
    scores.to_csv(new_file_path)
    scores.to_csv(new_file_path)
    plot_success = plot_f1(new_file_path)

    return plot_success


def main():
    summary_df = pd.DataFrame(columns=['trial_folder', 'device_name', 'f1', 'precision', 'recall'])
    blocks_stats = pd.DataFrame(columns=BLOCK_STATS_COLUMNS + ['is_dryer', "small_high_power_std"])
    dryer_cycles = pd.DataFrame(columns=STATISTICS_COLUMNS)
    gt_start_end_times = pd.DataFrame(columns=["trial_folder", "block_in_cycle", "start_time", "end_time", "start_utc",
                                               "end_utc"])

    for trial_fol, dev_name, old_fmt, subm_file_name in FOLDER_PARAMS:
        # try:
        potential_dryer_cycles, gt_start_end_times_f, summary, decision_matrix, device_blocks_stats = \
            output_f1_and_plot(trial_fol, dev_name, subm_file_name)
        blocks_stats = pd.concat([blocks_stats, device_blocks_stats], ignore_index=True)
        gt_start_end_times = pd.concat([gt_start_end_times, gt_start_end_times_f], ignore_index=True)

        if sum(decision_matrix["is_dryer"]) >= 1:
            summary_df.loc[len(summary_df)] = summary
            dryer_cycles = pd.concat([dryer_cycles, potential_dryer_cycles], ignore_index=True)

        # except Exception as e:
        #     logger.error("Error running dryer identification: %s, %s " % (trial_fol, e))

    save_df(summary_df, "Dryer_f1_scores")
    save_df(blocks_stats, "blocks_analysis_stats")
    save_df(dryer_cycles, "potential_dryer_cycles")
    save_df(gt_start_end_times, "gt_start_end_times")


if __name__ == "__main__":
    main()