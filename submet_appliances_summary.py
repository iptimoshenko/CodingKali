import json
import pandas as pd

"""
This script is intended to create a summary submeters JSON file with one dictionary per appliance
"""

source_path ="/mnt/DataDisk/submeter_data/"
#sub_folders = ['Bill_2017_08_14', 'Conrad_2017_08_21', 'Edward_2017_08_14',
#               'Irina_2017_09_15', 'Jon_2017_07_10', 'Jon_2017_09_15', 'Lionel_2017_08_31', 'Maria_2017_08_30',
#               'Matt_2017_08_25', 'Matt_2017_09_15', 'Niki_2017_08_14', 'Pete_2017_08_07']
                # 'Aistis_2017_09_25', 'Becky_2017_08_07',
sub_folders = ['Irina_2017_09_15']

appliance_summary = {'kettle': {}, 'toaster': {}, 'food_processor': {}, 'laptop': {}, 'haidryer': {}}

try:
    for folder in sub_folders:
        full_path = source_path + folder + '/Data/correct_labels.csv'
        correct_labels = pd.read_csv(full_path, names=['#Channel', 'Appliance', 'Brand', 'Model', 'Valid from', 'Valid to'])
        for row in correct_labels:
            channel = row[0]
            appliance = row[1].lower().replace(' ', '_'). replace('-', '_')
            if appliance not in appliance_summary.keys():
                appliance_summary[appliance] = {folder: channel}
            else:
                appliance_summary[appliance][folder] = channel
            print(appliance_summary)
except Exception as ex:
    print('Error opening file ', ex)


