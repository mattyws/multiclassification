import json
import os
import time
import pprint
import statistics
from collections import Counter
import sys
import csv

import trabalho_karin.helper as helper

pp = pprint.PrettyPrinter(indent=5)
date_pattern = "%Y-%m-%d"
datetime_pattern = "%Y-%m-%d %H:%M:%S"
itemid_label = 'ITEMID'
valuenum_label = 'valuenum'
value_label = 'value'
labitems_prefix = 'lab_'
items_prefix = 'item_'
mean_key = 'mean'
std_key = 'std'
csv_file_name = "sepsis_file.csv"
class_label = "organism_resistence"
interpretation_label = "interpretation"
org_item_label = "ORG_ITEMID"
microbiologyevent_label = "microbiologyevents"




def transform_values(row, features_type):
    for key in row:
        itemid = helper.get_itemid_from_key(key)
        if row[key] is not None and features_type[itemid] == helper.MEAN_LABEL:
            mean = 0
            std = 0
            if len(row[key]) > 1:
                mean = sum(row[key]) / len(row[key])
                std = statistics.stdev(row[key])
            elif len(row[key]) != 0:
                mean = row[key][0]
                std = 0
            row[key] = {mean_key: mean, std_key: std}
        elif row[key] is not None and features_type[itemid] == helper.CATEGORICAL_LABEL:
                row[key] = Counter(row[key]).most_common(1)[0][0]
    return row


def split_into_columns(row, features_type):
    new_row = dict()
    for key in row:
        itemid = helper.get_itemid_from_key(key)
        if features_type[itemid] == helper.MEAN_LABEL:
            if row[key] is not None:
                for key2 in row[key]:
                    new_row[key+"_"+key2] = row[key][key2]
            else:
                new_row[key+"_"+mean_key] = None
                new_row[key+"_"+std_key] = 0
        else:
            new_row[key] = row[key]
    return new_row


def has_equal(itemid):
    for item in helper.ARE_EQUAL:
        if itemid in item:
            return item[0]
    return itemid


def transform_equal_columns(row, features_type, prefix=""):
    keys_to_remove = []
    for key in row.keys():
        itemid = helper.get_itemid_from_key(key)
        standard_itemid = has_equal(itemid)
        if standard_itemid != itemid:
            if row[key] is not None:
                if row[prefix+standard_itemid] is None:
                    row[prefix + standard_itemid] = row[key]
                else:
                    if features_type[standard_itemid] == helper.MEAN_LABEL or features_type[standard_itemid] == helper.CATEGORICAL_LABEL:
                        row[prefix+standard_itemid].extend(row[key])
                    elif features_type[standard_itemid] == helper.YESNO_LABEL:
                        if row[key] == 1:
                            row[prefix + standard_itemid] = row[key]
            keys_to_remove.append(key)
    for key in keys_to_remove:
        row.pop(key)
    return row


def diastolic_systolic_pressure(row, prefix):
    # Using variables to make more readable and for performance purpose
    blood_pressure = prefix + '220052'
    BPs_L = prefix + '224167'
    BPs_R = prefix + '227243'
    BPd_R = prefix + '227242'
    BPd_L = prefix + '224643'
    has_BPs_L = BPs_L in row.keys()
    has_BPs_R = BPs_R in row.keys()
    has_BPd_R = BPd_R in row.keys()
    has_BPd_L = BPd_L in row.keys()

    # Check if this rows has any of the known mean pressure ids, if it has, drop any manual pressure and return the row
    for key in helper.PRESSURE_IDS:
        if prefix+key in row.keys():
            if has_BPd_L:
                row.pop(BPd_L)
            if has_BPs_L:
                row.pop(BPs_L)
            if has_BPd_R:
                row.pop(BPd_R)
            elif has_BPs_R:
                row.pop(BPs_R)
            return row

    # It doesn't has any of the known mean pressure ids, check if has the manual pressure for both left and right arms, if it has,
    # try to get the one with the higher quantity of values
    if has_BPs_R and has_BPs_L:
        if len(row[BPs_R]) > len(row[BPs_L]):
            # Drop the lowest sized, and if drop the other measure for the same arm
            row.pop(BPs_L)
            if has_BPd_L:
                row.pop(BPd_L)
        else:
            # Drop the lowest sized, and if drop the other measure for the same arm
            row.pop(BPs_R)
            if has_BPd_R:
                row.pop(BPd_R)

    # If it has the both measures for right arm,
    # add the mean to the main id for blood pressure,
    # and remove the measures for manual
    if has_BPs_R and has_BPd_R:
        print("Has both")
        for i in range(len(row[BPs_R])):
            if row[blood_pressure] is None:
                row[blood_pressure] = []
            mean_pressure = helper.MEAN_PRESSURE( float(row[BPd_R][i]), float(row[BPs_R][i]) )
            row[blood_pressure].append(mean_pressure)
        row.pop(BPd_R)
        row.pop(BPs_R)
    else:
        if has_BPd_R:
            row.pop(BPd_R)
        if has_BPs_R:
            row.pop(BPs_R)

    # The same as before but for the left arm
    if has_BPs_L and has_BPd_L:
        print("Has both")
        for i in range(len(row[BPs_L])):
            if row[blood_pressure] is None:
                row[blood_pressure] = []
            mean_pressure = helper.MEAN_PRESSURE( float(row[BPd_L][i]), float(row[BPs_L][i]) )
            row[blood_pressure].append(mean_pressure)
        row.pop(BPd_L)
        row.pop(BPs_L)
    else:
        if has_BPd_L:
            row.pop(BPd_L)
        if has_BPs_L:
            row.pop(BPs_L)
    return row


def transform_to_row(filtered_events, features_type, prefix=""):
    row = dict()
    for event in filtered_events:
        itemid = event[itemid_label]
        event_type  = features_type[itemid]
        if prefix+itemid not in row.keys() and event_type == helper.MEAN_LABEL :
            row[prefix+itemid] = []
        elif event_type == helper.CATEGORICAL_LABEL:
            row[prefix+itemid] = []
        elif event_type == helper.YESNO_LABEL:
            row[prefix+itemid] = 0

        if event_type == helper.MEAN_LABEL :
            try:
                if itemid in helper.FARENHEIT_ID:
                    row[prefix + itemid].append(helper.CELCIUS(float(event[valuenum_label])))
                else:
                    row[prefix+itemid].append(float(event[valuenum_label]))
            except:
                row[prefix + itemid].append(0)
        elif event_type == helper.CATEGORICAL_LABEL:
            row[prefix+itemid].append(event[value_label])
        elif event_type == helper.YESNO_LABEL and row[prefix+itemid] == 0:
            row[prefix+itemid] = 1
    for key in features_type.keys():
        if prefix+key not in row:
            row[prefix+key] = None
    row = diastolic_systolic_pressure(row, prefix)
    row = transform_equal_columns(row, features_type, prefix=prefix)
    row = transform_values(row, features_type)
    row = split_into_columns(row, features_type)
    return row

def get_data_from_admitday(json_object, date_str, key="charttime", date=False):
    admittime = time.strptime(date_str, datetime_pattern)
    filtered_objects = []
    for event in json_object:
        event_date = time.strptime(event[key], datetime_pattern)
        if date:
            difference = event_date.tm_mday - admittime.tm_mday
            if difference < 2:
                filtered_objects.append(event)
        else:
            difference = time.mktime(event_date) - time.mktime(admittime)
            if abs(difference) / 86400 <= 1:
                filtered_objects.append(event)
        # break
    return filtered_objects

print(len(helper.FEATURES_ITEMS_LABELS.keys()) + len(helper.FEATURES_LABITEMS_LABELS.keys()))


def get_organism_class(events):
    organism_count = dict()
    for event in events:
        if org_item_label in event.keys():
            if event[org_item_label] not in organism_count.keys():
                organism_count[event[org_item_label]] = 0
            if event[interpretation_label] == 'R':
                organism_count[event[org_item_label]] += 1
                if organism_count[event[org_item_label]] == 3:
                    return "R"
    return "S"


with open('sepsis_patients4', 'r') as patients_w_sepsis_handler:
    all_size = 0
    filtered_objects_total_size = 0
    table = []
    not_processes_files = 0
    patients_with_pressure = 0
    for line in patients_w_sepsis_handler:
        print(line.strip().split('/')[-1])
        all_size += os.path.getsize(line.strip())
        patient = json.load(open(line.strip(), 'r'))
        if microbiologyevent_label in patient.keys():
            filtered_chartevents_object = []
            if 'chartevents' in patient.keys():
                filtered_chartevents_object = get_data_from_admitday(patient['chartevents'], patient['admittime'],
                                                                     key='charttime', date=False)
                filtered_objects_total_size += sys.getsizeof(filtered_chartevents_object)

            # filtered_prescriptions_object = None
            # if 'prescriptions' in patient.keys():
            #     filtered_prescriptions_object = get_data_from_admitday(patient['prescriptions'], patient['admittime'],
            #                                                            key='startdate', date=True)
            #     filtered_objects_total_size += sys.getsizeof(filtered_prescriptions_object)

            filtered_labevents_object = []
            if 'labevents' in patient.keys():
                filtered_labevents_object = get_data_from_admitday(patient['labevents'], patient['admittime'],
                                                                       key='charttime', date=False)
                filtered_objects_total_size += sys.getsizeof(filtered_labevents_object)

            new_filtered_chartevents = []
            for event in filtered_chartevents_object:
                if event[itemid_label] in helper.FEATURES_ITEMS_LABELS.keys():
                    new_filtered_chartevents.append(event)

            new_filtered_labevents = []
            for event in filtered_labevents_object:
                if event[itemid_label] in helper.FEATURES_LABITEMS_LABELS.keys():
                    new_filtered_labevents.append(event)

            row_object = transform_to_row(new_filtered_chartevents, helper.FEATURES_ITEMS_TYPE, prefix=items_prefix)
            row_labevent = transform_to_row(new_filtered_labevents, helper.FEATURES_LABITEMS_TYPE, prefix=labitems_prefix)

            for key in row_labevent.keys():
                row_object[key] = row_labevent[key]
            row_object[class_label] = get_organism_class(patient[microbiologyevent_label])
            table.append(row_object)
        else:
            not_processes_files += 1
    pp.pprint(table)
    with open(csv_file_name, 'w') as csv_file_handler:
        writer = csv.DictWriter(csv_file_handler, table[0].keys())
        writer.writeheader()
        for row in table:
            writer.writerow(row)

    print("Number of files that do not had microbiologyevents : {}".format(not_processes_files))
    print("Size of files processed : {} bytes".format(all_size))
    print("Total size of filtered variables : {} bytes".format(filtered_objects_total_size))