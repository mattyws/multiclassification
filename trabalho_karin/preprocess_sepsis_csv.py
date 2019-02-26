import csv
import re

import numpy
import pandas as pd


# table = pd.read_csv('sepsis_file3.csv')
# table = table.filter(regex="lab_")
# for column in table.columns:
#     print(column, table[column].dtype, table[column].unique())
# exit()


range_re = re.compile('\d+-\d+')
with open('sepsis_file2.csv', 'r') as sepsisHandler:
    dictReader = csv.DictReader(sepsisHandler, quoting=csv.QUOTE_NONNUMERIC)
    new_data = []
    dictTypes = dict()
    for key in dictReader.fieldnames:
        dictTypes[key] = set()
    for row in dictReader:
        for key in row.keys():
            if row[key] != "?" and row[key] is not None and key is not None:
                try:
                    typeof = type(int(row[key]))
                except:
                    typeof = type(row[key])
                dictTypes[key].add(typeof)
    for key in dictTypes.keys():
        dictTypes[key] = list(dictTypes[key])
with open('sepsis_file2.csv', 'r') as sepsisHandler:
    dictReader = csv.DictReader(sepsisHandler, quoting=csv.QUOTE_NONNUMERIC)
    for row in dictReader:
        empty_key = None
        for key in row.keys():
            if isinstance(row[key], str) and range_re.match(row[key]):
                numbers = re.findall('\d+', row[key])
                numbers = [int(n) for n in numbers]
                row[key] = sum(numbers) / len(numbers)
                if type(int) not in dictTypes[key]:
                    dictTypes[key].append(type(int))
            if isinstance(row[key], str) and len(dictTypes[key]) > 1 :
                if range_re.match(row[key])	:
                    numbers = re.findall('\d+',row[key])
                    numbers = [int(n) for n in numbers]
                    row[key] = sum(numbers) / len(numbers)
                elif row[key].startswith('LESS THAN') or row[key].startswith('<') or \
                        row[key].startswith('LESS THEN'):
                    numbers = re.findall('\d+',row[key])
                    row[key] = float(numbers[0])
                elif row[key].startswith('GREATER THAN') or row[key].startswith('>') or \
                        row[key].startswith('GREATER THEN'):
                    numbers = re.findall('\d+',row[key])
                    row[key] = float(numbers[0])
                elif 'HEMOLYSIS FALSELY INCREASES THIS RESULT' == row[key]:
                    row[key] = "?"
                elif row[key] == 'NEG':
                    row[key] = '-1'
                elif 'ERROR' in row[key]:
                    row[key] = "?"
                elif row[key].lower() == 'tr':
                    row[key] = "?"
                elif row[key].lower() == 'n':
                    row[key] = "?"
                elif row[key] == numpy.nan:
                    row[key] = "?"
                elif row[key] == 'NONE' or row[key].lower() == 'notdone':
                    row[key] = "?"
                elif row[key] == '-' or row[key] == '.':
                    row[key] = "?"
                elif row[key].lower() == 'tntc' :
                    row[key] = "?"
        row.pop(None, None)
        new_data.append(row)
    with open('sepsis_file3.csv', 'w') as newFileHandler:
        print(new_data[0].keys())
        header = list(new_data[0].keys())
        header.sort()
        dictWriter = csv.DictWriter(newFileHandler, header, quoting=csv.QUOTE_NONNUMERIC)
        dictWriter.writeheader()
        for row in new_data:
            print(row['lab_51213'])
            dictWriter.writerow(row)
