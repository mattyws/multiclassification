import csv
import re
import pandas as pd


table = pd.read_csv('sepsis_file2.csv')
table = table.filter(regex="lab_")
for column in table.columns:
	print(column, table[column].dtype, table[column].unique())
exit()


with open('sepsis_file2.csv', 'r') as sepsisHandler:
	dictReader = csv.DictReader(sepsisHandler, quoting=csv.QUOTE_NONNUMERIC)
	with open('features_lt', 'r') as featuresHandler:
		features = featuresHandler.read().split('\n')
	new_data = []
	range_re = re.compile('\d-\d')
	for row in dictReader:
		empty_key = None
		for key in row.keys():			
			if type(row[key]) == type(str()):
				if range_re.match(row[key])	:
					numbers = re.findall('\d+',row[key])
					numbers = [int(n) for n in numbers]					
					row[key] = sum(numbers) / len(numbers)
				elif row[key].startswith('LESS THAN') or row[key].startswith('<'):
					if row[key].startswith('<'):
						print(row[key])
					numbers = re.findall('\d+',row[key])[0]
					if len(numbers) == 0:
						row[key] = "?"
					else:
						row[key] = numbers[0]
				elif row[key].startswith('GREATER THAN') or row[key].startswith('>'):
					numbers = re.findall('\d+',row[key])[-1]
					if row[key].startswith('>'):
						print(row[key])
					if len(numbers) == 0:
						row[key] = "?"
					else:
						row[key] = numbers[0]
		row.pop("", None)
		new_data.append(row)
	with open('sepsis_file3.csv', 'w') as newFileHandler:
		header = list(new_data[0].keys())
		header.sort()
		dictWriter = csv.DictWriter(newFileHandler, header, quoting=csv.QUOTE_NONNUMERIC)
		dictWriter.writeheader()
		for row in new_data:
			dictWriter.writerow(row)
