import csv

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

csv_file_path = 'sepsis_file.csv'
class_label = "organism_resistence"

data = []
classes = []
with open(csv_file_path, 'r') as csv_file_handler:
    reader = csv.DictReader(csv_file_handler)
    keys = reader.fieldnames
    for row in reader:
        row_list = []
        for key in keys:
            if key == class_label:
                classes.append(row[key])
            else:
                row_list.append(row[key])
        data.append(row_list)

kf = KFold(n_splits=10)
results = []
for train_index, test_index in kf.split(data):
    data_train, data_test = data[train_index], data[test_index]
    classes_train, classes_test = classes[train_index], classes[test_index]
    classifiers = [DecisionTreeClassifier(max_depth=3)]
    kfold_result = dict()
    for classifier in classifiers:
        classifier.fit(data_train, classes_train)
        predicted = classifier.predict(data_test)
        accuracy = accuracy_score(classes_test, predicted)
        kfold_result[type(classifier)] = accuracy
    results.append(kfold_result)