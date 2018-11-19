import csv
import itertools
import numpy
import pandas
from sklearn import svm
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing.label import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import numpy as np

from trabalho_karin import helper

def write_on_log(text):
    print(text)
    with open('result.log', 'a+') as result_file_handler:
        result_file_handler.write(text+'\n')

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            write_on_log("Model with rank: {0}".format(i))
            write_on_log("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            write_on_log("Parameters: {0}".format(results['params'][candidate]))
            write_on_log("")

def preprocess(data):
    for column in data:
        if data.dtypes[column] == object:
            data[column].fillna("", inplace=True)
            encoder = LabelEncoder()
            encoder.fit(data[column].tolist())
            data[column] = encoder.transform(data[column])
        elif data.dtypes[column] == float:
            data[column].fillna(0, inplace=True)
        elif data.dtypes[column] == int:
            data[column].fillna(0, inplace=True)
    return data

def normalize(df):
    normalized_df = (df - df.mean()) / df.std()
    return normalized_df

def preprocess_classes(classes):
    encoder = LabelEncoder()
    encoder.fit(classes)
    return encoder.transform(classes)

csv_file_path = 'sepsis_file.csv'
class_label = "organism_resistence"

data, classes = helper.load_csv_file(csv_file_path, class_label)
data = preprocess(data)
classes = preprocess_classes(classes)

data, search_data, classes, search_classes = train_test_split(data, classes, test_size=.20, stratify=classes)

classifiers = [MLPClassifier()]#, DecisionTreeClassifier(), svm.SVC(), RandomForestClassifier()]
search_iterations = 40
i = 0


while i < len(classifiers):
    print("======= Param search {} ======".format(type(classifiers[i])))
    random_search = RandomizedSearchCV(classifiers[i], param_distributions=helper.PARAM_DISTS[type(classifiers[i])],
                                       n_iter=search_iterations, cv=5)
    search_data = normalize(search_data)
    random_search.fit(search_data, search_classes)
    # report(random_search.cv_results_)
    classifiers[i].set_params(**random_search.best_params_)
    i += 1
write_on_log("========== Begin algorithm params =========")
for classifier in classifiers:
    write_on_log(str(type(classifier)))
    write_on_log(str(classifier.get_params()))
write_on_log("========== End algorithm params =========")

kf = KFold(n_splits=10)
results = []
for train_index, test_index in kf.split(data):
    data_train, data_test = data.iloc[train_index], data.iloc[test_index]
    classes_train, classes_test = classes[train_index], classes[test_index]
    kfold_result = dict()
    i = 0
    while i < len(classifiers):
        classifier = classifiers[i]
        print("======= Training {} ======".format(type(classifier)))
        classifier.fit(data_train, classes_train)
        predicted = classifier.predict(data_test)
        accuracy = accuracy_score(classes_test, predicted)
        kfold_result[type(classifier)] = accuracy
        i+=1
    results.append(kfold_result)
with open('result_{}.csv'.format(csv_file_path.split('.')[0]), 'w') as result_file_handler:
    writer = csv.DictWriter(result_file_handler, fieldnames=results[0].keys())
    writer.writeheader()
    for row in results:
        writer.writerow(row)