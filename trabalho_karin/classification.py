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

def write_on_log(file, text):
    print(text)
    with open("log_{}".format(file), 'a+') as result_file_handler:
        result_file_handler.write(text+'\n')

def preprocess(data):
    for column in data:
        if data.dtypes[column] == object:
            data[column].fillna("Não mensurado", inplace=True)
            encoder = LabelEncoder()
            encoder.fit(data[column].tolist())
            data[column] = encoder.transform(data[column])
        elif data.dtypes[column] == float:
            data[column].fillna(0, inplace=True)
        elif data.dtypes[column] == int:
            data[column].fillna(0, inplace=True)
    return data

def normalize(df, mean, std):
    normalized_df = (df - mean) / std
    normalized_df.fillna(0, inplace=True)
    return normalized_df

def preprocess_classes(classes):
    encoder = LabelEncoder()
    encoder.fit(classes)
    return encoder.transform(classes)

csv_file_paths = [
    'original/sepsis_feature_selection_filter_0_5_pearson.csv',
    'original/sepsis_feature_selection_filter_0_75_pearson.csv',
    'original/sepsis_feature_selection_ga.csv',
    'original/sepsis_file.csv'
    # 'original_mean/sepsis_feature_selection_filter_0_75_pearson.csv'
    # 'original_mean/sepsis_file_header_sem_desvio_padrao.csv',
    # 'second/sepsis_feature_selection_filter_0_5_pearson.csv',
    # 'second/sepsis_feature_selection_filter_0_75_pearson (1).csv',
    # 'second/sepsis_file2.csv'
                  ]
class_label = "organism_resistence"

for csv_file_path in csv_file_paths:
    print("============================= {} ============================".format(csv_file_path))
    data, classes = helper.load_csv_file(csv_file_path, class_label)
    data = preprocess(data)
    classes = preprocess_classes(classes)

    data, search_data, classes, search_classes = train_test_split(data, classes, test_size=.20, stratify=classes)

    classifiers = [MLPClassifier(), DecisionTreeClassifier(), svm.SVC(), RandomForestClassifier()]
    search_iterations = 140
    i = 0

    mean_std_pair = None
    while i < len(classifiers):
        print("======= Param search {} ======".format(type(classifiers[i])))
        random_search = RandomizedSearchCV(classifiers[i], param_distributions=helper.PARAM_DISTS[type(classifiers[i])],
                                           n_iter=search_iterations, cv=5)
        mean = search_data.mean()
        std = search_data.std()
        search_data = normalize(search_data, mean, std)
        random_search.fit(search_data, search_classes)
        classifiers[i].set_params(**random_search.best_params_)
        i += 1
    write_on_log(csv_file_path.replace('/', '_').split('.')[0], "========== Begin algorithm params {} =========".format(csv_file_path))
    for classifier in classifiers:
        write_on_log(csv_file_path.replace('/', '_').split('.')[0], str(type(classifier)))
        write_on_log(csv_file_path.replace('/', '_').split('.')[0], str(classifier.get_params()))
    write_on_log(csv_file_path.replace('/', '_').split('.')[0], "========== End algorithm params =========")

    kf = KFold(n_splits=10)
    results = []
    for train_index, test_index in kf.split(data):
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]
        mean = data_train.mean()
        std = data_train.std()
        data_train = normalize(data_train, mean, std)
        data_test = normalize(data_test, mean, std)
        classes_train, classes_test = classes[train_index], classes[test_index]
        kfold_result = dict()
        i = 0
        while i < len(classifiers):
            classifier = classifiers[i]
            print("======= Training {} ======".format(type(classifier)))
            classifier.fit(data_train, classes_train)
            try:
                predicted = classifier.predict(data_test)
                accuracy = accuracy_score(classes_test, predicted)
                kfold_result[type(classifier)] = accuracy
            except Exception as e:
                print(e)
                for column in data_test.columns:
                    print(data_test[column])
                kfold_result[type(classifier)] = 0
            i+=1
        results.append(kfold_result)
    with open(csv_file_path.split('/')[0]+'/result_{}.csv'.format(csv_file_path.split('/')[1].split('.')[0]), 'w') \
            as result_file_handler:
        writer = csv.DictWriter(result_file_handler, fieldnames=results[0].keys())
        writer.writeheader()
        for row in results:
            writer.writerow(row)