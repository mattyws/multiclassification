import csv
import numpy as np
import pandas
from scipy.stats import randint as sp_randint
from scipy.stats import expon as sp_expon
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree.tree import DecisionTreeClassifier


def get_itemid_from_key(key):
    return key.split("_")[1]

def load_csv_file(csv_file_path, class_label):
    data = pandas.read_csv(csv_file_path)
    classes = data[class_label]
    data = data.drop(class_label, axis=1)
    return data, classes

def generate_random_numbers_tuple():
    while True:
        result = []
        size = sp_randint(1, 10).rvs()
        for i in range(size):
            result.append(sp_expon(scale=50).rvs())
        yield tuple(result)

class RandIntMatrix(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.shape = sp_randint(1, 3).rvs()

    def rvs(self, random_state=None):
        return sp_randint(self.low, self.high).rvs(size=self.shape)

# generate_random_numbers_tuple()

PARAM_DISTS = {
    type(DecisionTreeClassifier()): {
        'criterion': ['gini', 'entropy'],
        'max_depth': sp_randint(3, 8)
    },
    type(SVC()): {
        'C': sp_expon(scale=100),
        'gamma': sp_expon(scale=.1),
        'max_iter': [300],
        'kernel': ['rbf', 'linear', 'sigmoid'],
    },
    type(MLPClassifier()): {
        'hidden_layer_sizes': RandIntMatrix(12, 128),
        'max_iter': [500],
        'activation': ['relu', 'tanh', 'logistic']
    },
    type(RandomForestClassifier()): {
        'n_estimators': sp_randint(10, 25),
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7, None]
    }
}

YESNO_LABEL = "yes/no"
CATEGORICAL_LABEL = "categorical"
MEAN_LABEL = "mean"

FEATURES_ITEMS_LABELS = {
    # '5656' : 'Phenylephrine',
    # '221749' : 'Phenylephrine',
    # '6752' : 'Phenylephrine',
    # '221906' : 'Norepinephrine',
    # '1136' : 'Vasopressin',
    # '1222' : 'Vasopressin',
    # '2445' : 'Vasopressin',
    # '30051' : 'Vasopressin',
    # '222315' : 'Vasopressin',
    '220052' : 'Arterial Blood Pressure mean',
    '52' : 'Arterial Blood Pressure mean',
    '3312' : 'Arterial Blood Pressure mean',
    '5679' : 'Arterial Blood Pressure mean',
    '225312' : 'Arterial Blood Pressure mean',
    '5600' : 'Arterial Blood Pressure mean',
    '3256' : 'Arterial Blood Pressure mean',
    '3314' : 'Arterial Blood Pressure mean',
    '3316' : 'Arterial Blood Pressure mean',
    '3320' : 'Arterial Blood Pressure mean',
    '3322' : 'Arterial Blood Pressure mean',
    '5731' : 'Arterial Blood Pressure mean',
    '2732' : 'Arterial Blood Pressure mean',
    '7618' : 'Arterial Blood Pressure mean',
    '7620' : 'Arterial Blood Pressure mean',
    '7622' : 'Arterial Blood Pressure mean',
    '53' : 'Arterial Blood Pressure mean',
    '443' : 'Arterial Blood Pressure mean',
    '224167' : 'BPs L',
    '224643' : 'BPd L',
    '227243' : 'BPs R',
    '227242' : 'BPd R',
    '228332' : 'Delirium assessment',
    '220739' : 'GCS - Eye Opening',
    '184' : 'GCS - Eye Opening',
    '223901' : 'GCS - Motor Response',
    '454' : 'GCS - Motor Response',
    '223900' : 'GCS - Verbal Response',
    '723' : 'GCS - Verbal Response',
    '211' : 'Heart Rate',
    '220045' : 'Heart Rate',
    '223835' : 'Inspired O2 Fraction',
    '448' : 'Minute Volume',
    '224687' : 'Minute Volume',
    '220181' : 'Non Invasive Blood Pressure mean',
    '226732' : 'O2 Delivery Device(s)',
    '467' : 'O2 Delivery Device(s)',
    '220277' : 'O2 saturation pulseoxymetry',
    '1046' : 'Pain Present',
    '223781' : 'Pain Present',
    '535' : 'Peak Insp. Pressure',
    '224695' : 'Peak Insp. Pressure',
    '505' : 'PEEP',
    '6924' : 'PEEP',
    '543' : 'Plateau Pressure',
    '224696' : 'Plateau Pressure',
    '616' : 'Respiratory Effort',
    '223990' : 'Respiratory Effort',
    '618' : 'Respiratory Rate',
    '220210' : 'Respiratory Rate',
    '224690' : 'Respiratory Rate (Total)',
    '676' : 'Temperature Celsius',
    '223762' : 'Temperature Celsius',
    '678' : 'Temperature Celsius',
    '223761' : 'Temperature Celsius',
    '227687' : 'Tobacco Use History',
    '225108' : 'Tobacco Use History',
    '720' : 'Ventilator Mode',
    '223849' : 'Ventilator Mode'
}

FEATURES_ITEMS_TYPE = {
    # '5656' : YESNO_LABEL,
    # '221749' : YESNO_LABEL,
    # '6752' : YESNO_LABEL,
    # '221906' : YESNO_LABEL,
    # '1136' : YESNO_LABEL,
    # '1222' : YESNO_LABEL,
    # '2445' : YESNO_LABEL,
    # '30051' : YESNO_LABEL,
    # '222315' : YESNO_LABEL,
    '220052' : MEAN_LABEL,
    '52' : MEAN_LABEL,
    '3312' : MEAN_LABEL,
    '5679' : MEAN_LABEL,
    '225312' : MEAN_LABEL,
    '5600' : MEAN_LABEL,
    '3256' : MEAN_LABEL,
    '3314' : MEAN_LABEL,
    '3316' : MEAN_LABEL,
    '3320' : MEAN_LABEL,
    '3322' : MEAN_LABEL,
    '5731' : MEAN_LABEL,
    '2732' : MEAN_LABEL,
    '7618' : MEAN_LABEL,
    '7620' : MEAN_LABEL,
    '7622' : MEAN_LABEL,
    '53' : MEAN_LABEL,
    '443' : MEAN_LABEL,
    '224167' : MEAN_LABEL,
    '224643' : MEAN_LABEL,
    '227243' : MEAN_LABEL,
    '227242' : MEAN_LABEL,
    '228332' : YESNO_LABEL,
    '220739' : MEAN_LABEL,
    '184' : MEAN_LABEL,
    '223901' : MEAN_LABEL,
    '454' : MEAN_LABEL,
    '223900' : MEAN_LABEL,
    '723' : MEAN_LABEL,
    '211' : MEAN_LABEL,
    '220045' : MEAN_LABEL,
    '223835' : MEAN_LABEL,
    '448' : MEAN_LABEL,
    '224687' : MEAN_LABEL,
    '220181' : MEAN_LABEL,
    '226732' : CATEGORICAL_LABEL,
    '467' : CATEGORICAL_LABEL,
    '220277' : MEAN_LABEL,
    '1046' : YESNO_LABEL,
    '223781' : YESNO_LABEL,
    '535' : MEAN_LABEL,
    '224695' : MEAN_LABEL,
    '505' : MEAN_LABEL,
    '6924' : MEAN_LABEL,
    '543' : MEAN_LABEL,
    '224696' : MEAN_LABEL,
    '616' : YESNO_LABEL,
    '223990' : YESNO_LABEL,
    '618' : MEAN_LABEL,
    '220210' : MEAN_LABEL,
    '224690' : MEAN_LABEL,
    '676' : MEAN_LABEL,
    '223762' : MEAN_LABEL,
    '678' : MEAN_LABEL,
    '223761' : MEAN_LABEL,
    '227687' : YESNO_LABEL,
    '225108' : YESNO_LABEL,
    '720' : CATEGORICAL_LABEL,
    '223849' : CATEGORICAL_LABEL
}

FEATURES_LABITEMS_LABELS = {
    '50861' : 'Alanine Aminotransferase (ALT)',
    '50862' : 'Albumin',
    '50863' : 'Alkaline Phosphatase',
    '50801' : 'Alveolar-arterial Gradient',
    '50866' : 'Ammonia',
    '50868' : 'Anion Gap',
    '50878' : 'Asparate Aminotransferase (AST)',
    '51144' : 'Bands',
    '50802' : 'Base Excess',
    '50882' : 'Bicarbonate',
    '50803' : 'Bicarbonate',
    '50885' : 'Bilirubin, Total',
    '50893' : 'Calcium, Total',
    '50902' : 'Chloride',
    '50806' : 'Chloride',
    '50910' : 'Creatine Kinase (CK)',
    '50912' : 'Creatinine',
    '51200' : 'Eosinophils',
    '50809' : 'Glucose',
    '51221' : 'Hematocrit',
    '51222' : 'Hemoglobin',
    '50811' : 'Hemoglobin',
    '51237' : 'INR(PT)',
    '50813' : 'Lactate',
    '50954' : 'Lactate Dehydrogenase (LD)',
    '51244' : 'Lymphocytes',
    '50960' : 'Magnesium',
    '51256' : 'Neutrophils',
    '50963' : 'NTproBNP',
    '50816' : 'Oxygen',
    '50817' : 'Oxygen Saturation',
    '50818' : 'pCO2',
    '50820' : 'pH',
    '50970' : 'Phosphate',
    '51265' : 'Platelet Count',
    '50821' : 'pO2',
    '50971' : 'Potassium',
    '50822' : 'Potassium, Whole Blood',
    '50889' : 'Protein',
    '51274' : 'PT',
    '51275' : 'PTT',
    '51277' : 'RDW',
    '51279' : 'Red Blood Cells',
    '50983' : 'Sodium',
    '51003' : 'Troponin T',
    '51006' : 'Urea Nitrogen',
    '51301' : 'White Blood Cells',
    '50824' : 'Sodium, Whole Blood'
}

FEATURES_LABITEMS_TYPE = {
    '50861' : MEAN_LABEL,
    '50862' : MEAN_LABEL,
    '50863' : MEAN_LABEL,
    '50801' : MEAN_LABEL,
    '50866' : MEAN_LABEL,
    '50868' : MEAN_LABEL,
    '50878' : MEAN_LABEL,
    '51144' : MEAN_LABEL,
    '50802' : MEAN_LABEL,
    '50882' : MEAN_LABEL,
    '50803' : MEAN_LABEL,
    '50885' : MEAN_LABEL,
    '50893' : MEAN_LABEL,
    '50902' : MEAN_LABEL,
    '50806' : MEAN_LABEL,
    '50910' : MEAN_LABEL,
    '50912' : MEAN_LABEL,
    '51200' : MEAN_LABEL,
    '50809' : MEAN_LABEL,
    '51221' : MEAN_LABEL,
    '51222' : MEAN_LABEL,
    '50811' : MEAN_LABEL,
    '51237' : MEAN_LABEL,
    '50813' : MEAN_LABEL,
    '50954' : MEAN_LABEL,
    '51244' : MEAN_LABEL,
    '50960' : MEAN_LABEL,
    '51256' : MEAN_LABEL,
    '50963' : MEAN_LABEL,
    '50816' : MEAN_LABEL,
    '50817' : MEAN_LABEL,
    '50818' : MEAN_LABEL,
    '50820' : MEAN_LABEL,
    '50970' : MEAN_LABEL,
    '51265' : MEAN_LABEL,
    '50821' : MEAN_LABEL,
    '50971' : MEAN_LABEL,
    '50822' : MEAN_LABEL,
    '50889' : MEAN_LABEL,
    '51274' : MEAN_LABEL,
    '51275' : MEAN_LABEL,
    '51277' : MEAN_LABEL,
    '51279' : MEAN_LABEL,
    '50983' : MEAN_LABEL,
    '51003' : MEAN_LABEL,
    '51006' : MEAN_LABEL,
    '51301' : MEAN_LABEL,
    '50824' : MEAN_LABEL
}

ARE_EQUAL = [
    ('467', '226732'),
    ('448', '224687'),
    ('50971', '50822'),
    ('618', '224690'),
    # ('5656', '221749'),
    # ('5656', '6752'),
    # ('1136', '1222'),
    # ('1136', '2445'),
    # ('1136', '30051'),
    # ('1136', '222315'),
    ('184', '220739'),
    ('454', '223901'),
    ('723', '223900'),
    ('211', '220045'),
    ('1046', '223781'),
    ('535', '224695'),
    ('505', '6924' ),
    ('543', '224696'),
    ('616', '223990'),
    ('618', '220210'),
    ('720', '223849'),
    ('51222', '50811'),
    ('676', '678'),
    ('676', '223761'),
    ('676', '223762'),
    ('225108', '227687'),
    ('50803', '50882'),
    ('50806', '50902'),
    ('50824', '50983'),
    ('220052', '220181'),
    ('220052', '52'),
    ('220052', '3312'),
    ('220052', '5679'),
    ('220052', '225312'),
    ('220052', '5600'),
    ('220052', '3256'),
    ('220052', '3314'),
    ('220052', '3316'),
    ('220052', '3320'),
    ('220052', '3322'),
    ('220052', '5731'),
    ('220052', '2732'),
    ('220052', '7618'),
    ('220052', '7620'),
    ('220052', '7622'),
    ('220052', '53'),
    ('220052', '443'),
]

FARENHEIT_ID = ['678', '223761']
PRESSURE_IDS = ['220052', '220181', '52', '3312', '5679', '225312', '5600', '3256', '3314', '3316', '3320',
                '3322', '5731', '2732', '7618', '7620', '7622', '53', '443']
MANUAL_PRESSURES_IDS = ['224167', '224643', '227243', '227242']
CELCIUS = lambda Tf : ((Tf - 32)*5)/9
MEAN_PRESSURE = lambda D, S: (2*D + S)/3
