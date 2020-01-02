from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import AdaBoostClassifier


class TrainEnsembleAdaBoosting():
    def __init__(self, data, classes, model_build_fn, epochs=100, batch_size=10, verbose=0):
        self.data = data
        self.classes = classes
        self.build_fn = model_build_fn
        self.keras_adapter = KerasClassifier(build_fn=model_build_fn, epochs=epochs, batch_size=batch_size,
                                             verbose=verbose)
        self.ensemble_classifier = AdaBoostClassifier(base_estimator=self.keras_adapter)

    def fit(self, n_estimatiors=15):
        self.ensemble_classifier.fit(self.data, self.classes, n_estimatiors=n_estimatiors)


    def get_classifiers(self):
        return self.ensemble_classifier.estimators_
