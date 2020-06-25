import numpy as np
from sklearn.metrics.classification import f1_score, precision_score, recall_score, cohen_kappa_score, accuracy_score, \
    classification_report
from sklearn.metrics.ranking import roc_auc_score


class ModelEvaluation():
    """
    Evaluate model trained in binary classification
    """
    def __init__(self, model, files, true_classes, predictions_scores):
        self.model = model
        self.files = files
        self.true_classes = true_classes
        self.predictions_scores = predictions_scores
        self.predictions_classes = np.array([score > 0.5 for score in predictions_scores]).astype(np.int)
        self.__get_metrics()

    def __get_metrics(self):
        metrics = dict()
        metrics['w_fscore'] = f1_score(self.true_classes, self.predictions_classes, average='weighted')
        metrics['w_precision'] = precision_score(self.true_classes, self.predictions_classes, average='weighted')
        metrics['w_recall'] = recall_score(self.true_classes, self.predictions_classes, average='weighted')

        metrics['mi_fscore'] = f1_score(self.true_classes, self.predictions_classes, average='micro')
        metrics['mi_precision'] = precision_score(self.true_classes, self.predictions_classes, average='micro')
        metrics['mi_recall'] = recall_score(self.true_classes, self.predictions_classes, average='micro')

        metrics['ma_fscore'] = f1_score(self.true_classes, self.predictions_classes, average='macro')
        metrics['ma_precision'] = precision_score(self.true_classes, self.predictions_classes, average='macro')
        metrics['ma_recall'] = recall_score(self.true_classes, self.predictions_classes, average='macro')

        metrics['fscore_b'] = f1_score(self.true_classes, self.predictions_classes)
        metrics['precision_b'] = precision_score(self.true_classes, self.predictions_classes)
        metrics['recall_b'] = recall_score(self.true_classes, self.predictions_classes)

        metrics['auc'] = roc_auc_score(self.true_classes, self.predictions_scores)
        metrics['kappa'] = cohen_kappa_score(self.true_classes, self.predictions_classes)
        metrics['accuracy'] = accuracy_score(self.true_classes, self.predictions_classes)
        self.metrics = metrics
        print(classification_report(self.true_classes, self.predictions_classes))
        # if return_predictions:
        #     result_dict = dict()
        #     for f, r in zip(files, result):
        #         result_dict[f] = r
        #     return metrics, result_dict
        # return metrics