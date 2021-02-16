import pandas as pd
from sklearn.calibration import calibration_curve
import os
import matplotlib.pyplot as plt

def plot_model_calibrations(models_predictions_probabilities:pd.DataFrame, label_column:str, analysis_dir_path:str, figure_prefix:str):
    if not os.path.exists(os.path.join(analysis_dir_path, "{}_calibration_curve.jpg".format(figure_prefix))):
        y_true = models_predictions_probabilities[label_column]
        fig = plt.figure(0, figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        for column in models_predictions_probabilities.columns:
            if 'Unnamed' in column or 'episode' in column or column == label_column:
                continue
            if '_' in column:
                model_name = column.split('_')[0]
            else:
                model_name = column
            y_pred = models_predictions_probabilities[column]
            prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
            calibration_df = pd.DataFrame({"prob_true":prob_true, "prob_pred":prob_pred})
            calibration_df.to_csv(os.path.join(analysis_dir_path, "{}_{}_calibration_output.csv".format(figure_prefix, column)))
            ax1.plot(prob_pred, prob_true, "s-",
                     label="%s" % (model_name))
        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir_path, "{}_calibration_curve.jpg".format(figure_prefix)))

