import pandas as pd

result_file_path = "/home/mattyws/Documents/mimic/ensemble_results/representation/"
result_file_name = result_file_path + "result.csv"

result = pd.read_csv(result_file_name)

result_groupby = result.groupby(by=["num_models"]).mean()
result_groupby.loc["mean"] = result_groupby.mean()
print(result_groupby)
result_groupby.to_csv(result_file_path + "result_groupby_nummodels.csv")
metrics_tolatex = ["w_precision", "w_recall", "w_fscore", "recall_b", "w_auc"]
with open(result_file_path + "result_groupby_tolatex.txt", "w") as file:
    file.write(result_groupby[metrics_tolatex].to_latex(float_format="%.2f"))