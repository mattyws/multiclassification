import pickle
import pandas as pd

from resources.functions import print_with_time
import os

def extract_metrics(analysis_directory:str, level0_results_path:str, structured_results_path:str, ensemble_results_path:str, balanced_test_set_results:str=None):
    metrics = ["w_fscore", "w_precision", "w_recall", "ma_fscore", "recall_b", "auc"]
    columns_order = ["Model", "Data Type", "Precision", "Recall", "W F-Score", "U F-Score", "TP Rate", "AUC"]

    if not os.path.exists(os.path.join(analysis_directory, "filtered_media_kfold.csv")):
        all_metrics = pd.read_csv(level0_results_path)
        all_metrics_mean = all_metrics.groupby(by="model").mean().reset_index()
        print(all_metrics_mean)
        results_df = all_metrics_mean[metrics + ['model']]
        print(results_df)
        results_df.loc[:, "Precision"] = results_df["w_precision"]
        results_df.loc[:, "Recall"] = results_df["w_recall"]
        results_df.loc[:, "W F-Score"] = results_df["w_fscore"]
        results_df.loc[:, "U F-Score"] = results_df["ma_fscore"]
        results_df.loc[:, "TP Rate"] = results_df["recall_b"]
        results_df.loc[:, "AUC"] = results_df["auc"]
        results_df = results_df.drop(columns=metrics)
        results_df.loc[:, "Data Type"] = results_df['model'].apply(lambda x: 'TTS' if 'textual' in x else 'STS')
        results_df.loc[:, "Model"] = results_df['model'].apply(lambda x: x.split('_')[0])
        results_df = results_df.drop(columns=['model'])
        results_df = results_df.round(decimals=3)
        results_df = results_df.sort_values(by='Data Type')
        results_df[columns_order].to_csv(os.path.join(analysis_directory, "filtered_media_kfold.csv"), index=False)

    if not os.path.exists(os.path.join(analysis_directory, "filtered_structured_only.csv")):
        structured_models = ["LinearSVC", "LogisticRegression", "GaussianNB", "DecisionTreeClassifier", "MLPClassifier"]
        structured_results = pd.read_csv(structured_results_path)
        structured_results.loc[:, "Precision"] = structured_results["w_precision"]
        structured_results.loc[:, "Recall"] = structured_results["w_recall"]
        structured_results.loc[:, "W F-Score"] = structured_results["w_fscore"]
        structured_results.loc[:, "U F-Score"] = structured_results["ma_fscore"]
        structured_results.loc[:, "TP Rate"] = structured_results["recall_b"]
        structured_results.loc[:, "AUC"] = structured_results["auc"]
        structured_results = structured_results.drop(columns=metrics)
        structured_results.loc[:, "Data Type"] = "Structured"
        structured_results.loc[:, "Model"] = structured_results['model']
        structured_results = structured_results.drop(columns=['model'])
        structured_results = structured_results.round(decimals=3)
        structured_results = structured_results.sort_values(by='Model')
        structured_results[columns_order].to_csv(os.path.join(analysis_directory, "filtered_structured_only.csv"),
                                                 index=False)

    if not os.path.exists(os.path.join(analysis_directory, 'filtered_ensemble_results.csv')):
        ensemble_models = ["LinearSVC", "LogisticRegression", "GaussianNB", "DecisionTreeClassifier", "MLPClassifier"]
        columns_order = ["Model", "Origin", "Precision", "Recall", "W F-Score", "U F-Score", "TP Rate", "AUC"]
        ensemble_results = pd.read_csv(ensemble_results_path)
        ensemble_results = ensemble_results[(ensemble_results['origin'] == 'struct_pred') |
                                            (ensemble_results['origin'] == 'both_pred')]
        ensemble_results = ensemble_results[ensemble_results['model'].isin(ensemble_models)]
        ensemble_results.loc[:, 'Origin'] = ensemble_results['origin'].apply(
            lambda x: "Structured + STS" if x == 'struct_pred' else "Structured + STS + TTS")
        ensemble_results.loc[:, 'Model'] = ensemble_results['model']
        ensemble_results.loc[:, "Precision"] = ensemble_results["w_precision"]
        ensemble_results.loc[:, "Recall"] = ensemble_results["w_recall"]
        ensemble_results.loc[:, "W F-Score"] = ensemble_results["w_fscore"]
        ensemble_results.loc[:, "U F-Score"] = ensemble_results["ma_fscore"]
        ensemble_results.loc[:, "TP Rate"] = ensemble_results["recall_b"]
        ensemble_results.loc[:, "AUC"] = ensemble_results["auc"]
        ensemble_results = ensemble_results.round(decimals=3)
        ensemble_results = ensemble_results.sort_values(by='Origin')
        ensemble_results[columns_order].to_csv(os.path.join(analysis_directory, 'filtered_ensemble_results.csv'),
                                               index=False)

    if not os.path.exists(os.path.join(analysis_directory, "filtered_test_data.csv")):
        if balanced_test_set_results is not None:
            columns_order = ["Model", "Data Type", "Precision", "Recall", "W F-Score", "U F-Score", "TP Rate", "AUC"]
            evaluation_metrics = None
            for fold in range(5):
                fold_evaluation_metrics_path = balanced_test_set_results.format(fold)
                fold_evaluation_df = pd.read_csv(fold_evaluation_metrics_path)
                if evaluation_metrics is None:
                    evaluation_metrics = fold_evaluation_df
                else:
                    evaluation_metrics = evaluation_metrics.append(fold_evaluation_df)
            results_df = evaluation_metrics[metrics + ['model']]
            print(results_df)
            results_df.loc[:, "Precision"] = results_df["w_precision"]
            results_df.loc[:, "Recall"] = results_df["w_recall"]
            results_df.loc[:, "W F-Score"] = results_df["w_fscore"]
            results_df.loc[:, "U F-Score"] = results_df["ma_fscore"]
            results_df.loc[:, "TP Rate"] = results_df["recall_b"]
            results_df.loc[:, "AUC"] = results_df["auc"]
            results_df = results_df.drop(columns=metrics)
            results_df.loc[:, "Data Type"] = results_df['model'].apply(lambda x: 'TTS' if 'textual' in x else 'STS')
            results_df.loc[:, "Model"] = results_df['model'].apply(lambda x: x.split('_')[0])
            results_df = results_df.drop(columns=['model'])
            results_df = results_df.round(decimals=3)
            results_df = results_df.sort_values(by='Data Type')
            results_df[columns_order].to_csv(os.path.join(analysis_directory, "filtered_test_data.csv"), index=False)