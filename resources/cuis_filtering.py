import pandas as pd
import os
import pickle
from tqdm import tqdm

class FilterCUI():

    def __init__(self, cuis_paths:pd.DataFrame, num_top_cuis:int, num_bottom_cuis:int, filtered_dir_path:str):
        self.cuis_paths = cuis_paths
        self.num_top_cuis = num_top_cuis
        self.num_bottom_cuis = num_bottom_cuis
        self.filtered_dir_path = filtered_dir_path

    def filter(self, cuis_idf:pd.Series)-> pd.DataFrame:
        cuis_idf = cuis_idf.sort_values()
        new_paths = []
        if self.num_top_cuis + self.num_bottom_cuis > len(cuis_idf):
            return
        elif self.num_top_cuis + self.num_bottom_cuis < len(cuis_idf):
            top_cuis = cuis_idf.iloc[:self.num_top_cuis]
            bottom_cuis = cuis_idf.iloc[-self.num_bottom_cuis:]
            cuis_idf = top_cuis.append(bottom_cuis)
        columns = cuis_idf.index.tolist() + ['bucket']
        for index, row in tqdm(self.cuis_paths.iterrows(), total=len(self.cuis_paths)):
            tfidf_cuis = pd.read_csv(row['path'])
            for column in columns:
                if column not in tfidf_cuis.columns:
                    tfidf_cuis.loc[:, column] = None
            columns_to_remove = []
            for column in tfidf_cuis.columns:
                if column not in columns:
                    columns_to_remove.append(column)
            tfidf_cuis = tfidf_cuis.drop(columns=columns_to_remove)
            tfidf_cuis = tfidf_cuis.sort_values(by=["bucket"])
            tfidf_cuis = tfidf_cuis.drop(columns=["bucket"])
            tfidf_cuis = tfidf_cuis.reindex(sorted(tfidf_cuis.columns), axis=1)
            tfidf_cuis = tfidf_cuis.values
            filtered_episode_path = os.path.join(self.filtered_dir_path, '{}.pkl'.format(row['episode']))
            with open(filtered_episode_path, 'wb') as pkl_file:
                pickle.dump(tfidf_cuis, pkl_file)
            new_paths.append({"episode":row['episode'], 'path':filtered_episode_path})
        new_paths = pd.DataFrame(new_paths)
        return new_paths