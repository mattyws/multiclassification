import pandas as pd
from tqdm import tqdm

def analyse_missing_values(dataset:pd.DataFrame, file_path_column:str):
    print("Analysing missing values")
    total_events = 0
    missing_events_features = dict()
    total_events_features_time = dict()
    missing_events_features_time = dict()
    for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        episode_df = pd.read_csv(row[file_path_column])
        episode_df = episode_df.set_index(["bucket"], drop=False)
        if "Unnamed: 0" in episode_df.columns:
            episode_df = episode_df.drop(columns=["Unnamed: 0"])
        total_events += len(episode_df)
        total_events_features_time['bucket'] = [x for x in range(48)]
        missing_events_features_time['bucket'] = [x for x in range(48)]
        for column in episode_df.columns:
            if column == "bucket" or "Unnamed" in column or column == "starttime" or column == "endtime":
                continue
            if column not in missing_events_features.keys():
                missing_events_features[column] = 0
            if column not in missing_events_features_time.keys():
                missing_events_features_time[column] = [0 for x in range(48)]
            if column not in total_events_features_time.keys():
                total_events_features_time[column] = [0 for x in range(48)]
            missing_events_features[column] += len(episode_df[episode_df[column].isna()])
            for n, value in enumerate(episode_df[column].values):
                if not pd.isna(value):
                    missing_events_features_time[column][n] += 1
                total_events_features_time[column][n] += 1
    total_events_features_time = pd.DataFrame(total_events_features_time)
    missing_events_features_time = pd.DataFrame(missing_events_features_time)
    events_features_time = pd.merge(total_events_features_time, missing_events_features_time, left_on="bucket", right_on='bucket',
                                    suffixes=["_total", "_missing"])

    events_features = []
    for key in missing_events_features.keys():
        events_features.append({"feature":key, "total_missing":missing_events_features[key], "total":total_events, "missing_rate":total_events/missing_events_features[key]})
    events_features = pd.DataFrame(events_features)
    events_features = events_features.sort_values(by=["missing_rate"], ascending=False)
    return events_features, events_features_time