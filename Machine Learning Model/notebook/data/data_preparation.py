from src.utils import save_object
import pandas as pd
import os
'''
6 -> jun
7 -> jul
8 -> aug
9 -> sep
'''

def prepare_dataset(df:pd.DataFrame)->pd.DataFrame:
    
    dropped_columns = ["year", "BUI", "FWI"]
    df = df.drop(columns=dropped_columns)

    month_mapping = {6: "jun", 7: "jul", 8: "aug", 9: "sep"}
    df["month"] = df['month'].map(month_mapping)
    
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names
    df.rename(columns={"Temperature": "temp", "Ws": "wind", "Rain": "rain", "Classes": "area"}, inplace=True)
    
    df["area"] = df["area"].str.strip()
    df["area"] = df["area"].map({"not fire": 0, "fire": 1})

    return df



df_1 = pd.read_csv("notebook/data/forestfires.csv")
df_2 = pd.read_csv("notebook/data/forestfire_dif.csv")
df_3 = pd.read_csv("notebook/data/forestfire_dif_1.csv")
df_4 = pd.read_csv("notebook/data/forestfire_dif_2.csv")

df_2 = prepare_dataset(df_2)
df_3 = prepare_dataset(df_3)
df_4 = prepare_dataset(df_4)

df_1["area"] = (df_1["area"] >= 10).astype(int)

result_df = pd.concat([df_1, df_2, df_3, df_4], axis=0)

path = os.path.join('notebook/data/', "forestfire_raw.csv")
os.makedirs(os.path.dirname(path), exist_ok=True)
result_df.to_csv(path, index=False, header=True)