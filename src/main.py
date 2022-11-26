from datascience import cleaning, exploratory, feature_eng, model, read_data

stroke_df = read_data.read_csv("Data\strokedata_raw.csv")
print(stroke_df.head())
