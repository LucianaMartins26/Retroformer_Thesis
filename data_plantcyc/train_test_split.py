import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../../../../data_processing_retroformer_pipeline/data_processing_pipeline/res/retroformer_format_ready_data.csv')

raw_train, raw_temp = train_test_split(df, test_size=0.4, random_state=42)

raw_test, raw_val = train_test_split(raw_temp, test_size=0.5, random_state=42)

raw_train.to_csv('raw_train.csv', index=False)
raw_test.to_csv('raw_test.csv', index=False)
raw_val.to_csv('raw_val.csv', index=False)