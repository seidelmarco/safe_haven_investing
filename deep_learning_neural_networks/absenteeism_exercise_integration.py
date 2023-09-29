import pandas as pd

from absenteeism_module_final import *

df = pd.read_csv('data_models/Absenteeism_new_data.csv')
# print(df)

model = AbsenteeismModel(model_file='data_models/absenteeism_model.pickle',
                         scaler_file='data_models/absenteeism_scaler.pickle')


model.load_and_clean_data(data_file='data_models/Absenteeism_new_data.csv')
print(model.predicted_outputs())

preprocessed_data = model.predicted_outputs()
# print(preprocessed_data)
model.predicted_outputs().to_csv('data_models/Absenteeism_predictions.csv', index=False)

preprocessed_data.to_excel('data_models/Absenteeism_predictions.xlsx', index=False)

