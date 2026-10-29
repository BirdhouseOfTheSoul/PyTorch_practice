import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import re
import numpy as np
from sklearn.preprocessing import OneHotEncoder

energy=pd.read_csv('Energy_Usage_2010_20261010.csv', dtype={'AVERAGE HOUSESIZE': 'str', 'ZERO KWH ACCOUNTS': 'str', 'CENSUS BLOCK': 'str'})
predictors=['COMMUNITY AREA NAME', 'BUILDING TYPE', 'BUILDING_SUBTYPE', 'ELECTRICITY ACCOUNTS', 'GAS ACCOUNTS', 'TOTAL POPULATION', 'TOTAL UNITS', 'AVERAGE STORIES', 'AVERAGE BUILDING AGE', 'AVERAGE HOUSESIZE', 'OCCUPIED UNITS', 'OCCUPIED UNITS PERCENTAGE', 'RENTER-OCCUPIED HOUSING UNITS', 'RENTER-OCCUPIED HOUSING PERCENTAGE', 'OCCUPIED HOUSING UNITS']



strings_to_floats=['AVERAGE HOUSESIZE', 'OCCUPIED UNITS', 'RENTER-OCCUPIED HOUSING UNITS', 'OCCUPIED HOUSING UNITS', 'ELECTRICITY ACCOUNTS', 'GAS ACCOUNTS', 'TOTAL POPULATION', 'TOTAL UNITS', 'TOTAL KWH', 'TOTAL THERMS']
for col in strings_to_floats:
    energy[col]=pd.to_numeric(energy[col].apply(lambda x: re.sub(',','',str(x))), errors='coerce')

categorical_predictors=['COMMUNITY AREA NAME', 'BUILDING TYPE', 'BUILDING_SUBTYPE']


to_predict = ['TOTAL KWH', 'TOTAL THERMS']


energy.dropna(subset=to_predict, inplace=True)


one_hot_encoder=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
predictor_encodings={}
new_categorical_predictor_names=[]
for predictor in categorical_predictors:
    predictor_encodings[predictor] = one_hot_encoder.fit(energy[[predictor]])
    encoded=predictor_encodings[predictor].transform(energy[[predictor]])
    encoded_df=pd.DataFrame(encoded, columns=predictor_encodings[predictor].get_feature_names_out([predictor]))
    energy=pd.concat([energy, encoded_df], axis=1)
    energy.drop(columns=[predictor], inplace=True)
    new_categorical_predictor_names.extend(predictor_encodings[predictor].get_feature_names_out([predictor]))

predictors = [p for p in predictors if p not in categorical_predictors] + new_categorical_predictor_names
n_predictors = len(predictors)

x_tensor = torch.from_numpy(energy[predictors].values.astype(np.float32))
y_tensor = torch.from_numpy(energy[to_predict].values.astype(np.float32))
y_tensor = (y_tensor - y_tensor.mean())/y_tensor.std()
x_tensor = (x_tensor - x_tensor.mean())/x_tensor.std()

x_tensor.shape, y_tensor.shape

model = nn.Sequential(nn.Linear(n_predictors, 2), nn.ReLU())
for x in model.parameters():
    print(x)


# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
x_tensor.shape
output=model(x_tensor)
#print(' '.join([str(x) for x in model.parameters()]))
#print(y_tensor)
print(output)
