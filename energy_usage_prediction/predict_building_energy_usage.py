# A simple practice exercise with pytorch to build a linear regression model to predict energy usage by building using city of chicago energy usage data

from time import time
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import re
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader



class EnergyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



class EnergyPredictor(nn.Module):
    def __init__(self, input_dim, dropout=0):
        super(EnergyPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(32, 2)  # Output layer for two targets

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x    


def process_data_and_features(csv_file):
    energy=pd.read_csv(csv_file, dtype={'AVERAGE HOUSESIZE': 'str', 'ZERO KWH ACCOUNTS': 'str', 'CENSUS BLOCK': 'str'})
    predictors=['COMMUNITY AREA NAME', 'BUILDING TYPE', 'BUILDING_SUBTYPE', 'ELECTRICITY ACCOUNTS', 'GAS ACCOUNTS', 'TOTAL POPULATION', 'TOTAL UNITS', 'AVERAGE STORIES', 'AVERAGE BUILDING AGE', 'AVERAGE HOUSESIZE', 'OCCUPIED UNITS', 'OCCUPIED UNITS PERCENTAGE', 'RENTER-OCCUPIED HOUSING UNITS', 'RENTER-OCCUPIED HOUSING PERCENTAGE', 'OCCUPIED HOUSING UNITS']

    strings_to_floats=['AVERAGE HOUSESIZE', 'OCCUPIED UNITS', 'RENTER-OCCUPIED HOUSING UNITS', 'OCCUPIED HOUSING UNITS', 'ELECTRICITY ACCOUNTS', 'GAS ACCOUNTS', 'TOTAL POPULATION', 'TOTAL UNITS', 'TOTAL KWH', 'TOTAL THERMS']
    for col in strings_to_floats:
        energy[col]=pd.to_numeric(energy[col].apply(lambda x: re.sub(',','',str(x))), errors='coerce')

    categorical_predictors=['COMMUNITY AREA NAME', 'BUILDING TYPE', 'BUILDING_SUBTYPE']
    to_predict = ['TOTAL KWH', 'TOTAL THERMS']


    energy=energy.dropna(subset=to_predict, ignore_index=True)

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

    all_used_columns = predictors + to_predict

    means = energy[all_used_columns].mean(axis=0)
    stds = energy[all_used_columns].std(axis=0)
    energy[all_used_columns] = energy[all_used_columns].fillna(means)
    energy[all_used_columns] = (energy[all_used_columns] - means) / stds

    return energy[predictors], energy[to_predict]





def train_epoch(model, train_loader, loss_function, optimizer, device):
    model.train()
    model=model.to(device)
    total_loss = 0.0
    total_samples = 0
    for _, (x_batch, y_batch) in enumerate(train_loader):
        x_tensor = x_batch.to(device)
        y_tensor = y_batch.to(device)
        optimizer.zero_grad()
        # Make predictions (forward pass)
        outputs = model(x_tensor)
        #print("outputs:", outputs)
        # Calculate the loss
        loss = loss_function(outputs, y_tensor)
        # Calculate adjustments (backward pass)
        loss.backward()
        # Update the model's parameters
        optimizer.step()
        total_loss += loss.detach() * x_tensor.size(0)
        total_samples += x_tensor.size(0)

    return model, total_loss / total_samples


def evaluate(model, test_loader, loss_function, device):
    """
    Evaluates the model's accuracy on a test dataset.

    This function sets the model to evaluation mode, iterates through the test data,
    and calculates the percentage of correct predictions.

    Args:
        model: The neural network model to be evaluated.
        test_loader: A data loader containing the test dataset.
        device: The device (e.g., 'cpu' or 'cuda') to run the evaluation on.

    Returns:
        The accuracy of the model on the test dataset as a percentage.
    """
    # Sets the model to evaluation mode.
    model.eval()
    total_loss = 0.0
    total_samples = 0

    # Disables gradient calculation to reduce memory usage and speed up computations.
    with torch.no_grad():
        # Iterates over all batches in the test data loader.
        for inputs, targets in test_loader:
            # Moves the input data and targets to the specified device.
            inputs, targets = inputs.to(device), targets.to(device)
            # Performs a forward pass to get the model's output.
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            total_loss += loss.detach() * inputs.size(0)
            total_samples += inputs.size(0)
             

    # Calculates the overall mean squared error (MSE) loss on the validation set.
    mse_loss = total_loss / total_samples
    return mse_loss
            

def main():
    energy_predictors_df, energy_to_predict_df = process_data_and_features('Energy_Usage_2010_20261010.csv')
    X_train, X_test, y_train, y_test = train_test_split(energy_predictors_df, energy_to_predict_df, test_size=0.2, random_state=14)

    train_dataset = EnergyDataset(X_train, y_train)
    val_dataset = EnergyDataset(X_test, y_test)
    print('train dataset size:', len(train_dataset))
    print('validation dataset size:', len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    model = EnergyPredictor(X_train.shape[1], dropout=0.2)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")



    # Define the loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    train_loss_history = []
    val_loss_history = []
    for epoch in range(10):
        start_time = time()
        #torch.mps.synchronize()
        model, loss = train_epoch(model, train_loader, loss_function, optimizer, device)
        train_loss_history.append(loss.item())
        #torch.mps.synchronize()
        print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}")#, Time: {time() - start_time:.2f}")
        val_loss = evaluate(model, val_loader, loss_function,device)
        print(f"Epoch {epoch}, Validation Loss: {val_loss.item():.4f}")
        val_loss_history.append(val_loss.item())
        scheduler.step()


    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

