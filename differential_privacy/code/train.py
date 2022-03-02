import numpy as np
import pandas as pd
import argparse
import os
import json
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler 

#library for privacy engine
from opacus import PrivacyEngine

class ChurnDataset(Dataset):
 
    def __init__(self, df):
  
        df = pd.read_csv(csv_file)
        
        df = df.drop(["Surname", "CustomerId", "RowNumber"], axis=1)

        # Grouping variable names
        self.categorical = ["Geography", "Gender"]
        self.target = "Exited"

        # One-hot encoding of categorical variables
        self.churn_frame = pd.get_dummies(df, prefix=self.categorical)

        # Save target and predictors
        self.X = self.churn_frame.drop(self.target, axis=1)
        self.y = self.churn_frame["Exited"]
        
        scaler = StandardScaler()
        X_array  = scaler.fit_transform(self.X)
        self.X = pd.DataFrame(X_array)

    def __len__(self):
        return len(self.churn_frame)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values, self.y[idx]]
    
def get_CHURN_model():
    model = nn.Sequential(nn.Linear(13, 64), 
                    nn.ReLU(), 
                    nn.Linear(64, 64), 
                    nn.ReLU(), 
                    nn.Linear(64, 1)) 
    return model

def get_dataloader(csv_file, batch_size):
     # Load dataset
    dataset = ChurnDataset(csv_file)

    # Split into training and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    
    return trainloader, testloader, trainset, testset

def train(trainloader, net, optimizer, n_epochs=100):
     
    device = "cpu"

    # Define the model
    #net = get_CHURN_model()
    net = net.to(device)
    
    #criterion = nn.CrossEntropyLoss() 
    criterion = nn.BCEWithLogitsLoss()

    # Train the net
    loss_per_iter = []
    loss_per_batch = []
    for epoch in range(n_epochs):

        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs.float())
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()

            # Save loss to plot
            running_loss += loss.item()
            loss_per_iter.append(loss.item())

        
        print("Epoch {} - Training loss: {}".format(epoch, running_loss/len(trainloader))) 
        
        running_loss = 0.0
        
    return net

def write_diffpriv(epsilon, target_delta):
    header = ['Epsilon', 'target_delta']
    data = [epsilon, target_delta]
    
    with open(os.path.join(args.model_dir, 'differential_privacy.csv'), 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(data)

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()
    
    batch_size = 50
    csv_file = os.path.join(args.train, 'churn.csv')
    trainloader, testloader, train_ds, test_ds = get_dataloader(csv_file, batch_size)
    
    ## train model with NO privacy engine
    net = get_CHURN_model()
    optimizer = optim.Adam(net.parameters(), weight_decay=0.0001, lr=0.003)
    model = train(trainloader, net, optimizer, 50)
    
    print("#### Model training with Differential Privacy ####")
    
    ## train model with privacy engine
    max_per_sample_grad_norm = 1.5
    sample_rate = batch_size/len(train_ds)
    noise_multiplier = 0.8
    
    net = get_CHURN_model()
    optimizer = optim.Adam(net.parameters(), weight_decay=0.0001, lr=0.003)

    privacy_engine = PrivacyEngine(
        net,
        max_grad_norm=max_per_sample_grad_norm,
        noise_multiplier = noise_multiplier,
        sample_rate = sample_rate,
    )

    privacy_engine.attach(optimizer)
    model = train(trainloader, net, optimizer, batch_size)

    # Save the Model
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
        
    # Save the Privacy engine values
    epsilon, best_alpha = privacy_engine.get_privacy_spent()
    write_diffpriv(epsilon, privacy_engine.target_delta)
    
    print (f" ε = {epsilon:.2f}, δ = {privacy_engine.target_delta}")