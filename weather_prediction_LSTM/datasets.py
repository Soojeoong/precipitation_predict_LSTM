# datasets.py

# import
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import argparse

# Dataset 클래스 만들기
class WeatherDataset(Dataset):
    def __init__(self, csv_path, flag, sequence_length=5, split_ratio=(0.8, 0.1, 0.1), scaler=None):
        super(WeatherDataset, self).__init__()
        # 데이터 불러오기
        # X(종속변수), y(예측변수) 저장
        # Float Tensor 타입으로 변환
        self.sequence_length = sequence_length
        self.csv_path = csv_path # 데이터 경로
        self.flag = flag # train, valid, test
        self.scaler = scaler
        
        # Load the dataset
        self.data = pd.read_csv(self.csv_path)
        
        # feature engineering
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = pd.get_dummies(self.data, columns=['weather'], drop_first=True)
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        self.data['day'] = self.data['date'].dt.day
        self.data.drop('date', axis=1, inplace=True)
        
        # target variable : 'precipitation'
        self.X = self.data.drop('precipitation', axis=1).values
        self.y = self.data['precipitation'].values
        
        # Scaling
        #if self.flag == "Train":
        #    self.scaler = StandardScaler()
        #    self.X_scaled = self.scaler.fit_transform(self.X)
        #else: # Valid, Test
        #    self.X_scaled = self.scaler.transform(self.X) 
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
       # Generate sequences
        self.X_sequences, self.y_sequences = self.create_sequences(self.X_scaled, self.y)

        # Split the data
        total_size = len(self.X_sequences)
        train_end = int(split_ratio[0] * total_size)
        valid_end = train_end + int(split_ratio[1] * total_size)

        if self.flag == "Train":
            self.X_sequences = self.X_sequences[:train_end]
            self.y_sequences = self.y_sequences[:train_end]
        elif self.flag == "Valid":
            self.X_sequences = self.X_sequences[train_end:valid_end]
            self.y_sequences = self.y_sequences[train_end:valid_end]
        elif self.flag == "Test":
            self.X_sequences = self.X_sequences[valid_end:]
            self.y_sequences = self.y_sequences[valid_end:]

    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)

    def __len__(self):
        return len(self.y_sequences)

    def __getitem__(self, idx):
        return self.X_sequences[idx], self.y_sequences[idx]

if __name__ == "__main__":
    sequence_length = 5

    # Train dataset
    train_dataset = WeatherDataset(csv_path='seattle-weather.csv', flag='Train', sequence_length=sequence_length)
    print(f'Dataset length for Train split: {len(train_dataset)}')

    # Check if sequence length is correct
    print("\nChecking sequence lengths in Train dataset:")
    for i in range(3):  # Print first 3 sequences
        X_sample, y_sample = train_dataset[i]
        print(X_sample)
        print(f"X_sample[{i}] shape: {X_sample.shape}")  # Should be (sequence_length, num_features)
        print(f"y_sample[{i}]: {y_sample}\n")
