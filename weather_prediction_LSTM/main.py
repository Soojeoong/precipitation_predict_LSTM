### main.py

#pip freeze > requirements.txt

import torch
from torch.utils.data import DataLoader
import argparse
from datasets import WeatherDataset
from LSTM import LSTM
from Trainer import Trainer
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    parser = argparse.ArgumentParser()
    
    # system args
    parser.add_argument("--seed",type=int, default="42")
    parser.add_argument("--device",type=str, default="cpu")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--do_eval",action="store_true")
    
    # model args
    parser.add_argument("--sequence_length", type=int, default=1, help="Sequence length for LSTM")
    parser.add_argument("--batch_size", type=int,default=4, help="batch size")
    parser.add_argument("--input_size", type=int, default=10, help="input size") # feature 수 (col 수)
    parser.add_argument("--hidden_size", type=int, default=2, help="hidden size") # 각 timestep에서 hidden_size차원의 벡터를 사용해 과거 정보 기억하고 처리함
    parser.add_argument("--num_layers", type=int, default=1, help="number of LSTM layers") # stack LSTM layer 수
    parser.add_argument("--output_size", type=int, default=1, help="output size") # 타겟 변수의 수
    
    # train args
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs" )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate model")
    parser.add_argument("--output_dir", type=str, default="output", help="output directory")
    
    # data args
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file")
    
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    set_seed(args.seed)

    check_path(args.output_dir)
    
    # Initialize dataset with sequence_length
    print("Train Dataset Load")
    train_dataset = WeatherDataset(csv_path=args.csv_path, flag="Train", sequence_length=args.sequence_length)

    print("Valid Dataset Load")
    valid_dataset = WeatherDataset(csv_path=args.csv_path, flag="Valid", sequence_length=args.sequence_length, scaler=train_dataset.scaler)

    print("Test Dataset Load")
    test_dataset = WeatherDataset(csv_path=args.csv_path, flag="Test", sequence_length=args.sequence_length, scaler=train_dataset.scaler)

    # Create DataLoader instances for train, valid, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Using Cuda:", torch.cuda.is_available())
    
    # Initialize model
    model = LSTM(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=args.output_size
    ).to(args.device)

    # Initialize trainer
    trainer = Trainer(model, train_loader, valid_loader, test_loader, args, scaler=train_dataset.scaler)

    if args.do_eval:
        # Load and evaluate model
        model_path = os.path.join(args.output_dir, "best_model.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
            trainer.test()
        else:
            print(f"No saved model found at {model_path}")
    else:
        # Train and evaluate model
        trainer.train()

if __name__ == "__main__":
    main()