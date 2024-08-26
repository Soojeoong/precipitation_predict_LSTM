### Trainer.py

# import
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class Trainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, args, scaler):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.args = args
        self.scaler = scaler
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_losses = []
        self.valid_losses = []
        self.best_valid_loss = float('inf')

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(self.train_loader, desc="Training"):
            inputs = inputs.to(self.args.device)#.unsqueeze(1)
            targets = targets.to(self.args.device).unsqueeze(1)  # (batch_size, 1)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(self.valid_loader, desc="Validating"):
                inputs = inputs.to(self.args.device)#.unsqueeze(1)
                targets = targets.to(self.args.device).unsqueeze(1)  # (batch_size, 1)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
        return running_loss / len(self.valid_loader)

    def test(self):
        self.model.eval()
        running_loss = 0.0
        predictions = []
        true_values = []
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Testing"):
                inputs = inputs.to(self.args.device)#.unsqueeze(1)
                targets = targets.to(self.args.device).unsqueeze(1)  # (batch_size, 1)

                outputs = self.model(inputs)
                
                # Inverse transform the predictions and targets
                #outputs = outputs.cpu().numpy() * self.scaler.scale_[-1] + self.scaler.mean_[-1] ###
                #targets = self.scaler.inverse_transform(targets.cpu().numpy()) ###
                
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                
                predictions.append(outputs.cpu().numpy())
                true_values.append(targets.cpu().numpy())

        average_loss = running_loss / len(self.test_loader)
        print(f"Test Loss: {average_loss:.4f}")
        
        # Flatten predictions and true values
        predictions = [item for sublist in predictions for item in sublist]
        true_values = [item for sublist in true_values for item in sublist]
        
        # Plotting predictions vs true values
        plt.figure(figsize=(10, 6))
        plt.plot(true_values, label='True Values')
        plt.plot(predictions, label='Predictions')
        plt.xlabel('Samples')
        plt.ylabel('Precipitation')
        plt.title('Predictions vs True Values')
        plt.legend()
        plot_path = os.path.join(self.args.output_dir, 'test_results.png')
        plt.savefig(plot_path)
        print(f"Test results plot saved at {plot_path}")
        
        return average_loss

    def train(self):
        log_file_path = os.path.join(self.args.output_dir, "training_log.txt")
        model_save_path = os.path.join(self.args.output_dir, "best_model.pth")

        with open(log_file_path, "w") as log_file:
            log_file.write(f"num_layers: {self.args.num_layers}\n")
            log_file.write(f"hidden_size: {self.args.hidden_size}\n")
            log_file.write(f"sequence_length: {self.args.sequence_length}\n")
            
            log_file.write(f"Batch Size: {self.args.batch_size}\n")
            log_file.write(f"Epochs: {self.args.epochs}\n")
            log_file.write(f"Learning Rate: {self.args.lr}\n\n")
            log_file.write("Epoch, Train Loss, Validation Loss\n")
            log_file.flush()  # 파일에 즉시 기록
            print("******************log writing 1")
            for epoch in range(self.args.epochs):
                print(f"Epoch {epoch+1}/{self.args.epochs}")
                train_loss = self.train_epoch()
                valid_loss = self.validate()

                self.train_losses.append(train_loss)
                self.valid_losses.append(valid_loss)

                print(f"Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")

                log_file.write(f"{epoch+1}, {train_loss:.4f}, {valid_loss:.4f}\n")
                log_file.flush()  # 파일에 즉시 기록
                print("******************log writing 2")
                # Save the model if validation loss decreases
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), model_save_path)
                    print(f"Model saved at {model_save_path}")

        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.valid_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        loss_plot_path = os.path.join(self.args.output_dir, 'loss_plot.png')
        plt.savefig(loss_plot_path)
        print(f"Loss plot saved at {loss_plot_path}")

        # Load the best model and test
        self.model.load_state_dict(torch.load(model_save_path))
        print(f"Model loaded from {model_save_path} for testing.")
        test_loss = self.test()
        with open(log_file_path, "a") as log_file:
            log_file.write(f"\nTest Loss: {test_loss:.4f}\n")
            log_file.flush()