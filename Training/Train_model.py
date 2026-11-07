import torch
import numpy as np

class TrainModel():

    def __init__(
        self, 
        model, 
        criterion, 
        optimizer, 
        train_loader, 
        device, 
        valid_loader,
        start_epoch,
        number_epochs,
        checkpoint_path
        ):
        super().__init__
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device
        self.valid_loader = valid_loader
        self.start_epoch = start_epoch
        self.number_epochs = number_epochs 
        self.checkpoint_path = checkpoint_path

    def train_mod(self):
        total_train_loss = 0

        for bag, label in self.train_loader:
            bag = bag.squeeze(0).unsqueeze(1).to(self.device)
            label = label.to(self.device).reshape(())
            
            self.optimizer.zero_grad()
    
            output = self.model(bag)
            logit = output.mean(dim=0)
    
            loss = self.criterion(
                logit,
                label.float().reshape_as(logit)
            )
            loss.backward()
    
            self.optimizer.step()
    
            total_train_loss+=loss.item()
        
        return total_train_loss/len(self.train_loader)

    def eval_mode(self):
        total_valid_loss = 0

        with torch.no_grad():
            for bag, label in self.valid_loader:
                label = label.to(self.device).reshape(())
                bag = bag.squeeze(0).unsqueeze(1).to(self.device)
                output = self.model(bag)
                logit = output.mean(dim=0)

                loss = self.criterion(
                  logit,
                  label.float().reshape_as(logit)
                )

                total_valid_loss += loss.item()

        return total_valid_loss/len(self.valid_loader)

    def train_model(self, fold):
        loss_res_train = []
        loss_res_validation = []
        
        for epoch in range(self.start_epoch, self.number_epochs):
          #Training mode
          self.model.train()
          mean_loss_train = self.train_mod()
          loss_res_train.append(mean_loss_train)

          self.model.eval()
          #Evaluation mode
          mean_loss_validation = self.eval_mode()
          loss_res_validation.append(mean_loss_validation)

          checkpoint = {
                "fold": fold,
                "epoch" : epoch+1,
                "train loss" : mean_loss_train,
                "validation loss" : mean_loss_validation, 
                "num_epochs" : self.number_epochs,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_state_dict" : self.model.state_dict()
            }
          torch.save(checkpoint, self.checkpoint_path)
          print("checkpoint saved .. !")
          print(
             f"Fold : {fold} | Train loss : {mean_loss_train:.4f} | Validation loss : {mean_loss_validation:.4f} | epoch : {epoch + 1}/{self.number_epochs}"
          )

    