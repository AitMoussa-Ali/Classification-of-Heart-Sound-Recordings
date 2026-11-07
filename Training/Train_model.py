import torch
import numpy as np
from Training.Early_stopping import EarlyStopping
import mlflow
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

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
        checkpoint_path,
        patience = 5,
        min_delta = 5
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
        self.patience = patience
        self.min_delta = min_delta

    def train_mod(self):
        total_train_loss = 0
        all_labels = []
        all_probs = []
        for bag, label in self.train_loader:
            all_labels.append(int(label.item()))

            bag = bag.squeeze(0).unsqueeze(1).to(self.device)
            label = label.to(self.device).reshape(())
            
            self.optimizer.zero_grad()
    
            output = self.model(bag)
            logit = output.mean(dim=0)
            proba = torch.sigmoid(logit).item()
            all_probs.append(proba)

            loss = self.criterion(
                logit,
                label.float().reshape_as(logit)
            )
            loss.backward()
    
            self.optimizer.step()
    
            total_train_loss+=loss.item()

        y_true = np.asarray(all_labels, dtype=np.int64)
        y_prob = np.asarray(all_probs)
        y_pred = (y_prob >= 0.5).astype(np.int64)
        
        
        return total_train_loss/len(self.train_loader), accuracy_score(y_true, y_pred)

    def eval_mode(self):
        total_valid_loss = 0
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for bag, label in self.valid_loader:
                label = label.to(self.device).reshape(())
                bag = bag.squeeze(0).unsqueeze(1).to(self.device)
                output = self.model(bag)
                logit = output.mean(dim=0)
                proba = torch.sigmoid(logit).item()
                all_probs.append(proba)
                all_labels.append(int(label.item()))

                loss = self.criterion(
                  logit,
                  label.float().reshape_as(logit)
                )

                total_valid_loss += loss.item()

            y_true = np.asarray(all_labels, dtype=np.int64)
            y_prob = np.asarray(all_probs)
            y_pred = (y_prob >= 0.5).astype(np.int64)
            tn, fp, fn, tp = confusion_matrix(
                y_true, y_pred, labels=[0, 1]
            ).ravel()

            mlflow.log_metrics({
                "validation_accuracy" :  accuracy_score(y_true, y_pred),
                "precision_abnormal_class" : precision_score(y_true=y_true, y_pred=y_pred, pos_label=1), 
                "precision_normal_class" : precision_score(y_true=y_true, y_pred=y_pred, pos_label=0), 
                "recall_sensitivity_abnormal" : recall_score(y_true, y_pred, pos_label=1),
                "recall_sensitivity_normal" : recall_score(y_true, y_pred, pos_label=0),
            })
        return total_valid_loss/len(self.valid_loader), accuracy_score(y_true, y_pred)

    def train_model(self, fold):
        loss_res_train = []
        loss_res_validation = []

        if self.patience is not None and self.min_delta is not None:
            early_stopping = EarlyStopping(
                patience=self.patience,
                min_delta=self.min_delta
            )
        
        for epoch in range(self.start_epoch, self.number_epochs):
            #Training mode
            self.model.train()
            mean_loss_train, accuracy_train = self.train_mod()
            loss_res_train.append(mean_loss_train)

            self.model.eval()
            #Evaluation mode
            mean_loss_validation, accuracy_validation = self.eval_mode()
            loss_res_validation.append(mean_loss_validation)
            
            mlflow.log_metrics(
                {
                    "train_loss" : mean_loss_train,
                    "validation_loss" : mean_loss_validation
                },
                step=epoch+1
            )

            print(
               f"Fold : {fold} | "
               f"Train loss : {mean_loss_train:.4f} | "
               f"Train accuracy : {accuracy_train:.4f} | "
               f"Validation loss : {mean_loss_validation:.4f} | "
               f"Validation accuracy : {accuracy_validation:.4f} | "
               f"epoch : {epoch + 1}/{self.number_epochs}"
            )

            if early_stopping(mean_loss_validation):
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

            if early_stopping.early_stop:
                print(f"Early stopping detecter a lepoch {epoch}. l'entrainement est suspendue")
                break
            
        return loss_res_train, loss_res_validation

    