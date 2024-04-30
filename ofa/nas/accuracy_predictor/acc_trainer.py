# 1K ~ 4K samples per resolution are usually sufficient for training the accuracy predictor.

# In the data preprocessing phase, it is important to make sure the accuracy scale is [0, 1] instead of [0, 100].

# The optimizer is adam. The learning rate is 1e-3. The weight decay is 1e-4. This training setting works well with different batch sizes (e.g., 500, 1000, etc).

# Besides, setting the bias term of the output layer as the average accuracy can improve the training stability.
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


class AccPredictorTrainer(object):
    def __init__(self,
                 predictor,
                 train_dataloader,
                 eval_dataloader,
                 save_name,
                 num_epochs,
                 device="cuda:0"):
        self.predictor = predictor
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        self.num_epochs = num_epochs
        self.lr = 1e-3
        self.wd = 1e-4
        self.batch_size = 1000
        
        self.loss_fn = RMSELoss()
        self.device = device
        
        self.save_dir = "model_zoo/accpredictors"
        self.save_name = save_name
        
        
    def train(self, base_acc):
        torch.nn.init.constant_(
            self.predictor.base_acc,
            torch.tensor(base_acc)
        )
        self.optimizer = optim.Adam(self.predictor.parameters(),
                               lr=self.lr,
                               weight_decay=self.wd)
        
        for t in tqdm(range(self.num_epochs)):
            self.train_one_epoch()
            self.evaluate()
        
        
    def train_one_epoch(self):
        model = self.predictor.to(self.device)
        loss_fn = self.loss_fn.to(self.device)
        
        size = len(self.train_dataloader.dataset)
        
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)
            pred = model(X)
            loss = loss_fn(pred, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def evaluate(self):
        model = self.predictor.to(self.device)
        loss_fn = self.loss_fn.to(self.device)
        
        model.eval()
        rmse_total = 0
        
        size = len(self.eval_dataloader.dataset)
        with torch.no_grad():
             for batch, (X, y) in enumerate(self.eval_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                loss = loss_fn(pred, y)
                
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                rmse = np.sqrt(mean_squared_error(pred.cpu().numpy()*100, y.cpu().numpy()*100))
                rmse_total += rmse
        
        rmse_f = rmse_total / len(self.eval_dataloader)
        print(f"rmse: {rmse_f:.4f}")

    def save(self):
        import os
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        save_name =os.path.join(self.save_dir,self.save_name+'.pth')
        torch.save(self.predictor.state_dict(), save_name)
        print(f'save model in {save_name}')
    
    def load(self, model_path):
        self.predictor.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f'loaded model in {model_path}')