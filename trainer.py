import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm

class Trainer():
    def __init__(self, train_loader, valid_loader, model, loss_fn, optimizer, scheduler, device, patience, epochs, result_path, fold_logger, **kargs):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.epochs = epochs
        self.logger = fold_logger
        self.best_model_path = os.path.join(result_path, 'best_model.pt')
        self.last_model_path = os.path.join(result_path, 'last_model.pt')

        self.start_epoch = kargs['start_epoch'] if 'start_epoch' in kargs else 0
    
    def train(self):
        best = np.inf
        for epoch in range(self.start_epoch + 1, self.epochs+1):
            epoch_start_time = time.time()

            print(f'lr: {self.scheduler.get_last_lr()}')
            loss_train, score_train = self.train_step()
            loss_val, score_val, val_throughput = self.valid_step()
            self.scheduler.step()

            epoch_time = time.time() - epoch_start_time
            self.logger.info(f'Epoch {str(epoch).zfill(5)}: t_loss:{loss_train:.3f} t_score:{score_train:.3f} v_loss:{loss_val:.3f} v_score:{score_val:.3f} throughput:{val_throughput:.2f} samples/sec time:{epoch_time:.2f}s')

            if loss_val < best:
                best = loss_val
                torch.save({
                    'model':self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'epoch': epoch,
                    'score_val': score_val.item(),
                    'loss_val': loss_val.item(), 
                }, self.best_model_path)
                bad_counter = 0

            else:
                bad_counter += 1

            if bad_counter == self.patience:
                break

            torch.save({
                'model':self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': epoch,
                'score_val': score_val.item(),
                'loss_val': loss_val.item(), 
            }, self.last_model_path)

    def train_step(self):
        self.model.train()

        total_loss = 0
        correct = 0
        for batch in tqdm(self.train_loader, file=sys.stdout): #tqdm output will not be written to logger file(will only written to stdout)
            x, y = batch['data'].to(self.device), batch['label'].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)            
            loss = self.loss_fn(output, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.shape[0]
            correct += sum(output.argmax(dim=1) == y).item() # classification task
        
        return total_loss/len(self.train_loader.dataset), correct/len(self.train_loader.dataset)
    
    def valid_step(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            correct = 0
            val_start_time = time.time()
            for batch in self.valid_loader:
                x, y = batch['data'].to(self.device), batch['label'].to(self.device)

                output = self.model(x)
                loss = self.loss_fn(output, y)

                total_loss += loss.item() * x.shape[0]
                correct += sum(output.argmax(dim=1) == y).item() # classification task
            
            val_time = time.time() - val_start_time
            throughput = len(self.valid_loader.dataset) / val_time

        return total_loss/len(self.valid_loader.dataset), correct/len(self.valid_loader.dataset), throughput
    
    def test(self, test_loader):
        # best_model.pt 파일이 없을 경우를 대비하여 last_model.pt 로드 시도
        model_path = self.best_model_path if os.path.exists(self.best_model_path) else self.last_model_path
        if not os.path.exists(model_path):
            self.logger.error("No model file found to load for testing.")
            return np.array([])
        
        self.model.load_state_dict(torch.load(model_path)['model'])
        self.model.eval()
        with torch.no_grad():
            result = []
            for batch in test_loader:
                # test_loader의 배치 형식에 'data' 키가 있다고 가정
                x = batch['data'].to(self.device)
                # 모델은 softmax를 포함하지 않은 로짓을 반환한다고 가정
                output = self.model(x)
                # softmax를 적용하여 확률로 변환
                probabilities = torch.softmax(output, dim=1)
                result.append(probabilities.detach().cpu().numpy())

        return np.concatenate(result,axis=0)