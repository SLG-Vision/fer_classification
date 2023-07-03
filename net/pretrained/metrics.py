from sklearn.metrics import precision_score, recall_score, f1_score, log_loss
import torch
import matplotlib.pyplot as plt

class compute_metric:
    
    def __init__(self,db) -> None:
        self.db = db
        
        # Training metrics
        self.epoch_loss= []
        self.epoch_acc = []
        self.epoch_y_true = []
        self.epoch_y_pred = []
        self.loss_values= []
        self.acc_values= []
        self.precision = [] 
        self.recall = [] 
        self.f1 = [] 

        
        # Testing metrics
        self.epoch_test_loss = []
        self.epoch_test_acc = []
        self.epoch_test_y_true = []
        self.epoch_test_y_pred = []
        self.test_acc = []
        self.test_loss = []
        self.test_precision = [] 
        self.test_recall = [] 
        self.test_f1 = [] 

    
    def add(self, outputs, acc, pred, labels):
        self.epoch_acc.append(acc)
        self.epoch_y_true.extend(labels.cpu().numpy())
        self.epoch_y_pred.extend(pred.cpu().numpy())
        logloss = log_loss(labels.cpu().numpy(), torch.nn.functional.softmax(outputs.detach().cpu(), dim=1), labels=[0,1,2,3,4,5,6])
        self.epoch_loss.append(logloss)

        
    def update(self):
        self.loss_values.append(sum(self.epoch_loss)/len(self.epoch_loss))
        self.acc_values.append(sum(self.epoch_acc)/len(self.epoch_acc))
        self.precision.append(precision_score(  self.epoch_y_true, self.epoch_y_pred, average='macro', zero_division=0))
        self.recall.append(recall_score(        self.epoch_y_true, self.epoch_y_pred, average='macro'))
        self.f1.append(f1_score(                self.epoch_y_true, self.epoch_y_pred, average='macro'))

        self.epoch_loss.clear()
        self.epoch_acc.clear()
        self.epoch_y_pred.clear()
        self.epoch_y_true.clear()
    
    def add_test(self, outputs, acc, pred, labels):
        self.epoch_test_acc.append(acc)
        self.epoch_test_y_true.extend(labels.cpu().numpy())
        self.epoch_test_y_pred.extend(pred.cpu().numpy())
        logloss = log_loss(labels.cpu().numpy(), torch.nn.functional.softmax(outputs.detach().cpu(), dim=1), labels=[0,1,2,3,4,5,6])
        self.epoch_test_loss.append(logloss)
    
    def update_test(self):
        self.test_loss.append(sum(self.epoch_test_loss)/len(self.epoch_test_loss))
        self.test_acc.append(sum(self.epoch_test_acc)/len(self.epoch_test_acc))   
        self.test_precision.append(precision_score( self.epoch_test_y_true, self.epoch_test_y_pred, average='macro', zero_division=0))
        self.test_recall.append(recall_score(       self.epoch_test_y_true, self.epoch_test_y_pred, average='macro'))
        self.test_f1.append(f1_score(               self.epoch_test_y_true, self.epoch_test_y_pred, average='macro'))

        
        self.epoch_test_loss.clear()
        self.epoch_test_acc.clear()
        self.epoch_test_y_pred.clear()
        self.epoch_test_y_true.clear()

    def plot(self, epochs, metric):
        plt.title(self.db + '_' + metric)
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        if metric == 'Loss':
            print(max(self.test_loss))
            plt.plot(epochs, self.loss_values, label='Train')
            plt.plot(epochs, self.test_loss  , label='Test')
        if metric == 'Accuracy':
            print(max(self.test_acc))
            plt.plot(epochs, self.acc_values, label='Train')
            plt.plot(epochs, self.test_acc  , label='Test')
        if metric == 'Precision':
            print(max(self.test_precision))
            plt.plot(epochs, self.precision, label='Train')
            plt.plot(epochs, self.test_precision  , label='Test')
        if metric == 'Recall':
            print(max(self.test_recall))
            plt.plot(epochs, self.recall, label='Train')
            plt.plot(epochs, self.test_recall  , label='Test')
        if metric == 'F1-Score':
            print(max(self.test_f1))
            plt.plot(epochs, self.f1, label='Train')
            plt.plot(epochs, self.test_f1  , label='Test')
            
        plt.legend()
        plt.savefig(f"plot_{self.db}_{metric}.png")
        plt.clf()