import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch


class DBLoader():
    
    def __init__(self, db) -> None:
        self.db_name = db
        self.transformations = transforms.Compose([
            transforms.Grayscale(1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ToTensor()
        ])
        
        if db == 'CK+':
            self.db_path = '/homes/vlapadula/fer_classification/net/CK+48'
            self.train_batch_size = 20
            self.test_batch_size = 20
            self.db = datasets.DatasetFolder(root=self.db_path, transform=self.transformations, loader=datasets.folder.default_loader, extensions='.png')

        if db == 'FER2013':
            self.db_path = '/homes/vlapadula/Dataset/FER2013/'
            self.train_batch_size = 200
            self.test_batch_size = 150
        
        if db == 'BigFER':
            self.db_path = '/homes/vlapadula/fer_classification/net/bigfer/dataset/images/'
            self.train_batch_size = 200
            self.test_batch_size = 150
        
        
    def load(self):
        if self.db_name == 'CK+':
            trainset_len = int(0.8*len(self.db))
            trainset, testset = torch.utils.data.random_split(self.db, lengths=[trainset_len, len(self.db)-trainset_len])

        else:
            trainset = datasets.DatasetFolder(root=self.db_path + 'train', transform=self.transformations, loader=datasets.folder.default_loader, extensions='.jpg')
            testset = datasets.DatasetFolder(root=self.db_path + 'validation' if self.db_name == 'BigFER' else self.db_path+'test', transform=self.transformations, loader=datasets.folder.default_loader, extensions='.jpg')

                
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.train_batch_size, shuffle=True, num_workers=1)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.test_batch_size, shuffle=False, num_workers=1)
        
        return trainloader, testloader