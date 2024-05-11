# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 12:22:46 2024

@author: aidan kim
"""

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import datasets, transforms

    
from torchvision import utils
import matplotlib.pyplot as plt

#function to get one hot encoding lables
def convert_to_one_hot(labels, num_classes):

    one_hot_labels = np.zeros((labels.size, num_classes))

    one_hot_labels[np.arange(labels.size), labels] = 1
    return one_hot_labels

#function to yeild random minibatch
def get_random_mini_batch(dataset, mini_batch_size):
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    
    for start_idx in range(0, dataset_size, mini_batch_size):
        end_idx = min(start_idx + mini_batch_size, dataset_size)
        mini_batch_indices = indices[start_idx:end_idx]
        
        mini_batch_data = []
        mini_batch_labels = []
        
        for idx in mini_batch_indices:
            data, label = dataset[idx]
            mini_batch_data.append(data)
            mini_batch_labels.append(label)
        
        mini_batch_data = np.array(mini_batch_data)
        mini_batch_labels = np.array(mini_batch_labels)
        
        yield torch.tensor(mini_batch_data), torch.tensor(mini_batch_labels)


class MyDataset(Dataset):
    def __init__(self, data, labels):

        self.data = data
        self.labels = labels

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        
        return self.data[idx], self.labels[idx]

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.conv1_bn = nn.BatchNorm2d(16)
        # Max Pooling Layer 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.conv2_bn = nn.BatchNorm2d(32)
        # Max Pooling Layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv3_bn = nn.BatchNorm2d(64)
        
        #Drop out
        self.dropout1 = nn.Dropout2d(p=0.05)#for Concolutional layer
        self.dropout2 = nn.Dropout(p=0.1)#for fully connected layer
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(64*3*3, 500)
        # Output Layer
        self.fc2 = nn.Linear(500, 10)
        

    def forward(self, x):
        # Apply Convolutional layer 1, ReLU activation, batchnorm and Max pooling
        x = self.pool1(F.relu(self.conv1_bn(self.conv1(x))))
        # Apply Convolutional layer 2, ReLU activation, batchnorm and Max pooling
        x = self.pool2(self.conv2_bn(F.relu(self.conv2(x))))
        # Apply drop out
        x = self.dropout1(x)
        # Apply Convolutional layer 3 and ReLU activation
        x = F.relu(self.conv3_bn(self.conv3(x)))

        # Flatten the tensor for the fully connected layer
        x = x = torch.flatten(x, 1)

        # Apply the first fully connected layer with ReLU activation
        x = self.dropout2(x)
        x = F.relu(self.fc1(x))
        # Apply drop out
        x = self.dropout2(x)
        # Apply the output layer
        x = self.fc2(x)
        return x

##function to plot kernel
def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))




# Create the model
def main():
    
    num_classes = 10
    
    # Load the training data and labels
    training_data = np.load('training_data.npy')
    training_label = np.load('training_label.npy')
    
    
    # Normalize the training data
    training_data_normalized = training_data.astype(np.float32) / 255.0
    training_data_normalized = training_data_normalized.transpose([0, 3, 1, 2])

    training_label_one_hot = convert_to_one_hot(training_label, num_classes)

    
    # Load the test data and labels
    
    testing_data   =   np.load('test_data.npy') 
    testing_data_normalized = testing_data.astype(np.float32) / 255.0
    testing_data_normalized = testing_data_normalized.transpose([0, 3, 1, 2])
    testing_data_normalized = torch.tensor(testing_data_normalized)
    
    
    testing_label    =    np.load('test_label.npy')
    testing_label_one_hot = convert_to_one_hot(testing_label, num_classes)
    testing_label_one_hot = torch.tensor(testing_label_one_hot)
    
    CIFAR_dataset = MyDataset(training_data_normalized, training_label_one_hot)
    
    #initialize model with parameters
    model = CustomCNN()
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 
    model.to(device)
    


    
    
    train_loss = []
    train_acc = []
    test_acc = []
    
    max_epoch = 100
    for epoch in range(max_epoch):
        print("epoch: ",epoch+1)
        train_loss_epoch = []
        train_correct = np.array([0,0,0,0,0,0,0,0,0,0])
        train_appearances = np.array([0,0,0,0,0,0,0,0,0,0])


        for train_features, train_labels in get_random_mini_batch( CIFAR_dataset, 200):

            train_features, train_labels = train_features.to(device), train_labels.to(device)
            #calculate loss 
            optimizer.zero_grad()
            pred = model(train_features)
            loss = criterion(pred, train_labels.float())
            train_loss_epoch.append(loss.cpu().detach().numpy())

            #apply softmax to get Y-hat
            pred = torch.softmax(pred, dim =1)
            indices = torch.argmax(pred, dim=1)
            one_hot_pred = F.one_hot(indices, num_classes=10)
            #calculate accuracy
            for k in range(10):

                correct = torch.sum(one_hot_pred[:, k] * train_labels[:, k])
                count = torch.sum(train_labels[:, k])

                

                train_correct[k] += correct.item()  
                train_appearances[k] += count.item()
            
            
            loss.backward()
            optimizer.step()
        train_loss.append(sum(train_loss_epoch)/250)   
        train_accuracy = [train_correct[i] / train_appearances[i] if train_appearances[i] != 0 else 0 for i in range(10)]
        avg_train_accuracy = sum(train_accuracy) / 10
        print("train accuracy: ",avg_train_accuracy)
        train_acc.append(avg_train_accuracy)
    
    
        #test data
        test_correct = np.array([0,0,0,0,0,0,0,0,0,0])
        test_appearances = np.array([0,0,0,0,0,0,0,0,0,0])
        testing_data_normalized = testing_data_normalized.to(device)
        testing_label_one_hot = testing_label_one_hot.to(device)
        pred = model(testing_data_normalized)
        
        pred = torch.softmax(pred, dim =1)
        indices = torch.argmax(pred, dim=1)
        one_hot_pred = F.one_hot(indices, num_classes=10)
        for k in range(10):
            
            correct = torch.sum(one_hot_pred[:, k] * testing_label_one_hot[:, k])
            count = torch.sum(testing_label_one_hot[:, k])
            
            
            
            test_correct[k] += correct.item() 
            test_appearances[k] += count.item()
        
        
        test_accuracy = [test_correct[i] / test_appearances[i] if test_appearances[i] != 0 else 0 for i in range(10)]
        avg_test_accuracy = sum(test_accuracy) / 10
        print("test accuracy: ", avg_test_accuracy)
        test_acc.append(avg_test_accuracy)
        
    
    #save model
    torch.save(model.state_dict(), 'CIFAR_classifier.pth')
    
    
    #visualize kernel of first convolution layer
    filter_conv1 = model.cpu().conv1.weight.data.clone()
    visTensor(filter_conv1 , ch=0, allkernels=False)

    plt.axis('off')
    plt.ioff()
    plt.show()
    
    # plot Training loss, accuracy and test accuracy over epoch
    ep = np.arange(0, 100)
    plt.plot(ep, train_loss)
    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.title('training loss')

    plt.show()
    
    plt.plot(ep, train_acc)
    plt.xlabel('epoch')

    plt.ylabel('accuracy')

    plt.title('training accuracy')

    plt.show()
    
    plt.plot(ep, test_acc)
    plt.xlabel('epoch')

    plt.ylabel('accuracy')

    plt.title('test accuracy')

    plt.show()
    
    
    
    
def test():
    print("------------test------------")
    model = CustomCNN()
    model.load_state_dict(torch.load('CIFAR_classifier.pth'))
    num_classes = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    testing_data   =   np.load('test_data.npy') 
    testing_data_normalized = testing_data.astype(np.float32) / 255.0
    testing_data_normalized = testing_data_normalized.transpose([0, 3, 1, 2])
    testing_data_normalized = torch.tensor(testing_data_normalized)
    
    
    testing_label    =    np.load('test_label.npy')
    testing_label_one_hot = convert_to_one_hot(testing_label, num_classes)
    testing_label_one_hot = torch.tensor(testing_label_one_hot)
    
    test_correct = np.array([0,0,0,0,0,0,0,0,0,0])
    test_appearances = np.array([0,0,0,0,0,0,0,0,0,0])
    testing_data_normalized = testing_data_normalized.to(device)
    testing_label_one_hot = testing_label_one_hot.to(device)
    pred = model(testing_data_normalized)
    
    pred = torch.softmax(pred, dim =1)
    indices = torch.argmax(pred, dim=1)
    one_hot_pred = F.one_hot(indices, num_classes=10)
    for k in range(10):

        correct = torch.sum(one_hot_pred[:, k] * testing_label_one_hot[:, k])
        count = torch.sum(testing_label_one_hot[:, k])
        
        
       
        test_correct[k] += correct.item()  
        test_appearances[k] += count.item()
    
    
    test_accuracy = [test_correct[i] / test_appearances[i] if test_appearances[i] != 0 else 0 for i in range(10)]
    avg_test_accuracy = sum(test_accuracy) / 10
    print(test_accuracy)
    print("average: ", avg_test_accuracy)
    
    y_pred = []
    y_true = []
    y_pred.extend(one_hot_pred.argmax(axis=1).data.cpu().numpy())
    y_true.extend(testing_label_one_hot.argmax(axis=1).data.cpu().numpy())
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                         columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    
if __name__ == '__main__':
    #main()
    test()