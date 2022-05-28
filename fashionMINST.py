import time
import torchvision
import torch
from torchvision import datasets,transforms
import torch.nn.functional as F
from torch import nn, optim
import os
import pandas as pd


root="./Fashion-MNIST/data"

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    torchvision.transforms.ToTensor(),
    ])
test_transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.FashionMNIST(root=root, train=True, download=True, transform=train_transform)
mnist_test = datasets.FashionMNIST(root=root, train=False, download=True, transform=test_transform)

train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True, num_workers=2)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

from torch import nn
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,64,3)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64,128,3)
        self.conv4 = nn.Conv2d(128,128,3,padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)

        self.fc5 = nn.Linear(128*6*6,512)
        self.drop1 = nn.Dropout()
        self.fc7 = nn.Linear(512,10)

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)


        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(-1,128*6*6)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.fc7(x)
       
        return x

def train(net, device, epochs,learning_rate,weight_decay):
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    initepoch = 0
    loss = nn.CrossEntropyLoss()
    best_test_acc = 0
    for epoch in range(initepoch, epochs):  # loop over the dataset multiple times
        timestart = time.time()

        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(train_iter, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
            running_loss += l.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('epoch %d, loss: %.4f,tran Acc: %.3f%%,time:%3f sec'
            % (epoch+1, running_loss / 500, 100.0 * correct / total, time.time() - timestart))

		#test
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_iter:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_acc = 100.0 * correct / total
            print('test Acc: %.3f%%' % (test_acc))
            if test_acc > best_test_acc:
                print('find best! save at checkpoint/cnn_best.pth')
                best_test_acc = test_acc
                torch.save(net.state_dict(), './checkpoint/cnn_best.pth')


    print('Finished Training')

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('train on', device)
    net = Net()
    net = net.to(device)
    epochs, lr, weight_decay = 60, 0.01, 10e-4
    train(net,device,epochs, lr, weight_decay)
    id = 0
    preds_list = []
    net.load_state_dict(torch.load('./checkpoint/cnn_best.pth'))
    with torch.no_grad():
        for X, y in test_iter:
            batch_pred = list(net(X.to(device)).argmax(dim=1).cpu().numpy())
            for y_pred in batch_pred:
                preds_list.append((id, y_pred))
                id += 1

    print('生成提交结果文件')
    with open('cnn_submission.csv', 'w') as f:
        f.write('ID,Prediction\n')
        for id, pred in preds_list:
            f.write('{},{}\n'.format(id, pred))


#https://www.freesion.com/article/1549498538/


