# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 21:17:06 2020

@author: PRATEEK BHANDARI
"""
import numpy as np               #to store training data that is to be loaded
import cv2                       #for image modification
import os                        #to acess path of images
from tqdm import tqdm            #to see bars/progress during loading
import torch                     #numpy to pytorch tensor,save model
import torch.nn as nn            #loss function,conv2d,linear
import torch.nn.functional as F  #maxpooling2d,softmax,relu
import torch.optim as optim      #optimizer function

img_size=75
#loading custom dataset
REBUILD_DATA = False
class Mathoperators():
    zero="extracted_images/0"
    one="extracted_images/1"
    two="extracted_images/2"
    three="extracted_images/3"
    four="extracted_images/4"
    five="extracted_images/5"
    six="extracted_images/6"
    seven="extracted_images/7"
    eight="extracted_images/8"
    nine="extracted_images/9"
    add="extracted_images/+"
    sub="extracted_images/-"
    mul="extracted_images/times"
    div="extracted_images/forward_slash"
    left_brac="extracted_images/("
    right_brac="extracted_images/)"
    img_size=75
    labels = {zero:0,one:1,two:2,three:3,four:4 ,five:5 ,six:6,seven:7,eight:8,nine:9,add:10,sub:11,mul:12,div:13,left_brac:14,right_brac:15}
    training_data =[]
    #to make modify taining data
    def make_training_data(self):
        for label in self.labels:
            for f,q in zip(tqdm(os.listdir(label)),range(5000)):
                if "jpg" in f:
                    try:
                        path=os.path.join(label,f)
                        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                        img=cv2.resize(img,(self.img_size,self.img_size))
                        self.training_data.append([np.array(img),np.eye(16)[self.labels[label]]])
                    except Exception as e:
                        print(e)          
        np.random.shuffle(self.training_data)
        np.save("training_data_final.npy",self.training_data)
        
if REBUILD_DATA :
    maths = Mathoperators()
    maths.make_training_data()

#Loading already made dataset
training_data=np.load("training_data_final.npy" ,allow_pickle=True)

#Nueral Network of the ML model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,5)
        self.conv2 = nn.Conv2d(32,64,5)        
        self.conv3 = nn.Conv2d(64,128,5)  
        x = torch.randn(img_size,img_size).view(-1,1,img_size,img_size)
        self._to_linear = None
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear,512)
        self.fc2 = nn.Linear(512,16)
    def convs(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    def forward(self,x):
        x = self.convs(x)
        x = x.view(-1,self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x,dim=1)
net = Net()

#inbuilt optimizer and loss calculator
optimizer = optim.Adam(net.parameters(),lr=0.001)
loss_function = nn.MSELoss()

#make train and test datasets
X = torch.Tensor([i[0] for i in training_data]).view(-1,img_size,img_size)
#X = X/255.0
y = torch.Tensor([i[1] for i in training_data])
VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)
train_X = X[:-val_size]
train_y = y[:-val_size]
test_X = X[-val_size:]
test_y = y[-val_size:]

#Training the model
batch_size=25
EPOCHS = 5
for epoch in range(EPOCHS):
    for i in tqdm(range(0,len(train_X),batch_size)):
        batch_X = train_X[i:i+batch_size].view(-1, 1, img_size, img_size)
        batch_y = train_y[i:i+batch_size]
        net.zero_grad()
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()    # Does the update
    print(f"Epoch: {epoch}. Loss: {loss} \n")

#saving the model for future use
torch.save(net.state_dict(),"getthenumber.pth")  
   
#To check the accuracy    
correct = 0
total = 0
with torch.no_grad(): 
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 1, img_size, img_size))[0]  # returns a list, 
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct += 1
        total += 1
print("Accuracy: ", round(correct/total, 3))   
