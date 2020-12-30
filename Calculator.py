# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 21:17:06 2020

@author: PRATEEK BHANDARI
"""
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

'''f = open("path.txt","r")
if f.mode == "r":
    contents= f.read()
f.close()'''    
    
Path = input("Enter path of ur image(add relative path) : ") 
#Load image and get contours
img_size=75
#Path="samples/two1.jpg";
im=cv2.imread(Path,cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(im, (5, 5), 0)
edged = cv2.Canny(blur, 0, 150)
contours, order = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#For sortin contours left to right
def greater(a,b):
    momA = cv2.moments(a)       
    xa= int(momA['m10']/momA['m00'])
    momB = cv2.moments(b)        
    xb = int(momB['m10']/momB['m00'])
    if xa>xb:
        return 1  
    else:
        return 0
    
#Used bubble sort to soort contours in order
for i in range(len(contours)):
    for j in range(len(contours)-1-i):   
        if greater(contours[j],contours[j+1]) :
            contours[j],contours[j+1]=contours[j+1],contours[j]
            
#to save expression            
expression = []

#ML model we trained
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
        return F.softmax(x, dim=1)
model = Net()

#load the model here
model.load_state_dict(torch.load("getthenumber.pth"))
model.eval()

#Get contour 
for contour in contours:
    #draw a rectangle around those contours on main image
    [x,y,w,h] = cv2.boundingRect(contour)
    #cv2.rectangle(im, (x-5,y-5), (x+w+5,y+h+5), (0, 255, 0), 1)
    #crop the required parts
    if h<50 :
        cv2.rectangle(im, (x-5,y-50), (x+w+5,y+h+50), (0, 255, 0), 1)
        im1=im[y-50:y+h+50 , x-5:x+w+50]
    elif w<50 :
        cv2.rectangle(im, (x-50,y-5), (x+w+50,y+h+5), (0, 255, 0), 1)
        im1=im[y-5:y+h+5 , x-50:x+w+50]
    else :
        cv2.rectangle(im, (x-5,y-5), (x+w+5,y+h+5), (0, 255, 0), 1)
        im1=im[y-5:y+h+5 , x-5:x+w+5]
    im1=cv2.resize(im1,(img_size,img_size))
    #get the prediction
    with torch.no_grad():
        data = torch.Tensor(im1)
        net_out = model(data.view(-1, 1, img_size, img_size))[0]    # returns a list,
        predicted_class = torch.argmax(net_out)
        expression.append(predicted_class.item())    

#Gives the expression
print(expression)

cv2.imshow('Final Image with Contours', im)
cv2.waitKey()
cv2.imwrite('samples/final.jpg', im)

#Forms the equation
fin_exp = str()
for i in expression:
    if i in range(0,10):
        fin_exp = fin_exp+str(i)
    else:
        if i == 10:
            fin_exp = fin_exp +str('+')
        elif i == 11:
            fin_exp= fin_exp + str('-')
        elif i == 12:
            fin_exp= fin_exp + str('*')
        elif i == 13:
            fin_exp= fin_exp + str('/')
        elif i == 14:
            fin_exp= fin_exp + str('(')
        else :
            fin_exp= fin_exp + str(')')

#Solve the equation

print(eval(fin_exp))
ans=eval(fin_exp)  
 
'''p = open("path.txt","w")
p.write(str(ans))
p.close()'''

    
    
    
    
    

