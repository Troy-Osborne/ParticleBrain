from ParticleBrain import Node, BrainLobe,ActivationFunctions
from ParticleBrain import ActivationFunctions as AF
from math import cos,sin,pi
from random import random
from matplotlib import pyplot as plt
tau=pi*2

##Demo Train a Radius Output from Azimuth Inclination Input        

##initialise object with layers of different nodes. No need to use traditional activation functions, insert functions you suspet might best model the training set
#here I've used some sine waves within the layers themself as well as an abs function.
#identity just returns the input, however there is a node bias and the connections are weighted so it can model all lines
b=BrainLobe([[Node(AF.identity),Node(AF.identity)],
             [Node(sin),Node(sin),Node(AF.identity),Node(sin),Node(AF.identity),Node(sin)],
             [Node(abs),Node(abs)],
             [Node(AF.identity)]])



##Create the training set
train=[(lambda a,b:([a,b],[1+(abs(sin(3*a)*2)+abs(1+sin(2*b)*3))]))(random()*tau,random()*pi) for i in range(1000)]
LR=5
threshold=100
while 1: 
    LR,Score=b.Learn(train,Steps=20,particles=1000,MaxEntries=200,LearningRate=LR,RateDecay=.92,DrawingMode=True,Drag=0.3)
    LR=LR*2
    print(LR)
    if Score<threshold:
        break
    if LR<0.1:
        LR=5


####Draw 25000 points from the trained network
x=[];y=[];z=[]
for i in range(25000):
    az=random()*tau
    inc=random()*pi
    rad=b.Run([az,inc])
    x.append(cos(az)*sin(inc)*rad[0])
    y.append(sin(az)*sin(inc)*rad[0])
    z.append(cos(inc)*rad[0])
ax = plt.axes(projection='3d')

ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)
plt.show()
