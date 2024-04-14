import math
from random import random
from copy import deepcopy
from matplotlib import pyplot as plt
from math import cos, sin,pi

#Connections have 1 variable changed by the swarm
#And 4 uint8s and a float-double for export
#Nodes have 1 variable changed by the swarm
#And 1 uint8 and a float-double for export

def randomiseposition(pos,LearningRate):
    newpos=[i+(random()-.5)*LearningRate for i in pos]
    return newpos

def randomvelocity(LearningRate,Axes):
    Rate=LearningRate*.2
    newpos=[(random()-.5)*Rate for i in range(Axes)]
    return newpos

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def identity(x):
    return x
tanh=math.tanh

def tanhnormal(x):
    return (math.tanh(x)+1)/2

def fastact(x):
    return max(0,min(x,1))*min(x,1)

def step(x):
    return 0 if x<.5 else 1

def RandomConnections(layershape,Weightinterval=(-1,1)):
    #[1,9,10,6,1] is an example shape, 1 input, 9 nodes in the second layer, 10 in the third, 6 in the third, 1 output
    connections=[]
    for i in range(len(layershape)-1):
        row=[]
        for a in range(layershape[i]):
            for b in range(layershape[i+1]):
                row.append(Connection((i,a),(i+1,b),Weightinterval[0]+random()*(Weightinterval[1]-Weightinterval[0])))
        connections.append(row)
    return connections

class node:
    def __init__(self,Function,Bias=0):
        self.Function=Function
        self.Input=0
        self.Output=0
        self.Bias=Bias

class Connection:
    def __init__(self,InputNode,OutputNode,Weight):
        self.InputCol=InputNode[0]
        self.InputRow=InputNode[1]
        self.OutputCol=OutputNode[0]
        self.OutputRow=OutputNode[1]
        self.Weight=Weight


def Accelerate(velocity,position,globalbest,particlebest,LearningRate):
    if particlebest==None:
        return velocity
    if globalbest==None:
        return velocity
    acceleration=LearningRate*1
    outvelocity=velocity
    for n in range(len(velocity)):
        outvelocity[n]+=(globalbest[1][n]-position[n])*acceleration
        ##Accelerate in the direction of the current best value,
        #the rate of acceleration is proportional to distance so particles
        ##further from the ideal will do wider sweeps to look for new local extrema.
        #particles close to the ideal will sweep at close range around that value to
        #find any slight tweaking that may improve it
        outvelocity[n]+=(particlebest[1][n]-position[n])*acceleration
        ##Accelerate in the direction of the best value this particle has seen
        #the rate of acceleration works the same as above, but it goes towards
        #a mixture of its personal best and global best prioritising the one
        #that's furthest away so that it can chance upon pockets or seams of
        #optima between the two known points.
    return outvelocity



class BrainLobe:
    def __init__(self,layers,biases="Random",connections="Random"):
        self.layers=layers
        if biases=="Random":
            self.RandomCellBiases()
        ##Randomise Biases if set to random, otherwise leave them as they are.
        self.layershape=[len(i) for i in layers]
        if connections=="Random":
            self.connections=RandomConnections(self.layershape)
        else:
            self.connections=connections
        self.nodecount=sum(self.layershape)
        self.connectioncount=sum([len(i) for i in self.connections])
        self.axes=self.nodecount+self.connectioncount
    def RandomCellBiases(self):
        for i in self.layers:
            for j in i:
                j.Bias=random()*4-2
    def Center(self):
        #Connections have 1 variable changed by the swarm
        #Nodes have 1 variable changed by the swarm
        out=[]
        for col in self.layers:
            for currentnode in col:
                out.append(currentnode.Bias)
        for col in self.connections:
            for conn in col:
                out.append(conn.Weight)
        return out
    def GetScore(self,TrainingData):
        Score=0
        for datum in TrainingData:
                In=datum[0]
                Output=self.Run(In)
                diff=0
                for i in range(len(Output)):
                    diff+=abs(Output[i]-datum[1][i])
                    #calculate the difference for this one piece of data
                Score+=diff
                ##add each difference to the end score
        return Score #return the score, lower is better (like golf)
        
    def Learn(self,TrainingData,Steps=40,particles=100,MaxEntries=400,LearningRate=1):
        RateDecay=.975
        
        #Make 10 Candidate Networks with associated velocities and extrema
        Particles=[[randomiseposition(self.Center(),LearningRate),randomvelocity(LearningRate,self.axes),None] for p in range(particles)]
        Candidates=[BrainLobe(deepcopy(self.layers),biases=None,connections=deepcopy(self.connections)) for p in range(particles)]
        print("Training for %04d steps"%Steps)
        length=len(TrainingData)
        print("Training Data contains %08d entries"%length)
        if length>MaxEntries:
            print("This is larger than recommended, creating a subset for training")
        
            Subset=[]
            n=0
            while n<MaxEntries:
                Subset.append(TrainingData[int(random()*length)])
                n+=1
            TrainingData=Subset
        Best=(self.GetScore(TrainingData),self.Center())
        for i in range(Steps):
            print("Step Number %03d"%i)
            #Move the particles
            for n in range(len(Particles)):
                Particles[n][1]=Accelerate(Particles[n][1],Particles[n][0],Best,Particles[n][2],LearningRate)
                for axis in range(self.axes):
                    Particles[n][0][axis]+=Particles[n][1][axis]
            ##UPDATE CANDIDATES
                axis=0
                for col in Candidates[n].layers:
                    for currentnode in col:
                        currentnode.Bias=Particles[n][0][axis]
                        axis+=1
                for col in Candidates[n].connections:
                    for conn in col:
                        conn.Weight=Particles[n][0][axis]
                        axis+=1
            ##TEST CANDIDATES
                Score=0
                for datum in TrainingData:
                    In=datum[0]
                    Output=Candidates[n].Run(In)
                    Diff=sum(map(lambda actual,expected:abs(actual-expected),Output,datum[1]))
                    Score+=Diff
                #check if score is a global best (Lower scores are better)
                if Score<Best[0]:
                    Best=(Score,deepcopy(Particles[n][0]))
                #check if score is a local best (Lower scores are better)
                PersonalBest=Particles[n][2]
                if PersonalBest==None or PersonalBest[0]>Score:
                    Particles[n][2]=(Score,deepcopy(Particles[n][0]))
            ###Update Bests
            LearningRate*=RateDecay

        ###Overwrite self with Best
        BestPos=Best[1]
        axis=0
        for col in self.layers:
            for currentnode in col:
                currentnode.Bias=BestPos[axis]
                axis+=1
        for col in self.connections:
            for conn in col:
                conn.Weight=BestPos[axis]
                axis+=1
        print("UPDATED")
        print("Best Score was %02d"%int(Best[0]))
                ####Delete All Particles Delete All Candidates
        del Particles
        del Candidates
        return LearningRate,Best[0]
                
    def Run(self,inputs):
        #Run First Layer
        firstlayer=self.layers[0]
        if len(firstlayer)!=len(inputs):
            print("ERROR")
            return None
        for node in range(len(firstlayer)):
            firstlayer[node].Input=inputs[node]
            firstlayer[node].Output=firstlayer[node].Function(firstlayer[node].Input+firstlayer[node].Bias)
        for layer in range(1,len(self.layers)):
            #Reset Values
            for node in self.layers[layer]:
                node.Input=node.Bias
            ###Feed Values Forward From Prior Layers Connections
            for i in self.connections[layer-1]:
                INpos=i.InputCol,i.InputRow
                OUTpos=i.OutputCol,i.OutputRow
                self.layers[OUTpos[0]][OUTpos[1]].Input+=self.layers[INpos[0]][INpos[1]].Output*i.Weight
            #### Calculate the outputs of all nodes in the layer
            for node in self.layers[layer]:
                node.Output=node.Function(node.Input)## set the nodes output to it's input (which already includes the bias from when it was reset
        """for conn in self.connections[-1]:
            INpos=i.InputCol,i.InputRow
            OUTpos=i.OutputCol,i.OutputRow
            self.layers[OUTpos[0]][OUTpos[1]].Input+=self.layers[INpos[0]][INpos[1]].Output*i.Weight"""
        outputs=[i.Output for i in self.layers[-1]]
        return outputs

tau=math.pi
pi=math.pi
        

b=BrainLobe([[node(identity),node(identity)],
             [node(sin),node(sin),node(fastact),],
             [node(abs),node(abs),node(identity)],
             [node(identity)]])


##Demo Train a Radius Output from Azimuth Inclination Input
train=[(lambda a,b:([a,b],[1+(abs(sin(3*a)*2)+abs(1+sin(2*b)*3))]))(random()*2*pi,random()*pi) for i in range(1000)]
LR=2
threshold=50
while 1: 
    LR,Score=b.Learn(train,LearningRate=LR)
    LR=LR*2
    print(LR)
    if Score<threshold:
        break
    if LR<0.01:
        LR=2



x=[];y=[];z=[]
for i in range(1000):
    az=random()*pi*2
    inc=random()*pi
    rad=b.Run([az,inc])
    x.append(cos(az)*sin(inc)*rad[0])
    y.append(sin(az)*sin(inc)*rad[0])
    z.append(cos(inc)*rad[0])
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)
plt.show()
