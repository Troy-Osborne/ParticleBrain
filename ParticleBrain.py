import math
from random import random
from copy import deepcopy

#Connections have 1 variable changed by the swarm
#And 4 uint8s and a float-double for export
#Nodes have 1 variable changed by the swarm
#And 1 uint8 and a float-double for export

##Custom class which we can create instances of and assign values to
class functionlist(object):
    def __init__(self):
        pass
ActivationFunctions=functionlist()


def DrawScores(Scores,globalbest,Resolution=(1020,1020)):
    from PIL import Image,ImageDraw
    n=0
    length=len(Scores)
    im=Image.new("RGB",Resolution,(100,100,100))
    dr=ImageDraw.Draw(im)
    for i in Scores:
        curr,best=i
        if (curr-best)>-10:
            dr.rectangle((n/length*Resolution[0],best,(n+1)/length*Resolution[0],curr+10),(255,0,0))
        n+=1
    ######DRAW LINE
    dr.rectangle((0,globalbest-4,Resolution[0],globalbest),(255,255,255))
    return im
    

def randomiseposition(pos,LearningRate):
    Jump=LearningRate*.4
    newpos=[i+(random()-.5)*Jump for i in pos]
    return newpos

def randomvelocity(LearningRate,Axes):
    Rate=LearningRate*.1
    newpos=[(random()-.5)*Rate for i in range(Axes)]
    return newpos

def sigmoid(x):
    if x<50:
        return 0
    if x>-20:
        return 1
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


#MAKE THE ACTIVATION FUNCTIONS ATTRIBUTES OF `ActivationFunctions OUR INSTANCE OF THE CUSTOM CLASS `functionlist
setattr(ActivationFunctions,'step',step)#assignment can be done like this
ActivationFunctions.fastact=fastact #
ActivationFunctions.identity=identity
ActivationFunctions.sigmoid=sigmoid
ActivationFunctions.identity=identity
ActivationFunctions.tanh=tanh
ActivationFunctions.tanhnormal=tanhnormal

#NOW THEY CAN SIMPLY BE IMPORTED AS:
##from ParticleBrain import Node,BrainLobe,ActivationFunctions

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

class Node:
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

AR=0.05


##Accelerate to the global best and personal best, with attraction proportional to distance
def Accelerate1(velocity,position,globalbest,particlebest,LearningRate):
    if particlebest==None:
        return velocity
    if globalbest==None:
        return velocity
    acceleration=LearningRate*AR
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



###Accelerate towards the global best only
def Accelerate2(velocity,position,globalbest,particlebest,LearningRate):
    if particlebest==None:
        return velocity
    if globalbest==None:
        return velocity
    acceleration=LearningRate*AR
    outvelocity=velocity
    for n in range(len(velocity)):
        outvelocity[n]+=(globalbest[1][n]-position[n])*acceleration
        ##Accelerate in the direction of the current best value,
        #the rate of acceleration is proportional to distance so particles
        ##further from the ideal will do wider sweeps to look for new local extrema.
        #particles close to the ideal will sweep at close range around that value to
        #find any slight tweaking that may improve it
    return outvelocity

###Accelerate to particle's best only, giving each particle it's own search domain with no cooperation.
def Accelerate3(velocity,position,globalbest,particlebest,LearningRate):
    if particlebest==None:
        return velocity
    if globalbest==None:
        return velocity
    acceleration=LearningRate*AR
    outvelocity=velocity
    for n in range(len(velocity)):
        outvelocity[n]+=(particlebest[1][n]-position[n])*acceleration
        ##Accelerate in the direction of the current best value,
        #the rate of acceleration is proportional to distance so particles
        ##further from the ideal will do wider sweeps to look for new local extrema.
        #particles close to the ideal will sweep at close range around that value to
        #find any slight tweaking that may improve it
    return outvelocity


###Accelerate to whichever is closer, particle best or global best
def Accelerate4(velocity,position,globalbest,particlebest,LearningRate):
    if particlebest==None:
        return velocity
    if globalbest==None:
        return velocity
    acceleration=LearningRate*AR
    outvelocity=velocity
    gdist=sum(map(lambda a,b:abs(a-b),position,globalbest[1])) ###sum the distance between position and global best on each axis (manhattan distance)
    pdist=sum(map(lambda a,b:abs(a-b),position,particlebest[1]))###sum the distance between position and particle's best on each axis (manhattan distance)
    for n in range(len(velocity)):
        if gdist<pdist:
            outvelocity[n]+=(globalbest[1][n]-position[n])*acceleration
        else:
            outvelocity[n]+=(particlebest[1][n]-position[n])*acceleration
        ##Accelerate in the direction of the current best value,
        #the rate of acceleration is proportional to distance so particles
        ##further from the ideal will do wider sweeps to look for new local extrema.
        #particles close to the ideal will sweep at close range around that value to
        #find any slight tweaking that may improve it
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
        self.globalbest=None
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
        
    def Learn(self,TrainingData,Steps=100,particles=200,MaxEntries=250,LearningRate=1,RateDecay=.97,DrawingMode=False,Drag=0.1,ParticleTypes=[0,1,2,4],MinVal=-3,MaxVal=3):
        #Make Candidate Networks with associated velocities and extrema
        RangeSize=MaxVal-MinVal
        ##I've recently changed it so that there are multiple particle types
        ####There are different particle types which will be randomly assigned at the beginning.
        Particles=[[randomiseposition(self.Center(),LearningRate),randomvelocity(LearningRate,self.axes),None,ParticleTypes[int(random()*len(ParticleTypes))]] for p in range(particles)]
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
        if self.globalbest==None:
            self.globalbest=(self.GetScore(TrainingData),self.Center())
        for i in range(Steps):
            print("Step Number %03d"%i)
            #Move the particles
            ########TEST TO GRAPH THE CANDIDATE SCORES
            #
            if DrawingMode:
                StepScores=[0 for part in range(len(Particles))]
            for n in range(len(Particles)):
                #####UPDATE ACCELERTION
                pMode=Particles[n][3]  #current particle's mode
                if pMode==0:
                    if random()<1/4: ###1 in 4 chance of setting random velocity
                        Particles[n][1]=[(MinVal+random()*RangeSize)*.3 for i in range(self.axes)]
                    #randomise velocity every so often
                else:
                    Particles[n][1]=[Accelerate1,Accelerate2,Accelerate3,Accelerate4][pMode-1](Particles[n][1],Particles[n][0],self.globalbest,Particles[n][2],LearningRate)
                #Particles[n][1]=Accelerate4(Particles[n][1],Particles[n][0],self.globalbest,Particles[n][2],LearningRate)
                ###UPDATE POSITION AND APPLY DRAG
                if pMode==0:
                    if random()<1/4: ###1 in 10 chance of setting random new position
                        Particles[n][0]=[(MinVal+random()*RangeSize) for i in range(self.axes)]
                for axis in range(self.axes):
                    Particles[n][1][axis]*=1-Drag
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
                Score=Candidates[n].GetScore(TrainingData)
                #check if score is a global best (Lower scores are better)
                if Score<self.globalbest[0]:
                    self.globalbest=(Score,deepcopy(Particles[n][0]))
                if DrawingMode:
                    StepScores[n]=Score
                #check if score is a local best (Lower scores are better)
                PersonalBest=Particles[n][2]
                if PersonalBest==None or PersonalBest[0]>Score:
                    Particles[n][2]=(Score,deepcopy(Particles[n][0]))
            ###Update Bests
            LearningRate*=RateDecay
            if DrawingMode:
                graph=[]
                for n in range(len(StepScores)):
                    L=StepScores[n]
                    H=Particles[n][2][0]
                    graph.append((L,H))
                DrawScores(graph, self.globalbest[0]).save("%04d.png"%i)
        ###Overwrite self with Best
        BestPos=self.globalbest[1]
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
        print("Best Score was %02d"%int(self.globalbest[0]))
                ####Delete All Particles Delete All Candidates
        del Particles
        del Candidates
        return LearningRate,self.globalbest[0]
                
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
        outputs=[i.Output for i in self.layers[-1]]
        return outputs


