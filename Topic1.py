from math import sqrt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import random
import math
import collections
import queue
# region SearchAlgorithms

class Node:
    id = None
    up = None
    down = None
    left = None
    right = None
    previousNode = None


    def __init__(self, value):
        self.value = value


class SearchAlgorithms:
    ''' * DON'T change Class, Function or Parameters Names and Order
        * You can add ANY extra functions,
          classes you need as long as the main
          structure is left as is '''
    path = []  # Represents the correct path from start node to the goal node.
    fullPath = []  # Represents all visited nodes from the start node to the goal node.
    array2D=[]
    row=[]
    startNode=None
    endNode=None
    lencols=0
    lenrows=0
    def __init__(self, mazeStr):
        ''' mazeStr contains the full board
         The board is read row wise,
         the nodes are numbered 0-based starting
         the leftmost node'''
        global charsOfRow
        id=0
        listOfRows = mazeStr.split(" ")
        for i in listOfRows :
            charsOfRow=i.split(",")
            self.row=[]
            for j in charsOfRow:
                element=Node(j)
                element.id=id
                if element.value=="S":
                    self.startNode=element
                elif element.value=="E":
                    self.endNode=element
                self.row.append(element)
                id+=1
            self.array2D.append(self.row)
        self.lenrows =len(listOfRows)
        self.lencols = len(charsOfRow)

        for i in range(self.lenrows):
            for j in range(self.lencols):
                if i!=0:
                    if self.array2D[i-1][j].value!="#":
                        self.array2D[i][j].up=self.array2D[i-1][j]
                if j!=0:
                    if self.array2D[i][j-1].value != "#":
                        self.array2D[i][j].left = self.array2D[i][j-1]
                if i!=(self.lenrows-1):
                    if self.array2D[i+1][j].value != "#":
                        self.array2D[i][j].down=self.array2D[i+1][j]
                if j!=(self.lencols-1):
                    if self.array2D[i][j+1].value != "#":
                        self.array2D[i][j].right = self.array2D[i][j+1]

    def BFS(self):
        '''Implement Here'''

        notvisit = collections.deque([self.startNode])
        visit=[]
        i = 0
        while len(notvisit) > 0:
            currentNode = notvisit.pop()
            visit.append(currentNode.id)
            if currentNode.id == self.endNode.id:
                self.fullPath = visit
                pre=currentNode
                while pre.id!=self.startNode.id:
                 self.path.append(pre.id)
                 pre=pre.previousNode
                self.path.append(self.startNode.id)
                self.path.reverse()
                return self.fullPath, self.path

            if currentNode.up != None:
                if currentNode.up.id not in visit and currentNode.up not in notvisit:
                    notvisit.appendleft(currentNode.up)
                    x = int(currentNode.up.id / 7)
                    y = currentNode.up.id % 7
                    self.array2D[x][y].previousNode = currentNode
            if currentNode.down != None:
                if currentNode.down.id not in visit and currentNode.down not in notvisit:
                    notvisit.appendleft(currentNode.down)
                    x = int(currentNode.down.id / 7)
                    y = currentNode.down.id % 7
                    self.array2D[x][y].previousNode = currentNode
            if currentNode.left != None:
                if currentNode.left.id not in visit and currentNode.left not in notvisit:
                    notvisit.appendleft(currentNode.left)
                    x = int(currentNode.left.id / 7)
                    y = currentNode.left.id % 7
                    self.array2D[x][y].previousNode = currentNode
            if currentNode.right != None:
                if currentNode.right.id not in visit and currentNode.right not in notvisit:
                    notvisit.appendleft(currentNode.right)
                    x = int(currentNode.right.id / 7)
                    y = currentNode.right.id % 7
                    self.array2D[x][y].previousNode = currentNode

    #return self.fullPath, self.path


# endregion

# region NeuralNetwork
class NeuralNetwork():

    def __init__(self, learning_rate, threshold):
        self.learning_rate = learning_rate
        self.threshold = threshold
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((2, 1)) - 1

    def step(self, x):
        if x>float(self.threshold):
            return 1
        else:
            return 0


    def train(self, training_inputs, training_outputs, training_iterations):
        for i in range (training_iterations):
         output = self.think(training_inputs)
        error=training_outputs-output
        updateweights=np.dot(training_inputs.T,error*self.learning_rate)
        self.synaptic_weights+=updateweights
    def think(self, inputs):
        inputs= inputs.astype(float)
        outputAct=np.sum(np.dot(inputs,self.synaptic_weights))
        output=self.step(outputAct)
        return output


# endregion

#################################### Algorithms Main Functions #####################################
# region Search_Algorithms_Main_Fn

def SearchAlgorithm_Main():
    searchAlgo = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.')
    fullPath, path = searchAlgo.BFS()
    print('**BFS**\n Full Path is: ' + str(fullPath) + "\n Path: " + str(path)+'\n')


# endregion

# region Neural_Network_Main_Fn
def NN_Main():
    learning_rate = 0.1
    threshold = -0.2
    neural_network = NeuralNetwork(learning_rate, threshold)
    print('**Neural_Network**')
    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0, 0],
                                [0, 1],
                                [1, 0],
                                [1, 1]])

    training_outputs = np.array([[0, 0, 0, 1]]).T

    neural_network.train(training_inputs, training_outputs, 100)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    inputTestCase = [1, 1]

    print("Considering New Situation: ", inputTestCase[0], inputTestCase[1], end=" ")
    print("New Output data: ", end=" ")
    print(neural_network.think(np.array(inputTestCase)))
    print("Wow, we did it!")


# endregion


######################## MAIN ###########################33
if __name__ == '__main__':

    SearchAlgorithm_Main()
    NN_Main()

