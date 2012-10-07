# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
from util import PriorityQueueWithFunction

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

from util import Queue
from util import Stack
import util
import game

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def CostFunction(state):
    return state[2]

def printDebug(*arg):
    debug = 0
    if(debug):
        print(arg)
    return
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


class StackP(Stack):
    """
    """
    def  __init__(self):
      Stack.__init__(self)        # super-class initializer
    def push(self, item, priority=0):
        Stack.push(self, item) #Ignoring Priority

class QueueP(Queue):
    """
    """
    def  __init__(self):
      Queue.__init__(self)        # super-class initializer
    def push(self, item, priority=0):
        Queue.push(self, item) #Ignoring Priority

class PriorityQueueWithFunctionP(PriorityQueueWithFunction):
    """
    """
    def  __init__(self, priorityFunction):
      PriorityQueueWithFunction.__init__(self,priorityFunction)        # super-class initializer
    def push(self, item, priority=0):
      print self,item
      PriorityQueueWithFunction.push(self, item) #Ignoring Priority

def genericSearch(problem,container,heuristic=nullHeuristic):
    myPositionCounter = util.Counter() 
    startState= problem.getStartState()
    container.push([[startState],[],0],0)
    while(not(container.isEmpty())):
        printDebug("start iteration",container)
        currStateArr,currDirectionArr,currCost = container.pop()
        printDebug("after pop" ,currStateArr," ", currDirectionArr," ",currCost)
        currState=currStateArr[len(currStateArr)-1]
        #First of all check if we reached the end state
        if(problem.isGoalState(currState)):
            printDebug("Found solution!!!",currState,currCost)
            return currDirectionArr
        #Checking if we visited this position
        if(myPositionCounter[currState]!=0): #First time we encounter this position
          continue
        myPositionCounter[currState]+=1
        printDebug("New State Encountered:" , currState)
        successors = problem.getSuccessors(currState)
        for successor in successors:
          state = successor[0]
          direction = successor[1]
          cost = successor[2]
          myCurrPosition = game.Configuration(state,direction).getPosition()
          tempArr = currStateArr[:]
          tempArr.append(state)
          printDebug("TempArr: ", tempArr) 
          tempDir = currDirectionArr[:]
          tempDir.append(direction)
          hCost = heuristic(myCurrPosition,problem)
          successorCost = currCost+cost
          totalCost = successorCost+hCost
          printDebug("hCost: " , hCost) 
          container.push([tempArr,tempDir,successorCost],totalCost)
    return []
    
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    myQueue = StackP()
    return genericSearch(problem, myQueue)

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    myQueue = QueueP()#PriorityQueueWithFunction(depthFirstSearchQueueFunction)
    return genericSearch(problem, myQueue)

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    myQueue = util.PriorityQueue()
    return genericSearch(problem, myQueue)

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    myQueue = util.PriorityQueue()
    return genericSearch(problem,myQueue,heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
