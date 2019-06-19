# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

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

class nodeClass:

    def __init__(self, coord):
        self.path = []
        self.coord = coord
        self.cost = 0

    def inheritPath(self, path):
        self.path = path[:]

    def appendMove(self, move):
        self.path.append(move)

    def inheritCost(self, cost):
        self.cost = cost

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

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

    # create the fringe
    fringe = util.Stack()
    startState = problem.getStartState()
    fringe.push(nodeClass(startState))

    finished = False
    explored = set()

    while finished == False:

        # Loop through following:
        # 1. check that there are items left on the fringe,
        # 2. remove leaf node from fringe according to [strategy] (i.e. LIFO for dfs),
        # 3. if the node contains a goal state, return solution, otherwise
        # 4. add the node to the explored set
        # 5. expand chosen node and add the resulting nodes to the fringe

        if fringe.isEmpty():
            return []

        currentNode = fringe.pop()

        if problem.isGoalState(currentNode.coord):
            return currentNode.path
            finished = True

        if currentNode.coord not in explored:
            explored.add(currentNode.coord)
            newNodes = getChildren(currentNode, problem)
            addExpanded(fringe, newNodes)


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    # create the fringe
    fringe = util.Queue()
    startState = problem.getStartState()
    fringe.push(nodeClass(startState))

    finished = False
    explored = set()

    while finished == False:

        # Loop through following:
        # 1. check that there are items left on the fringe,
        # 2. remove leaf node from fringe according to [strategy] (i.e. FIFO for bfs),
        # 3. if the node contains a goal state, return solution, otherwise
        # 4. add the node to the explored set
        # 5. expand chosen node and add the resulting nodes to the fringe

        if fringe.isEmpty():
            return []

        currentNode = fringe.pop()

        if problem.isGoalState(currentNode.coord):
            return currentNode.path
            finished = True

        if currentNode.coord not in explored:
            explored.add(currentNode.coord)
            newNodes = getChildren(currentNode, problem)
            addExpanded(fringe, newNodes)


def uniformCostSearch(problem):
    "Search the node of least total cost first. "

    # create the fringe
    fringe = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = nodeClass(startState)
    fringe.push(startNode, startNode.cost)

    finished = False
    explored = set()

    while finished == False:

        # Loop through following:
        # 1. check that there are items left on the fringe,
        # 2. remove leaf node from fringe according to [strategy] (i.e. lowest cum cost for ucs),
        # 3. if the node contains a goal state, return solution, otherwise
        # 4. add the node to the explored set
        # 5. expand chosen node and add the resulting nodes to the fringe

        if fringe.isEmpty():
            return []

        currentNode = fringe.pop()

        if problem.isGoalState(currentNode.coord):
            return currentNode.path
            finished = True

        if currentNode.coord not in explored:
            explored.add(currentNode.coord)
            newNodes = getChildren(currentNode, problem)
            addExpandedCost(fringe, newNodes)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    fringe = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = nodeClass(startState)
    combinedCost = startNode.cost + heuristic(startNode.coord, problem)
    fringe.push(startNode, combinedCost)

    finished = False
    explored = set()

    while finished == False:

        # Loop through following:
        # 1. check that there are items left on the fringe,
        # 2. remove leaf node from fringe according to [strategy] (i.e. lowest cum cost for ucs),
        # 3. if the node contains a goal state, return solution, otherwise
        # 4. add the node to the explored set
        # 5. expand chosen node and add the resulting nodes to the fringe

        if fringe.isEmpty():
            return []

        currentNode = fringe.pop()

        if problem.isGoalState(currentNode.coord):
            return currentNode.path
            finished = True

        if currentNode.coord not in explored:
            explored.add(currentNode.coord)
            newNodes = getChildren(currentNode, problem)
            addExpandedCostHeuristic(fringe, newNodes, heuristic, problem)

def getChildren(node, problem):
    parentNode = node
    childList = []
    successors = problem.getSuccessors(node.coord)
    # print successors
    for i in range(len(successors)):
        successor = successors[i]
        child = nodeClass(successor[0])
        child.inheritPath(parentNode.path)
        child.inheritCost(parentNode.cost + successor[2])
        child.appendMove(successor[1])
        childList.append(child)

    return childList

def addExpanded(stack, newNodes):
    for node in newNodes:
        stack.push(node)

def addExpandedCost(queue, newNodes):
    for node in newNodes:
        queue.push(node, node.cost)

def addExpandedCostHeuristic (queue, newNodes, heuristic, problem):
    for node in newNodes:
        queue.push(node, node.cost + heuristic(node.coord, problem))

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch