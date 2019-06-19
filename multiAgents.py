# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = list(successorGameState.getPacmanPosition())
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"


        foodDist = []
        foodLoc = currentGameState.getFood().asList()

        if action == 'Stop':
            return -float("inf")

        for ghostState in newGhostStates:
            if ghostState.getPosition() == tuple(newPos) and ghostState.scaredTimer is 0:
                return -float("inf")

        for food in foodLoc:
            manDist = -manhattanDistance(food, newPos)
            foodDist.append(manDist)

        return max(foodDist)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction, which is always legal

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        depth = 0
        agentIndex = 0
        av = self.minimax(gameState, agentIndex, depth)
        # print (av)
        return av[0]

    def minimax(self, state, agentIndex, depth):

        if agentIndex >= state.getNumAgents():
            agentIndex = 0
            depth += 1

        if depth == self.depth:
            return self.evaluationFunction(state)

        if agentIndex == 0:
            return self.maxVal(state, agentIndex, depth)
        else:
            return self.minVal(state, agentIndex, depth)



    def maxVal(self, state, agentIndex, depth):

        bestYet = ('action', -float('inf'))

        if not state.getLegalActions(agentIndex):
            return self.evaluationFunction(state)

        for a in state.getLegalActions(agentIndex):


            # print ('a: ', a, 'agentIndex: ', agentIndex, 'depth: ', depth)

            if a == 'Stop':
                continue

            newState = state.generateSuccessor(agentIndex, a)
            val = self.minimax(newState, agentIndex + 1, depth)
            if type(val) is not int:
                val = val[1]
            newVal = max(bestYet[1], val)
            if newVal is not bestYet[1]:
                bestYet = (a, newVal)

        # print('maxval: ', bestYet)
        return bestYet

    def minVal(self, state, agentIndex, depth):

        bestYet = ('action', float('inf'))

        if not state.getLegalActions(agentIndex):
            return self.evaluationFunction(state)

        for a in state.getLegalActions(agentIndex):

            # print ('a: ', a, 'agentIndex: ', agentIndex, 'depth: ', depth)

            if a == 'Stop':
                continue

            newState = state.generateSuccessor(agentIndex, a)
            val = self.minimax(newState, agentIndex + 1, depth)

            if type(val) is not int:
                val = val[1]
            newVal = min(bestYet[1], val)
            if newVal is not bestYet[1]:
                bestYet = (a, newVal)

        # print ('minVal: ', bestYet)
        return bestYet

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        depth = 0
        agentIndex = 0
        alpha = -float('inf')
        beta = float('inf')
        av = self.minimaxab(gameState, agentIndex, depth, alpha, beta)
        # print (av)
        return av[0]


    def minimaxab(self, state, agentIndex, depth, alpha, beta):
        if agentIndex >= state.getNumAgents():
            agentIndex = 0
            depth += 1

        if depth == self.depth:
            return self.evaluationFunction(state)

        if agentIndex == 0:
            return self.maxValab(state, agentIndex, depth, alpha, beta)
        else:
            return self.minValab(state, agentIndex, depth, alpha, beta)


    def maxValab(self, state, agentIndex, depth, alpha, beta):
        bestYet = ('action', -float('inf'))

        if not state.getLegalActions(agentIndex):
            return self.evaluationFunction(state)

        for a in state.getLegalActions(agentIndex):

            # print ('a: ', a, 'agentIndex: ', agentIndex, 'depth: ', depth)

            if a == 'Stop':
                continue

            newState = state.generateSuccessor(agentIndex, a)
            val = self.minimaxab(newState, agentIndex + 1, depth, alpha, beta)
            if type(val) is not int:
                val = val[1]
            newVal = max(bestYet[1], val)
            if newVal is not bestYet[1]:
                bestYet = (a, newVal)

            if beta <= bestYet[1]:
                return bestYet
            alpha = max(alpha, bestYet[1])


        # print('maxval: ', bestYet)
        return bestYet


    def minValab(self, state, agentIndex, depth, alpha, beta):
        bestYet = ('action', float('inf'))

        if not state.getLegalActions(agentIndex):
            return self.evaluationFunction(state)

        for a in state.getLegalActions(agentIndex):

            # print ('a: ', a, 'agentIndex: ', agentIndex, 'depth: ', depth)

            if a == 'Stop':
                continue

            newState = state.generateSuccessor(agentIndex, a)
            val = self.minimaxab(newState, agentIndex + 1, depth, alpha, beta)

            if type(val) is not int:
                val = val[1]
            newVal = min(bestYet[1], val)
            if newVal is not bestYet[1]:
                bestYet = (a, newVal)

            if alpha >= bestYet[1]:
                return bestYet
            beta = min(beta, bestYet[1])


        return bestYet


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        depth = 0
        agentIndex = 0
        av = self.expectimax(gameState, agentIndex, depth)
        # print (av)
        return av[0]


    def expectimax(self, state, agentIndex, depth):
        if agentIndex >= state.getNumAgents():
            agentIndex = 0
            depth += 1

        if depth == self.depth:
            return self.evaluationFunction(state)

        if agentIndex == 0:
            return self.maxVal(state, agentIndex, depth)
        else:
            return self.ghostVal(state, agentIndex, depth)


    def maxVal(self, state, agentIndex, depth):
        bestYet = ('action', -float('inf'))

        if not state.getLegalActions(agentIndex):
            return self.evaluationFunction(state)

        for a in state.getLegalActions(agentIndex):

            # print ('a: ', a, 'agentIndex: ', agentIndex, 'depth: ', depth)

            if a == 'Stop':
                continue

            newState = state.generateSuccessor(agentIndex, a)
            val = self.expectimax(newState, agentIndex + 1, depth)
            if type(val) is tuple:
                val = val[1]
            newVal = max(bestYet[1], val)
            if newVal is not bestYet[1]:
                bestYet = (a, newVal)

        # print('maxval: ', bestYet)
        return bestYet


    def ghostVal(self, state, agentIndex, depth):
        v = 0

        if not state.getLegalActions(agentIndex):
            return self.evaluationFunction(state)

        for a in state.getLegalActions(agentIndex):

            # print ('a: ', a, 'agentIndex: ', agentIndex, 'depth: ', depth)
            numActions = float(len(state.getLegalActions(agentIndex)))
            weight = 1/numActions

            if a == 'Stop':
                continue

            newState = state.generateSuccessor(agentIndex, a)
            val = self.expectimax(newState, agentIndex + 1, depth)

            if type(val) is tuple:
                val = val[1]

            v = v + weight*val


        # print ('minVal: ', bestYet)
        return v


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"


    # Initialize
    foodDist = []
    foodList = currentGameState.getFood().asList()

    ghostDist = []
    ghostStates = currentGameState.getGhostStates()
    scaredGhosts = 0

    position = currentGameState.getPacmanPosition()

    # Calculate a ghost score
    for ghostState in ghostStates:
        if ghostState.scaredTimer is not 0:
            scaredGhosts += 1
            ghostDist.append(0)
            continue

        ghost = ghostState.getPosition()
        gdist = manhattanDistance(ghost, position)
        if gdist == 0:
            ghostDist.append(0)
        else:
            ghostDist.append(-1.0 / gdist)
    ghostScore = min(ghostDist) + 100 * scaredGhosts

    # Calculate a food score
    for food in foodList:
        fdist = manhattanDistance(food, position)
        foodDist.append(-fdist)
    if not foodDist:
        foodDist.append(0)
    foodScore = max(foodDist)

    # game score
    gameScore = currentGameState.getScore()


    # weighted addition
    # print('ghostScore: ', ghostScore, 'foodScore: ', foodScore, 'gameScore: ', gameScore)
    totalScore = ghostScore + foodScore + gameScore

    return totalScore


# Abbreviation
better = betterEvaluationFunction


class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """

