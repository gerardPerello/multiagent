# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import math

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        #print(scores)
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Volem que s'apropi a les fruites i s'allunyi dels fantasmes

        foodDistance = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
        if foodDistance:
            foodMinima = min(foodDistance)
        else:
            foodMinima = -1  # perque si la llista esta buida vol dir que hem hem d'anar cap aquesta direcció, i per tant necessitem un valor molt gran.
        ghostDistance = [util.manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        fantasmaMoltAprop = 0
        for i in ghostDistance:
            if i <= 1:
                fantasmaMoltAprop += 1
        distanciaFantasmes = sum(ghostDistance)
        if distanciaFantasmes == 0:
            distanciaFantasmes = -1  # perque aixo voldra dir que tenim els fantasmes al voltant, i per tant ens en volem allunyar si o si d'aquesta direcció
        #print(foodMinima, distanciaFantasmes, fantasmaMoltAprop)

        result = successorGameState.getScore() + 1 / float(foodMinima) - 1 / float(
            distanciaFantasmes) - fantasmaMoltAprop

        return result


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

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        result = float("-inf")
        action = 1
        for agentState in gameState.getLegalActions(0):
            valorminimax = self.miniMaxDecision(1, 0, gameState.generateSuccessor(0, agentState))
            if valorminimax > result:
                result = valorminimax
                action = agentState
        return action

    def miniMaxDecision(self, agente, profundidad, gameState):
        if gameState.isLose() or gameState.isWin() or profundidad == self.depth:
            return self.evaluationFunction(gameState)
        if agente == 0:  # En cas de que estiguem en el agent pacman, el maximitzem
            return self.maxValue(agente, profundidad, gameState)
        else:  # En cas de que estiguem en algun dels agents fantasmes, els minimitzem
            return self.minValue(agente, profundidad, gameState)

    def maxValue(self, agente, profundidad, gameState):
        v = float("-inf")
        for newState in gameState.getLegalActions(agente):
            v = max(v, self.miniMaxDecision(1, profundidad, gameState.generateSuccessor(agente, newState)))
        return v

    def minValue(self, agente, profundidad, gameState):
        nextAgente = agente + 1  # Proxim fantasma a veure
        if gameState.getNumAgents() == nextAgente:  # Ja haurem recorregut tots els agents, tornem a començar pel
            # pacman, comença un nou tir.
            nextAgente = 0
        if nextAgente == 0:  # Com que comença un nou tir, la profunditat augmenta.
            profundidad += 1;
        v = float("inf")
        for newState in gameState.getLegalActions(agente):
            v = min(v, self.miniMaxDecision(nextAgente, profundidad, gameState.generateSuccessor(agente, newState)))
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        result = float("-inf")
        action = 1
        alfa = float("-inf")
        beta = float("inf")
        for agentState in gameState.getLegalActions(0):
            valorminimax = self.alfaBeta(1, 0, alfa, beta, gameState.generateSuccessor(0, agentState))
            if valorminimax > result:
                result = valorminimax
                action = agentState
            if result > beta:
                return result
            alfa = max(alfa,result)
        return action
    def alfaBeta(self, agente, profundidad, alfa, beta, gameState):
        if gameState.isLose() or gameState.isWin() or profundidad == self.depth:
            return self.evaluationFunction(gameState)
        if agente == 0:  # En cas de que estiguem en el agent pacman, el maximitzem
            return self.maxValueAB(agente, profundidad, alfa, beta, gameState)
        else:  # En cas de que estiguem en algun dels agents fantasmes, els minimitzem
            return self.minValueAB(agente, profundidad, alfa, beta, gameState)

    def maxValueAB(self, agente, profundidad, alfa, beta, gameState):
        v = float("-inf")
        for newState in gameState.getLegalActions(agente):
            v = max(v, self.alfaBeta(1, profundidad, alfa, beta, gameState.generateSuccessor(agente, newState)))
            if v > beta:
                return v
            alfa = max(v,alfa)
        return v

    def minValueAB(self, agente, profundidad, alfa, beta, gameState):
        nextAgente = agente + 1  # Proxim fantasma a veure
        if gameState.getNumAgents() == nextAgente:  # Ja haurem recorregut tots els agents, tornem a començar pel
            # pacman, comença un nou tir.
            nextAgente = 0
        if nextAgente == 0:  # Com que comença un nou tir, la profunditat augmenta.
            profundidad += 1;
        v = float("inf")
        for newState in gameState.getLegalActions(agente):
            v = min(v, self.alfaBeta(nextAgente, profundidad, alfa, beta, gameState.generateSuccessor(agente, newState)))
            if v < alfa:
                return v
            beta = min(v, beta)
        return v

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
        result = float("-inf")
        action = 1
        for agentState in gameState.getLegalActions(0):
            valorminimax = self.expectiMaxDecision(1, 0, gameState.generateSuccessor(0, agentState))
            if valorminimax > result:
                result = valorminimax
                action = agentState
        return action

    def expectiMaxDecision(self, agente, profundidad, gameState):
        if gameState.isLose() or gameState.isWin() or profundidad == self.depth:
            return self.evaluationFunction(gameState)
        if agente == 0:  # En cas de que estiguem en el agent pacman, el maximitzem
            return self.maxValue(agente, profundidad, gameState)
        else:  # En cas de que estiguem en algun dels agents fantasmes, els minimitzem
            return self.expectValue(agente, profundidad, gameState)

    def maxValue(self, agente, profundidad, gameState):
        v = float("-inf")
        for newState in gameState.getLegalActions(agente):
            v = max(v, self.expectiMaxDecision(1, profundidad, gameState.generateSuccessor(agente, newState)))
        return v

    def expectValue(self, agente, profundidad, gameState):
        nextAgente = agente + 1  # Proxim fantasma a veure
        if gameState.getNumAgents() == nextAgente:  # Ja haurem recorregut tots els agents, tornem a començar pel
            # pacman, comença un nou tir.
            nextAgente = 0
        if nextAgente == 0:  # Com que comença un nou tir, la profunditat augmenta.
            profundidad += 1;
        v = float("inf")
        estados = []
        for newState in gameState.getLegalActions(agente):
            estados.append(self.expectiMaxDecision(nextAgente, profundidad, gameState.generateSuccessor(agente, newState)))
        v = sum(estados) / float(len(gameState.getLegalActions(agente)))
        return v


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    # Volem que s'apropi a les fruites i s'allunyi dels fantasmes en cas que aquests ens puguin matar, si no, hem d'intentar menjar-nos-els, pensant en seguir optant a la fruita.

    foodDistance = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
    if foodDistance:
        foodMinima = min(foodDistance)
    else:
        foodMinima = -1  # perque si la llista esta buida vol dir que hem hem d'anar cap aquesta direcció, i per tant necessitem un valor molt gran.

    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    ghostDistance = [util.manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]

    distanciaFantasmes = 0
    fantasmaMoltAprop = 0

    for i in range(len(ghostDistance)):
        if newScaredTimes[i] >= 2:
            distanciaFantasmes -= ghostDistance[i]
            if ghostDistance[i] <= 1:
                fantasmaMoltAprop -= 1
        else:
            distanciaFantasmes += ghostDistance[i]
            if ghostDistance[i] <= 1:
                fantasmaMoltAprop += 1

    if distanciaFantasmes == 0:
        distanciaFantasmes = -1  # perque aixo voldra dir que tenim els fantasmes al voltant, i per tant ens en volem allunyar si o si d'aquesta direcció

    capsulesDistances = [util.manhattanDistance(newPos, capsuleState) for capsuleState in newCapsules]

    if capsulesDistances:
        capsulaMinima = min(capsulesDistances)
        itemMinim = min(capsulaMinima, foodMinima)
    else:
        itemMinim = foodMinima

    result = currentGameState.getScore() + 1 / float(itemMinim) - 1 / float(distanciaFantasmes) - fantasmaMoltAprop


    return result


# Abbreviation
better = betterEvaluationFunction
