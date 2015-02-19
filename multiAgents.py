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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if action == 'Stop':
            return -float("inf")

        for ghost in newGhostStates:
            if ghost.getPosition() == newPos and not ghostState.scaredTimer:
                return -float("inf")

        minDisplacement = float("inf")
        for foodPos in currentGameState.getFood().asList():
            x_disp = abs(foodPos[0] - newPos[0])
            y_disp = abs(foodPos[1] - newPos[1])
            displacement = x_disp+y_disp
            if minDisplacement > displacement:
                minDisplacement = displacement

        return -minDisplacement

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        best_score = float('-inf')
        best_action = Directions.STOP
        available_actions = gameState.getLegalActions(0)
        for current_action in available_actions:
            child = gameState.generateSuccessor(0, current_action)
            current_score = self.minimizer(0, 1, child)
            if current_score > best_score and current_action != Directions.STOP:
                best_score = current_score
                best_action = current_action
        return best_action

    def maximizer(self, depth, agent_index, game_state):
        if depth == self.depth:
            return self.evaluationFunction(game_state)
        else:
            available_actions = game_state.getLegalActions(agent_index)
            if available_actions:
                best_score = float('-inf')
            else:
                best_score = self.evaluationFunction(game_state)
            for next_action in available_actions:
                child = game_state.generateSuccessor(agent_index, next_action)
                current_score = self.minimizer(depth, agent_index+1, child)
                if current_score > best_score:
                    best_score = current_score
            return best_score

    def minimizer(self, depth, agent_index, game_state):
        if depth == self.depth:
            return self.evaluationFunction(game_state)
        else:
            available_actions = game_state.getLegalActions(agent_index)
            if available_actions:
                best_score = float('inf')
            else:
                best_score = self.evaluationFunction(game_state)
            for action in available_actions:
                if agent_index == game_state.getNumAgents() - 1:
                    child = game_state.generateSuccessor(agent_index, action)
                    current_score = self.maximizer(depth+1, 0, child)
                    if current_score < best_score:
                        best_score = current_score
                else:
                    child = game_state.generateSuccessor(agent_index, action)
                    current_score = self.minimizer(depth, agent_index+1, child)
                    if current_score < best_score:
                        best_score = current_score
            return best_score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        best_score = float('-inf')
        best_action = Directions.STOP
        available_actions = gameState.getLegalActions(0)
        a = float('-inf')
        b = float('inf')
        for current_action in available_actions:
            if current_action != Directions.STOP:
                child = gameState.generateSuccessor(0, current_action)
                current_score = self.minimizer(0, 1, child, a, b)
                if current_score > best_score and current_action != Directions.STOP:
                    best_score = current_score
                    best_action = current_action
                if best_score > b:
                    return best_score
                a = max(a, best_score)
        return best_action

    def maximizer(self, depth, agent_index, game_state, a, b):
        #print(str(a) + " " + str(b) + "@" + str(depth) + "for maximizer")
        if depth == self.depth:
            return self.evaluationFunction(game_state)
        else:
            available_actions = game_state.getLegalActions(agent_index)

            best_score = float('-inf')
            if available_actions:
                best_score = float('-inf')
            else:
                best_score = self.evaluationFunction(game_state)
            for next_action in available_actions:
                if next_action != Directions.STOP:
                    child = game_state.generateSuccessor(agent_index, next_action)
                    best_score = max(best_score, self.minimizer(depth, agent_index+1, child, a, b))
                    if best_score > b:
                        return best_score
                    #print("Best score: " + str(best_score))
                    a = max(a, best_score)
            return best_score

    def minimizer(self, depth, agent_index, game_state, a, b):
        #print(str(a) + " " + str(b) + "@" + str(depth) + "for minimizer")
        if depth == self.depth:
            return self.evaluationFunction(game_state)
        else:
            available_actions = game_state.getLegalActions(agent_index)

            best_score = float('inf')
            if available_actions:
                best_score = float('inf')
            else:
                best_score = self.evaluationFunction(game_state)
            for action in available_actions:


                if action != Directions.STOP:
                    if agent_index == game_state.getNumAgents() - 1:
                        child = game_state.generateSuccessor(agent_index, action)
                        best_score = min(best_score, self.maximizer(depth+1, 0, child, a, b))
                        if best_score < a:
                            return best_score
                        b = min(best_score, b)
                    else:
                        child = game_state.generateSuccessor(agent_index, action)
                        best_score = min(best_score, self.minimizer(depth, agent_index+1, child, a, b))
                        if best_score < a:
                            return best_score
                        b = min(best_score, b)
            return best_score

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
        best_score = float('-inf')
        best_action = Directions.STOP
        available_actions = gameState.getLegalActions(0)
        for current_action in available_actions:
            child = gameState.generateSuccessor(0, current_action)
            current_score = self.expect(0, 1, child)
            if current_score > best_score and current_action != Directions.STOP:
                best_score = current_score
                best_action = current_action
        return best_action

    def maximizer(self, depth, agent_index, game_state):
        if depth == self.depth:
            return self.evaluationFunction(game_state)
        else:
            available_actions = game_state.getLegalActions(agent_index)
            if available_actions:
                best_score = float('-inf')
            else:
                best_score = self.evaluationFunction(game_state)
            for next_action in available_actions:
                child = game_state.generateSuccessor(agent_index, next_action)
                current_score = self.expect(depth, agent_index+1, child)
                if current_score > best_score:
                    best_score = current_score
            return best_score

    def expect(self, depth, agent_index, game_state):
        if depth == self.depth:
            return self.evaluationFunction(game_state)
        else:
            available_actions = game_state.getLegalActions(agent_index)
            total_score = 0
            if available_actions:
                for action in available_actions:
                    if agent_index == game_state.getNumAgents() - 1:
                        child = game_state.generateSuccessor(agent_index, action)
                        total_score += self.maximizer(depth+1, 0, child)
                    else:
                        child = game_state.generateSuccessor(agent_index, action)
                        total_score += self.expect(depth, agent_index+1, child)
                    average_score = total_score / len(available_actions)
                return average_score
            return self.evaluationFunction(game_state)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #distance to nearest food
    #print dir(currentGameState)


    if len(currentGameState.getFood().asList()) == 0:
        return float('inf')

    def calc_dist(p1, p2):
      return ((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2)

    score = currentGameState.getScore()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


    #ghost code
    minGhostDisp = float('inf')
    for ghost in newGhostStates:
        disp = calc_dist(newPos, ghost.getPosition())
        if disp < minGhostDisp:
            minGhostDisp = disp

    if ghostState.scaredTimer:
        if minGhostDisp == 0:
            return float("inf")
    else:
        if minGhostDisp == 0:
            return float("-inf")
        minGhostDisp *= -1

    score += .6/float(minGhostDisp)

    #nearest food
    minDisplacement = float("inf")
    for foodPos in currentGameState.getFood().asList():
        displacement = calc_dist(newPos, foodPos)
        if displacement < minDisplacement:
            minDisplacement = displacement

    score += .4/float(minDisplacement)

    return score

# Abbreviation
better = betterEvaluationFunction

