# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]




class SearchNode:

    def __init__(self, state, action = None, parent = None, cost = None, heuristic = None, problem = None):
        self.parent = parent
        self.a = action  # From what action
        self.s = state
        self.g = cost
        self.h = heuristic
        self.p = problem
        if self.h is not None:
            if self.p is None:
                raise AttributeError("A star is active, need problem parameter to be non-null!")

    def construct_path(self):
        res = [self]
        curr = self
        while curr is not None:
            res.append(curr.get_state())
            curr = curr.parent
        return list(reversed(res))

    def construct_action(self):
        res = []
        curr = self
        while curr.parent is not None:
            res.append(curr.a)
            curr = curr.parent

        return list(reversed(res))

    def get_state(self):
        return self.s

    def get_cost(self):
        return self.g

    def get_heuristic(self):
        return 0 if self.h is None else self.h(self.s, self.p)

    def get_priority(self):
        return self.get_heuristic() + self.get_cost()


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    fringe = util.Stack()
    visited = set()

    s = problem.getStartState()
    fringe.push(SearchNode(state = s, action =None, parent =None, cost = 0))
    while not fringe.isEmpty():
        curr_node = fringe.pop()
        curr_state = curr_node.get_state()
        if problem.isGoalState(curr_state):
            return curr_node.construct_action()

        # Prevent multiple path, multi-expanded problem(2nd version)
        if curr_state not in visited:
            visited.add(curr_state)
            # getSuccessors() return (child_state, action, cost_of_action)
            for child_state_tuple in problem.getSuccessors(curr_state):
                child_state, action, cost = child_state_tuple
                fringe.push(SearchNode(child_state, action, curr_node, curr_node.get_cost() + cost))



def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()
    visited = set()

    s = problem.getStartState()
    fringe.push(SearchNode(state=s, action=None, parent=None, cost=0))
    while not fringe.isEmpty():
        curr_node = fringe.pop()
        curr_state = curr_node.get_state()
        if problem.isGoalState(curr_state):
            return curr_node.construct_action()
        # This if statement is important to prevent duplicate visits to the same state
        # with different path history
        if curr_state not in visited:
            visited.add(curr_state)
            # getSuccessors() return (child_state, action, cost_of_action)
            for child_state_tuple in problem.getSuccessors(curr_state):
                child_state, action, cost = child_state_tuple
                fringe.push(SearchNode(child_state, action, curr_node, curr_node.get_cost() + cost))


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()
    visited = set()

    s = problem.getStartState()
    s_node = SearchNode(state=s
                        ,action=None
                        ,parent=None
                        ,cost=0)
    fringe.push(s_node, s_node.get_priority())
    while not fringe.isEmpty():
        curr_node = fringe.pop()
        curr_state = curr_node.get_state()
        if problem.isGoalState(curr_state):
            return curr_node.construct_action()
        # This if statement is important to prevent duplicate visits to the same state
        # with different path history
        if curr_state not in visited:
            visited.add(curr_state)
            # getSuccessors() return (child_state, action, cost_of_action)
            for child_state_tuple in problem.getSuccessors(curr_state):
                child_state, action, cost = child_state_tuple
                child_node = SearchNode(state = child_state
                                        ,action = action
                                        ,parent = curr_node
                                        ,cost = curr_node.get_cost() + cost)
                fringe.push(child_node, child_node.get_priority())


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()
    visited = set()

    s = problem.getStartState()
    s_node = SearchNode(state=s, action=None, parent=None, cost=0, heuristic=heuristic, problem=problem)
    fringe.push(s_node, s_node.get_priority())
    while not fringe.isEmpty():
        curr_node = fringe.pop()
        curr_state = curr_node.get_state()
        if problem.isGoalState(curr_state):
            return curr_node.construct_action()
        # This if statement is important to prevent duplicate visits to the same state
        # with different path history
        if curr_state not in visited:
            visited.add(curr_state)
            # getSuccessors() return (child_state, action, cost_of_action)
            for successor in problem.getSuccessors(curr_state):
                child_state, action, cost = successor
                child_node = SearchNode(state=child_state
                                        ,action=action
                                        ,parent=curr_node
                                        ,cost=curr_node.get_cost() + cost
                                        ,heuristic=heuristic
                                        ,problem=problem)
                fringe.push(child_node, child_node.get_priority())


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
