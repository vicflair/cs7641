from mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning
import mdptoolbox.util as util
import mdptoolbox.example
import numpy as np


class Stocks(object):

    """Defines the stock trading MDP."""

    def __init__(self, N=7):
        self.A = 3  # 3 actions
        self.N = N  # N stock states

    def transitions(self):
        """Create transition matrices."""
        shape = (self.N + 1, self.N + 1)
        transitions = []

        # Define for action "buy"
        matrix = np.identity(self.N + 1)  # buy is the same as "hold"
        matrix[0, 0] = 0
        matrix[0, (self.N + 1) / 2] = 1  # always buy "average" stock
        util.checkSquareStochastic(matrix)
        transitions.append(matrix)

        # Define for action "hold". Basically a random walk.
        matrix = np.zeros(shape)
        matrix[0, 0] = 1
        matrix[1, 0:2] = 1. / 2
        for i in range(2, self.N):
            for j in range(i - 1, i + 2, 1):
                matrix[i, j] = 1. / 3
        matrix[self.N, self.N - 1:self.N + 1] = 1. / 2
        util.checkSquareStochastic(matrix)
        transitions.append(matrix)

        # Define for action "sell"
        matrix = np.zeros(shape)
        matrix[:, 0] = 1  # always reset to initial state, i.e. no stock
        util.checkSquareStochastic(matrix)
        transitions.append(matrix)

        return transitions

    def rewards(self, rval=None):
        """Define rewards matrix."""

        # Define reward values
        if rval is None:
            rval = [-1, -0.05, 2]
        shape = (self.N + 1, self.A)
        rewards = np.zeros(shape)

        # Define for action "buy". Always incur some purchase cost
        # greater than the opportunity cost of holding.
        rewards[:, 0] = rval[0]

        # Define for action "hold". Always incur opportunity cost.
        rewards[:, 1] = rval[1]

        # Define rewards for selling.
        xmin = -(self.N - 1) / 2
        xmax = -xmin + 1
        rewards[1:, 2] = [rval[2] ** x for x in range(xmin, xmax, 1)]
        rewards[0, 2] = -999  # Never do this

        return rewards


class Maze(object):

    """The Maze MDP."""

    def __init__(self, maze=None, goal=None, theseus=None, minotaur=None):
        # The pre-defined maze
        if maze is None:
            self.maze = np.asarray(
                [[0, 1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 1, 1, 0, 1, 0],
                 [1, 0, 0, 0, 0, 1, 0],
                 [0, 0, 1, 1, 1, 1, 0],
                 [0, 0, 1, 0, 0, 1, 0],
                 [0, 0, 1, 1, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0]])
        else:
            self.maze = maze
        self.X = self.maze.shape[0]
        self.Y = self.maze.shape[1]

        # Goal position
        if goal is None:
            self.goal = (0, 0)
        else:
            self.goal = goal

        # Theseus starting position
        if theseus is None:
            self.theseus = (5, 3)
        else:
            assert self.maze[theseus] == 0
            assert theseus != self.goal
            self.theseus = theseus

        # Minotaur starting position
        if minotaur is None:
            self.minotaur = (5, 1)
        else:
            assert self.maze[minotaur] == 0
            self.minotaur = minotaur

    def transitions(self):
        """Return transition matrices, one for each action."""

        # Initialize transition matrices
        shape = self.maze.shape
        num_states = (shape[0] * shape[1]) ** 2
        matrix_size = (4, num_states, num_states)
        T = np.zeros(matrix_size)

        # All possible positions on the map
        pos = [(i, j) for i in range(shape[0]) for j in range(shape[1])]

        # For every pair of positions, get transition probabilities
        pos2 = ((theseus, minotaur) for theseus in pos for minotaur in pos)
        for theseus, minotaur in pos2:
            # Get Theseus's new positions (deterministic)
            theseus_next = self.get_moves(theseus)

            # Get Minotaur's possible new positions (random)
            minotaur_next = self.get_moves(minotaur)

            # Update transition probabilities for each action matrix
            current_state = self.global_state(theseus, minotaur)
            for a in range(4):
                # Get current and next states
                next_states = [self.global_state(theseus_next[a], M)
                               for M in minotaur_next]
                # Update transition probabilities
                for ns in next_states:
                    T[a, current_state, ns] += 0.25

        # "Reset" to initial state when reaching goal or meeting minotaur.
        initial_state = self.global_state(self.theseus, self.minotaur)
        # All states where Theseus and minotaur are co-located
        for p in pos:
            current_state = self.global_state(p, p)

            # Reset to initial state is guaranateed
            T[:, current_state, :] = 0
            T[:, current_state, initial_state] = 1

        # Next state after goal is always initial state, i.e. reset
        T[:, self.goal, :] = 0
        T[:, self.goal, initial_state] = 1

        # Confirm stochastic matrices
        for a in range(4):
            util.checkSquareStochastic(T[a])
        return T

    def rewards(self, rval=None):
        """Returns reward matrix."""

        # Define reward values
        if rval is None:
            # Reward for goal, penalty for minotaur, step penalty
            self.rval = [1, -1, -0.01]
        else:
            self.rval = rval

        # Initialize rewards matrix with step penalty
        shape = self.maze.shape
        num_states = (shape[0] * shape[1]) ** 2
        R = np.ones((num_states, 4)) * self.rval[2]

        # All possible positions on the map
        pos = [(i, j) for i in range(shape[0]) for j in range(shape[1])]

        # Positions adjacent to goal and the goal-reaching action. Adjacent
        # positions found by moving from the goal without rebounds off walls.
        # Optimal action for an adjacent position is the opposite of the
        # action which led from the goal to the adjacent position. E.g, if
        # "South" led to position X, then optimal action for X is "North".
        penultimate = [(s, (a + 2) % 4)
                       for a, s in enumerate(self.get_moves(self.goal))
                       if s != self.goal]

        # Reward for taking a goal-reaching action, no matter where the
        # minotaur is going to be. Do not consider case where minotaur
        # sits at goal, or collides with Theseus.
        for adj, a_star in penultimate:
            for minotaur in pos:
                R[self.global_state(adj, minotaur), a_star] = self.rval[0]

        # Penalty for Theseus sharing the same state as the minotaur, no
        # matter what action is taken next :(
        for p in pos:
            R[self.global_state(p, p), :] = self.rval[1]

        return R

    def get_moves(self, pos):
        """Get result of N, E, S, W moves."""
        x = pos[0]
        y = pos[1]
        moves = []
        # Check North
        if (x > 0) and (self.maze[x - 1, y] == 0):
            moves.append((x - 1, y))
        else:
            moves.append((x, y))
        # Check East
        if (y < self.Y - 1) and (self.maze[x, y + 1] == 0):
            moves.append((x, y + 1))
        else:
            moves.append((x, y))
        # Check South
        if (x < self.X - 1) and (self.maze[x + 1, y] == 0):
            moves.append((x + 1, y))
        else:
            moves.append((x, y))
        # Check West
        if (y > 0) and (self.maze[x, y - 1] == 0):
            moves.append((x, y - 1))
        else:
            moves.append((x, y))

        return moves

    def local_state(self, pos):
        """Convert position on map to a local state."""
        state = (self.Y * pos[0] + pos[1])
        return state

    def global_state(self, pos1, pos2):
        """Convert a pair of positions on map to a unique state."""
        num_local_states = (self.X * self.Y)
        state1 = self.local_state(pos1)
        state2 = self.local_state(pos2)
        global_state = state1 * num_local_states + state2
        return global_state


    def visualize(self, state):
        pass

    def visualize_policy(self, state, policy):
        pass

    def unit_test_global_state(self):
        """Test proper functionality of global_state()."""
        height = self.X
        width = self.Y
        pos = [(i, j) for i in range(height) for j in range(width)]
        global_states = [self.global_state(pos1, pos2)
                         for pos1 in pos
                         for pos2 in pos]
        total = (height ** 2) * (width ** 2)  # Expected state size
        assert len(set(global_states)) == total
        assert set(global_states).difference(set(range(total))) == set()


def unit_test_Maze():
    """Test proper functionality of Maze()."""
    maze = np.asarray([[0, 0],
                       [0, 1]])
    goal = (0, 0)
    theseus = (0, 1)
    minotaur = (1, 0)

    M = Maze(maze=maze, goal=goal, theseus=theseus, minotaur=minotaur)
    print "Transitions"
    print M.transitions()
    print "\nRewards"
    print M.rewards()


def example():
    """Run the MDP Toolbox forest example."""
    transitions, rewards = mdptoolbox.example.forest()
    viter = ValueIteration(transitions, rewards, 0.9)
    viter.run()
    print viter.policy


def solve_stocks():
    """Solve the Stocks MDP."""
    tmp = Stocks()
    discount = 0.9

    T = tmp.transitions()
    print "\nAction: Buy"
    print T[0]
    print "\nAction: Hold"
    print T[1]
    print "\nAction: Sell"
    print T[2]

    R = tmp.rewards()
    print "\nRewards"
    print R

    viter = ValueIteration(T, R, discount)
    viter.run()
    print "\nValue iteration: {}".format(viter.policy)

    piter = PolicyIteration(T, R, discount)
    piter.run()
    print "\nPolicy iteration: {}".format(piter.policy)

    qlearn = QLearning(T, R, discount, n_iter=10000)
    qlearn.run()
    print "\nQ-learning: {}".format(qlearn.policy)
    print "\nQ: \n{}".format(qlearn.Q)


def solve_mini_maze():
    """Solve miniature Maze MDP."""
    maze = np.asarray([[0, 0],
                       [0, 1]])
    goal = (0, 0)
    theseus = (0, 1)
    minotaur = (1, 0)

    M = Maze(maze=maze, goal=goal, theseus=theseus, minotaur=minotaur)
    T = M.transitions()
    R = M.rewards()
    discount = 0.9

    viter = ValueIteration(T, R, discount)
    viter.run()
    print "\nValue iteration: {}".format(viter.policy)

    piter = PolicyIteration(T, R, discount)
    piter.run()
    print "\nPolicy iteration: {}".format(piter.policy)

    qlearn = QLearning(T, R, discount, n_iter=10000)
    qlearn.run()
    print "\nQ-learning: {}".format(qlearn.policy)
    print "\nQ: \n{}".format(qlearn.Q)


def main():
    """Run everything."""
    solve_mini_maze()

if __name__ == "__main__":
    main()
