"""
CS50's Introduction to AI with Python:

Find the course notes and source code here: https://cs50.harvard.edu/ai/2023/
"""

import sys
import os
import inspect

import warnings

from PIL import Image, ImageDraw

"""
Use stacks and queues in search problems:
- initial state
- actions
- transition model
- goal test
- path cost function
- we look for the optimal solution of coming from inital state via actions to goal (goal test)

node: a data strucure that keeps track of a state, a parent, an action, path cost - in search problems keeping
track of these four specific values

The frontier comprises all possibilities we could explore next and what we haven't explored yet

For search algos we need stacks and queues.
We have to decide what approach/order we choose for taking a node off a frontier:
For Depth-First-Search DFS we take a stack: search algo that always expands the deepest node in the frontier
For Breadth-First-Search BFS we take a queue: search algo that always expands the shallowest node in the frontier
"""

global fx, i


class Node:
    """
    comprises state, parent, action and costs for DFS and BFS
    """
    def __init__(self, state, parent, action, costs):
        self.state = state
        self.parent = parent
        self.action = action
        # Path costs: not used in course - just for later purposes
        self.costs = costs


class StackFrontier:
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            warnings.warn(
                'You reached an empty frontier',
                UserWarning)
            raise Exception('empty frontier')
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node

    def __repr__(self):
        for i in self.frontier: print(i)
        return f'Frontier:({self.frontier}, {i})'


class QueueFrontier(StackFrontier):
    def remove(self):
        if self.empty():
            warnings.warn(
                'You reached an empty frontier',
                UserWarning)
            raise Exception('empty frontier')
        else:   # First in first out
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node


class Maze:
    def __init__(self, filename):
        # Read textfile and set height and width of maze
        global fx
        if '__file__' not in locals():
            fx = inspect.getframeinfo(inspect.currentframe())[0]
        else:
            fx = __file__

        # dieser Befehl zeigt mir mein working directory:
        os_dir = os.path.dirname(os.path.abspath(fx))
        print("That's my working dir:\n", os_dir)

        with open(os.path.join(os_dir, filename)) as f:
            # we read the text-file line for line
            contents = f.read()
            print('Contents of textfile:\n', contents)

        # Validate start and goal
        if contents.count('A') != 1:
            raise Exception('maze must have exactly one start point...')
        if contents.count('B') != 1:
            raise Exception('maze must have exactly one goal...')

        # Determine height and width of maze
        # contents are staggered lines - splitlines flatten it - like in one row
        # ['#####B#', '##### #', '####  #', '#### ##', '     ##', 'A######']

        # or
        # index [0] is the top-line:

        # ['###                 #########', '#   ###################   # #', '# ####                # # # #',
        # '# ################### # # # #', '#                     # # # #', '##################### # # # #',
        # '#   ##                # # # #', '# # ## ### ## ######### # # #', '# #    #   ##B#         # # #',
        # '# # ## ################ # # #', '### ##             #### # # #', '### ############## ## # # # #',
        # '###             ##    # # # #', '###### ######## ####### # # #', '###### ####             #   #',
        # 'A      ######################']

        contents = contents.splitlines()
        print(contents)

        # that's why now the elements of contents are originally the height
        self.height = len(contents)
        # and the width is now the length of the longest line since a piece of the way is NULL and a hash(wall) is
        # like a letter
        self.width = max(len(line) for line in contents)

        # Keep track of walls - we loop through div. lines: height and width
        # every hashtag a.k.a wall in the textfile is a True-value:

        self.walls = []
        for i in range(self.height):
            # we construct the grid/maze - it's made of rows and columns depicted by lists:
            row = []
            for j in range(self.width):
                try:
                    # we look for A in a line i and line-position j, if found we set a tuple with the xy-coords (i,j)
                    # we look into every imagined cell - xy-coord (i,j)
                    if contents[i][j] == 'A':
                        self.start = (i, j)
                        print(f"""
                        
                            SELF START: {self.start}
                        
                        """)
                        #  (15, 0) - 16th line, first position
                        row.append(False)
                    elif contents[i][j] == 'B':
                        # we set a tuple with the goal-coords
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j] == ' ':
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)
        print(f"""

                            SELF Walls: {self.walls}

                                """)
        #  SELF Walls - a representation of the maze by True and False values:
        #  we start with top row 0 - True means there is a hashtag a.k.a wall:
        #  [[True, True, True, True, True, False, True], [True, True, True, True, True, False, True],
        #  [True, True, True, True, False, False, True], [True, True, True, True, False, True, True],
        #  [False, False, False, False, False, True, True], [False, True, True, True, True, True, True]]

        # we already construct the empty var solution:
        self.solution = None

    def print(self):
        solution = self.solution[1] if self.solution is not None else None
        print() # an empty terminal row

        # walls is a list of lists:
        """
        https://www.simplilearn.com/tutorials/python-tutorial/enumerate-in-python
        Enumerate-func:
        my_list = ['a', 'b', 'c']

        for index, item in enumerate (my_list):
        print(index, item)

        This function would print out the following: 0 a, 1 b, 2 c
        """
        # rows are the lists within the walls-list, i is the counter:
        for i, row in enumerate(self.walls):
            # now row is our iterable (list) and every item in it stands for a column, j is the counter
            for j, col in enumerate(row):
                # if col is True - same as:
                if col:
                    print('█', end='')
                elif (i, j) == self.start:
                    print('A', end='')
                elif (i, j) == self.goal:
                    print('B', end='')
                elif solution is not None and (i, j) in solution:
                    print('*', end='')
                else:
                    # ' ' is an aisle
                    print(' ', end='')
            print()
        print()

    def neighbors(self, state):
        # state are the coordinates
        row, col = state
        candidates = [
            ('up', (row - 1, col)),
            ('down', (row + 1, col)),
            ('left', (row, col - 1)),
            ('right', (row, col + 1)),
        ]

        result = []
        # the next line indexes this
        # ('up', (row - 1, col)):
        for action, (r, c) in candidates:
            # in walls we index the item [c] within the list [r]
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                result.append((action, (r, c)))
        print('Result of func neighbors:\n', result)
        return result
    """
    Solving...
    Result of func neighbors:
    [('up', (4, 0))]
    Result of func neighbors:
    Why do we have two entries here? The algo tries both ways - A is the start so that it is the wrong way
    [('down', (5, 0)), ('right', (4, 1))]
    Result of func neighbors:
    Left had already been explored, so right is the correct way:
    [('left', (4, 0)), ('right', (4, 2))]
    """

    def solve(self):
        """ Finds a solution to maze, if one exists. """

        # Keep track of number of states explored
        self.num_explored = 0

        # Initialize frontier to just the starting position
        start = Node(state=self.start, parent=None, action=None, costs=None)

        # we instance just an empty frontier-list - choose StackFrontier or QueueFrontier
        frontier = QueueFrontier()
        frontier.add(start)
        print('Start-Frontier:', frontier)

        # Initialize an empty explored set
        self.explored = set()

        # Keep looking until solution found
        while True:

            # If nothing left in frontier, then no path
            if frontier.empty():
                raise Exception('no solution')

            # Choose a node from the frontier
            node = frontier.remove()
            self.num_explored += 1
            # print('Chosen node from frontier:', node) #wie kann ich die Objekte drucken???

            # If node is the goal, then we have a solution
            if node.state == self.goal:
                actions = []
                cells = []
                # this loop keeps repeating looking through all the parent nodes until we get back to the initial state
                # which has no parent
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                # when the loop has finished I reverse the actions and cells since I want the solution go from
                # the start to the finish - like reading the list backwards
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return

            # Mark node as explored
            self.explored.add(node.state)

            # Add neighbors to frontier
            for action, state in self.neighbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    child = Node(state=state, parent=node, action=action, costs=None)
                    frontier.add(child)

    def output_image(self, filename, show_solution=True, show_explored=False):

        cell_size = 50
        cell_border = 2

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.width * cell_size, self.height * cell_size),
            "black"
        )
        draw = ImageDraw.Draw(img)

        solution = self.solution[1] if self.solution is not None else None
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):

                # Walls
                if col:
                    fill = (40, 40, 40)

                # Start
                elif (i, j) == self.start:
                    fill = (255, 0, 0)
                # Goal
                elif (i, j) == self.goal:
                    fill = (0, 171, 28)
                # Solution
                elif solution is not None and show_solution and (i, j) in solution:
                    fill = (220, 235, 113)
                # Explored
                elif solution is not None and show_explored and (i, j) in self.explored:
                    fill = (212, 97, 85)
                # Empty
                else:
                    fill = (237, 240, 252)

                # draw cell
                draw.rectangle(
                    ([(j * cell_size + cell_border, i * cell_size + cell_border),
                      ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)]),
                    fill=fill
                )
        img.save(filename)
        img.show()


if len(sys.argv) != 2:
    sys.exit("Usage: python maze.py maze<1 - 3>.txt")


m = Maze(sys.argv[1])
print('Maze:')
m.print()
print('Solving...')
m.solve()
print("States Explored:", m.num_explored)
print("Solution:")
m.print()
m.output_image("maze.png", show_explored=True)

# Todo: above-mentioned is uninformed search; still to research informed search like:
#  manhattan-distance, A*-search, Minimax etc.

"""
greedy best-first search: uses heuristic function h(n) - Manhattan distance; which one is closer to the goal
neglecting walls, just geographically 

A* search: algo, that expands node with lowest value of g(n) + h(n)
g(m) = cost to reach node
h(n) = estimated cost to goal

Adversarial search - Minimax: Max player aims to maximize score - Min player aims to minimize score.
"""



