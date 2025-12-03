from collections import deque
from heapq import heappush, heappop

from .graph import Cell
from .utils import trace_path


"""
General graph search instructions:

First, define the correct data type to keep track of your visited cells
and add the start cell to it. If you need to initialize any properties
of the start cell, do that too.

Next, implement the graph search function. When you find a path, use the
trace_path() function to return a path given the goal cell and the graph. You
must have kept track of the parent of each node correctly and have implemented
the graph.get_parent() function for this to work. If you do not find a path,
return an empty list.

To visualize which cells are visited in the navigation webapp, save each
visited cell in the list in the graph class as follows:
     graph.visited_cells.append(Cell(cell_i, cell_j))
where cell_i and cell_j are the cell indices of the visited cell you want to
visualize.
"""


def depth_first_search(graph, start, goal):
    """Depth First Search (DFS) algorithm. This algorithm is optional for P3.
    Args:
        graph: The graph class.
        start: Start cell as a Cell object.
        goal: Goal cell as a Cell object.
    """
    graph.init_graph()  # Reset all node-related data in the graph.

    stack = [start]
    visited = set()

    # Start node has no parent.
    graph.parent[(start.i, start.j)] = None
    visited.add((start.i, start.j))
    graph.visited_cells.append(Cell(start.i, start.j))

    while stack:
        current = stack.pop()
        ci, cj = current.i, current.j

        # Check goal.
        if ci == goal.i and cj == goal.j:
            return trace_path(current, graph)

        for nbr in graph.find_neighbors(ci, cj):
            key = (nbr.i, nbr.j)
            if key in visited:
                continue

            visited.add(key)
            graph.parent[key] = current
            graph.visited_cells.append(Cell(nbr.i, nbr.j))
            stack.append(nbr)

    # No path found.
    return []


def breadth_first_search(graph, start, goal):
    """Breadth First Search (BFS) algorithm.
    Args:
        graph: The graph class.
        start: Start cell as a Cell object.
        goal: Goal cell as a Cell object.
    """
    graph.init_graph()  # Reset all node-related data in the graph.

    queue = deque()
    visited = set()

    queue.append(start)
    visited.add((start.i, start.j))
    graph.parent[(start.i, start.j)] = None
    graph.visited_cells.append(Cell(start.i, start.j))

    while queue:
        current = queue.popleft()
        ci, cj = current.i, current.j

        # Check goal.
        if ci == goal.i and cj == goal.j:
            return trace_path(current, graph)

        for nbr in graph.find_neighbors(ci, cj):
            key = (nbr.i, nbr.j)
            if key in visited:
                continue

            visited.add(key)
            graph.parent[key] = current
            graph.visited_cells.append(Cell(nbr.i, nbr.j))
            queue.append(nbr)

    # No path found.
    return []


def heuristic(a, b):
    """Simple Manhattan distance heuristic for grid maps."""
    return abs(a.i - b.i) + abs(a.j - b.j)


def a_star_search(graph, start, goal):
    """A* Search algorithm.
    Args:
        graph: The graph class.
        start: Start cell as a Cell object.
        goal: Goal cell as a Cell object.
    """
    graph.init_graph()  # Reset all node-related data in the graph.

    # Priority queue storing: (f_cost, g_cost, counter, Cell)
    open_heap = []

    # Cost from start to each cell: (i, j) -> g_cost
    g_cost = {}

    # Simple counter to break ties so we never compare Cell objects.
    counter = 0

    start_key = (start.i, start.j)
    g_cost[start_key] = 0.0
    graph.parent[start_key] = None

    f_start = g_cost[start_key] + heuristic(start, goal)
    heappush(open_heap, (f_start, g_cost[start_key], counter, start))
    counter += 1

    # Closed set: cells we already expanded.
    closed = set()

    while open_heap:
        f, g, _, current = heappop(open_heap)
        ci, cj = current.i, current.j
        key = (ci, cj)

        # Skip if we already processed this cell.
        if key in closed:
            continue

        closed.add(key)
        graph.visited_cells.append(Cell(ci, cj))

        # Check for goal.
        if ci == goal.i and cj == goal.j:
            return trace_path(current, graph)

        # Explore neighbors.
        for nbr in graph.find_neighbors(ci, cj):
            nk = (nbr.i, nbr.j)
            if nk in closed:
                continue

            tentative_g = g + 1.0  # uniform step cost on the grid

            # If this path to neighbor is better, remember it.
            if nk not in g_cost or tentative_g < g_cost[nk]:
                g_cost[nk] = tentative_g
                graph.parent[nk] = current
                f_nbr = tentative_g + heuristic(nbr, goal)
                heappush(open_heap, (f_nbr, tentative_g, counter, nbr))
                counter += 1

    # No path found.
    return []
