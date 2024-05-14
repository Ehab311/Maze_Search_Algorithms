import math

import queue
from queue import LifoQueue


class Node:
    id = None
    up = None
    down = None
    left = None
    right = None
    previousNode = None
    edgeCost = None
    gOfN = None  # total edge cost
    hOfN = None  # heuristic value
    heuristicFn = None

    def __init__(self, value):
        self.value = value


class SearchAlgorithms:
    Path = []
    fullPath = []
    totalCost = -1
    mazeStr = ""
    maze_rows = 0
    maze_cols = 0
    maze = []
    adj_list = {}
    weighted_adj_list = {}
    heuristic_Euclidean = {}
    heuristic_Manhattan = {}
    maze_length = 0
    end_index = 0
    start_index = 0
    cost = {}

    def __init__(self, mazeStr, cost=None):
        self.mazeStr = mazeStr
        self.create_maze(mazeStr)
        self.Path.clear()
        self.adj_list.clear()
        self.fullPath.clear()
        self.heuristic_Euclidean.clear()
        self.heuristic_Manhattan.clear()
        self.cost.clear()
        self.create_adj_list()
        if cost != None:
            self.create_cost_dict(cost)
            self.create_weighted_adj_list()
            self.create_heuristic()

    def get_row_col(self, mazeStr):
        index = mazeStr.index(' ')
        if index % 2 == 0:
            cols = int(index / 2)
        else:
            cols = int((index - 1) / 2) + 1
        rows = mazeStr.count(' ') + 1
        self.maze_cols = cols
        self.maze_rows = rows
        self.maze_length = cols * rows

    def create_maze(self, mazeStr):
        self.get_row_col(mazeStr)
        self.maze = [[0 for _ in range(self.maze_cols)] for _ in range(self.maze_rows)]
        pop = 0
        for i, char in enumerate(mazeStr):
            if char == ',' or char == ' ':
                i += 1
                pop += 1
            elif char == 'S':
                self.maze[(i - pop) // self.maze_cols][(i - pop) % self.maze_cols] = 'S'
            elif char == 'E':
                self.maze[(i - pop) // self.maze_cols][(i - pop) % self.maze_cols] = 'E'
            elif char == '.':
                self.maze[(i - pop) // self.maze_cols][(i - pop) % self.maze_cols] = 0
            elif char == '#':
                self.maze[(i - pop) // self.maze_cols][(i - pop) % self.maze_cols] = -1

    def create_adj_list(self):
        self.adj_list = {}
        for i in range(self.maze_rows):
            for j in range(self.maze_cols):
                if self.maze[i][j] != -1:  # Ignore walls
                    neighbors = []
                    if i > 0 and self.maze[i - 1][j] != -1:  # Up
                        neighbors.append(((i - 1) * self.maze_cols) + j)
                    if i < self.maze_rows - 1 and self.maze[i + 1][j] != -1:  # Down
                        neighbors.append(((i + 1) * self.maze_cols) + j)
                    if j > 0 and self.maze[i][j - 1] != -1:  # Left
                        neighbors.append((i * self.maze_cols) + j - 1)
                    if j < self.maze_cols - 1 and self.maze[i][j + 1] != -1:  # Right
                        neighbors.append((i * self.maze_cols) + j + 1)
                    if self.maze[i][j] == 'E':
                        self.end_index = (i * self.maze_cols) + j
                    if self.maze[i][j] == 'S':
                        self.start_index = (i * self.maze_cols) + j

                    self.adj_list[(i * self.maze_cols) + j] = neighbors

    def create_cost_dict(self, cost):
        for i in range(self.maze_rows):
            for j in range(self.maze_cols):
                self.cost[(i * self.maze_cols) + j] = cost[(i * self.maze_cols) + j]

    def create_weighted_adj_list(self):
        self.weighted_adj_list = {}
        for i in range(self.maze_rows):
            for j in range(self.maze_cols):
                if self.maze[i][j] != -1:  # Ignore walls
                    neighbors = []
                    if i > 0 and self.maze[i - 1][j] != -1:  # Up
                        upper = ((i - 1) * self.maze_cols) + j
                        neighbors.extend([(upper, self.cost[upper])])
                    if i < self.maze_rows - 1 and self.maze[i + 1][j] != -1:  # Down
                        down = ((i + 1) * self.maze_cols) + j
                        neighbors.extend([(down, self.cost[down])])
                    if j > 0 and self.maze[i][j - 1] != -1:  # Left
                        left = (i * self.maze_cols) + j - 1
                        neighbors.extend([(left, self.cost[left])])
                    if j < self.maze_cols - 1 and self.maze[i][j + 1] != -1:  # Right
                        right = (i * self.maze_cols) + j + 1
                        neighbors.extend([(right, self.cost[right])])
                    if self.maze[i][j] == 'E':
                        self.end_index = (i * self.maze_cols) + j
                    if self.maze[i][j] == 'S':
                        self.start_index = (i * self.maze_cols) + j
                    self.weighted_adj_list[(i * self.maze_cols) + j] = neighbors

    def manhattan_distance(self, node1, node2):
        """
        Compute the Manhattan distance between two nodes.

        Parameters:
            node1 (tuple): Coordinates of the first node in the format (x1, y1).
            node2 (tuple): Coordinates of the second node in the format (x2, y2).

        Returns:
            int: Manhattan distance between the two nodes.
        """
        x1, y1 = node1
        x2, y2 = node2
        return abs(x1 - x2) + abs(y1 - y2)

    def euclidean_distance(self, node1, node2):
        """
        Compute the Euclidean distance between two nodes.

        Parameters:
            node1 (tuple): Coordinates of the first node in the format (x1, y1).
            node2 (tuple): Coordinates of the second node in the format (x2, y2).

        Returns:
            float: Euclidean distance between the two nodes.
        """
        x1, y1 = node1
        x2, y2 = node2
        return int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    def create_heuristic(self):
        row_end = self.end_index // self.maze_cols
        col_end = self.end_index % self.maze_cols
        end_node = (row_end, col_end)
        for key, values in self.adj_list.items():
            row1 = key // self.maze_cols
            col1 = key % self.maze_cols
            node1 = (row1, col1)
            self.heuristic_Manhattan[key] = self.manhattan_distance(node1, end_node)
            self.heuristic_Euclidean[key] = self.euclidean_distance(node1, end_node)

    def DFS(self):
        visited = ([False] * self.maze_length * 2)
        not_visited = []
        visited[self.start_index] = True
        not_visited.extend(reversed(self.adj_list[self.start_index]))
        self.fullPath.append(self.start_index)  # Assuming self.fullPath is a list
        parent = {self.start_index: 'S'}
        for adj in self.adj_list[self.start_index]:
            parent[adj] = self.start_index
            visited[adj] = True
        while not_visited:
            current_node = not_visited.pop()  # Dequeue the Last node from not_visited
            self.fullPath.append(current_node)  # Append the index of the neighbor
            if current_node == self.end_index:  # If the exit node is reached, terminate the DFS
                break
            # visited[current_node] = True
            if current_node in self.adj_list:
                for neighbor in reversed(self.adj_list[current_node]):
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        not_visited.append(neighbor)  # Append the index of the neighbor
                        parent[neighbor] = current_node

        child = self.end_index
        while child != 'S':
            self.Path.insert(0, child)
            child = parent[child]
        return self.Path, self.fullPath

    def BFS(self):
        visited = ([False] * self.maze_length * 2)
        not_visited = []
        visited[self.start_index] = True
        not_visited.extend(self.adj_list[self.start_index])
        self.fullPath.append(self.start_index)  # Assuming self.fullPath is a list
        parent = {self.start_index: 'S'}
        for neighbor in self.adj_list[self.start_index]:
            parent[neighbor] = self.start_index
            visited[neighbor] = True

        while not_visited:
            current_node = not_visited.pop(0)  # Dequeue the first node from not_visited
            self.fullPath.append(current_node)  # Append the index of the neighbor
            if current_node == self.end_index:  # If the exit node is reached, terminate the BFS
                break
            if current_node in self.adj_list:
                for neighbor in self.adj_list[current_node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        not_visited.append(neighbor)  # Append the index of the neighbor
                        parent[neighbor] = current_node

        child = self.end_index
        while child != "S":
            self.Path.insert(0, child)
            child = parent[child]
        return self.Path, self.fullPath

    def bidirectional_search(self):
        visited_start = [False] * (self.maze_length * 2)
        visited_end = [False] * (self.maze_length * 2)
        not_visited_start = []
        not_visited_end = []

        visited_start[self.start_index] = True
        visited_end[self.end_index] = True

        not_visited_start.extend(self.adj_list[self.start_index])
        not_visited_end.extend(self.adj_list[self.end_index])

        parent_start = {self.start_index: 'S'}
        for adj in self.adj_list[self.start_index]:
            parent_start[adj] = self.start_index
            visited_start[adj] = True

        parent_end = {self.end_index: 'E'}
        for adj in self.adj_list[self.end_index]:
            parent_end[adj] = self.end_index
            visited_end[adj] = True

        intersection = -1

        while not_visited_start and not_visited_end:
            current_node_start = not_visited_start.pop(0)
            if visited_end[current_node_start]:
                intersection = current_node_start
                break
            visited_start[current_node_start] = True

            if current_node_start in self.adj_list:
                for neighbor in self.adj_list[current_node_start]:
                    if not visited_start[neighbor]:
                        not_visited_start.append(neighbor)
                        parent_start[neighbor] = current_node_start

            current_node_end = not_visited_end.pop(0)
            if visited_start[current_node_end]:
                intersection = current_node_end
                break
            visited_end[current_node_end] = True

            if current_node_end in self.adj_list:
                for neighbor in self.adj_list[current_node_end]:
                    if not visited_end[neighbor]:
                        not_visited_end.append(neighbor)
                        parent_end[neighbor] = current_node_end

        if intersection == -1:
            return "No path found"

        path_start = []
        path_end = []

        node = intersection
        while node != "S":
            path_start.insert(0, node)
            node = parent_start[node]

        node = intersection
        while node != "E":
            path_end.append(node)
            node = parent_end[node]

        return path_start, path_end

    def IDDFS(self):
        depth_limit = 0
        while True:
            self.Path.clear()
            self.fullPath.clear()
            Path, fullPath = self.DFS_with_depth_limit(depth_limit)
            if len(self.Path) != 0:
                self.Path = Path
                self.fullPath = fullPath
                return self.Path, self.fullPath, depth_limit
            depth_limit += 1

    def DFS_with_depth_limit(self, depth_limit):
        visited = [False] * (self.maze_length * 2)
        not_visited = []
        visited[self.start_index] = True
        not_visited.extend(reversed(self.adj_list[self.start_index]))
        self.fullPath.append(self.start_index)  # Assuming self.fullPath is a list
        parent = {self.start_index: 'S'}
        for adj in self.adj_list[self.start_index]:
            parent[adj] = self.start_index
            visited[adj] = True
        depth = {self.start_index: 0}
        for deep in self.adj_list[self.start_index]:
            depth[deep] = 1

        # depth = {0: 0, 1: 1, 7: 1}
        while not_visited:
            current_node = not_visited.pop()  # Dequeue the Last node from not_visited
            self.fullPath.append(current_node)  # Append the index of the neighbor
            if current_node == self.end_index:  # If the exit node is reached, terminate the DFS
                child = self.end_index
                while child != 'S':
                    self.Path.insert(0, child)
                    child = parent[child]

            if depth[current_node] < depth_limit:
                # visited[current_node] = True
                if current_node in self.adj_list:
                    for neighbor in reversed(self.adj_list[current_node]):
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            not_visited.append(neighbor)  # Append the index of the neighbor
                            parent[neighbor] = current_node
                            # Ensure neighbor is initialized in the depth dictionary
                            if neighbor not in depth:
                                depth[neighbor] = depth[current_node] + 1
        return self.Path, self.fullPath

    def UCS(self):
        visited = [False] * (self.maze_length * 2)
        not_visited = []
        visited[self.start_index] = True
        path_cost = 0
        not_visited.extend(self.weighted_adj_list[self.start_index])
        self.fullPath.append(self.start_index)
        parent = {self.start_index: 'S'}
        for adj in self.adj_list[self.start_index]:
            parent[adj] = self.start_index
            visited[adj] = True

        while not_visited:
            not_visited = sorted(not_visited, key=lambda x: x[1])
            current_node = not_visited.pop(0)
            self.fullPath.append(current_node[0])
            if current_node[0] == self.end_index:  # Terminate when goal node is reached
                break
            # visited[current_node[0]] = True
            if current_node[0] in self.weighted_adj_list:
                for neighbor in self.weighted_adj_list[current_node[0]]:
                    if not visited[neighbor[0]]:
                        visited[neighbor[0]] = True
                        iso = (neighbor[0], neighbor[1] + current_node[1])
                        not_visited.append(iso)
                        parent[neighbor[0]] = current_node[0]

        # Reconstruct path
        child = self.end_index
        while child != 'S':
            self.Path.insert(0, child)
            child = parent[child]
            if child == 'S':
                path_cost += self.cost[0]
            else:
                path_cost += self.cost[child]

        # Update total cost
        self.totalCost = path_cost
        return self.Path, self.fullPath, self.totalCost

    def dijkstra(self):
        visited = [False] * (self.maze_length * 2)
        not_visited = []
        visited[self.start_index] = True
        path_cost = 0
        not_visited.extend(self.weighted_adj_list[self.start_index])
        self.fullPath.append(self.start_index)
        parent = {self.start_index: 'S'}
        for adj in self.adj_list[self.start_index]:
            parent[adj] = self.start_index
            visited[adj] = True

        while not_visited:
            not_visited = sorted(not_visited, key=lambda x: x[1])
            current_node = not_visited.pop(0)
            self.fullPath.append(current_node[0])
            # visited[current_node[0]] = True
            if current_node[0] in self.weighted_adj_list:
                for neighbor in self.weighted_adj_list[current_node[0]]:
                    if not visited[neighbor[0]]:
                        visited[neighbor[0]] = True
                        iso = (neighbor[0], neighbor[1] + current_node[1])
                        not_visited.append(iso)
                        parent[neighbor[0]] = current_node[0]

        # Reconstruct path
        child = self.end_index
        while child != 'S':
            self.Path.insert(0, child)
            child = parent[child]
            if child == 'S':
                path_cost += self.cost[0]
            else:
                path_cost += self.cost[child]

        # Update total cost
        self.totalCost = path_cost
        return self.Path, self.fullPath, self.totalCost

    def AstarEcludianHeuristic(self):
        visited = ([False] * self.maze_length * 2)
        not_visited = []
        visited[self.start_index] = True
        path_cost = 0
        for node in self.weighted_adj_list[self.start_index]:
            zeros_neighbor = (node[0], node[1] + self.heuristic_Euclidean[node[0]])
            not_visited.append(zeros_neighbor)
        self.fullPath.append(self.start_index)  # Assuming self.fullPath is a list
        parent = {self.start_index: 'S'}
        for adj in self.adj_list[self.start_index]:
            parent[adj] = self.start_index
            visited[adj] = True
        while not_visited:
            not_visited = sorted(not_visited, key=lambda x: x[1])
            current_node = not_visited.pop(0)  # Dequeue the first node from not visited
            self.fullPath.append(current_node[0])  # Append the index of the neighbor
            if current_node[0] == self.end_index:  # If the exit node is reached, terminate the BFS
                break
            # visited[current_node[0]] = True
            if current_node[0] in self.weighted_adj_list:
                for neighbor in self.weighted_adj_list[current_node[0]]:
                    if not visited[neighbor[0]]:
                        visited[neighbor[0]] = True
                        old_heuristic = self.heuristic_Euclidean[current_node[0]]
                        new_heuristic = self.heuristic_Euclidean[neighbor[0]]
                        iso = (neighbor[0], (current_node[1] - old_heuristic) + neighbor[1] + new_heuristic)
                        not_visited.append(iso)  # Append the index of the neighbor
                        parent[neighbor[0]] = current_node[0]

        child = self.end_index
        while child != 'S':
            self.Path.insert(0, child)
            child = parent[child]
            if child != 'S':
                path_cost += self.cost[child]
        end_heuristic = self.heuristic_Euclidean[self.end_index]
        self.totalCost = path_cost + end_heuristic
        return self.Path, self.fullPath, self.totalCost

    def AstarManhattanHeuristic(self):
        visited = ([False] * self.maze_length * 2)
        not_visited = []
        visited[self.start_index] = True
        path_cost = 0
        for node in self.weighted_adj_list[self.start_index]:
            zeros_neighbor = (node[0], node[1] + self.heuristic_Manhattan[node[0]])
            not_visited.append(zeros_neighbor)
        self.fullPath.append(self.start_index)  # Assuming self.fullPath is a list
        parent = {self.start_index: 'S'}
        for adj in self.adj_list[self.start_index]:
            parent[adj] = self.start_index
            visited[adj] = True
        while not_visited:
            not_visited = sorted(not_visited, key=lambda x: x[1])
            current_node = not_visited.pop(0)  # Dequeue the first node from not visited
            self.fullPath.append(current_node[0])  # Append the index of the neighbor
            if current_node[0] == self.end_index:  # If the exit node is reached, terminate the BFS
                break
            # visited[current_node[0]] = True
            if current_node[0] in self.weighted_adj_list:
                for neighbor in self.weighted_adj_list[current_node[0]]:
                    if not visited[neighbor[0]]:
                        visited[neighbor[0]] = True
                        old_heuristic = self.heuristic_Manhattan[current_node[0]]
                        new_heuristic = self.heuristic_Manhattan[neighbor[0]]
                        iso = (neighbor[0], (current_node[1] - old_heuristic) + neighbor[1] + new_heuristic)
                        not_visited.append(iso)  # Append the index of the neighbor
                        parent[neighbor[0]] = current_node[0]

        child = self.end_index
        while child != 'S':
            self.Path.insert(0, child)
            child = parent[child]
            if child != 'S':
                path_cost += self.cost[child]
        end_heuristic = self.heuristic_Manhattan[self.end_index]
        self.totalCost = path_cost + end_heuristic
        return self.Path, self.fullPath, self.totalCost

    def GreedyManhattanHeuristic(self):
        visited = ([False] * self.maze_length * 2)
        not_visited = []
        visited[self.start_index] = True
        path_cost = 0
        for node in self.weighted_adj_list[self.start_index]:
            zeros_neighbor = (node[0], self.heuristic_Manhattan[node[0]])
            not_visited.append(zeros_neighbor)
        self.fullPath.append(self.start_index)  # Assuming self.fullPath is a list
        parent = {self.start_index: 'S'}
        for adj in self.adj_list[self.start_index]:
            parent[adj] = self.start_index
            visited[adj] = True
        while not_visited:
            not_visited = sorted(not_visited, key=lambda x: x[1])
            current_node = not_visited.pop(0)  # Dequeue the first node from not visited
            self.fullPath.append(current_node[0])  # Append the index of the neighbor
            if current_node[0] == self.end_index:  # If the exit node is reached, terminate the BFS
                path_cost = self.heuristic_Manhattan[self.end_index]
                break
            # visited[current_node[0]] = True
            if current_node[0] in self.weighted_adj_list:
                for neighbor in self.weighted_adj_list[current_node[0]]:
                    if not visited[neighbor[0]]:
                        visited[neighbor[0]] = True
                        new_heuristic = self.heuristic_Manhattan[neighbor[0]]
                        iso = (neighbor[0], new_heuristic)
                        path_cost = new_heuristic
                        not_visited.append(iso)  # Append the index of the neighbor
                        parent[neighbor[0]] = current_node[0]

        child = self.end_index
        while child != 'S':
            self.Path.insert(0, child)
            child = parent[child]
        self.totalCost = path_cost
        return self.Path, self.fullPath, self.totalCost

    def GreedyEcludianHeuristic(self):
        visited = ([False] * self.maze_length * 2)
        not_visited = []
        visited[self.start_index] = True
        path_cost = 0
        for node in self.weighted_adj_list[self.start_index]:
            zeros_neighbor = (node[0], self.heuristic_Euclidean[node[0]])
            not_visited.append(zeros_neighbor)
        self.fullPath.append(0)  # Assuming self.fullPath is a list
        parent = {self.start_index: 'S'}
        for adj in self.adj_list[self.start_index]:
            parent[adj] = self.start_index
            visited[adj] = True
        while not_visited:
            not_visited = sorted(not_visited, key=lambda x: x[1])
            current_node = not_visited.pop(0)  # Dequeue the first node from not visited
            self.fullPath.append(current_node[0])  # Append the index of the neighbor
            if current_node[0] == self.end_index:  # If the exit node is reached, terminate the BFS
                path_cost = self.heuristic_Euclidean[self.end_index]
                break
            # visited[current_node[0]] = True
            if current_node[0] in self.weighted_adj_list:
                for neighbor in self.weighted_adj_list[current_node[0]]:
                    if not visited[neighbor[0]]:
                        visited[neighbor[0]] = True
                        new_heuristic = self.heuristic_Euclidean[neighbor[0]]
                        path_cost = new_heuristic
                        iso = (neighbor[0], new_heuristic)
                        not_visited.append(iso)  # Append the index of the neighbor
                        parent[neighbor[0]] = current_node[0]

        child = self.end_index
        while child != 'S':
            self.Path.insert(0, child)
            child = parent[child]
        self.totalCost = path_cost
        return self.Path, self.fullPath, self.totalCost


def main():
    s0 = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.')
    print('adjacency  list: ' + str(s0.adj_list))

    s1 = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.')
    path, fullPath = s1.BFS()
    print('')
    print('BFS Path: ' + str(path), end='\nFull Path is: ')
    print(fullPath)

    s2 = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.')
    path, fullPath = s2.DFS()
    print('')
    print('DFS Path: ' + str(path), end='\nFull Path is: ')
    print(fullPath)

    s3 = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.')
    path_start, path_end = s3.bidirectional_search()
    print('')
    print('bidirectional_Search:\nPath from start to intersection  ' + str(path_start),
          end='\nPath from  intersection to end : ')
    print(path_end)

    s4 = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.')
    depth_limit = 15
    path, fullPath = s4.DFS_with_depth_limit(depth_limit)
    print('')
    if (len(path) == 0):
        print('We can not reach to end with DFS with depth limit = ' + str(depth_limit), end='\nTraversed Path is: ')
    else:
        print('Depth limited Search: ' + str(path), end='\nFull Path is: ')
    print(fullPath)

    s5 = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.')
    path, fullPath, deepth = s5.IDDFS()
    print('')
    print('The exit found at deep = ' + str(deepth))
    print('Iterative Deeping Search Path: ' + str(path), end='\nFull Path is: ')
    print(fullPath)

    s6 = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.',
                          [0, 15, 2, 100, 60, 35, 30, 3
                              , 100, 2, 15, 60, 100, 30, 2
                              , 100, 2, 2, 2, 40, 30, 2, 2
                              , 100, 100, 3, 15, 30, 100, 2
                              , 100, 0, 2, 100, 30])
    path, fullPath, cost = s6.UCS()
    print('')
    print('UCS Path: ' + str(path), end='\nFull Path is: ')
    print(fullPath)
    print('Total Cost: ' + str(cost))

    s7 = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.',
                          [0, 15, 2, 100, 60, 35, 30, 3
                              , 100, 2, 15, 60, 100, 30, 2
                              , 100, 2, 2, 2, 40, 30, 2, 2
                              , 100, 100, 3, 15, 30, 100, 2
                              , 100, 0, 2, 100, 30])
    path, fullPath, cost = s7.AstarEcludianHeuristic()
    print('')
    print('AstarEcludianHeuristic Path: ' + str(path), end='\nFull Path is: ')
    print(fullPath)
    print('Total Cost: ' + str(cost))

    # [1, 1, 1, 1, 1, 1, 1, 1
    #     , 1, 1, 1, 1, 1, 1, 1
    #     , 1, 1, 1, 1, 1,1, 1, 1
    #     , 1, 1, 1, 1, 1, 1, 1
    #     , 1, 1, 1, 1, 1]
    # -----------------------------------
    # [0, 15, 2, 100, 60, 35, 30, 3
    #     , 100, 2, 15, 60, 100, 30, 2
    #     , 100, 2, 2, 2, 40, 30, 2, 2
    #     , 100, 100, 3, 15, 30, 100, 2
    #     , 100, 0, 2, 100, 30]
    # -----------------------------------
    s8 = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.',
                          [1, 1, 1, 1, 1, 1, 1, 1
                              , 1, 1, 1, 1, 1, 1, 1
                              , 1, 1, 1, 1, 1, 1, 1, 1
                              , 1, 1, 1, 1, 1, 1, 1
                              , 1, 1, 1, 1, 1]
                          )
    path, fullPath, cost = s8.AstarManhattanHeuristic()
    print('')
    print('AstarManhattanHeuristic Path: ' + str(path), end='\nFull Path is: ')
    print(fullPath)
    print('Total Cost: ' + str(cost))

    s9 = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.',
                          [0, 15, 2, 100, 60, 35, 30, 3
                              , 100, 2, 15, 60, 100, 30, 2
                              , 100, 2, 2, 2, 40, 30, 2, 2
                              , 100, 100, 3, 15, 30, 100, 2
                              , 100, 0, 2, 100, 30])
    path, fullPath, cost = s9.GreedyManhattanHeuristic()
    print('')
    print('GreedyManhattanHeuristic Path: ' + str(path), end='\nFull Path is: ')
    print(fullPath)
    print('Total Cost: ' + str(cost))

    s10 = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.',
                           [0, 15, 2, 100, 60, 35, 30, 3
                               , 100, 2, 15, 60, 100, 30, 2
                               , 100, 2, 2, 2, 40, 30, 2, 2
                               , 100, 100, 3, 15, 30, 100, 2
                               , 100, 0, 2, 100, 30])
    path, fullPath, cost = s10.GreedyEcludianHeuristic()
    print('')
    print('GreedyEcludianHeuristic Path: ' + str(path), end='\nFull Path is: ')
    print(fullPath)
    print('Total Cost: ' + str(cost))

    s11 = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.',
                           [0, 15, 2, 100, 60, 35, 30, 3
                               , 100, 2, 15, 60, 100, 30, 2
                               , 100, 2, 2, 2, 40, 30, 2, 2
                               , 100, 100, 3, 15, 30, 100, 2
                               , 100, 0, 2, 100, 30])
    path, fullPath, cost = s11.dijkstra()
    print('')
    print('dijkstra Path: ' + str(path), end='\nFull Path is: ')
    print(fullPath)
    print('Total Cost: ' + str(cost))


main()
