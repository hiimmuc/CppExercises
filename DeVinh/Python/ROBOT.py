import math
import time
import numpy as np

'''
Idea: DFS in tree
0
a[0,0]        |      a[0,1]          |  ...     a[0,n]
a[1,0] a[1,1] | a[1,0] a[1,1] a[1,2] | a[1,n-1] a[1,n]
...
...
'''


class Vertex:
    """The vertex used in the graph below"""

    def __init__(self, key, data):
        self.adjancency_list = {}
        self.key = key
        self.data = data
        self.current_cost = 0  # stores own weight added with followers in path

    def connect(self, otherVertex, weight):
        self.adjancency_list[otherVertex] = weight

    def get_connections(self):
        return self.adjancency_list.keys()

    def get_cost(self, vertex):
        return self.adjancency_list[vertex]


class Graph:
    """graph used to find all paths between two nodes using DFS"""

    def __init__(self):
        self.number_of_vertices = 0
        self.vertices = {}

    def add(self, key, data):
        """adds a vertex to graph and saves vertex based on unique key"""
        if key not in self.vertices:
            self.number_of_vertices += 1
            self.vertices[key] = Vertex(key, data)
            return True

        return False

    def addEdge(self, vertex_head, vertex_tail, weight):
        """connects two vertices"""
        if vertex_head in self.vertices and vertex_tail in self.vertices:
            self.vertices[vertex_head].connect(vertex_tail, weight)
            return True

        return False

    def getAllPaths(self, start, end):
        return self.dfs(start, end, [], [], [])

    def getAllPathsSorted(self, start, end):
        res = self.dfs(start, end, [], [], [])
        return sorted(res, key=lambda k: k['cost'])

    def dfs(self, current_vertex, dest_vertex, visited, path, full_path):
        """finds all paths between two nodes, returns all paths with their respective cost"""

        # get vertex, it is now visited and should be added to path
        vertex = self.vertices[current_vertex]
        visited.append(current_vertex)
        path.append(vertex.data)

        # save current path if we found end
        if current_vertex == dest_vertex:
            full_path.append({"path": list(path), "cost": vertex.current_cost})

        for i in vertex.get_connections():
            if i not in visited:
                self.vertices[i].current_cost = vertex.get_cost(
                    i) + vertex.current_cost
                self.dfs(i, dest_vertex, visited, path, full_path)

        # continue finding paths by popping path and visited to get accurate paths
        path.pop()
        visited.pop()

        if not path:
            return full_path


def createGraph(mat, n, m):
    """Calculate all possible cases n x m x 3

    Args:
        matrix (np.array): matrix with padded
        n (int): number of rows in org matrix
        m (int): number of columns in org matrix
    """
    node_set = set()
    tree = [[]]
    # root to 1 layer
    for i in range(1, n + 1):
        tree[-1].extend([[[(0, 0), (i, 1), mat[i][1]]]])

    node_set.add((0, 0))

    # 1st column to m
    for j in range(1, m):
        tree.append([])
        for i in range(1, n + 1):
            tree[-1].extend([[[(i, j), (i - 1, j + 1), mat[i - 1][j + 1]],
                              [(i, j), (i, j + 1), mat[i][j + 1]],
                              [(i, j), (i + 1, j + 1), mat[i + 1][j + 1]]]])
            node_set.add((i, j))
            node_set.add((i - 1, j + 1))
            node_set.add((i, j + 1))
            node_set.add((i + 1, j + 1))

    # size: (m-1) * n  * 3 + n | ex: n = 3, m = 5 -> 39
    # remove redundances
    for i in range(len(tree)):
        for j, edges in enumerate(tree[i]):
            for k, edge in enumerate(edges):
                node0, node1, weight = edge
                if 0 == node1[0] or n + 1 == node1[0]:
                    edges.pop(k)
            tree[i][j] = edges
    # size: (m-1) * n  * 3 + n - m * 2 | ex: n = 3, m = 5 -> 31

    # layer by columns
    print("\nTree structure:")
    for i in range(len(tree)):
        print(f"{i + 1}th layer:", tree[i])

    # create graph
    graph = Graph()
    # add vertices
    for node in node_set:
        graph.add(node, node)

    for i, layer in enumerate(tree):
        for j, edges in enumerate(layer):
            for k, edge in enumerate(edges):
                node0, node1, weight = edge
                graph.addEdge(node0, node1, weight)

    return graph


def evaluateGraph(graph, n, m):
    paths = []
    for i in range(1, n + 1):
        res = graph.getAllPaths((0, 0), (i, m))
        paths.extend(res)

    # sort by cost
    paths.sort(key=lambda x: x['cost'])

    print("\nPossible paths:", len(paths))
    for path in paths:
        print(path)

    return paths[-1]


def solution(n: int, m: int, matrix):
    # padding with zero
    matrix = np.array(matrix)
    matrix_padded = np.pad(matrix, pad_width=1,
                           constant_values=0, mode="constant")
    print("Padded matrix:\n", matrix_padded)
    # create graph
    graph = createGraph(matrix_padded, n, m)
    # compute longest path by DFS
    results = evaluateGraph(graph, n, m)
    return results['cost']


if __name__ == "__main__":
    with open('./INP/ROBOT.INP') as f:
        data = [line.replace('\n', '') for line in f.readlines()]

    n, m = data[0].split(" ")
    mat = [[int(value) for value in line.split(" ")] for line in data[1:]]

    result = solution(int(n), int(m), mat)
    print(result)

    with open("./OUT/ROBOT.OUT", 'w') as wf:
        wf.write(str(result))
