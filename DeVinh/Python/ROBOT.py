import math
import time
import numpy as np

from collections import deque

'''
Idea: BFS in tree
0
a[0,0]        |      a[0,1]          |  ...     a[0,n]
a[1,0] a[1,1] | a[1,0] a[1,1] a[1,2] | a[1,n-1] a[1,n]
...
...
'''


class Graph:

    # Initialisation of graph
    def __init__(self, vertices):

        # No. of vertices
        self.vertices = vertices

        # adjacency list
        self.adj = {i: [] for i in range(self.vertices)}

        #
        self.graph = {}

    def addEdge(self, u, v, w):
        # add u to v's list
        self.adj[u].append(v)
        # since the graph is undirected
        self.adj[v].append(u)
        #
        self.graph.setdefault(u, {}).setdefault(v, w)
        self.graph.setdefault(v, {}).setdefault(u, w)

    # method return farthest node and its distance from node u

    def BFS(self, u):
        # marking all nodes as unvisited
        visited = [False for i in range(self.vertices + 1)]
        # mark all distance with -1
        distance = [-1 for i in range(self.vertices + 1)]

        # distance of u from u will be 0
        distance[u] = 0
        # in-built library for queue which performs fast operations on both the ends
        queue = deque()
        queue.append(u)
        # mark node u as visited
        visited[u] = True

        while queue:

            # pop the front of the queue(0th element)
            front = queue.popleft()
            # loop for all adjacent nodes of node front

            for i in self.adj[front]:
                if not visited[i]:
                    # mark the ith node as visited
                    visited[i] = True
                    # make distance of i , one more than distance of front
                    distance[i] = distance[front]+self.graph[front][i]
                    # Push node into the stack only if it is not visited already
                    queue.append(i)

        maxDis = 0

        # get farthest node distance and its index
        for i in range(self.vertices):
            if distance[i] > maxDis:

                maxDis = distance[i]
                nodeIdx = i

        return nodeIdx, maxDis

    # method prints longest path of given tree
    def LongestPathLength(self):

        # first DFS to find one end point of longest path
        node, Dis = self.BFS(0)

        # second DFS to find the actual longest path
        node_2, LongDis = self.BFS(node)

        print('Longest path is from', node, 'to', node_2, 'of length', LongDis)

        return LongDis


def createGraphFromMat(mat, n, m):
    tree = []
    # root to 1 layer
    for i in range(1, n + 1):
        tree.append([(0, 0), (i, 1), mat[i][1]])

    # 1st layer to n
    for i in range(1, n + 1):
        for j in range(1, m):
            tree.append([(i, j), (i, j + 1), mat[i][j + 1]])
            tree.append([(i, j), (i - 1, j + 1), mat[i - 1][j + 1]])
            tree.append([(i, j), (i + 1, j + 1), mat[i - 1][j + 1]])

    vertices = len(tree)
    # print(vertices)
    G = Graph(vertices)  # ((m - 1) * n + 1) * 3

    for i, node in enumerate(tree):
        pass

    # for i in range(1,  m + 1):
    #     for j in range(1, m):
    #         G.addEdge(j, j + 1, mat[i][j+1])
    #         G.addEdge(j, j + 1, mat[i - 1][j+1])
    #         G.addEdge(j, j + 1, mat[i + 1][j+1])
    return tree


def solution(n: int, m: int, matrix):
    # padding with zero
    matrix = np.array(matrix)
    matrix_padded = np.pad(matrix, 1, constant_values=0, mode="constant")
    print(matrix_padded)
    # creat graph
    graph = createGraphFromMat(matrix_padded, n, m)
    print(graph)
    # return graph.LongestPathLength()


if __name__ == "__main__":
    with open('./INP/ROBOT.INP') as f:
        data = [line.replace('\n', '') for line in f.readlines()]

    n, m = data[0].split(" ")
    mat = [[int(value) for value in line.split(" ")] for line in data[1:]]

    result = solution(int(n), int(m), mat)
    print(result)

    with open("./OUT/ROBOT.OUT", 'w') as wf:
        wf.write(str(result))
