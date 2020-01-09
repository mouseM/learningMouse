'''
@Time: 2019/12/25 19:29
@Author: mih
@Des: 
'''
import scipy.sparse as sp
import numpy
import networkx
import matplotlib.pyplot as plt
import json

# 定义一张用于测试的图, A表示改图的邻接矩阵
A = [
    [0, 1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 0]
]
# 根据图的邻接矩阵绘制改图
def draw_graph(A):
    # 节点数
    N = len(A)
    nodes = [i + 1 for i in range(N)]
    edges = []
    for i in range(N):
        j = i
        while(j < N):
            if A[i][j] == 1:
                edges.append((i + 1, j + 1))
            j = j + 1
    # 绘制无向图
    G = networkx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    for edge in G.edges:
        print(edge)
    networkx.draw(G, with_labels=True)
    plt.show()

def Test3():
    G = networkx.Graph()
    G.add_edges_from([[1, 2], [2, 1]])
    for edge in G.edges:
        print(edge)
def Test1():
    A_ = sp.csr_matrix(A, dtype=int)
    print(A_)
def Test2():
    A_ = numpy.array(A).flatten()
    print(A_)

def JsonRead():
    with open(r"./params.json", 'r') as f:
        params = json.load(f)
        batchSize = params["batchSize"]
        epochs = params["epochs"]
        print("batchSize: {0}".format(batchSize))
        print("epochs: {0}".format(epochs))

if __name__ == '__main__':
    Test3()