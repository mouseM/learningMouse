'''
@Time: 2019/12/31 15:49
@Author: mih
@Des: 
'''
import random
import torch
from gensim.models import Word2Vec
import numpy
from imblearn.over_sampling import SMOTE
from GraphEmbedding_RandomWalk.NN import LineNetwork
from torch import nn
import networkx

class Deep_Walk(nn.Module):
    # walk_length: 随机游走的步数
    # start_node: 随机游走开始的节点
    # A: 图的邻接矩阵
    def __init__(self, G, A, walk_length, embed_size=30, window_size=7, workers=3, iter=10):
        super(Deep_Walk, self).__init__()
        self.walk_length = walk_length
        self.embed_size = embed_size
        # self.G = G
        # self.A = A
        # self.window_size = window_size
        # self.workers = workers
        # self.iter = iter
        # self.nodes = sorted(G.nodes(), key=lambda x: int(x))
        # self.edges = G.edges()
        # self.get_embedding(embed_size=embed_size, window_size=window_size, workers=workers, iter=iter)
        #
        # self.line = LineNetwork(feature_dim = self.embed_size * 2, hidden_layer_dim = self.embed_size, output_dim = 2)
        # self.softMax = nn.Softmax(dim = -1)
        # self.loss = nn.CrossEntropyLoss()
        self.G = networkx.Graph(self.get_edges(A))
        self.A = A
        self.window_size = window_size
        self.workers = workers
        self.iter = iter
        self.nodes = [str(i) for i in range(A.shape[0])]
        self.G.add_nodes_from(self.nodes)
        self.edges = list(self.G.edges())
        self.get_embedding(embed_size=embed_size, window_size=window_size, workers=workers, iter=iter)
        self.line = LineNetwork(input_features=self.embed_size, hidden_features=self.embed_size, output_features=2)
        self.softMax = nn.Softmax(dim=-1)
        self.loss = nn.CrossEntropyLoss()

        # self.edge_generator_forward = nn.Sequential(
        #     nn.Linear(embed_size * 2, embed_size * 2, bias=True),
        #     nn.BatchNorm1d(num_features=embed_size * 2),
        #     nn.ReLU()
        # )
        # self.edge_generator_backward = nn.Sequential(
        #     nn.Linear(embed_size * 2, embed_size * 2, bias=True),
        #     nn.BatchNorm1d(num_features=embed_size * 2),
        #     nn.ReLU()
        # )

    def get_edges(self, A):
        edges = []
        for i, row in enumerate(A):
            for j, col in enumerate(row):
                if (A[i][j] != 0):
                    edges.append((str(i), str(j)))
        numpy.random.shuffle(edges)
        return edges

    def forward(self, *input):
        edges = input[0]
        labels = input[1]
        edges = edges.permute([1, 0])
        src_nodes = edges[0]
        dst_nodes = edges[1]
        src_embeddings = self.word_embeddings.index_select(index=src_nodes, dim=0)
        dst_embeddings = self.word_embeddings.index_select(index=dst_nodes, dim=0)
        # edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)

        # edge_embeddings_forward = torch.cat([src_embeddings, dst_embeddings], dim=-1)
        # edge_embeddings_backward = torch.cat([dst_embeddings, src_embeddings], dim=-1)
        # # 添加一个边的生成层
        # edge_embeddings_forward = self.edge_generator_forward(edge_embeddings_forward)
        # edge_embeddings_backward = self.edge_generator_backward(edge_embeddings_backward)
        # edge_embeddings = edge_embeddings_forward + edge_embeddings_backward

        # 哈达玛积表示边
        edge_embeddings = torch.mul(src_embeddings, dst_embeddings)

        output = self.line(edge_embeddings)
        output = self.softMax(output)
        loss = self.loss(output, labels)
        return loss

    def get_sentence(self, start_node):
        sentence = [start_node]
        current_walk_length = 0
        while (current_walk_length <= self.walk_length):
            current_walker_index = sentence[-1]
            current_walker_index_neighbors = list(self.G.neighbors(current_walker_index))
            if (len(current_walker_index_neighbors) > 0):
                sentence.append(random.choice(current_walker_index_neighbors))
            else:
                # 如果是孤立点
                pass
            current_walk_length = current_walk_length + 1
        return sentence

    def get_sentences(self, nodes):
        sentences = []
        for node in nodes:
            sentences.append(self.get_sentence(node))
        self.sentences = sentences
        return sentences

    def get_embedding(self, embed_size=30, window_size=7, workers=3, iter=10):
        self.get_sentences(self.nodes)
        print("Learning embedding vectors...")
        self.word2Vec = Word2Vec(sentences=self.sentences, size=self.embed_size, min_count = 1, hs=0, sg = 1, workers=self.workers,
                         window=self.window_size, iter=self.iter)
        print("Learning embedding vectors done!")
        self.word_embeddings = []
        for word in self.nodes:
            neighbors = self.G[word]
            if(len(neighbors) > 0):
                self.word_embeddings.append(self.word2Vec.wv[word])
            else:
                self.word_embeddings.append([0.0] * self.embed_size)
        self.word_embeddings = torch.tensor(self.word_embeddings, dtype=torch.float)

    def test(self, edges):
        edges = edges.permute([1, 0])
        src_nodes = edges[0]
        dst_nodes = edges[1]
        src_embeddings = self.word_embeddings.index_select(index=src_nodes, dim=0)
        dst_embeddings = self.word_embeddings.index_select(index=dst_nodes, dim=0)
        # 直接拼接形成边
        # edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)

        # # 使用生成便的一个独立结构
        # edge_embeddings_forward = torch.cat([src_embeddings, dst_embeddings], dim=-1)
        # edge_embeddings_backward = torch.cat([dst_embeddings, src_embeddings], dim=-1)
        # # 添加一个边的生成层
        # edge_embeddings_forward = self.edge_generator_forward(edge_embeddings_forward)
        # edge_embeddings_backward = self.edge_generator_backward(edge_embeddings_backward)
        # edge_embeddings = edge_embeddings_forward + edge_embeddings_backward

        # 哈达玛积表示边
        edge_embeddings = torch.mul(src_embeddings, dst_embeddings)

        output = self.line(edge_embeddings)
        output = self.softMax(output)
        return output


