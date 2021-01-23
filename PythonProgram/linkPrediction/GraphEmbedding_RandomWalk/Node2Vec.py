'''
@Time: 2020/1/9 15:02
@Author: mih
@Des: 
'''
import numpy
import torch
from gensim.models import Word2Vec
from torch import nn
from GraphEmbedding_RandomWalk.NN import LineNetwork
import networkx

class Node2vec(nn.Module):
    def __init__(self, G, A, walk_length, p, q, embed_size, iter, window_size, workers):
        super(Node2vec, self).__init__()
        # self.G = G
        # self.A = A
        # self.walk_length = walk_length
        # self.p = p
        # self.q = q
        # self.embed_size = embed_size
        # self.iter = iter
        # self.window_size = window_size
        # self.workers = workers
        # self.nodes = sorted(G.nodes(), key=lambda x: int(x))
        # self.edges = list(G.edges())
        # self.preprocess_transition_probs()
        # self.get_embeddings(embed_size = embed_size, window_size = window_size, workers = workers, iter = iter)
        #
        # self.line = LineNetwork(input_features=self.embed_size * 2, hidden_features=self.embed_size, output_features=2)
        # self.softMax = nn.Softmax(dim = -1)
        # self.loss = nn.CrossEntropyLoss()


        self.G = networkx.Graph(self.get_edges(A))

        self.A = A
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.embed_size = embed_size
        self.iter = iter
        self.window_size = window_size
        self.workers = workers
        # self.nodes = sorted(self.G.nodes(), key=lambda x: int(x))
        self.nodes = [str(i) for i in range(A.shape[0])]
        self.G.add_nodes_from(self.nodes)
        self.edges = list(self.G.edges())
        self.preprocess_transition_probs()
        self.get_embeddings(embed_size=embed_size, window_size=window_size, workers=workers, iter=iter)

        self.line = LineNetwork(input_features=self.embed_size * 2, hidden_features=self.embed_size, output_features=2)
        self.softMax = nn.Softmax(dim=-1)
        self.loss = nn.CrossEntropyLoss()

        self.edge_generator_forward = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size * 2, bias=True),
            nn.BatchNorm1d(num_features=embed_size * 2),
            nn.ReLU()
        )
        self.edge_generator_backward = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size * 2, bias=True),
            nn.BatchNorm1d(num_features=embed_size * 2),
            nn.ReLU()
        )

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

        edge_embeddings_forward = torch.cat([src_embeddings, dst_embeddings], dim=-1)
        edge_embeddings_backward = torch.cat([dst_embeddings, src_embeddings], dim=-1)
        # 添加一个边的生成层
        edge_embeddings_forward = self.edge_generator_forward(edge_embeddings_forward)
        edge_embeddings_backward = self.edge_generator_backward(edge_embeddings_backward)
        edge_embeddings = edge_embeddings_forward + edge_embeddings_backward

        # 哈达玛积表示边
        # edge_embeddings = torch.mul(src_embeddings, dst_embeddings)

        output = self.line(edge_embeddings)
        output = self.softMax(output)
        loss = self.loss(output, labels)
        return loss

    def create_alias_table(self, area_ratio):
        l = len(area_ratio)
        accept, alias = [0] * l, [0] * l
        small, large = [], []
        area_ratio_ = numpy.array(area_ratio) * l
        for i, prob in enumerate(area_ratio_):
            if prob < 1.0:
                small.append(i)
            else:
                large.append(i)
        while small and large:
            small_idx, large_idx = small.pop(), large.pop()
            accept[small_idx] = area_ratio_[small_idx]
            alias[small_idx] = large_idx
            area_ratio_[large_idx] = area_ratio_[large_idx] - \
                                     (1 - area_ratio_[small_idx])
            if area_ratio_[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)
        while large:
            large_idx = large.pop()
            accept[large_idx] = 1
        while small:
            small_idx = small.pop()
            accept[small_idx] = 1
        return [accept, alias]

    def preprocess_transition_probs(self):
        G = self.G
        self.alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr].get('weight', 1.0)
                                  for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            self.alias_nodes[node] = self.create_alias_table(normalized_probs)
        self.alias_edges = {}

        # alias_edges 用于存储当又走到edge的dst节点时， 下一个节点的游走情况， 返回这个dst节点对应的accept, alias
        # 其中accept 的索引表示直接游走到这个索引节点， alias表示另一个节点
        for edge in self.edges:
            src = edge[0]
            dst = edge[1]
            forward_edge = (src, dst)
            backward_edge = (dst, src)
            self.alias_edges[forward_edge] = self.get_alias_edge(src, dst)
            self.alias_edges[backward_edge] = self.get_alias_edge(dst, src)

    def get_alias_edge(self, t, v):
        G = self.G
        p = self.p
        q = self.q
        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)
            if x == t:
                unnormalized_probs.append(weight / p)
            elif G.has_edge(x, t):
                unnormalized_probs.append(weight)
            else:
                unnormalized_probs.append(weight / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return self.create_alias_table(normalized_probs)

    def get_sentences(self):
        self.sentences = []
        for v in self.nodes:
            self.sentences.append(self.node2vec_walk(walk_length=self.walk_length, start_node=v))

    def node2vec_walk(self, walk_length, start_node):
        G = self.G

        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[self.alias_sample(self.alias_nodes[cur][0], self.alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    # try:
                    alias_edge =  self.alias_edges[edge]
                    src = alias_edge[0]
                    dst = alias_edge[1]
                    next_node = cur_nbrs[self.alias_sample(src, dst)]
                    # except:
                    #     print("edge: {0}".format(edge))
                    #     print("alias_edges: {0}".format(self.alias_edges[edge]))
                    walk.append(next_node)
            else:
                break
        return walk

    def alias_sample(self, accept, alias):
        try:
            N = len(accept)
            i = int(numpy.random.random() * N)
            r = numpy.random.random()
            if r < accept[i]:
                return i
            else:
                return alias[i]
        except:
            print("error in alias sample!")


    def get_embeddings(self, embed_size=30, window_size=7, workers=3, iter=10):
        self.get_sentences()
        print("Learning embedding vectors...")
        self.word2Vec = Word2Vec(sentences=self.sentences, size=self.embed_size, min_count=1, hs=0, sg=1,
                                 workers=self.workers,
                                 window=self.window_size, iter=self.iter)
        print("Learning embedding vectors done!")
        self.word_embeddings = []
        for node in self.G.nodes():
            self.word_embeddings.append(self.word2Vec.wv[node])
        self.word_embeddings = torch.tensor(self.word_embeddings, dtype = torch.float)

    def test(self, edges):
        edges = edges.permute([1, 0])
        src_nodes = edges[0]
        dst_nodes = edges[1]
        src_embeddings = self.word_embeddings.index_select(index=src_nodes, dim=0)
        dst_embeddings = self.word_embeddings.index_select(index=dst_nodes, dim=0)
        # edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)

        edge_embeddings_forward = torch.cat([src_embeddings, dst_embeddings], dim=-1)
        edge_embeddings_backward = torch.cat([dst_embeddings, src_embeddings], dim=-1)
        # 添加一个边的生成层
        edge_embeddings_forward = self.edge_generator_forward(edge_embeddings_forward)
        edge_embeddings_backward = self.edge_generator_backward(edge_embeddings_backward)
        edge_embeddings = edge_embeddings_forward + edge_embeddings_backward

        # 哈达玛积表示边
        # edge_embeddings = torch.mul(src_embeddings, dst_embeddings)

        output = self.line(edge_embeddings)
        output = self.softMax(output)
        return output