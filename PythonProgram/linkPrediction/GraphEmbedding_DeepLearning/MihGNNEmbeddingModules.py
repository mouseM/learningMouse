import torch
from torch import nn
import numpy
from torch.nn import Parameter
from Tools import Matrix_pre_handle
import math
from GraphEmbedding_DeepLearning.NN import LineNetwork
from GraphEmbedding_DeepLearning.AggregationModule import GNNAggregationModule
from GraphEmbedding_DeepLearning.AggregationModule import MihGNNAggregationModule
from GraphEmbedding_DeepLearning.AggregationModule import MihGNNAggregationModule2
from GraphEmbedding_DeepLearning.AggregationModule import MihGNNAggregationModuleWithConv1d
from GraphEmbedding_DeepLearning.AggregationModule import GNNAggregationForNegativeModule
from GraphEmbedding_DeepLearning.AggregationModule import GNNAggregationWithAttentionModule
from GraphEmbedding_DeepLearning.AggregationModule import GCNAggregationModule
from GraphEmbedding_DeepLearning.LossModule import MihOutputModule1
from GraphEmbedding_DeepLearning.LossModule import MihOutputModule2
from GraphEmbedding_RandomWalk.Node2Vec import Node2vec
"""
GNN 优化的目标函数是：labaels - math.e ** (-sum((x_i - x_j) ** 2))
改进负样本的嵌入状态更新
"""
class MihGNNEmbedding1(nn.Module):
    def __init__(self,A, As, all_nodes_neighbors, N, d, layers, steps, delay, GPU = False):
        super(MihGNNEmbedding1, self).__init__()
        self.A = Matrix_pre_handle(A, steps, delay)
        self.A = torch.tensor(self.A, dtype = torch.float)
        self.d = d
        self.e = torch.tensor(math.e, dtype=torch.float)
        self.layers = layers
        embedding_state = numpy.random.randn(N, d)
        embedding_state = torch.tensor(data=embedding_state, dtype=torch.float)
        self.aggregationModule = GNNAggregationForNegativeModule(A=self.A, As = As,
                                                                 all_nodes_neighbors = all_nodes_neighbors,
                                                                 convolution_layers=layers, d=d,
                                                                 embedding_states=embedding_state)
        self.lossFunction = MihOutputModule1(d = self.d, e = self.e, layers = layers)


    def forward(self, *input):
        pairs = input[0]
        labels = input[1]

        node_indexes = pairs.permute([1, 0])
        src_node_indexes = node_indexes[0]
        dst_node_indexes = node_indexes[1]
        ordered_labels, src_node_embeddings = self.aggregationModule(labels, src_node_indexes)
        ordered_labels, dst_node_embeddings = self.aggregationModule(labels, dst_node_indexes)
        loss = self.lossFunction(ordered_labels, src_node_embeddings, dst_node_embeddings)
        return loss

    def test(self, edges, preference="positive"):
        edges = edges.permute([1, 0])
        src_node_indexes = edges[0]
        dst_node_indexes = edges[1]
        if (preference == "positive"):
            preference_labels = torch.ones_like(src_node_indexes, dtype=torch.long)
        if (preference == "negative"):
            preference_labels = torch.zeros_like(src_node_indexes, dtype=torch.long)
        _, src_node_embeddings = self.aggregationModule(preference_labels, src_node_indexes)
        _, dst_node_embeddings = self.aggregationModule(preference_labels, dst_node_indexes)
        differcences_sum = (src_node_embeddings - dst_node_embeddings) ** 2
        differcences_sum = torch.sum(differcences_sum, dim=1) / (self.d * self.layers)
        predicts = self.e ** (-differcences_sum)
        return predicts


"""
GNN 优化的目标函数是：labaels - math.e ** (-sum((x_i - x_j) ** 2))
"""
class MihGNNEmbedding2(nn.Module):
    def __init__(self,A, As, all_nodes_neighbors, N, d, layers, steps, delay, GPU = False):
        super(MihGNNEmbedding2, self).__init__()
        self.A = Matrix_pre_handle(A, steps, delay)
        self.A = torch.tensor(self.A, dtype = torch.float)
        self.d = d
        self.e = torch.tensor(math.e, dtype=torch.float)
        self.layers = layers
        self.As = As
        self.all_nodes_neighbors = all_nodes_neighbors
        embedding_state = numpy.random.randn(N, d)
        embedding_state = torch.tensor(data=embedding_state, dtype=torch.float)
        self.aggregationModule = GNNAggregationModule(A = self.A, As=As, all_nodes_neighbors = all_nodes_neighbors,
                                                      convolution_layers=layers, d=d,
                                                      embedding_states=embedding_state)
        self.lossFunction = MihOutputModule1(d = self.d, e = self.e, layers = layers)

    def forward(self, *input):
        pairs = input[0]
        labels = input[1]

        node_indexes = pairs.permute([1, 0])
        src_node_indexes = node_indexes[0].numpy()
        dst_node_indexes = node_indexes[1].numpy()
        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        loss = self.lossFunction(labels, src_node_embeddings, dst_node_embeddings)
        return loss

    def test(self, edges):
        edges = edges.permute([1, 0])
        src_node_indexes = edges[0]
        dst_node_indexes = edges[1]

        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        differcences_sum = (src_node_embeddings - dst_node_embeddings) ** 2
        differcences_sum = torch.sum(differcences_sum, dim=1) / (self.d * self.layers)
        predicts = self.e ** (-differcences_sum)
        return predicts

# GNN 使用余弦相似度作为目标函数
class MihGNNEmbedding3(nn.Module):
    def __init__(self, A, As, all_nodes_neighbors, N, d, layers, steps, delay, GPU = False):
        super(MihGNNEmbedding3, self).__init__()
        self.A = torch.tensor(Matrix_pre_handle(A, steps=steps, delay=delay), dtype=torch.float)
        embedding_state = numpy.random.randn(N, d)
        embedding_state = torch.tensor(data=embedding_state, dtype=torch.float)
        self.aggregationModule = GNNAggregationModule(A=self.A, As= As, all_nodes_neighbors = all_nodes_neighbors,
                                                      convolution_layers=layers, d=d,
                                                      embedding_states=embedding_state)
        self.lossFunction = MihOutputModule2()
        self.relu = nn.ReLU()

    def forward(self, *input):
        pairs = input[0]
        labels = input[1]
        node_indexes = pairs.permute([1, 0])
        src_node_indexes = node_indexes[0]
        dst_node_indexes = node_indexes[1]
        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        loss = self.lossFunction(labels, src_node_embeddings, dst_node_embeddings)
        return loss

    def test(self, edges):
        edges = edges.permute([1, 0])
        src_node_indexes = edges[0]
        dst_node_indexes = edges[1]
        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        predicts = torch.sum(src_node_embeddings * dst_node_embeddings, dim=0, keepdim=False) / (
                torch.norm(src_node_embeddings, p=2) * torch.norm(dst_node_embeddings, p=2))
        return predicts

# 引入注意力机制节点j对节点i的注意力为f(h_i, h_j), h_i, h_j为嵌入状态， f可以是一个全连接层， 输入维度是2 * D， 输出维度是1
class MihGNNEmbedding4(nn.Module):
    def __init__(self, A, As, all_nodes_neighbors, N, d, layers, steps, delay, GPU = False):
        super(MihGNNEmbedding4, self).__init__()
        self.A = torch.tensor(Matrix_pre_handle(A, steps=steps, delay=delay), dtype=torch.float)
        embedding_state = numpy.random.randn(N, d)
        embedding_state = torch.tensor(data=embedding_state, dtype=torch.float)
        self.aggregationModule = GNNAggregationWithAttentionModule(A=self.A, As = As,
                                                                   all_nodes_neighbors = all_nodes_neighbors,
                                                                   convolution_layers=layers, d=d,
                                                                   embedding_states=embedding_state)
        self.liner = LineNetwork(input_features=d * 2, output_features=2, hidden_features=d)
        self.soft_max = nn.Softmax(dim = -1)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, *input):
        pairs = input[0]
        labels = input[1]
        node_indexes = pairs.permute([1, 0])
        src_node_indexes = node_indexes[0]
        dst_node_indexes = node_indexes[1]
        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        liner_input = torch.cat([src_node_embeddings, dst_node_embeddings], dim=1)
        output = self.liner(liner_input)
        output = self.soft_max(output)
        loss = self.cross_entropy(output, labels)
        return loss

    def test(self, edges):
        edges = edges.permute([1, 0])
        src_node_indexes = edges[0]
        dst_node_indexes = edges[1]
        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        liner_input = torch.cat([src_node_embeddings, dst_node_embeddings], dim=1)
        predictions = self.liner(liner_input)
        return predictions

# 使用cross_entry损失函数， 前提保证训练数据的正负样本数量相同
class MihGNNEmbedding5(nn.Module):
    def __init__(self, A, As, all_nodes_neighbors, N, d, layers, steps, delay, GPU = False):
        super(MihGNNEmbedding5, self).__init__()
        self.A = torch.tensor(Matrix_pre_handle(A, steps=steps, delay=delay), dtype=torch.float)
        embedding_state = numpy.random.randn(N, d)
        embedding_state = torch.tensor(data = embedding_state, dtype=torch.float)
        self.aggregationModule = GNNAggregationForNegativeModule(A = self.A, As = As,
                                                                 all_nodes_neighbors = all_nodes_neighbors,
                                                                 convolution_layers = layers, d = d,
                                                                 embedding_states = embedding_state)
        self.liner = LineNetwork(input_features = d * 2, output_features = 2, hidden_features = d)
        self.soft_max = nn.Softmax(dim = 1)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.add_module(name = "AggregationModule", module = self.aggregationModule)
        self.add_module(name = "FullConnectionLiner", module = self.liner)


    def forward(self, *input):
        pairs = input[0]
        labels = input[1]
        node_indexes = pairs.permute([1, 0])
        src_node_indexes = node_indexes[0]
        dst_node_indexes = node_indexes[1]
        ordered_labels, src_node_embeddings = self.aggregationModule(labels, src_node_indexes)
        ordered_labels, dst_node_embeddings = self.aggregationModule(labels, dst_node_indexes)
        liner_input = torch.cat([src_node_embeddings, dst_node_embeddings], dim = 1)
        output = self.liner(liner_input)
        output = self.soft_max(output)
        loss = self.cross_entropy(output, labels)
        return loss


    def test(self, edges, preference = "negative"):
        edges = edges.permute([1, 0])
        src_node_indexes = edges[0]
        dst_node_indexes = edges[1]
        if (preference == "positive"):
            preference_labels = torch.ones_like(src_node_indexes, dtype = torch.long)
        if (preference == "negative"):
            preference_labels = torch.zeros_like(src_node_indexes, dtype = torch.long)
        _, src_node_embeddings = self.aggregationModule(preference_labels, src_node_indexes)
        _, dst_node_embeddings = self.aggregationModule(preference_labels, dst_node_indexes)
        liner_input = torch.cat([src_node_embeddings, dst_node_embeddings], dim = 1)
        predictions = self.soft_max(liner_input)
        predictions = self.liner(liner_input)
        return predictions
"""
GNN 模型 + cross_entory损失函数
将所有卷积层相加
"""
class MihGNNEmbedding6(nn.Module):
    def __init__(self,A, As, all_nodes_neighbors, N, d, layers, steps, delay, GPU = False):
        super(MihGNNEmbedding6, self).__init__()
        self.A = torch.tensor(Matrix_pre_handle(A, steps=steps, delay=delay), dtype=torch.float)
        embedding_states = numpy.random.randn(N, d)
        embedding_states = torch.tensor(data = embedding_states, dtype=torch.float)
        self.aggregationModule = GNNAggregationModule(A = self.A, As = As, all_nodes_neighbors = all_nodes_neighbors,
                                                      convolution_layers = layers, d = d,
                                                      embedding_states = embedding_states)
        self.liner = LineNetwork(input_features = d * 2, output_features = 2, hidden_features = d)
        self.soft_max = nn.Softmax(dim=1)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.add_module(name = "AggregationModule", module = self.aggregationModule)
        self.add_module(name = "FullConnectionLiner", module = self.liner)


    def forward(self, *input):
        pairs = input[0]
        labels = input[1]
        node_indexes = pairs.permute([1, 0])
        src_node_indexes = node_indexes[0]
        dst_node_indexes = node_indexes[1]
        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        liner_input = torch.cat([src_node_embeddings, dst_node_embeddings], dim = 1)
        output = self.liner(liner_input)
        output = self.soft_max(output)
        loss = self.cross_entropy(output, labels)
        return loss


    def test(self, edges):
        edges = edges.permute([1, 0])
        src_node_indexes = edges[0]
        dst_node_indexes = edges[1]
        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        liner_input = torch.cat([src_node_embeddings, dst_node_embeddings], dim = 1)
        predictions = self.soft_max(liner_input)
        predictions = self.liner(liner_input)
        return predictions

# 使用谱域的GCN模型
class MihGNNEmbedding7(nn.Module):
    def __init__(self, A, N, d, layers, steps, delay, GPU = False):
        super(MihGNNEmbedding7, self).__init__()
        I = numpy.eye(N)
        A = torch.tensor(A + I, dtype=torch.float)
        embedding_state = numpy.random.randn(N, d)
        embedding_state = torch.tensor(data = embedding_state, dtype=torch.float)
        self.aggregationModule = GCNAggregationModule(A = A, convolution_layers = layers, d = d, embedding_states = embedding_state)
        self.liner = LineNetwork(input_features = d * 2, output_features = 2, hidden_features = d)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.add_module(name = "AggregationModule", module = self.aggregationModule)
        self.add_module(name = "FullConnectionLiner", module = self.liner)


    def forward(self, *input):
        pairs = input[0]
        labels = input[1]
        node_indexes = pairs.permute([1, 0])
        src_node_indexes = node_indexes[0]
        dst_node_indexes = node_indexes[1]
        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        liner_input = torch.cat([src_node_embeddings, dst_node_embeddings], dim = 1)
        output = self.liner(liner_input)
        loss = self.cross_entropy(output, labels)
        return loss


    def test(self, edges):
        edges = edges.permute([1, 0])
        src_node_indexes = edges[0]
        dst_node_indexes = edges[1]
        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        liner_input = torch.cat([src_node_embeddings, dst_node_embeddings], dim = 1)
        predictions = self.liner(liner_input)
        return predictions

"""
GNN 优化的目标函数是：labaels - math.e ** (-sum((x_i - x_j) ** 2))
"""
class MihGNNEmbedding8(nn.Module):
    def __init__(self,A, As, all_nodes_neighbors, N, d, layers, steps, delay, GPU = False):
        super(MihGNNEmbedding8, self).__init__()
        self.A = Matrix_pre_handle(A, steps, delay)
        self.A = torch.tensor(self.A, dtype = torch.float)
        self.d = d
        self.e = torch.tensor(math.e, dtype=torch.float)
        self.layers = layers
        self.As = As
        self.all_nodes_neighbors = all_nodes_neighbors
        embedding_state = numpy.random.randn(N, d)
        embedding_state = torch.tensor(data=embedding_state, dtype=torch.float)
        self.aggregationModule = MihGNNAggregationModule(A = self.A, As=As, all_nodes_neighbors = all_nodes_neighbors,
                                                      convolution_layers=layers, d=d,
                                                      embedding_states=embedding_state)
        self.lossFunction = MihOutputModule1(d = self.d, e = self.e, layers = layers)

    def forward(self, *input):
        pairs = input[0]
        labels = input[1]

        node_indexes = pairs.permute([1, 0])
        src_node_indexes = node_indexes[0]
        dst_node_indexes = node_indexes[1]
        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        loss = self.lossFunction(labels, src_node_embeddings, dst_node_embeddings)
        return loss

    def test(self, edges):
        edges = edges.permute([1, 0])
        src_node_indexes = edges[0]
        dst_node_indexes = edges[1]

        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        differcences_sum = (src_node_embeddings - dst_node_embeddings) ** 2
        differcences_sum = torch.sum(differcences_sum, dim=1) / (self.d * self.layers)
        predicts = self.e ** (-differcences_sum)
        return predicts

# 使用谱域的GCN模型
class MihGNNEmbedding9(nn.Module):
    def __init__(self, A, As, all_nodes_neighbors, N, d, layers, steps, delay, GPU = False):
        super(MihGNNEmbedding9, self).__init__()

        # self.A = Matrix_pre_handle(A, steps, delay)
        # self.A = torch.tensor(self.A, dtype = torch.float)

        # I = numpy.eye(N)
        # self.A = torch.tensor(A + 5 * I, dtype=torch.float)

        self.A = torch.tensor(A, dtype = torch.float)
        self.A = self.pre_handle(self.A)

        self.layers = layers
        embedding_state = numpy.random.randn(N, d)
        embedding_state = torch.tensor(data = embedding_state, dtype=torch.float)

        self.tanh = nn.Tanh()
        self.embedding_state = embedding_state
        self.aggregationLayers = nn.Sequential()
        layers_embeddings = []
        for layer in range(layers):
            aggregation_layer = nn.Linear(in_features = d, out_features = d)
            self.aggregationLayers.add_module(name = "aggregation_layer_{0}".format(layer + 1), module = aggregation_layer)
            self.embedding_state = aggregation_layer(self.embedding_state)
            self.embedding_state = torch.matmul(self.A, self.embedding_state)
            self.embedding_state = self.tanh(self.embedding_state)
            # layers_embeddings.append(self.embedding_state * delay[layer])



        self.liner = LineNetwork(input_features = d * 2, output_features = 2, hidden_features = d)
        self.softMax = nn.Softmax(dim = -1)
        self.cross_entropy = nn.CrossEntropyLoss()



    def pre_handle(self, A):
        D = torch.sum(A, dim = -1, keepdim = False)
        D = D ** (-1 / 2)
        D = torch.diag(D)

        L = torch.matmul(D, A)
        L = torch.matmul(L, D)
        return L

    def forward(self, *input):
        pairs = input[0]
        labels = input[1]
        node_indexes = pairs.permute([1, 0])
        src_node_indexes = node_indexes[0]
        dst_node_indexes = node_indexes[1]
        src_node_embeddings = self.embedding_state.index_select(dim = 0, index = src_node_indexes)
        dst_node_embeddings = self.embedding_state.index_select(dim = 0, index = dst_node_indexes)
        liner_input = torch.cat([src_node_embeddings, dst_node_embeddings], dim = 1)
        output = self.liner(liner_input)
        output = self.softMax(output)
        loss = self.cross_entropy(output, labels)
        return loss


    def test(self, edges):
        edges = edges.permute([1, 0])
        src_node_indexes = edges[0]
        dst_node_indexes = edges[1]
        src_node_embeddings = self.embedding_state.index_select(dim=0, index=src_node_indexes)
        dst_node_embeddings = self.embedding_state.index_select(dim=0, index=dst_node_indexes)
        liner_input = torch.cat([src_node_embeddings, dst_node_embeddings], dim = 1)
        predictions = self.liner(liner_input)
        predictions = self.softMax(predictions)
        return predictions

class MihGNNEmbedding10(nn.Module):
    def __init__(self,A, As, all_nodes_neighbors, N, d, layers, steps, delay, GPU = False):
        super(MihGNNEmbedding10, self).__init__()
        self.A = Matrix_pre_handle(A, steps, delay)
        self.A = torch.tensor(self.A, dtype = torch.float)
        self.d = d
        self.e = torch.tensor(math.e, dtype=torch.float)
        self.layers = layers
        self.As = As
        self.all_nodes_neighbors = all_nodes_neighbors
        embedding_state = numpy.random.randn(N, d)
        embedding_state = torch.tensor(data=embedding_state, dtype=torch.float)
        self.aggregationModule = MihGNNAggregationModule(A = self.A, As=As, all_nodes_neighbors = all_nodes_neighbors,
                                                      convolution_layers=layers, d=d,
                                                      embedding_states=embedding_state)
        self.liner = LineNetwork(input_features=d * 2, output_features=2, hidden_features=d)
        # self.soft_max = nn.Softmax(dim = -1)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, *input):
        pairs = input[0]
        labels = input[1]
        node_indexes = pairs.permute([1, 0])
        src_node_indexes = node_indexes[0]
        dst_node_indexes = node_indexes[1]
        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        node_embeddings = torch.cat([src_node_embeddings, dst_node_embeddings], dim = -1)
        predictions = self.liner(node_embeddings)
        # predictions = self.soft_max(predictions)
        loss = self.cross_entropy(predictions, labels)
        return loss

    def test(self, edges):
        edges = edges.permute([1, 0])
        src_node_indexes = edges[0]
        dst_node_indexes = edges[1]

        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        node_embeddings = torch.cat([src_node_embeddings, dst_node_embeddings], dim=-1)
        predictions = self.liner(node_embeddings)
        predictions = self.soft_max(predictions)
        return predictions

class MihGNNEmbedding11(nn.Module):
    def __init__(self,A, As, all_nodes_neighbors, N, d, layers, steps, delay, GPU = False):
        super(MihGNNEmbedding11, self).__init__()
        self.A = Matrix_pre_handle(A, steps, delay)
        self.A = torch.tensor(self.A, dtype = torch.float)
        self.d = d
        self.e = torch.tensor(math.e, dtype=torch.float)
        self.layers = layers
        self.As = As
        self.all_nodes_neighbors = all_nodes_neighbors
        embedding_state = numpy.random.randn(N, d)
        embedding_state = torch.tensor(data=embedding_state, dtype=torch.float)
        self.aggregationModule = MihGNNAggregationModuleWithConv1d(A = self.A, As=As, all_nodes_neighbors = all_nodes_neighbors,
                                                      convolution_layers=layers, d=d,
                                                      embedding_states=embedding_state)
        self.liner = LineNetwork(input_features=d, output_features=2, hidden_features=d)
        self.soft_max = nn.Softmax(dim = -1)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, *input):
        pairs = input[0]
        labels = input[1]
        node_indexes = pairs.permute([1, 0])
        src_node_indexes = node_indexes[0]
        dst_node_indexes = node_indexes[1]
        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        node_embeddings = torch.cat([src_node_embeddings, dst_node_embeddings], dim = -1)
        predictions = self.liner(node_embeddings)
        predictions = self.soft_max(predictions)
        loss = self.cross_entropy(predictions, labels)
        return loss

    def test(self, edges):
        edges = edges.permute([1, 0])
        src_node_indexes = edges[0]
        dst_node_indexes = edges[1]

        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        node_embeddings = torch.cat([src_node_embeddings, dst_node_embeddings], dim=-1)
        predictions = self.liner(node_embeddings)
        predictions = self.soft_max(predictions)
        return predictions

class MihGNNEmbedding12(nn.Module):
    def __init__(self,A, As, all_nodes_neighbors, N, d, layers, steps, delay, weight, GPU = False):
        super(MihGNNEmbedding12, self).__init__()
        self.A = Matrix_pre_handle(A, steps, delay)
        self.A = torch.tensor(self.A, dtype = torch.float)
        self.d = d
        self.e = torch.tensor(math.e, dtype=torch.float)
        self.layers = layers
        self.As = As
        self.weight = torch.tensor(weight, dtype = torch.float)
        self.all_nodes_neighbors = all_nodes_neighbors
        embedding_state = numpy.random.randn(N, d)
        embedding_state = torch.tensor(data=embedding_state, dtype=torch.float)
        # embedding_state = torch.randn(size=[N, d], dtype=torch.float)
        self.aggregationModule = MihGNNAggregationModule2(A = self.A, As=As, all_nodes_neighbors = all_nodes_neighbors,
                                                      convolution_layers=layers, d=d,
                                                      embedding_states=embedding_state)
        self.liner = LineNetwork(input_features=d * 2, output_features=2, hidden_features=d)
        self.soft_max = nn.Softmax(dim = -1)
        # self.cross_entropy = nn.CrossEntropyLoss(weight = self.weight, reduction = 'sum')
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, *input):
        pairs = input[0]
        labels = input[1]
        node_indexes = pairs.permute([1, 0])
        src_node_indexes = node_indexes[0]
        dst_node_indexes = node_indexes[1]
        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        node_embeddings = torch.cat([src_node_embeddings, dst_node_embeddings], dim = -1)
        predictions = self.liner(node_embeddings)
        # predictions = self.soft_max(predictions)
        loss = self.cross_entropy(predictions, labels)
        return loss

    def test(self, edges):
        edges = edges.permute([1, 0])
        src_node_indexes = edges[0]
        dst_node_indexes = edges[1]

        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        node_embeddings = torch.cat([src_node_embeddings, dst_node_embeddings], dim=-1)
        predictions = self.liner(node_embeddings)
        predictions = self.soft_max(predictions)
        return predictions

class MihGNNEmbedding12WithJaccard(nn.Module):
    def __init__(self,A, As, all_nodes_neighbors, N, d, layers, steps, delay, weight, GPU = False):
        super(MihGNNEmbedding12WithJaccard, self).__init__()
        self.original_A = torch.tensor(A, dtype=torch.float)
        self.one_hop_matrix = torch.matmul(self.original_A, self.original_A)
        self.A = Matrix_pre_handle(A, steps, delay)
        self.A = torch.tensor(self.A, dtype = torch.float)
        self.d = d
        self.e = torch.tensor(math.e, dtype=torch.float)
        self.layers = layers
        self.As = As
        self.weight = torch.tensor(weight, dtype = torch.float)
        self.all_nodes_neighbors = all_nodes_neighbors
        embedding_state = numpy.random.randn(N, d)
        embedding_state = torch.tensor(data=embedding_state, dtype=torch.float)
        self.aggregationModule = MihGNNAggregationModule2(A = self.A, As=As, all_nodes_neighbors = all_nodes_neighbors,
                                                      convolution_layers=layers, d=d,
                                                      embedding_states=embedding_state)
        self.edge_generator_forward  = nn.Sequential(
            nn.Linear(d * 2, d * 2, bias=True),
            nn.BatchNorm1d(num_features = d * 2),
            nn.ReLU(),
            nn.Linear(d * 2, d * 2, bias=True),
            nn.BatchNorm1d(num_features=d * 2),
            nn.ReLU()
        )
        # self.edge_generator_backward = nn.Sequential(
        #     nn.Linear(d * 2, d * 2, bias=True),
        #     nn.BatchNorm1d(num_features=d * 2),
        #     nn.ReLU()
        # )
        self.liner = LineNetwork(input_features=d * 2, output_features=2, hidden_features=d)
        self.soft_max = nn.Softmax(dim = -1)
        self.cross_entropy = nn.CrossEntropyLoss(weight = self.weight, reduction = 'sum')
        # self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, *input):
        pairs = input[0]
        labels = input[1]
        node_indexes = pairs.permute([1, 0])
        src_node_indexes = node_indexes[0]
        dst_node_indexes = node_indexes[1]
        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)

        edge_embeddings_forward = torch.cat([src_node_embeddings, dst_node_embeddings], dim = -1)
        edge_embeddings_backward = torch.cat([dst_node_embeddings, src_node_embeddings], dim = -1)
        # 添加一个边的生成层
        edge_embeddings_forward = self.edge_generator_forward(edge_embeddings_forward)
        edge_embeddings_backward = self.edge_generator_forward(edge_embeddings_backward)
        edge_embeddings = edge_embeddings_forward + edge_embeddings_backward

        # edge_embeddings = torch.mul(src_node_embeddings,dst_node_embeddings)

        predictions = self.liner(edge_embeddings)

        loss = self.cross_entropy(predictions, labels)
        return loss

    def test(self, edges):
        edges = edges.permute([1, 0])
        src_node_indexes = edges[0]
        dst_node_indexes = edges[1]

        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        edge_embeddings_forward = torch.cat([src_node_embeddings, dst_node_embeddings], dim=-1)
        edge_embeddings_backward = torch.cat([dst_node_embeddings, src_node_embeddings], dim=-1)
        # 添加一个边的生成层
        edge_embeddings_forward = self.edge_generator_forward(edge_embeddings_forward)
        edge_embeddings_backward = self.edge_generator_forward(edge_embeddings_backward)
        edge_embeddings = edge_embeddings_forward + edge_embeddings_backward

        # edge_embeddings = torch.mul(src_node_embeddings, dst_node_embeddings)

        predictions = self.liner(edge_embeddings)
        predictions = self.soft_max(predictions)
        return predictions

class MihGNNEmbedding12WithTrainWeight(nn.Module):
    def __init__(self,A, As, all_nodes_neighbors, N, d, layers, steps, delay, GPU = False):
        super(MihGNNEmbedding12WithTrainWeight, self).__init__()
        self.A = Matrix_pre_handle(A, steps, delay)
        self.A = torch.tensor(self.A, dtype = torch.float)
        self.d = d
        self.e = torch.tensor(math.e, dtype=torch.float)
        self.layers = layers
        self.As = As
        self.weight = torch.tensor([1, 1], dtype = torch.float)
        self.weight = Parameter(self.weight, requires_grad = True)
        self.all_nodes_neighbors = all_nodes_neighbors
        embedding_state = numpy.random.randn(N, d)
        embedding_state = torch.tensor(data=embedding_state, dtype=torch.float)
        self.aggregationModule = MihGNNAggregationModule2(A = self.A, As=As, all_nodes_neighbors = all_nodes_neighbors,
                                                      convolution_layers=layers, d=d,
                                                      embedding_states=embedding_state)
        self.liner = LineNetwork(input_features=d * 2, output_features=2, hidden_features=d)
        self.soft_max = nn.Softmax(dim = -1)

        self.loss = nn.NLLLoss()
        # self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, *input):
        pairs = input[0]
        labels = input[1]
        node_indexes = pairs.permute([1, 0])
        src_node_indexes = node_indexes[0]
        dst_node_indexes = node_indexes[1]
        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        node_embeddings = torch.cat([src_node_embeddings, dst_node_embeddings], dim = -1)
        predictions = self.liner(node_embeddings)
        predictions = self.soft_max(predictions)
        predictions = torch.log(predictions)
        predictions = predictions * self.weight
        loss = self.loss(predictions, labels)
        return loss

    def test(self, edges):
        edges = edges.permute([1, 0])
        src_node_indexes = edges[0]
        dst_node_indexes = edges[1]

        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        node_embeddings = torch.cat([src_node_embeddings, dst_node_embeddings], dim=-1)
        predictions = self.liner(node_embeddings)
        predictions = self.soft_max(predictions)
        predictions = torch.log(predictions)
        predictions = predictions * self.weight
        return predictions

class MihGNNEmbedding12WithNoAggregation(nn.Module):
    def __init__(self, N, d, layers, GPU = False):
        super(MihGNNEmbedding12WithNoAggregation, self).__init__()
        self.d = d
        self.e = torch.tensor(math.e, dtype=torch.float)
        self.layers = layers
        embedding_state = numpy.random.randn(N, d)
        self.embedding_state = torch.tensor(data=embedding_state, dtype=torch.float)

        self.liner = LineNetwork(input_features=d * 2, output_features=2, hidden_features=d)
        # self.soft_max = nn.Softmax(dim = -1)
        # self.cross_entropy = nn.CrossEntropyLoss(weight = self.weight, reduction = 'sum')
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, *input):
        pairs = input[0]
        labels = input[1]
        node_indexes = pairs.permute([1, 0])
        src_node_indexes = node_indexes[0]
        dst_node_indexes = node_indexes[1]
        src_node_embeddings = self.embedding_state.index_select(dim = 0, index = src_node_indexes)
        dst_node_embeddings = self.embedding_state.index_select(dim = 0, index = dst_node_indexes)
        node_embeddings = torch.cat([src_node_embeddings, dst_node_embeddings], dim = -1)
        predictions = self.liner(node_embeddings)
        loss = self.cross_entropy(predictions, labels)
        return loss

    def test(self, edges):
        edges = edges.permute([1, 0])
        src_node_indexes = edges[0]
        dst_node_indexes = edges[1]
        src_node_embeddings = self.embedding_state.index_select(dim=0, index=src_node_indexes)
        dst_node_embeddings = self.embedding_state.index_select(dim=0, index=dst_node_indexes)
        node_embeddings = torch.cat([src_node_embeddings, dst_node_embeddings], dim=-1)
        predictions = self.liner(node_embeddings)
        return predictions

class MihGNNEmbedding12AferRandomWalk(nn.Module):
    def __init__(self,G, A, As, all_nodes_neighbors, N, d, walk_length, p, q, iter, window_size, workers, layers, steps, delay, GPU = False):
        super(MihGNNEmbedding12AferRandomWalk, self).__init__()
        self.A = Matrix_pre_handle(A, steps, delay)
        self.A = torch.tensor(self.A, dtype = torch.float)
        self.d = d
        self.e = torch.tensor(math.e, dtype=torch.float)
        self.layers = layers
        self.As = As
        self.all_nodes_neighbors = all_nodes_neighbors
        print("Getting word embedding!")
        node2Vec = Node2vec(G =G , A = A, walk_length = walk_length, p = p, q = q, embed_size = d, iter = iter, window_size = window_size, workers = workers)
        embedding_state = node2Vec.word_embeddings
        # embedding_state = numpy.random.randn(N, d)
        # embedding_state = torch.tensor(data=embedding_state, dtype=torch.float)
        self.aggregationModule = MihGNNAggregationModule2(A = self.A, As=As, all_nodes_neighbors = all_nodes_neighbors,
                                                      convolution_layers=layers, d=d,
                                                      embedding_states=embedding_state)
        self.liner = LineNetwork(input_features=d * 2, output_features=2, hidden_features=d)
        self.soft_max = nn.Softmax(dim = -1)
        # self.cross_entropy = nn.CrossEntropyLoss(weight = self.weight, reduction = 'sum')
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, *input):
        pairs = input[0]
        labels = input[1]
        node_indexes = pairs.permute([1, 0])
        src_node_indexes = node_indexes[0]
        dst_node_indexes = node_indexes[1]
        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        node_embeddings = torch.cat([src_node_embeddings, dst_node_embeddings], dim = -1)
        predictions = self.liner(node_embeddings)
        # predictions = self.soft_max(predictions)
        loss = self.cross_entropy(predictions, labels)
        return loss

    def test(self, edges):
        edges = edges.permute([1, 0])
        src_node_indexes = edges[0]
        dst_node_indexes = edges[1]

        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        node_embeddings = torch.cat([src_node_embeddings, dst_node_embeddings], dim=-1)
        predictions = self.liner(node_embeddings)
        predictions = self.soft_max(predictions)
        return predictions


class MihGNNEmbedding13(nn.Module):
    def __init__(self,A, As, all_nodes_neighbors, N, d, layers, steps, delay, GPU = False):
        super(MihGNNEmbedding13, self).__init__()
        self.A = Matrix_pre_handle(A, steps, delay)
        self.A = torch.tensor(self.A, dtype = torch.float)
        self.d = d
        self.e = torch.tensor(math.e, dtype=torch.float)
        self.layers = layers
        self.As = As
        self.all_nodes_neighbors = all_nodes_neighbors
        embedding_state = numpy.random.randn(N, d)
        embedding_state = torch.tensor(data=embedding_state, dtype=torch.float)
        self.aggregationModule = MihGNNAggregationModule2(A = self.A, As=As, all_nodes_neighbors = all_nodes_neighbors,
                                                      convolution_layers=layers, d=d,
                                                      embedding_states=embedding_state)
        self.lossFunction = MihOutputModule1(d = self.d, e = self.e, layers = layers)

    def forward(self, *input):
        pairs = input[0]
        labels = input[1]

        node_indexes = pairs.permute([1, 0])
        src_node_indexes = node_indexes[0]
        dst_node_indexes = node_indexes[1]
        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        loss = self.lossFunction(labels, src_node_embeddings, dst_node_embeddings)
        return loss

    def test(self, edges):
        edges = edges.permute([1, 0])
        src_node_indexes = edges[0]
        dst_node_indexes = edges[1]

        src_node_embeddings = self.aggregationModule(src_node_indexes)
        dst_node_embeddings = self.aggregationModule(dst_node_indexes)
        differcences_sum = (src_node_embeddings - dst_node_embeddings) ** 2
        differcences_sum = torch.sum(differcences_sum, dim=1) / (self.d * self.layers)
        predicts = self.e ** (-differcences_sum)
        return predicts