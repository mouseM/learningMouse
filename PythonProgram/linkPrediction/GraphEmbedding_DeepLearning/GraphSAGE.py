import torch
from torch import nn
import random
from torch import functional as F
import networkx
import numpy
import math
from torch import optim
from GraphEmbedding_DeepLearning.NN import LineNetwork

layers = 2
embedding_size = 128
train_radio = 0.9
sample_method = 'under_sample'
batch_size = 32
epochs = 50
file = r'../Data/USAir.gml'

def process_gml_file(file = r'./Data/bio-GE-GT.gml'):
    G = networkx.read_gml(file)
    A = numpy.array(networkx.adjacency_matrix(G).todense())
    N = A.shape[0]
    nodes = numpy.arange(N)
    numpy.random.shuffle(nodes)
    all_neighbors = {}
    for node in nodes:
        node_neighbors = set()
        for index, value in enumerate(A[node]):
            if value > 0:
                node_neighbors.add(index)
        all_neighbors[node] = node_neighbors
    positives = []
    negavives = []
    for i in range(N):
        for j in range(i, N):
            if i != j:
                label = A[i][j]
                if label == 1:
                    positives.append([i, j, 1])
                if label == 0:
                    negavives.append([i, j, 0])
    edges_size = len(positives)
    none_edges_size = len(negavives)
    if sample_method == 'all_sample':
        numpy.random.shuffle(positives)
        numpy.random.shuffle(negavives)

        train_positives = positives[: int(edges_size * train_radio)]
        test_positives = positives[int(edges_size * train_radio):]
        train_negatives = negavives[: int(none_edges_size * train_radio)]
        test_negatives = negavives[int(none_edges_size * train_radio):]

        train_positives.extend(train_negatives)
        train_data = train_positives
        test_positives.extend(test_negatives)
        test_data = test_positives
        numpy.random.shuffle(train_data)
        numpy.random.shuffle(test_data)
        weight = [1, int(none_edges_size / edges_size)]
    if sample_method == 'under_sample':
        numpy.random.shuffle(positives)
        numpy.random.shuffle(negavives)
        train_positives = positives[: int(edges_size * train_radio)]
        test_positives = positives[int(edges_size * train_radio):]
        train_negatives = negavives[: int(edges_size * train_radio)]
        # test_negatives = negavives[int(edges_size * radio):]
        test_negatives = negavives[int(edges_size * train_radio): edges_size]
        train_positives.extend(train_negatives)
        train_data = train_positives
        test_positives.extend(test_negatives)
        test_data = test_positives
        numpy.random.shuffle(train_data)
        numpy.random.shuffle(test_data)
        weight = [1, 1]
    if sample_method == 'over_sample':
        pass

    train_pairs = [pair[: 2] for pair in train_data]
    train_labels = [pair[-1] for pair in train_data]
    test_pairs = [pair[: 2] for pair in test_data]
    test_labels = [pair[-1] for pair in test_data]
    return nodes, all_neighbors, train_pairs, train_labels, test_pairs, test_labels

nodes, all_neighbors, train_pairs, train_labels, test_pairs, test_labels = process_gml_file(file)
N = len(nodes)
class GraphSage(nn.Module):
    """docstring for GraphSage"""

    def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, gcn=False, agg_func='MEAN'):
        super(GraphSage, self).__init__()

        self.input_size = input_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.gcn = gcn
        self.agg_func = agg_func

        self.raw_features = raw_features
        self.adj_lists = adj_lists

        for index in range(1, num_layers + 1):
            layer_size = out_size if index != 1 else input_size
            setattr(self, 'sage_layer' + str(index), SageLayer(layer_size, out_size, gcn=self.gcn))

    def forward(self, nodes_batch):
        """
        Generates embeddings for a batch of nodes.
        nodes_batch	-- batch of nodes to learn the embeddings
        """
        lower_layer_nodes = list(nodes_batch)
        nodes_batch_layers = [(lower_layer_nodes,)]
        # self.dc.logger.info('get_unique_neighs.')
        for i in range(self.num_layers):
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = self._get_unique_neighs_list(
                lower_layer_nodes)
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))

        assert len(nodes_batch_layers) == self.num_layers + 1

        pre_hidden_embs = self.raw_features
        for index in range(1, self.num_layers + 1):
            nb = nodes_batch_layers[index][0]
            pre_neighs = nodes_batch_layers[index - 1]
            # self.dc.logger.info('aggregate_feats.')
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
            sage_layer = getattr(self, 'sage_layer' + str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)
            # self.dc.logger.info('sage_layer.')
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
                                         aggregate_feats=aggregate_feats)
            pre_hidden_embs = cur_hidden_embs

        return pre_hidden_embs

    def _nodes_map(self, nodes, hidden_embs, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]
        return index

    def _get_unique_neighs_list(self, nodes, num_sample=10):
        _set = set
        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh
                           in to_neighs]
        else:
            samp_neighs = to_neighs
        samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        _unique_nodes_list = list(set.union(*samp_neighs))
        i = list(range(len(_unique_nodes_list)))
        unique_nodes = dict(list(zip(_unique_nodes_list, i)))
        return samp_neighs, unique_nodes, _unique_nodes_list

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs

        assert len(nodes) == len(samp_neighs)
        indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
        assert (False not in indicator)
        if not self.gcn:
            samp_neighs = [(samp_neighs[i] - set([nodes[i]])) for i in range(len(samp_neighs))]
        # self.dc.logger.info('2')
        if len(pre_hidden_embs) == len(unique_nodes):
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
        # self.dc.logger.info('3')
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        # self.dc.logger.info('4')

        if self.agg_func == 'MEAN':
            num_neigh = mask.sum(1, keepdim=True)
            mask = mask.div(num_neigh).to(embed_matrix.device)
            aggregate_feats = mask.mm(embed_matrix)

        elif self.agg_func == 'MAX':
            # print(mask)
            indexs = [x.nonzero() for x in mask == 1]
            aggregate_feats = []
            # self.dc.logger.info('5')
            for feat in [embed_matrix[x.squeeze()] for x in indexs]:
                if len(feat.size()) == 1:
                    aggregate_feats.append(feat.view(1, -1))
                else:
                    aggregate_feats.append(torch.max(feat, 0)[0].view(1, -1))
            aggregate_feats = torch.cat(aggregate_feats, 0)

        # self.dc.logger.info('6')

        return aggregate_feats

"""
Encodes a node's using 'convolutional' GraphSage approach
"""
class SageLayer(nn.Module):
    def __init__(self, input_size, out_size, gcn=False):
        super(SageLayer, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.gcn = gcn
        self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size))
        self.relu = nn.ReLU()
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    """
    Generates embeddings for a batch of nodes.
    nodes	 -- list of nodes
    """
    def forward(self, self_feats, aggregate_feats, neighs=None):

        if not self.gcn:
            combined = torch.cat([self_feats, aggregate_feats], dim=1)
        else:
            combined = aggregate_feats
        combined = self.relu(self.weight.mm(combined.t())).t()
        return combined
class MihModule(nn.Module):
    def __init__(self, N, embedding_size, layers):
        super(MihModule, self).__init__()
        embeddings = numpy.random.random([N, embedding_size])
        embeddings = torch.tensor(data = embeddings, dtype = torch.float)
        self.graphSage = GraphSage(num_layers = layers, input_size = embedding_size, out_size = embedding_size,
                              raw_features = embeddings, adj_lists = all_neighbors)
        self.liner = LineNetwork(input_features=embedding_size, output_features=2, hidden_features=embedding_size)
        self.soft_max = nn.Softmax(dim=-1)
        self.cross_entropy = nn.CrossEntropyLoss()

        self.edge_generator_forward = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size * 2, bias=True),
            nn.BatchNorm1d(num_features=embedding_size * 2),
            nn.ReLU()
        )
        self.edge_generator_backward = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size * 2, bias=True),
            nn.BatchNorm1d(num_features=embedding_size * 2),
            nn.ReLU()
        )

    def forward(self, *input):
        edges = input[0]
        srcNodes = [edge[0] for edge in edges]
        dstNodes = [edge[1] for edge in edges]
        labels = input[1]
        labels = torch.tensor(labels, dtype = torch.long)
        srcNodesEmbedding = self.graphSage(srcNodes)
        dstNodesEmbedding = self.graphSage(dstNodes)
        # 直接拼接形成边
        # edge_embeddings = torch.cat([srcNodesEmbedding, dstNodesEmbedding], dim=-1)

        # # 使用一个独立的模型生成边
        # edge_embeddings_forward = torch.cat([srcNodesEmbedding, dstNodesEmbedding], dim=-1)
        # edge_embeddings_backward = torch.cat([dstNodesEmbedding, srcNodesEmbedding], dim=-1)
        # # 添加一个边的生成层
        # edge_embeddings_forward = self.edge_generator_forward(edge_embeddings_forward)
        # edge_embeddings_backward = self.edge_generator_backward(edge_embeddings_backward)
        # edge_embeddings = edge_embeddings_forward + edge_embeddings_backward

        # 使用哈达玛积形成边
        edge_embeddings = torch.mul(srcNodesEmbedding, dstNodesEmbedding)

        predictions = self.liner(edge_embeddings)
        predictions = self.soft_max(predictions)
        loss = self.cross_entropy(predictions, labels)
        return loss

    def test(self, edges):
        srcNodes = [edge[0] for edge in edges]
        dstNodes = [edge[1] for edge in edges]
        srcNodesEmbedding = self.graphSage(srcNodes)
        dstNodesEmbedding = self.graphSage(dstNodes)
        # edge_embeddings = torch.cat([srcNodesEmbedding, dstNodesEmbedding], dim=-1)

        # edge_embeddings_forward = torch.cat([srcNodesEmbedding, dstNodesEmbedding], dim=-1)
        # edge_embeddings_backward = torch.cat([dstNodesEmbedding, srcNodesEmbedding], dim=-1)
        # # 添加一个边的生成层
        # edge_embeddings_forward = self.edge_generator_forward(edge_embeddings_forward)
        # edge_embeddings_backward = self.edge_generator_backward(edge_embeddings_backward)
        # edge_embeddings = edge_embeddings_forward + edge_embeddings_backward

        # 使用哈达玛积形成边
        edge_embeddings = torch.mul(srcNodesEmbedding, dstNodesEmbedding)

        predictions = self.liner(edge_embeddings)
        predictions = self.soft_max(predictions)
        return predictions


if __name__ == '__main__':
    module = MihModule(N=N, embedding_size=embedding_size, layers=layers)
    optimizer = optim.Adam(module.parameters(), lr=0.001)
    train_batches = math.ceil(len(train_pairs) / batch_size)
    for epoch in range(epochs):
        for index in range(train_batches):
            edges_batch = train_pairs[index * batch_size:(index + 1) * batch_size]
            labels_batch = train_labels[index * batch_size:(index + 1) * batch_size]
            loss = module(edges_batch, labels_batch)
            loss.backward(retain_graph=True)
            optimizer.step()
            if (index % 50 == 0):
                print("[epochs : {0}  batchs : {1} loss : {2}]".format(epoch, index, loss))
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    test_batches = math.ceil(len(test_pairs) / batch_size)
    all_labels = []
    all_predictions = []
    for index in range(test_batches):
        edges_batch = test_pairs[index * batch_size:(index + 1) * batch_size]
        labels_batch = test_labels[index * batch_size:(index + 1) * batch_size]
        output = module.test(edges_batch)
        output, predictions = torch.max(output.data, dim=1)
        predictions = predictions.numpy()
        output = output.detach().numpy()
        # output = output[:, 1]
        all_labels.extend(labels_batch)
        all_predictions.extend(predictions)
    for i, prediction in enumerate(all_predictions):
        test = all_labels[i]
        if (prediction == 1 and test == 1):
            TP = TP + 1
        if (prediction == 1 and test == 0):
            FP = FP + 1
        if (prediction == 0 and test == 1):
            FN = FN + 1
        if (prediction == 0 and test == 0):
            TN = TN + 1
    print("TP: {0}".format(TP))
    print("TN: {0}".format(TN))
    print("FP: {0}".format(FP))
    print("FN: {0}".format(FN))
    print("AP: {0}".format((TP + TN) / (TP + FP + TN + FN)))
    print("AC: {0}".format((TP) / (TP + FP)))