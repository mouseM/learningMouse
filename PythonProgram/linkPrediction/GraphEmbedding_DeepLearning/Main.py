import json
import torch
from torch import optim
import numpy
from GraphEmbedding_DeepLearning.PolysemousNetworkEmbedding import PriorDistribution
from GraphEmbedding_DeepLearning.PolysemousNetworkEmbedding import PolysemousNetwork
from GraphEmbedding_DeepLearning.PolysemousNetworkEmbedding import MihPolysemousNetwork
from Tools import process_gml_file
from Tools import get_data_loader
from GraphEmbedding_DeepLearning.MihGNNEmbeddingModules import MihGNNEmbedding1, MihGNNEmbedding12WithJaccard
from GraphEmbedding_DeepLearning.MihGNNEmbeddingModules import MihGNNEmbedding2
from GraphEmbedding_DeepLearning.MihGNNEmbeddingModules import MihGNNEmbedding3
from GraphEmbedding_DeepLearning.MihGNNEmbeddingModules import MihGNNEmbedding4
from GraphEmbedding_DeepLearning.MihGNNEmbeddingModules import MihGNNEmbedding5
from GraphEmbedding_DeepLearning.MihGNNEmbeddingModules import MihGNNEmbedding6
from GraphEmbedding_DeepLearning.MihGNNEmbeddingModules import MihGNNEmbedding7
from GraphEmbedding_DeepLearning.MihGNNEmbeddingModules import MihGNNEmbedding8
from GraphEmbedding_DeepLearning.MihGNNEmbeddingModules import MihGNNEmbedding9
from GraphEmbedding_DeepLearning.MihGNNEmbeddingModules import MihGNNEmbedding10
from GraphEmbedding_DeepLearning.MihGNNEmbeddingModules import MihGNNEmbedding11
from GraphEmbedding_DeepLearning.MihGNNEmbeddingModules import MihGNNEmbedding12
from GraphEmbedding_DeepLearning.MihGNNEmbeddingModules import MihGNNEmbedding12WithNoAggregation
from GraphEmbedding_DeepLearning.MihGNNEmbeddingModules import MihGNNEmbedding12WithTrainWeight
from GraphEmbedding_DeepLearning.MihGNNEmbeddingModules import MihGNNEmbedding12AferRandomWalk
from GraphEmbedding_DeepLearning.MihGNNEmbeddingModules import MihGNNEmbedding13
from GraphEmbedding_DeepLearning.CovENetwork import CovE
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

with open(r"./params.json", 'r') as file:

    params = json.load(file)
    data_set_name = params["data_set"]
    module_name = params["module_name"]
    batchSize = params["batchSize"]
    epochs = params["epochs"]
    radio = params["radio"]
    learning_rate = params["learning_rate"]
    print("module_name: {0}".format(module_name))
    if module_name == "MihGNNEmbedding1":
        module_params = params["MihGNNEmbedding1"]
    if module_name == "MihGNNEmbedding2":
        module_params = params["MihGNNEmbedding2"]
    if module_name == "MihGNNEmbedding3":
        module_params = params["MihGNNEmbedding3"]
    if module_name == "MihGNNEmbedding4":
        module_params = params["MihGNNEmbedding4"]
    if module_name == "MihGNNEmbedding5":
        module_params = params["MihGNNEmbedding5"]
    if module_name == "MihGNNEmbedding6":
        module_params = params["MihGNNEmbedding6"]
    if module_name == "MihGNNEmbedding7":
        module_params = params["MihGNNEmbedding7"]
    if module_name == "MihGNNEmbedding8":
        module_params = params["MihGNNEmbedding8"]
    if module_name == "MihGNNEmbedding9":
        module_params = params["MihGNNEmbedding9"]
    if module_name == "MihGNNEmbedding10":
        module_params = params["MihGNNEmbedding10"]
    if module_name == "MihGNNEmbedding11":
        module_params = params["MihGNNEmbedding11"]
    if module_name == "MihGNNEmbedding12":
        module_params = params["MihGNNEmbedding12"]
    if module_name == "MihGNNEmbedding12WithJaccard":
        module_params = params["MihGNNEmbedding12WithJaccard"]
    if module_name == "MihGNNEmbedding12WithNoAggregation":
        module_params = params["MihGNNEmbedding12WithNoAggregation"]
    if module_name == "MihGNNEmbedding12WithTrainWeight":
        module_params = params["MihGNNEmbedding12WithTrainWeight"]
    if module_name == "MihGNNEmbedding12AferRandomWalk":
        module_params = params["MihGNNEmbedding12AferRandomWalk"]
    if module_name == "MihGNNEmbedding13":
        module_params = params["MihGNNEmbedding13"]
    if module_name == "PolysemousNetworkEmbedding":
        module_params = params["PolysemousNetworkEmbedding"]
    if module_name == "MihPolysemousNetworkEmbedding":
        module_params = params["MihPolysemousNetworkEmbedding"]
    if module_name == "CovE":
        module_params = params["CovE"]

    embedding_size = module_params["embedding_size"]
    layers = module_params["layers"]
    if module_name == 'MihGNNEmbedding12AferRandomWalk':
        steps = module_params["steps"]
        delays = module_params["delays"]
        walk_length = module_params["walk_length"]
        num_walks = module_params["num_walks"]
        walker = module_params["walker"]
        p = module_params["p"]
        q = module_params["q"]
        window_size = module_params["window_size"]
        iter = module_params["iter"]
        workers = module_params["workers"]
        kwargs = {}
        kwargs["walk_length"] = walk_length
        kwargs["num_walks"] = num_walks
        kwargs["walker"] = walker
        kwargs["p"] = p
        kwargs["q"] = q
        kwargs["window_size"] = window_size
        kwargs["iter"] = iter
        kwargs["workers"] = workers
        print("steps: {0}".format(steps))
        print("delays: {0}".format(delays))

    if module_name != 'PolysemousNetworkEmbedding' and module_name != "MihPolysemousNetworkEmbedding":
        steps = module_params["steps"]
        delays = module_params["delays"]
        print("steps: {0}".format(steps))
        print("delays: {0}".format(delays))
    if module_name == 'PolysemousNetworkEmbedding':
        K = module_params["K"]
        alpha = module_params["alpha"]
        P_epochs = module_params["P_epochs"]
        print("K: {0}".format(K))
        print("alpha: {0}".format(alpha))
        print("P_epochs: {0}".format(P_epochs))
    if module_name == 'MihPolysemousNetworkEmbedding':
        K = module_params["K"]
        alpha = module_params["alpha"]
        P_epochs = module_params["P_epochs"]
        steps = module_params["steps"]
        delays = module_params["delays"]
        print("K: {0}".format(K))
        print("alpha: {0}".format(alpha))
        print("P_epochs: {0}".format(P_epochs))
    print("dataSet: {0}".format(data_set_name))
    print("radio: {0}".format(radio))
    print("batchSize: {0}".format(batchSize))
    print("epochs: {0}".format(epochs))
    print("embedding_size: {0}".format(embedding_size))
    print("layers: {0}".format(layers))

file_address = r"../Data/{0}.gml".format(data_set_name)


if torch.cuda.is_available():
    GPU = True
else:
    GPU = False

G, A, nodes, all_nodes_neighbors, As = process_gml_file(file_address)
# A = generate_random_graph(10, 0.1)
A_shape = A.shape
N = A_shape[0]
print("邻接矩阵尺寸：[{0}, {1}]".format(A_shape[0], A_shape[1]))

def Train(module, epochs):
    print("module parameters:")
    for name, parameter in module.named_parameters():
        print("name:{0}\nparameter:{1}".format(name, parameter))
    optimizer = optim.Adam(module.parameters(), lr=learning_rate)
    for epoch in tqdm(range(epochs)):
        sum_loss = 0.0
        for data in train_loader:
            inputs, labels = data
            labels = labels.to(torch.long)
            optimizer.zero_grad()
            loss = module(inputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()
            sum_loss = sum_loss + loss
        print("epochs : {0} loss : {1}".format(epoch, sum_loss))

def Test1(module):
    for name, parameter in module.named_parameters():
        print("name:{0}\nparameter:{1}".format(name, parameter))
    all_labels = []
    all_predictions = []
    for i, data in enumerate(test_loader, 0):
        edges, test_labels = data
        predictions = module.test(edges)
        test_labels = test_labels.numpy()
        all_labels.extend(test_labels)
        predictions = predictions.detach().numpy()
        all_predictions.extend(predictions)

    devide_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for devide_value in devide_values:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i, label in enumerate(all_labels):
            prediction = all_predictions[i]
            if (prediction >= devide_value and label == 1):
                TP = TP + 1
            if (prediction < devide_value and label == 1):
                FN = FN + 1
            if (prediction >= devide_value and label == 0):
                FP = FP + 1
            if (prediction < devide_value and label == 0):
                TN = TN + 1
        print("-------------devide value : {0} ---------------".format(devide_value))
        print("TP: {0}".format(TP))
        print("FP: {0}".format(FP))
        print("TN: {0}".format(TN))
        print("FN: {0}".format(FN))
        print("AP: {0}".format((TP + TN) / (TP + FP + TN + FN)))
        print("AC: {0}".format((TP) / (TP + FP)))
        print("AUC: {0}".format(roc_auc_score(all_labels, all_predictions)))

def Test2(module):
    for name, parameter in module.named_parameters():
        print("name:{0}\nparameter:{1}".format(name, parameter))
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    all_labels = []
    all_predictions = []
    all_score = []
    for test_data, test_label in test_loader:
        output = module.test(test_data)
        _, predictions = torch.max(output.data, dim=1)
        test_label = test_label.numpy()
        predictions = predictions.numpy()
        output = output.detach().numpy()
        scores = output[:, 1]
        all_labels.extend(test_label)
        all_predictions.extend(predictions)
        all_score.extend(scores)
        for index in range(len(predictions)):
            prediction = predictions[index]
            test = test_label[index]
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
    print("AUC: {0}".format(roc_auc_score(all_labels, all_score)))




cross_entropy_loss_function_Modules = ["MihGNNEmbedding4", "MihGNNEmbedding5", "MihGNNEmbedding6", "MihGNNEmbedding7",
                                       "MihGNNEmbedding9", "MihGNNEmbedding10", "MihGNNEmbedding11",
                                       "MihGNNEmbedding12", "MihGNNEmbedding12WithNoAggregation",
                                       "MihGNNEmbedding12WithJaccard",
                                       "MihGNNEmbedding12WithTrainWeight",
                                       "MihGNNEmbedding12AferRandomWalk",
                                       "MihPolysemousNetworkEmbedding", "CovE"]


def create_module(name, A, As, all_nodes_neighbors, N, embedding_size, layers, steps, delays, GPU, kwargs = None):
    if name == "MihGNNEmbedding1":
        module = MihGNNEmbedding1(A=A, As=As, all_nodes_neighbors=all_nodes_neighbors, N=N, d=embedding_size,
                                  layers=layers, steps=steps, delay=delays, GPU=GPU)
    if name == "MihGNNEmbedding2":
        module = MihGNNEmbedding2(A=A, As=As, all_nodes_neighbors=all_nodes_neighbors, N=N, d=embedding_size,
                                  layers=layers, steps=steps, delay=delays, GPU=GPU)
    if name == "MihGNNEmbedding3":
        module = MihGNNEmbedding3(A=A, As=As, all_nodes_neighbors=all_nodes_neighbors, N=N, d=embedding_size,
                                  layers=layers, steps=steps, delay=delays, GPU=GPU)
    if name == "MihGNNEmbedding4":
        module = MihGNNEmbedding4(A=A, As=As, all_nodes_neighbors=all_nodes_neighbors, N=N, d=embedding_size,
                                  layers=layers, steps=steps, delay=delays, GPU=GPU)
    if name == "MihGNNEmbedding5":
        module = MihGNNEmbedding5(A=A, As=As, all_nodes_neighbors=all_nodes_neighbors, N=N, d=embedding_size,
                                  layers=layers, steps=steps, delay=delays, GPU=GPU)
    if name == "MihGNNEmbedding6":
        module = MihGNNEmbedding6(A=A, As=As, all_nodes_neighbors=all_nodes_neighbors, N=N, d=embedding_size,
                                  layers=layers, steps=steps, delay=delays, GPU=GPU)
    if name == "MihGNNEmbedding7":
        module = MihGNNEmbedding7(A=A, N=N, d=embedding_size, layers=layers, steps=steps, delay=delays, GPU=GPU)
    if name == "MihGNNEmbedding8":
        module = MihGNNEmbedding8(A=A, As=As, all_nodes_neighbors=all_nodes_neighbors, N=N, d=embedding_size,
                                  layers=layers, steps=steps, delay=delays, GPU=GPU)
    if name == "MihGNNEmbedding9":
        module = MihGNNEmbedding9(A=A, As=As, all_nodes_neighbors=all_nodes_neighbors, N=N, d=embedding_size,
                                  layers=layers, steps=steps, delay=delays, GPU=GPU)
    if name == "MihGNNEmbedding10":
        module = MihGNNEmbedding10(A=A, As=As, all_nodes_neighbors=all_nodes_neighbors, N=N, d=embedding_size,
                                   layers=layers, steps=steps, delay=delays, GPU=GPU)
    if name == "MihGNNEmbedding11":
        module = MihGNNEmbedding11(A=A, As=As, all_nodes_neighbors=all_nodes_neighbors, N=N, d=embedding_size,
                                   layers=layers, steps=steps, delay=delays, GPU=GPU)
    if name == "MihGNNEmbedding12":
        module = MihGNNEmbedding12(A=A, As=As, all_nodes_neighbors=all_nodes_neighbors, N=N, d=embedding_size,
                                   layers=layers, steps=steps, delay=delays, weight=weight, GPU=GPU)
    if name == "MihGNNEmbedding12WithJaccard":
        module = MihGNNEmbedding12WithJaccard(A=A, As=As, all_nodes_neighbors=all_nodes_neighbors, N=N, d=embedding_size,
                                   layers=layers, steps=steps, delay=delays, weight=weight, GPU=GPU)

    if name == "MihGNNEmbedding12WithTrainWeight":
        module = MihGNNEmbedding12WithTrainWeight(A=A, As=As, all_nodes_neighbors=all_nodes_neighbors, N=N, d=embedding_size,
                                   layers=layers, steps=steps, delay=delays, GPU=GPU)
    if name == "MihGNNEmbedding12AferRandomWalk":
        walk_length = kwargs["walk_length"]
        p = kwargs["p"]
        q = kwargs["q"]
        iter = kwargs["iter"]
        window_size = kwargs["window_size"]
        workers = kwargs["workers"]

        module = MihGNNEmbedding12AferRandomWalk(G = G, A = A, As = As, all_nodes_neighbors = all_nodes_neighbors,
                                                 N = N, d = embedding_size, walk_length = walk_length, p = p, q = q,
                                                 iter = iter, window_size = window_size, workers = workers,
                                                 layers = layers, steps = steps, delay = delays, GPU = False)

    if name == "MihGNNEmbedding12WithNoAggregation":
        module = MihGNNEmbedding12WithNoAggregation(N=N, d=embedding_size,
                                   layers=layers, GPU=GPU)
    if name == "MihGNNEmbedding13":
        module = MihGNNEmbedding13(A=A, As=As, all_nodes_neighbors=all_nodes_neighbors, N=N, d=embedding_size,
                                   layers=layers, steps=steps, delay=delays, GPU=GPU)
    if name == "CovE":
        module = CovE(A=A, N=N, embeding_size=64, layers=2, steps=steps, delay=delays, width=8, height=8,
                      kernel_size=3, out_channel=3, GPU=False)

    if name == "PolysemousNetworkEmbedding":
        A_test = A + numpy.eye(N)
        P_module = PriorDistribution(A=A_test, K=K, alpha=alpha, GPU=GPU)
        optimizer = optim.Adam(P_module.parameters(), lr=0.001)
        for epoch in range(P_epochs):
            optimizer.zero_grad()
            loss = P_module()
            print("epoch : {0}   loss : {1}".format(epoch + 1, loss))
            loss.backward(retain_graph=True)
            optimizer.step()
        P = P_module.state_dict()["prior_distribution_matrix"]
        module = PolysemousNetwork(P=P, A=A_test, embedding_size=embedding_size, layers=layers, K=K, GPU=GPU)
    if name == "MihPolysemousNetworkEmbedding":
        A_test = A + numpy.eye(N)
        A_test = A_test + numpy.eye(N)
        P_module = PriorDistribution(A=A_test, K=K, alpha=alpha, GPU=GPU)
        optimizer = optim.Adam(P_module.parameters(), lr=0.001)
        for epoch in range(P_epochs):
            optimizer.zero_grad()
            loss = P_module()
            print("epoch : {0}   loss : {1}".format(epoch + 1, loss))
            loss.backward(retain_graph=True)
            optimizer.step()
        P = P_module.state_dict()["prior_distribution_matrix"]
        module = MihPolysemousNetwork(P=P, A=A_test, embedding_size=embedding_size, layers=layers, K=K, GPU=GPU)
    return module

if __name__ == '__main__':

    for i in range(1):
        print("---------------------{0} time ----------------------".format(i + 1))
        train_loader, test_loader, A_test, weight = get_data_loader(A, radio, batchSize, "under_sample", GPU)
        module = create_module(name=module_name, A=A_test, As=As, all_nodes_neighbors=all_nodes_neighbors, N=N,
                               embedding_size=embedding_size,layers=layers, steps=steps, delays=delays,
                               GPU=GPU, kwargs = None)
        Train(module, epochs)
        if (module_name in cross_entropy_loss_function_Modules):
            Test2(module)
        else:
            Test1(module)


