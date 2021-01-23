import torch
import json
from torch import optim
from GraphEmbedding_RandomWalk.DeepWalk import Deep_Walk
from GraphEmbedding_RandomWalk.Node2Vec import Node2vec
from Tools import process_gml_file
from Tools import get_data_loader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy
with open(r"./params.json", 'r') as file:
    params = json.load(file)
    data_set_name = params["data_set"]
    module_name = params["module_name"]
    batchSize = params["batchSize"]
    epochs = params["epochs"]
    radio = params["radio"]
    learning_rate = params["learning_rate"]
    print("module_name: {0}".format(module_name))
    if module_name == "DeepWalk":
        module_params = params["DeepWalk"]
    if module_name == "Node2Vec":
        module_params = params["Node2Vec"]
        p = module_params["p"]
        q = module_params["q"]
        print("p: {0}".format(p))
        print("q: {0}".format(q))
    embed_size = module_params["embed_size"]
    walk_length = module_params["walk_length"]
    num_walks = module_params["num_walks"]
    workers = module_params["workers"]
    window_size = module_params["window_size"]
    iter = module_params["iter"]
    sample_method = module_params['sample_method']
    print("dataSet: {0}".format(data_set_name))
    print("radio: {0}".format(radio))
    print("batchSize: {0}".format(batchSize))
    print("epochs: {0}".format(epochs))
    print("embedding_size: {0}".format(embed_size))
    print("walk_length: {0}".format(walk_length))
    print("num_walks: {0}".format(num_walks))
    print("workers: {0}".format(workers))
    print("window_size: {0}".format(window_size))
    print("iter: {0}".format(iter))
    print("sample_method: {0}".format(sample_method))
if data_set_name == 'PB':
    file_address = r"../Data/PB.gml"
if data_set_name == 'bio-CE-GT':
    file_address = r"../Data/bio-CE-GT.gml"
if data_set_name == 'hamster':
    file_address = r"../Data/hamster.gml"
if data_set_name == 'USAir':
    file_address = r"../Data/USAir.gml"
if data_set_name == 'Yeast':
    file_address = r"../Data/Yeast.gml"

if torch.cuda.is_available():
    GPU = True
else:
    GPU = False

G, A, nodes, all_neighbors, As = process_gml_file(file_address)
train_loader, test_loader, A_test, weight = get_data_loader(A, radio = radio, batch_size = batchSize, sample_method = sample_method, GPU = GPU)
if module_name == 'DeepWalk':
    module = Deep_Walk(G=G, A=A_test, walk_length=walk_length, embed_size=embed_size, window_size=window_size,
                       workers=workers)
if module_name == 'Node2Vec':
    module = Node2vec(G = G, A = A_test, walk_length = walk_length, p = p, q = q, embed_size = embed_size, iter = iter, window_size = window_size, workers = workers)


def Train(module, epochs):
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


def Test(module):

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
        output = output.detach().numpy()
        scores = output[:, 1]
        test_label = test_label.numpy()
        predictions = predictions.numpy()
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
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    torch.backends.cudnn.deterministic = True
if __name__ == '__main__':
    setup_seed(6)
    Train(module, epochs)
    Test(module)
