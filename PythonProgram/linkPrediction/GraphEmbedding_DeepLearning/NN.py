'''
@Time: 2019/12/18 15:49
@Author: mih
@Des:
'''
import torch
import torch.nn as nn

class LineNetwork(nn.Module):
    def __init__(self, input_features, output_features, hidden_features):
        super(LineNetwork, self).__init__()
        self.line1 = nn.Sequential(
            nn.Linear(input_features, hidden_features, bias=True),
            nn.Dropout(p=0.3),
            # nn.BatchNorm1d(num_features = hidden_features),
            nn.ReLU()
        )
        # self.line2 = nn.Sequential(
        #     nn.Linear(hidden_features * 2, hidden_features, bias=True),
        #     nn.Dropout(p=0.3),
        #     nn.ReLU()
        # )

        self.line2 = nn.Sequential(
            nn.Linear(hidden_features, output_features, bias=False),
            nn.BatchNorm1d(num_features = output_features),
            nn.ReLU()
        )


    def forward(self, input):
        ouput = self.line1(input)
        ouput = self.line2(ouput)
        # ouput = self.line3(ouput)
        return ouput







