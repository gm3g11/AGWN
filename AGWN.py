import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
import math
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import signal
from scipy.sparse import csgraph



# Load raw data
gt_file = np.load("./pems08/gt.npz")
data_file = np.load("./pems08/data.npz")
adjacent=np.load("./adjacent_new.npy")
edge_index=np.load("edge_index_traffic_new.npy")
weight=np.load("weight_new.npy")
gt=gt_file["arr_0"]
data=data_file["arr_0"]

device = torch.device("cuda")
gt=torch.tensor(gt, dtype=torch.float).to(device)
data=torch.tensor(data=data[:,:,40,:], dtype=torch.float).to(device)
edge_index=torch.tensor(edge_index, dtype=torch.int64).to(device)
weight=torch.tensor(weight, dtype=torch.float).to(device)

# Normalization
data[:,:,0]=(data[:,:,0]-data[:,:,0].mean())/data[:,:,0].var()
data[:,:,1]=(data[:,:,1]-data[:,:,1].mean())/data[:,:,1].var()

#Data Loader function
class LoadData(Dataset):
    def __init__(self, data, gt):
        self.data = data
        self.gt = gt

    def __getitem__(self, index):
        x = LoadData.to_tensor(data[index])
        target = LoadData.to_tensor(gt[index])
        return x, target

    def __len__(self):
        return len(self.data)

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)

#generate train,valid,test data
gt_torch=gt.cuda()
data_torch=data.cuda()
train_data = LoadData(data_torch[:7000],gt_torch[:7000])
test_data= LoadData(data_torch[7000:8000],gt_torch[7000:8000])
val_data=LoadData(data_torch[8000:],gt_torch[8000:])
train_loader = DataLoader(train_data, batch_size=50, shuffle=True)
test_loader = DataLoader(test_data, batch_size=50, shuffle=True)

#generate weight for the adjacent matrix
def process_graph(graph_data):
    N = graph_data.size(0)
    matrix_i = torch.eye(N)
    matrix_i = matrix_i.to(torch.device("cuda"))

    graph_data += matrix_i  # A~ [N, N]

    degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False)  # [N]
    degree_matrix = degree_matrix.pow(-1)
    degree_matrix[degree_matrix == float("inf")] = 0.  # [N]

    degree_matrix = torch.diag(degree_matrix)  # [N, N]

    return torch.mm(degree_matrix, graph_data)


graph_data = process_graph(adjacent)

#AGWN wavelet kernel generation
def sfwt_wavelet(A, tau=6):
    # '''
    # A : adjacency matrix
    # tau: scale factor
    # h: continuous wavelet filter
    # '''

    # Construct graph wavelet filter, W
    h = signal.ricker(1000, 100)[500:]
    W = np.zeros(A.shape).astype(float)
    # Generate geodesic distances
    spath = csgraph.shortest_path(A, directed=False, unweighted=True).astype(int)
    # Resample filter
    if not (h.size % tau):
        hm = h.reshape(tau, -1).mean(axis=1)
    else:
        hm = h[:-(h.size % tau)].reshape(tau, -1).mean(axis=1)

    for i in range(W.shape[0]):
        # N: histogram of distances from i
        # N_t: Number of vertices within k hops of i for all k < tau
        N = np.bincount(spath[i, :])
        N_t = np.where(spath[i, :] < tau, N[spath[i, :]], i)
        mask = (spath[i, :] < tau)
        # a : wavelet coefficients
        a = np.zeros_like(spath[i, :]).astype(float)
        a[mask] = hm[spath[i, :][mask]] / N_t[mask].astype(float)
        # W[:, i] = a+0.001
        W[:, i] = a

    return W


W = sfwt_wavelet(graph_data)

#Graph Model
class GraphSFWT(nn.Module):
    def __init__(self, hidden_c=1024, hidden_c_2=512, out_c=1):
        super(GraphSFWT, self).__init__()

        self.linear_1 = nn.Linear(1150 * 2, hidden_c)
        self.linear_2 = nn.Linear(hidden_c, hidden_c_2)
        self.linear_3 = nn.Linear(hidden_c_2, out_c)
        self.act = nn.ReLU()

    def forward(self, data, W, device):
        W = torch.from_numpy(np.asarray(W, dtype=np.float32)).cuda()
        flow_x = data.to(device)  # [B, N,  D]
        B, N = flow_x.size(0), flow_x.size(1)
        output_1 = self.act(torch.matmul(W, flow_x))  # [N, N], [B, N, D]=>[B, N, D]
        output_2 = self.act(torch.matmul(W, output_1))  # [N, N], [B, N, D]=>[B, N, D]
        output_2 = output_2.view(B, -1)
        output_3 = self.linear_1(output_2)
        output_3 = self.linear_2(output_3)
        output_3 = self.linear_3(output_3)
        return output_3



# Common practise for initialization.
def weights_init(model):

    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out',
            nonlinearity='relu')
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, val=0.0)
        elif isinstance(layer, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(layer.weight, val=1.0)
            torch.nn.init.constant_(layer.bias, val=0.0)
        elif isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, val=0.0)

my_net=GraphSFWT()
my_net.apply(weights_init)
my_net.cuda()



#Train Model

def MAPELoss(output, target):
    return torch.mean(torch.abs((target - output) / target))

device = torch.device("cuda")
train_data = LoadData(data_torch[:6000], gt_torch[:6000])
val_data = LoadData(data_torch[6000:8000], gt_torch[6000:8000])
test_data = LoadData(data_torch[8000:], gt_torch[8000:])

train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
val_loader = DataLoader(val_data, batch_size=5, shuffle=True)
test_loader = DataLoader(test_data, batch_size=5, shuffle=True)

criterion = nn.MSELoss()
criterion_mae_loss = nn.L1Loss()

optimizer = optim.Adam(params=my_net.parameters(), lr=1e-4)

# Train model
Epoch = 1000
loss_min = 200
my_net.train()
for epoch in range(Epoch):
    epoch_loss = 0.0
    start_time = time.time()
    for x, target in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
        my_net.zero_grad()
        predict_value = my_net(x, W, device)
        target = target.to(device)
        loss = criterion(predict_value, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    end_time = time.time()

    print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch, epoch_loss / len(train_data),
                                                                      (end_time - start_time) / 60))

    # Test Model
    # TODO: Visualize the Prediction Result
    # TODO: Measure the results with metrics MAE, MAPE, and RMSE
    my_net.eval()
    with torch.no_grad():
        total_mse_loss = 0.0
        total_mae_loss = 0.0
        total_mape_loss = 0.0
        for x, target in val_loader:
            # predict_value = my_net(x, W,device).to(torch.device("cpu"))  # [B, N, 1, D]
            predict_value = my_net(x, W, device)
            target = target.to(device)

            mse_loss = criterion(predict_value, target)
            mae_loss = criterion_mae_loss(predict_value, target)
            mape_loss = MAPELoss(predict_value, target)

            total_mse_loss += mse_loss.item()
            total_mae_loss += mae_loss.item()
            total_mape_loss += mape_loss.item()

        if total_mse_loss / len(val_data) < loss_min:
            loss_min = total_mse_loss / len(val_data)
            torch.save(my_net.state_dict(), "GraphSFWT_new.pkl")
            print("RMSE: {:02.4f}".format(math.sqrt(total_mse_loss) / len(val_data)))
            print("MAE: {:02.4f}".format(total_mae_loss / len(val_data)))
            print("MAPE: {:02.4f}".format(total_mape_loss / len(val_data)))


#Test model
my_net.eval()
my_net.load_state_dict(torch.load('GraphSFWT_new.pkl'))
test_data=LoadData(data_torch[8000:8590],gt_torch[8000:8590])
batch_size=10

test_loader = DataLoader(test_data, batch_size=10, shuffle=True)

criterion = nn.MSELoss()
criterion_mae_loss = nn.L1Loss()

def MAPELoss(output, target):
  return torch.mean(torch.abs((target - output) / target))
device = torch.device("cuda")
with torch.no_grad():

    total_mse_loss = []
    total_mae_loss = []
    total_mape_loss = []

    for x,target in test_loader :

        # predict_value = my_net(x, W,device).to(torch.device("cpu"))  # [B, N, 1, D]
        predict_value = my_net(x, W,device)
        target=target.to(device)
        mse_loss = criterion(predict_value, target)
        mae_loss=criterion_mae_loss(predict_value, target)
        mape_loss=MAPELoss(predict_value, target)
        print("RMSE: {:02.4f}".format(math.sqrt(mse_loss.item()/batch_size)))
        print("MAE: {:02.4f}".format( mae_loss.item()/batch_size))
        print("MAPE: {:02.4f}".format( mape_loss.item() /batch_size))
        total_mse_loss.append(math.sqrt(mse_loss.item())/batch_size)
        total_mae_loss.append(mae_loss.item()/batch_size)
        total_mape_loss.append(mape_loss.item() /batch_size)
        print("======")