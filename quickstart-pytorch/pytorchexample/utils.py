
from cProfile import label
import copy
import math
from pyexpat import features
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


def attack(model_server, model_client, learning_rate, batch_size_client, epoch):

    weights_server = model_server.state_dict()
    weights_client = model_client.state_dict() 

    weight_diff = {} 

    for name in weights_server: 
        weight_diff[name] = weights_server[name] - weights_client[name]

    converged = False
    #get sim features, sim labels

    while not converged:
        #change to sim features and sim labels
        model_sim_client = sim_update(features, labels, model_server, learning_rate, batch_size_client, epoch)
        #calculate sim wweight diff 
        #calcuclate loss = loss_sim(weight_diff, sim_weight_diff) + loss_inv(sim_features) 
        #update sim_features

    pass

def sim_update(features, labels, model_server, learning_rate, batch_size_client, epoch):
    N = features.shape[0]

    num_batches = math.ceil(N / batch_size_client)

    weights_server = model_server.state_dict()
    weights_init = {} 
    for name in weights_server: 
        weights_init[name] = weights_server[name].detach().clone()

    model_client = copy.deepcopy(model_server)
    model_client.load_state_dict(weights_init)

    loss_fn = nn.CrossEntropyLoss()

    # partition data into batches
    batch_indices = partition_data(features, batch_size_client)

    for e in range(epoch):
        for b in range(num_batches):

            #get batch 
            idx = batch_indices[b]
            x_batch = features[idx] 
            y_batch = labels[idx] 

            # calculate gradient
            model_client.zero_grad()

            outputs = model_client(x_batch)
            loss = loss_fn(outputs, y_batch)

            loss.backwards()

            # Extract gradients
            gradients = {
                name: param.grad.detach().clone()
                for name, param in model_client.named_parameters()
            }

            # Update weights
            with torch.no_grad():
                for name, param in model_client.named_parameters():
                    param -= learning_rate * gradients[name]   

    # return weights
    return model_client.state_dict()


def partition_data(features, batch_size_client):
    N = features.shape[0]

    indices = torch.randperm(N)
    chunks = indices.split(batch_size_client)

    return chunks
