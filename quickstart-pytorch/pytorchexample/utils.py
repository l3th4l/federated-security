
from cProfile import label
import copy
import math
from pyexpat import features
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

# Helper functions 
def cosine_distance(weight_diff, weight_diff_sim):

    # Flatten and concatenate all parameter differences
    w_true = torch.cat([
        tensor.reshape(-1)
        for tensor in weight_diff.values()
    ])

    w_sim = torch.cat([
        tensor.reshape(-1)
        for tensor in weight_diff_sim.values()
    ])

    # Cosine distance
    return 1 - F.cosine_similarity(w_true, w_sim, dim=0)


# attack

def rand_init(num_samples, input_shape, epochs, batch_size, device, mode="noise", dataset=None):
    num_batches = math.ceil(num_samples / batch_size)

    dummy_batches = []

    for epoch in range(epochs):
        epoch_batches = []

        for batch in range(num_batches):
            current_batch_size = min(batch_size, num_samples - batch * batch_size)

            if mode == "noise":
                # Random initialisation in [0, 1]
                x = torch.rand(current_batch_size, *input_shape, device=device)

            elif mode == "dataset":
                if dataset is None:
                    raise ValueError("A dataset must be provided when mode='dataset'")

                indices = torch.randint(low=0, high=len(dataset), size=(current_batch_size,))

                x = torch.stack([dataset[i][0] for i in indices]).to(device)

            else:
                raise ValueError(f"Unknown mode: {mode}")

            epoch_batches.append(nn.Parameter(x))

        dummy_batches.append(epoch_batches)

    return dummy_batches

def attack(model_server, model_client, learning_rate, batch_size_client, labels, epoch, init_mode="noise", dataset=None, input_shape=(3, 32, 32)):
    # Original server parameters
    params_server = {
        name: param.detach().clone()
        for name, param in model_server.named_parameters()
    }

    # Final client parameters
    params_client = {
        name: param.detach().clone()
        for name, param in model_client.named_parameters()
    }

    # True parameter difference
    weight_diff = {
        name: params_server[name] - params_client[name]
        for name in params_server
    }
    
    # Initialize dummy features and labels
    num_samples = len(labels)
    
    dummy_batches = rand_init(
        num_samples=num_samples,
        input_shape=input_shape,
        epochs=epoch,
        batch_size=batch_size_client,
        device=next(model_server.parameters()).device,
        mode=init_mode,
        dataset=dataset,
    )

    label_batches = [labels[i:i + batch_size_client] for i in range(0, num_samples, batch_size_client)]

    optimizer = torch.optim.Adam([x for epoch_batches in dummy_batches for x in epoch_batches], lr=0.1)
    
    # TODO Add convergence criterion
    #converged = False
    #get sim features, sim labels
    
    max_iter = 1000

    #while not converged:
    for i in range(max_iter):
        optimizer.zero_grad()

        #change to sim features and sim labels
        params_sim = sim_update(dummy_batches, label_batches, model_server, learning_rate, epoch)
                
        #calculate sim weight diff 
        weight_diff_sim = {
            name: params_server[name] - params_sim[name]
            for name in params_server
        }

        # Similarity loss
        loss_sim = cosine_distance(
            weight_diff,
            weight_diff_sim
        )

        # Add epoch-invariant prior later
        loss = loss_sim  # + lambda_inv * loss_inv(features_sim)

        # Backpropagate into dummy inputs
        loss.backward()

        optimizer.step()

    return dummy_batches

def sim_update(dummy_batches, label_batches, model_server, learning_rate, epoch):

    # Initial parameters (keep them as constants)
    params = {
        name: param.detach().clone().requires_grad_(True)
        for name, param in model_server.named_parameters()
    }

    # Model buffers (e.g. BatchNorm running statistics)
    buffers = {
        name: buf.detach().clone()
        for name, buf in model_server.named_buffers()
    }

    loss_fn = nn.CrossEntropyLoss()

    for e in range(epoch):
        for b in range(len(dummy_batches[e])):

            #get batch 
            x_batch = dummy_batches[e][b]
            y_batch = label_batches[b]
            
            # calls the server model but with the parameters oof our simulated client 
            outputs = functional_call(model_server, (params, buffers), (x_batch,))
            loss = loss_fn(outputs, y_batch)

            # loss.backwards() detaches the gradients 
            # Extract gradients while preserving the graph
            gradients = torch.autograd.grad(
                loss,
                tuple(params.values()),
                create_graph=True
            )

            # Differentiable SGD update
            params = {
                name: param - learning_rate * grad
                for (name, param), grad
                in zip(params.items(), gradients)
            }  

    # return weights
    return params 

