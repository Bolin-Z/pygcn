import numpy as np
from models import GCNNetwork
from utils import load_data, normalize_diffusion_matrix, xent
from optimizer import GradDescentOptim
from pathlib import Path
import pickle

if __name__ == "__main__":
    # parameter setting
    RESET_PARAM = False
    
    seed = 42
    epoches = 1000
    lr = 0.01
    wd = 5e-4

    loss_min = 1e6
    es_iters = 0
    es_steps = 50
    
    # load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    number_of_nodes = adj.shape[0]
    number_of_features = features.shape[1]
    number_of_classes = labels.shape[1]

    # load / creat network
    print(f'Loading model...')
    model_name = 'model.pth'
    model_path = Path('./{}'.format(model_name))
    if model_path.is_file():
        print('Load model from cache...')
        with model_path.open('rb') as f:
            network = pickle.load(f)
            if RESET_PARAM:
                network.reset_param()
    else:
        print('Creat model...')
        network = GCNNetwork(
            n_inputs=number_of_features,
            n_outputs=number_of_classes,
            n_layers=1,
            hidden_size=[16],
            activation=np.tanh,
            seed=seed
        )
    
    # optimizer
    optim = GradDescentOptim(
        lr=lr,
        wd=wd
    )

    # record
    val_accs = list()
    test_accs = list()
    train_losses = list()
    val_losses = list()

    # main training loop
    print(f"{'-' * 20} Start {'-' * 20}")
    A = normalize_diffusion_matrix(adj)
    X = np.array(features)
    for epoch in range(epoches):
        # forward
        y_pred = network.forward(A, X)
        optim(y_pred, labels, idx_train)
        # backward
        for layer in reversed(network.layers):
            layer.backward(optim, update=True)
        
        val_acc = (np.argmax(y_pred, axis=1) == np.argmax(labels, axis=1))[
            [i for i in range(labels.shape[0]) if (i in idx_val)]
        ]
        val_accs.append(np.mean(val_acc))

        test_acc = (np.argmax(y_pred, axis=1) == np.argmax(labels, axis=1))[
            [i for i in range(labels.shape[0]) if (i in idx_test)]
        ]
        test_accs.append(np.mean(test_acc))

        loss = xent(y_pred, labels)
        train_loss = np.mean(loss[idx_train])
        val_loss = np.mean(loss[idx_val])

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < loss_min:
            loss_min = val_loss
            es_iters = 0
        else:
            es_iters += 1
        
        if es_iters > es_steps:
            print("Early stopping")
            break
        
        # logging
        if epoch % 5 == 0:
            print(f"Epoch: {epoch + 1}")
            print(f"\tTrain loss: {train_loss}")
            print(f"\tVal loss: {val_loss}")
            print(f"\tVal acc: {val_accs[-1]}")
            print(f"\tTest acc: {test_accs[-1]}")
            print(f"{'-' * 50}")