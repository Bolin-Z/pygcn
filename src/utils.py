from pathlib import Path
from scipy.linalg import sqrtm
import pickle
import numpy as np

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_data(path="../cora/"):
    print(f'Loading {"cora"} dataset...')
    preprocess_data_path = Path('./data.tmp')

    if preprocess_data_path.is_file():
        print('Read from cache...')
        with preprocess_data_path.open('rb') as f:
            data = pickle.load(f)
    else:
        print('Read from raw data...')
        with preprocess_data_path.open('wb') as f:
            data = dict()

            content = path + "cora.content"
            idx_features_labels = np.genfromtxt(content, dtype=np.dtype(str))
            data['features'] = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
            data['labels'] = encode_onehot(idx_features_labels[:, -1])

            # build graph
            cites = path + "cora.cites"
            idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
            idx_map = {j: i for i, j in enumerate(idx)}
            edges_unordered = np.genfromtxt(cites, dtype=np.int32)
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

            adj = np.zeros((len(idx_map), len(idx_map)), dtype=np.float32)
            for edge in edges:
                adj[edge[0], edge[1]] = 1.0
                adj[edge[1], edge[0]] = 1.0
            data['adjacent'] = np.matrix(adj, dtype=np.float32)
            
            data['idx_train'] = np.arange(140)
            data['idx_val'] = np.arange(200, 500)
            data['idx_test'] = np.arange(500, 1500)

            pickle.dump(data, f)
    
    return data['adjacent'], data['features'], data['labels'], data['idx_train'], data['idx_val'], data['idx_test']

def normalize_diffusion_matrix(A:np.matrix):
    number_of_nodes = A.shape[0]
    A_mod = A + np.eye(number_of_nodes, dtype=np.float32)
    D_mod = np.zeros_like(A_mod)
    np.fill_diagonal(D_mod, np.asarray(A_mod.sum(axis=1)).flatten())
    D_mod_invroot = np.linalg.inv(sqrtm(D_mod))
    A_hat = D_mod_invroot @ A_mod @ D_mod_invroot
    return A_hat

def glorot_init(nin, nout):
    sd = np.sqrt(6.0 / (nin + nout))
    return np.random.uniform(-sd, sd, size=(nin, nout))

def xent(pred, labels):
    return -np.log(pred)[np.arange(pred.shape[0]), np.argmax(labels, axis=1)]

if __name__ == "__main__":
    import random
    idx_test = np.arange(3)
    label = np.eye(5)
    pred = np.array([
        [ random.randrange(0, 1000) for _ in range(5)] for _ in range(5)
    ])
    print(label)
    print(pred)
    tmp = (np.argmax(pred, axis=1) == np.argmax(label, axis=1))
    print(tmp)
    tmp = tmp[[i for i in range(label.shape[0]) if i in idx_test]]
    print(tmp)
    print(np.mean(tmp))