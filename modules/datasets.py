#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:07:19 2020.

@author: hashemghanem
"""
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import os
import os.path as osp


def dfs(u, adj_local, visited):
    """Return the set of nodes that can be reached from u."""
    indices = [u]
    # iterate over nodes
    for v in range(adj_local.shape[1]):
        if adj_local[u, v] == 1 and visited[v] == 0:
            visited[v] = 1
            indices = indices + dfs(v, adj_local, visited)
    return indices


def jumps_withgraph_randpos_fixamp(g, L=200, jmps_mxnum=1, amplitudes=[0., 1.],
                                   noise_std=0.01, seed=None):
    """
    Create a dataset: at-maximum jmps_mxnum, rand jumps position, fixed amp.

    Parameters
    ----------
    g : a networkx graph.
    L : number of observed/original signals. The default is 200.
    jmps_mxnum :  maximum number of jumps per signal.  The default is 1.
    amplitudes:  amplitudes of the jumps. Must have (jumps_num + 1) elements.
                Type: list/tensor/array. The default is [0.,1.].
    noise_std : noise standard deviation. The default is 0.01.
    seed: if None, no random-seed is specified for torch. Default is None.

    Returns
    -------
    w : pxL (original signals)
    y : pxL (observed signals)
    D : pxp (A true dictionary that fits the dataset without normalization).
    """
    # setting the torch seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
    # p is the size of the graph.
    p = len(g.nodes)
    # get a list of the edges of g and the adjecency matrix.
    adj = torch.zeros((p, p))
    edgelist = []
    for (u, v) in (g.edges):
        edgelist = edgelist + [(u, v)]
        adj[u, v] = 1
        adj[v, u] = 1

    w = torch.zeros((p, L))
    for k in range(L):
        # picking the edges where the jumps occur on g.
        # pay attention here that if this edge exists on a cycle, then
        # we won't notice the jumps unless 2 edges are chosen in that cycle.
        positions = torch.randint(0, len(edgelist), (jmps_mxnum,))
        # Locally delete the chosen edges in the adjecancy matrix.
        adj_local = adj.detach().clone()
        for pos in (positions):
            (u, v) = edgelist[pos]
            # print(u, v)
            adj_local[u, v] = 0
            adj_local[v, u] = 0
        # visited is to label visited nodes after each exploration of g.
        visited = torch.zeros(p)
        # In order not to start the exploration walks from the same node
        # each time, we randomly permute the nodes.
        nodes = torch.randperm(p).tolist()
        # Shuffle the order of the amplitudes to remove any bias
        amp_shuffle = torch.randperm(len(amplitudes))
        amplitudes = [amplitudes[amp_shuffle[i]]
                      for i in range(len(amplitudes))]
        # Start building w[k]
        jmp_id = 0
        for node in (nodes):
            if visited[node] == 0:
                visited[node] = 1
                # Get the indices of nodes to have the same value on it.
                indices = dfs(node, adj_local, visited)
                w[indices, k] = amplitudes[jmp_id]
                jmp_id = jmp_id + 1

    # Creating the observations
    y = w + noise_std * torch.randn(*w.shape)
    # creating the dictionary
    D_true = torch.zeros((p, len(edgelist)))
    for ind, (u, v) in enumerate(edgelist):
        D_true[u, ind] = 1
        D_true[v, ind] = -1
    return w, y, D_true


def jumps_fixnum_ranpos_fixamp(p=50, L=200, jumps_num=1, amplitudes=[0., 1.],
                               noise_std=0.01, seed=None):
    """
    Create a dataset: fixed jumps num, rand jumps positions, and fixed amp.

    Parameters
    ----------
    p : signal's dimensionality. The default is 50.
    L : number of observed/original signals. The default is 200.
    jumps_num :  #jumps per signal.  The default is 1.
    amplitudes:  amplitudes of the jumps. Must have (jumps_num + 1) elements.
                Type: list/tensor/array. The default is [0.,1.].
    noise_std : noise standard deviation. The default is 0.01.
    seed: if None, no random-seed is specified for torch. Default is None.

    Returns
    -------
    w : pxL (original signals)
    y : pxL (observed signals)
    D : pxp (A true dictionary that fits the dataset without normalization).
    """
    # setting the torch seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    w = torch.zeros((p, L))

    for k in range(L):
        # To assure having DISTINCT jumps_num jumps.
        while True:
            positions = torch.sort(torch.randint(0, p, (jumps_num,))).values
            if len(positions.unique()) == jumps_num:
                break
        w_current = amplitudes[0] * torch.ones(p)
        for j in range(jumps_num):
            w_current[positions[j]:p] = amplitudes[j+1]
        w[:, k] = w_current

    # Creating the observations
    y = w + noise_std * torch.randn(*w.shape)

    # creating the dictionary
    D_true = -torch.eye(p)
    D_true[p-1, 0] = 1
    for r in range(0, p-1):
        D_true[r, r+1] = 1
    return w, y, D_true.T


def jumps_fixnum_ranpos_ranamp(p=50, L=200, jumps_num=1, amplitude_rng=[0, 1.],
                               noise_std=0.01, seed=None):
    """
    Create a dataset: fixed jumps num, rand jumps positions, random amplitudes.

    Parameters
    ----------
    p : signal's dimensionality. The default is 50.
    L : number of observed/original signals. The default is 200.
    jumps_num :  #jumps per signal.  The default is 1.
    amplitude_rng:  The range in which each signal must lie [min_rng, max_rng].
                Type: list/tensor/array. The default is [0.,1.].
    noise_std : noise standard deviation. The default is 0.01.
    seed: if None, no random-seed is specified for torch. Default is None.

    Returns
    -------
    w : pxL (original signals)
    y : pxL (observed signals)
    D : pxp (A true dictionary that fits the dataset without normalization).
    """
    # setting the torch seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    w = torch.zeros((p, L))
    amplitudes = (amplitude_rng[1]-amplitude_rng[0]) * \
        torch.rand((jumps_num, L)) + amplitude_rng[0]

    for k in range(L):
        # To assure having DISTINCT jumps_num jumps.
        while True:
            positions = torch.sort(torch.randint(0, p, (jumps_num,))).values
            if len(positions.unique()) == jumps_num:
                break
        w_current = amplitudes[0, k] * torch.ones(p)
        for j in range(jumps_num-1):
            w_current[positions[j]:positions[j+1]] = amplitudes[j+1, k]
        w[:, k] = w_current

    # Creating the observations
    y = w + noise_std * torch.randn(*w.shape)

    # creating the dictionary
    D_true = -torch.eye(p)
    D_true[p-1, 0] = 1
    for r in range(0, p-1):
        D_true[r, r+1] = 1
    return w, y, D_true.T


def random_jumps(p=50, L=200, jumps_max_num=5, noise_std=0.01, seed=None):
    """
    Create a dataset that has random jumps number, position and amplitude.

    Parameters
    ----------
    p : signal's dimensionality. The default is 50.
    L : number of observed/original signals. The default is 200.
    jumps_max_num : maximum jumps per signal.  The default is 5.
    noise_std : noise standard deviation. The default is 0.01.
    seed: if None, no random-seed is specified for torch. Default is None.

    Returns
    -------
    w : pxL (original signal)
    y : pxL (observed signal)
    D : pxp (A true dictionary that fits the dataset without normalization).
    """
    # setting the torch seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # generating how many jumps per each observation.
    jumps_num = torch.randint(
        low=0, high=jumps_max_num+1, size=(L,))
    w = torch.zeros((p, L))

    for k in range(L):
        positions = torch.sort(torch.randint(1, p-1, (jumps_num[k],))).values
        amplitudes = torch.rand(jumps_num[k]+1)
        w_current = amplitudes[0] * torch.ones(p)
        for j in range(jumps_num[k]):
            w_current[positions[j]:p] = amplitudes[j+1]
        w[:, k] = w_current

    # Creating the observations
    y = w + noise_std * torch.randn(*w.shape)

    # creating the dictionary
    D_true = -torch.eye(p)
    D_true[p-1, 0] = 1
    for r in range(0, p-1):
        D_true[r, r+1] = 1
    return w, y, D_true.T


def syn_dataset(noise_std=0.1):
    """Build a simple dataset from ciricular-shift (very simple)."""
    # constructing w,y
    p = 50
    # first block
    w = np.ones([p, 1])
    w[10:20, :] = 2
    w[30:40, :] = 0
    for i in range(p-1):
        w = np.concatenate((w, np.roll(w[:, 0][:, None], i+1, axis=0)), axis=1)

    # second block
    w = np.concatenate((w, np.ones([p, 1])), axis=1)
    w[0:10, p] = 0
    w[10:20, p] = 1
    w[20:30, p] = 2
    w[30:40, p] = 3
    w[40:p, p] = 4
    for i in range(p-1):
        w = np.concatenate((w, np.roll(w[:, p][:, None], i+1, axis=0)), axis=1)

    # third block
    w = np.concatenate((w, np.ones([p, 1])), axis=1)
    w[0:10, 2*p] = 5
    w[10:20, 2*p] = 3
    w[20:30, 2*p] = -5
    w[30:40, 2*p] = 3
    w[40:p, 2*p] = 5
    for i in range(p-1):
        w = np.concatenate(
            (w, np.roll(w[:, 2*p][:, None], i+1, axis=0)), axis=1)

    # print('w size is : {}'.format(w.shape))
    w = w.T
    np.random.shuffle(w)
    w = w.T

    # generating the noisy observations
    y = w + noise_std * np.random.randn(*w.shape)

    # creating the dictionary
    D_true = -np.eye(p)
    D_true[p-1, 0] = 1
    for r in range(0, p-1):
        D_true[r, r+1] = 1
    return w, y, D_true


def bsds500img286092(p=128, noise_std=.1):
    """Return the 128x128 image 286092."""
    r, c = 100, 200
    data_path = osp.join(osp.dirname(osp.realpath(__file__)), "..",
                         "BSR/BSDS500/data/images/train/286092.jpg")
    w = Image.open(data_path).convert('L')
    w = torch.from_numpy(
        np.array(w.getdata()).reshape(w.size[1], w.size[0]))
    print(type(w))
    print(w.shape)
    w = 1.*(w - w.min())
    w = w/w.max()
    plt.imshow(w, cmap='gray')
    plt.title('full image before cropping')
    plt.show()
    w = w[r:r + p, c:c+p]
    plt.imshow(w, cmap='gray')
    plt.title('Ground truth image')
    plt.show()
    y = w + noise_std*torch.randn(*w.shape)
    plt.imshow(y, cmap='gray')
    plt.title('Noisy image')
    plt.show()
    D_true = -torch.eye(p)
    D_true[p-1, 0] = 1
    for r in range(0, p-1):
        D_true[r, r+1] = 1
    return w, y, D_true.T


def bsds500imgtest(p=128, noise_std=.1):
    """Return the 128x128 image 286092."""
    r, c = 150, 50
    data_path = osp.join(osp.dirname(osp.realpath(__file__)), "..",
                         "BSR/BSDS500/data/images/train/372047.jpg")
    w = Image.open(data_path).convert('L')
    w = torch.from_numpy(
        np.array(w.getdata()).reshape(w.size[1], w.size[0]))
    print(type(w))
    print(w.shape)
    w = 1.*(w - w.min())
    w = w/w.max()
    plt.imshow(w, cmap='gray')
    plt.title('full image before cropping')
    plt.show()
    w = w[r:r + p, c:c+p]
    plt.imshow(w, cmap='gray')
    plt.title('Ground truth image')
    plt.show()
    y = w + noise_std*torch.randn(*w.shape)
    plt.imshow(y, cmap='gray')
    plt.title('Noisy image')
    plt.show()
    D_true = -torch.eye(p)
    D_true[p-1, 0] = 1
    for r in range(0, p-1):
        D_true[r, r+1] = 1
    return w, y, D_true.T


def lena(noise_std=.2):
    # data_path = osp.join(osp.dirname(osp.realpath(__file__)), "..",
    #                  "BSR/BSDS500/data/images/train/286092.jpg")
    fname = os.path.join(os.path.dirname(__file__), '..', 'lena.dat')
    f = open(fname, 'rb')
    lena = np.array(pickle.load(f))
    f.close()
    lena = torch.from_numpy(lena)
    lena = lena.float()
    lena = lena - lena.min()
    lena = lena/lena.max()
    w = lena
    p = w.shape[0]
    plt.imshow(w, cmap='gray')
    plt.title('Ground truth image')
    plt.show()
    y = w + noise_std*torch.randn(*w.shape)
    plt.imshow(y, cmap='gray')
    plt.title('Noisy image')
    plt.show()
    D_true = -torch.eye(p)
    D_true[p-1, 0] = 1
    for r in range(0, p-1):
        D_true[r, r+1] = 1
    return w, y, D_true.T


def bsds500(p, noise_std=.1):
    w = torch.empty(p, 0)
    y = torch.empty(p, 0)
    ds_dir = osp.join(osp.dirname(osp.realpath(__file__)), "..")
    tr_dir = data_path = osp.join(ds_dir, "BSR/BSDS500/data/images/train")
    files = os.listdir(tr_dir)
    for fn in files:
        _, filename = os.path.split(fn)
        _, ext = os.path.splitext(filename)
        if ext.lower() == '.jpg':
            # import image
            w_cur = Image.open(osp.join(tr_dir, fn)).convert('L')
            w_cur = torch.from_numpy(
                np.array(w_cur.getdata()).reshape(w_cur.size[1], w_cur.size[0]))
            # crop a square pxp
            r0 = torch.randint(0, w_cur.shape[0]-p, (1,))
            c0 = torch.randint(0, w_cur.shape[1]-p, (1,))
            w_cur = w_cur[r0:r0+p, c0:c0+p]
            w_cur = w_cur.float()
            # scale to [0,1]
            w_cur = w_cur - w_cur.min()
            w_cur = w_cur / w_cur.max()
            # gen obs'
            y_cur1 = w_cur + noise_std * torch.randn(*w_cur.shape)
            y_cur2 = w_cur + noise_std * torch.randn(*w_cur.shape)
            # concatenate
            w = torch.cat((w, w_cur), dim=1)
            w = torch.cat((w, w_cur.T), dim=1)
            y = torch.cat((y, y_cur1), dim=1)
            y = torch.cat((y, y_cur2.T), dim=1)
    D_true = -torch.eye(p)
    D_true[p-1, 0] = 1
    for r in range(0, p-1):
        D_true[r, r+1] = 1
    return w, y, D_true.T


def mnist(p=28, noise_std=.05, train = True):
    transform = torchvision.transforms.Compose([
        # you can add other transformations in this list
        torchvision.transforms.ToTensor()
    ])
    imagenet_data = torchvision.datasets.MNIST(
        root='MNIST', download=True, transform=transform,
        train = train)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=4,
                                              shuffle=True)
    sz = imagenet_data.__len__()
    rows = torch.cat(tuple(imagenet_data[i][0].squeeze()
                           for i in range(sz)), dim=0)
    cols = torch.cat(tuple(imagenet_data[i][0].squeeze()
                           for i in range(sz)), dim=1).T
    w = torch.cat((rows, cols), dim=0)
    y = w + noise_std * torch.randn(*w.shape)
    plt.imshow(w[:28], cmap='gray')
    plt.title('Ground truth')
    plt.show()
    plt.imshow(y[:28], cmap='gray')
    plt.title('Observation')
    plt.show()
    w, y = w.T, y.T
    D_true = -torch.eye(p)
    D_true[p-1, 0] = 1
    for r in range(0, p-1):
        D_true[r, r+1] = 1
    return w, y, D_true.T
