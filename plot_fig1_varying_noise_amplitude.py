"""Figure 1: learn D while varying noise variance."""
import matplotlib.pylab as pylab
from matplotlib import pyplot as plt
import torch
from modules.datasets import jumps_fixnum_ranpos_fixamp
from modules.denoise import Denoiser
from modules.dict_learning import DictLearner

'''
In what follows:
    p: signal dimensionality.
    L: number of observations.
    m: the second dimension of D.
'''

# Generating the dataset
p, sz, jumps_num = 64, 10000, 4
amps = [0., 10, 0, 10., 0.]
# noise_std  of the noise will vary later in a for loop.
noise_std_values = [.5, .75, 1., 2., 3.]

# Defining the inner loop solver which learns w(D,y).
gamma = 1.
inner_algo = ("dual_FISTA", None)
# Learning rate.
eta1, inner_dec_r = None, None
tolerance, inner_num_of_iter = .0001, 2000
wlearner = Denoiser(gamma=gamma, algo=inner_algo,
                    lr=eta1, dec_r=inner_dec_r,
                    num_of_iter=inner_num_of_iter,
                    tolerance=tolerance, verbose=False)


# Setting up the outer loop params.
outer_algo = 'autograd'
eta2, outer_dec_r = 0.065, 0.002
outer_num_of_iter = 10000
batch_size, valid_size = 64, 256
col0sumconst, soft_threshold = True, 0


# Preparing the pyplot setting.
plt.style.use('tableau-colorblind10')
params = {'legend.fontsize': 'large',
          'axes.labelsize': 'xx-large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large',
          "axes.titlesize": "x-large"}
pylab.rcParams.update(params)
# Define the figure instance.
fig, axs = plt.subplots(1, 6, figsize=(20, 6), constrained_layout=True)
subcaptions = [r"(a) True dictionary $D_{TV}$", r"(b) $\sigma=0.5$",
               r"(c) $\sigma=0.75$", r"(d) $\sigma=1$",
               r"(e) $\sigma=2$", r"(f) $\sigma=3$"]

for i, noise_std in enumerate(noise_std_values):
    print(f"-----------------------------\n-----------------------------"
          f"\n Processing for noise std = {noise_std:.2f}")
    # Generating the dataset with the according noise variance
    w1, y1, Dtv_not_scaled = jumps_fixnum_ranpos_fixamp(p=p, L=sz,
                                                        jumps_num=jumps_num,
                                                        amplitudes=amps,
                                                        noise_std=noise_std,
                                                        seed=None)
    # Horizentally roll previous signals to better represent the input space.
    w, y = w1, y1
    for shft in range(1, p):
        w = torch.cat((w, torch.roll(w1, shft, 0)), dim=1)
        y = torch.cat((y, torch.roll(y1, shft, 0)), dim=1)
    # Shuffle
    perm = torch.randperm(w.shape[1])
    w, y = w[:, perm], y[:, perm]

    # instantiating a dictionary learner (outer loop solver)
    optimizer = DictLearner(wlearner, algo=outer_algo, eta2=eta2,
                            dec_r=outer_dec_r, num_of_iter=outer_num_of_iter,
                            batch_size=batch_size, valid_size=valid_size,
                            col0sumconst=col0sumconst,
                            soft_threshold=soft_threshold, verbose=True)
    # Initializing D
    D_init = .01*torch.randn(*Dtv_not_scaled.shape)

    # Training stage
    (training_loss, D_est) = optimizer.optimize_D(w, y, D_init, w_init=None)
    valid_loss = optimizer.get_validloss()
    # Permuting columns for better visualization.
    weights = (D_est-torch.roll(D_est, -1, 0)).abs()
    mx_indices = weights.max(dim=0).indices
    visited_dest = torch.zeros(D_est.shape[1])
    visited_src = torch.zeros(D_est.shape[1])
    D_perm = torch.zeros(D_est.shape)
    for j in range(D_est.shape[1]):
        dest_ind = mx_indices[j]
        if visited_dest[dest_ind] == 0:
            visited_dest[dest_ind] = 1
            visited_src[j] = 1
            D_perm[:, dest_ind] = D_est[:, j]
    D_perm[:, visited_dest == 0] = D_est[:, visited_src == 0]
    axs[i+1].imshow(D_perm)
    axs[i+1].set_title(subcaptions[i+1])

imsh = axs[0].imshow(Dtv_not_scaled)
fig.subplots_adjust(right=0.93)
cbar_ax = fig.add_axes([0.95, 0.35, 0.01, 0.3])
fig.colorbar(imsh, cax=cbar_ax)
plt.savefig("varying_noise.pdf", bbox_inches='tight')
plt.show()
