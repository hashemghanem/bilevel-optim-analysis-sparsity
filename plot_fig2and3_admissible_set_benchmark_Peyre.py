"""Figures 2&3:role of the admissible set+benchmark with Peyre et al, 2011."""
import torch
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
from modules.denoise import Denoiser
from modules.dict_learning import DictLearner
from modules.datasets import jumps_fixnum_ranpos_ranamp

'''
In what follows:
    p: signal dimensionality.
    L: number of observations.
    m: the second dimension of D.
'''

# Generate the dataset.
p, L, jumps_num = 64, 640000, 4
amplitude_rng = [0, 10.]
noise_std = .5
w, y, Dtv_not_scaled = jumps_fixnum_ranpos_ranamp(p=p, L=L,
                                                  jumps_num=jumps_num,
                                                  amplitude_rng=amplitude_rng,
                                                  noise_std=noise_std,
                                                  seed=None)

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# First run our algorithm with the admissible set.
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

# Defining the inner loop solver which learns w(D,y)
gamma = 1.
inner_algo = ("dual_FISTA", None)
eta1, inner_dec_r = None, None
tolerance, inner_num_of_iter = .0001, 2000
wlearner = Denoiser(gamma=gamma, algo=inner_algo, lr=eta1, dec_r=inner_dec_r,
                    num_of_iter=inner_num_of_iter, tolerance=tolerance, verbose=False)


# Setting up the outer loop params
outer_algo = 'autograd'
eta2, outer_dec_r = 0.04, 0.0025
outer_num_of_iter = 5000
batch_size, valid_size = 64, 256
col0sumconst, soft_threshold = True, 0

# instantiating a dictionary learner (outer loop solver)
optimizer = DictLearner(wlearner, algo=outer_algo, eta2=eta2,
                        dec_r=outer_dec_r, num_of_iter=outer_num_of_iter,
                        batch_size=batch_size, valid_size=valid_size,
                        col0sumconst=col0sumconst,
                        soft_threshold=soft_threshold, verbose=True)
# Initializing D
D_init = .01*torch.randn(*Dtv_not_scaled.shape)

# Training stage
(tloss_with0col, D_est) = optimizer.optimize_D(w, y, D_init, w_init=None)
valid_loss = optimizer.get_validloss()

# Define  a figure instance.
fig, axs = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
params = {'legend.fontsize': 'large',
          'axes.labelsize': 'xx-large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large'}
pylab.rcParams.update(params)

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

min_scl = D_est.min()
max_scl = D_est.max()

axs[0].imshow(D_perm, vmin=min_scl, vmax=max_scl)

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Second, run our algorithm without the admissible set.
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

# Defining the inner loop solver which learns w(D,y)
gamma = 1.
inner_algo = ("dual_FISTA", None)
eta1, inner_dec_r = None, None
tolerance, inner_num_of_iter = .0001, 2000
wlearner = Denoiser(gamma=gamma, algo=inner_algo,
                    lr=eta1, dec_r=inner_dec_r,
                    num_of_iter=inner_num_of_iter,
                    tolerance=tolerance, verbose=False)


# Setting up the outer loop params
outer_algo = 'autograd'
eta2, outer_dec_r = 0.004, 0.0065
outer_num_of_iter = 5000
batch_size, valid_size = 64, 256
col0sumconst, soft_threshold = False, 0

# instantiating a dictionary learner (outer loop solver)
optimizer = DictLearner(wlearner, algo=outer_algo, eta2=eta2,
                        dec_r=outer_dec_r, num_of_iter=outer_num_of_iter,
                        batch_size=batch_size, valid_size=valid_size,
                        col0sumconst=col0sumconst,
                        soft_threshold=soft_threshold, verbose=True)
# Initializing D
D_init = .01*torch.randn(*Dtv_not_scaled.shape)

# Training stage
(tloss_without0col, D_est) = optimizer.optimize_D(w, y, D_init, w_init=None)
valid_loss = optimizer.get_validloss()

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

im = axs[1].imshow(D_perm, vmin=min_scl, vmax=max_scl)
fig.subplots_adjust(right=1.1)
cbar_ax = fig.add_axes([0.58, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.savefig('with_out_0colsum.pdf', bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Third solve with the smoothing in Peyre et al, 2011
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

# Defining the inner loop solver which learns w(D,y)
gamma = 1.
epsilon = 1e-3
inner_algo = ("smooth", epsilon)
eta1, inner_dec_r = None, None
tolerance, inner_num_of_iter = .0001, 2000
wlearner = Denoiser(gamma=gamma, algo=inner_algo,
                    lr=eta1, dec_r=inner_dec_r,
                    num_of_iter=inner_num_of_iter,
                    tolerance=tolerance, verbose=False)


# Setting up the outer loop params
outer_algo = 'closed_form'
eta2, outer_dec_r = 0.004, 0.0065
outer_num_of_iter = 5000
batch_size, valid_size = 64, 256
col0sumconst, soft_threshold = False, 0

# instantiating a dictionary learner (outer loop solver)
optimizer = DictLearner(wlearner, algo=outer_algo, eta2=eta2,
                        dec_r=outer_dec_r, num_of_iter=outer_num_of_iter,
                        batch_size=batch_size, valid_size=valid_size,
                        col0sumconst=col0sumconst,
                        soft_threshold=soft_threshold, verbose=True)
# Initializing D
D_init = .01*torch.randn(*Dtv_not_scaled.shape)

# Training stage
(tloss_pyre, D_est) = optimizer.optimize_D(w, y, D_init, w_init=None)
valid_loss = optimizer.get_validloss()

# Define a figure instance.
fig, axs = plt.subplots(1, 2, figsize=(12, 5.1), constrained_layout=True)
line_width, marker_size = 2, 0
# Colorblind friendly color cycles
ccycles = ['tableau-colorblind10', 'seaborn-colorblind']

# markers in case needed
mrk = ['s', 'v', 'o', 'x', '3', 'p', '|']
# making the labels/ticks' font larger
params = {'legend.fontsize': 'xx-large',
          'axes.labelsize': 'xx-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)
# Set the color style (cycle + map) you want.
plt.style.use(ccycles[0])
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

im = axs[1].imshow(D_perm,
                   vmin=min_scl, vmax=max_scl)
cbar_ax = fig.add_axes([1., 0.15, 0.02, 0.8])
fig.colorbar(im, cax=cbar_ax)

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Finally, plot loss curves of different algo's.
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------


legends = ['Our algorithm (Alg.1)',  r'$D=0$ validation loss', r'$D_{TV}$ validation loss',
           r'Alg.1 w/o projection $\Pi_{\mathcal{C}}$', r'Alg. proposed in [11]']
xlabel, ylabel = 'Iteration index in the outer loop (batch-wise)', r'Training loss $\mathcal{E}~({D})$'
npoints = outer_num_of_iter

# Evaluated already by feeding D = 0 and D = D_tv_scaled to the inner optimizer
# So when changing the dataset, these losses will change, not only that
# but also the scaling factor \lambda in D_tv_scaled.
ED0 = torch.tensor([7.93]*npoints)
EDtv = torch.tensor([1.8390105962753296]*npoints)


# start plotting
axs[0].plot(tloss_with0col, linewidth=line_width, marker=mrk[2],
            markersize=marker_size, markerfacecolor="None")
next(axs[0]._get_lines.prop_cycler)
axs[0].plot(ED0, linewidth=line_width, marker=mrk[3],
            markersize=marker_size, markerfacecolor="None")
axs[0].plot(EDtv, linewidth=line_width+0.8, marker=mrk[4],
            markersize=marker_size, markerfacecolor="None")
axs[0].plot(tloss_without0col[0:npoints], linewidth=line_width, marker=mrk[1],
            markersize=marker_size, markerfacecolor="None")
axs[0].plot(tloss_pyre[0:npoints], linewidth=line_width, marker=mrk[0],
            markersize=marker_size, markerfacecolor="None")

axs[0].set_xlabel(xlabel)
axs[0].set_ylabel(ylabel)
axs[0].legend(legends)
axs[0].grid()
plt.savefig('peyre_benchmark.pdf', bbox_inches='tight')
plt.show()
