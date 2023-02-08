from modules.datasets import bsds500imgtest, lena, mnist
from modules.myplot import plot, imshow, plotlogxy, plotlogy
import torch
from modules.denoise import Denoiser
from matplotlib import pyplot as plt
from modules.dict_learning import DictLearner, DictLearner_2d_signals_inner_alternative
# from modules.myplot import plot
from modules.datasets import bsds500img286092, bsds500, mnist
p = 28
noise_std = .05
# Defining the inner loop solver which learns w(D,y)
gamma = 1.
epsilon = 1e-3
w, y, Dtv_not_scaled = mnist(p=p, noise_std=noise_std)

# comment/uncomment one of the next 2 lines to choose the solver you want.
inner_algo = ("dual_FISTA", None)
eta1, inner_dec_r = None, None
tolerance, inner_num_of_iter = .0001, 2000
wlearner = Denoiser(gamma=gamma, algo=inner_algo,
                    lr=eta1, dec_r=inner_dec_r,
                    num_of_iter=inner_num_of_iter,
                    tolerance=tolerance, verbose=False)


# Setting up the outer loop params
outer_algo = 'autograd'
eta2, outer_dec_r = 0.1, 0.008
outer_num_of_iter = 60000
batch_size, valid_size = 32, 16
col0sumconst, soft_threshold = True, 0

# instantiating a dictionary learner (outer loop solver)
optimizer = DictLearner(wlearner, algo=outer_algo, eta2=eta2,
                        dec_r=outer_dec_r, num_of_iter=outer_num_of_iter,
                        batch_size=batch_size, valid_size=valid_size,
                        col0sumconst=col0sumconst,
                        soft_threshold=soft_threshold, verbose=True)
# Initializing D
D_init = .01*torch.randn(*Dtv_not_scaled.shape)
noise_error = torch.mean(1/2 * torch.sum((w[:100] - y[:100])**2, dim=0))
print(f"As a reference, the error with D = 0 is : {noise_error:.4f}")

# Training stage
(training_loss, D_est) = optimizer.optimize_D(w, y, D_init, w_init=None)
valid_loss = optimizer.get_validloss()


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

# Search for the best scalar to scale Dtv_not_scaled into Dtv_scaled.
# Note that we use just num_pnts points from the dataset to do that.
sclr = torch.arange(.03, .08, .01)
ED_sclr = torch.zeros(sclr.shape)
for ind in range(len(sclr)):
    (inner_cost_1, w_est_1) = wlearner.optimize_w(
        y[:, :p], sclr[ind]*Dtv_not_scaled)
    (inner_cost_2, w_est_2) = wlearner.optimize_w(
        y[:, :p].T, sclr[ind]*Dtv_not_scaled)
    w_est_2 = w_est_2.T
    w_est = (w_est_1 + w_est_2)/2
    ED_sclr[ind] = torch.mean(1./2*torch.sum((w[:, :p]-w_est)**2, dim=0))
print("The best value to scale Dtv_not_scaled is: {:.5f}\n ".format(
    sclr[ED_sclr.argmin()].item()))
Dtv_scaled = sclr[ED_sclr.argmin()].item() * Dtv_not_scaled
# We want to visualize a comparison between E(D_sclr) as a fun of sclr
# against E(D=0). So we need to compute E(D=0) as follows.
ED0 = torch.mean(1./2*torch.sum((y-w)**2, dim=0))
ED0 = ED0 * torch.ones(sclr.shape)

# Now plot both curves
legends = [r"$D=\lambda \times D_{tv,not-scaled}$", "D=0"]
xlabel, ylabel = r"$\lambda$", r"$\mathcal{E}~(D)$"

plotlogxy(sclr[:, None].repeat(1, 2),
          torch.cat((ED_sclr[:, None], ED0[:, None]), dim=1),
          legend_labels=legends, xlabel=xlabel, ylabel=ylabel,
          file_name=None)

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

# plotting the val/training losses
ED0 = torch.tensor([ED0[0]]*len(training_loss))
EDtv = torch.tensor([ED_sclr.min().item()] * len(training_loss))

legends = ['Training loss', 'Validation Loss',
           'D=0 loss on val set', 'D_tv loss on val set']
all_vctrs = torch.cat((training_loss[:, None], valid_loss[:, None],
                       ED0[:, None], EDtv[:, None]), dim=1)
xlabel, ylabel = 'Iteration index in the outer loop (batch-wise)', r'$\mathcal{E}~(D)$'
title = '0-col-sum constraint'
saving_path = 'TrainValLoss.pdf'
plotlogy(all_vctrs, title=title, ylabel=ylabel, xlabel=xlabel, line_width=2.5,
         legend_labels=legends, marker_size=0, file_name=None)

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

# visualize the learnt D_est
weights = (D_est-torch.roll(D_est, -1, 0)).abs()
mx_indices = weights.max(dim=0).indices
visited_dest = torch.zeros(D_est.shape[1])
visited_src = torch.zeros(D_est.shape[1])

D_perm = torch.zeros(D_est.shape)
for i in range(D_est.shape[1]):
    dest_ind = mx_indices[i]
    if visited_dest[dest_ind] == 0:
        visited_dest[dest_ind] = 1
        visited_src[i] = 1
        D_perm[:, dest_ind] = D_est[:, i]
D_perm[:, visited_dest == 0] = D_est[:, visited_src == 0]

imshow(D_perm, xlabel='Edge index', ylabel='Node index',
       file_name=None, title='Columns-permuted variant of D_est')


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Peformance on an example from the dataset.
(inner_cost_1, w_est_1) = wlearner.optimize_w(y[:, :p], D_est)
(inner_cost_2, w_est_2) = wlearner.optimize_w(y[:, :p].T, D_est)
w_est_2 = w_est_2.T
w_est = (w_est_1 + w_est_2)/2

(inner_cost_1, w_est_1) = wlearner.optimize_w(y[:, :p], Dtv_scaled)
(inner_cost_2, w_est_2) = wlearner.optimize_w(y[:, :p].T, Dtv_scaled)
w_est_2 = w_est_2.T
w_tv = (w_est_1 + w_est_2)/2
w_tv = w_tv
psig = (w[:, :p]**2).mean()
ptv = ((w[:, :p]-w_tv)**2).mean()
pnoise = ((w[:, :p]-y[:, :p])**2).mean()
pest = ((w[:, :p]-w_est)**2).mean()
snrtv = 10 * torch.log10(psig/ptv)
snrobs = 10 * torch.log10(psig/pnoise)
snrest = 10 * torch.log10(psig/pest)


print(
    f"SNR noise {snrobs:.3f}, SNR tv {snrtv:.3f}, SNR estimated {snrest:.3f}")
fig, axs = plt.subplots(1, 4, figsize=(20, 6), constrained_layout=True)
plt.axis('off')
axs[0].imshow(w[:, :p].T, cmap='gray')
axs[0].set_title('Ground truth')
axs[1].imshow(y[:, :p].T, cmap='gray')
axs[1].set_title('Noisy image')
axs[2].imshow(w_est.T, cmap='gray')
axs[2].set_title(r"Denoised by learned $D_{est}$")
axs[3].imshow(w_tv.T, cmap='gray')
axs[3].set_title(r"Denoised by  $D_{tv}$")
plt.savefig("MNIST_performance_on_example_photo.pdf", bbox_inches='tight')
plt.show()
