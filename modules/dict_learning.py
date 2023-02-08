#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 10:45:12 2020.

@author: hashemghanem
"""

import torch

# memory measuring libs
# import os
# import psutil
# import tracemalloc


class DictLearner:
    r"""This class learns the matrix D.

    Given a dataset $(W,Y)=(w_i, y_i)_{i \in {1,\dots, L}$ such that:
            $ y_i = w_ i + noise $
    where both $ w_i, y_i $ are of dimensions: (px1), we solve the
    following problem in this class:
        $ D = argmin_D E_D(W,Y) = argmin \sum_i || w_i - w(D, y_i) ||_2^2 $
    such that:
            $ w(D,y)= argmin_w (0.5*||y-w||_2^2 ) + \gamma || D'w ||_1 $
    where the dimension of D is: (pxm) (refer to denoise.py for more info).

    Till now we use gradient descent, whose gradient is computed by either:
        1: closed-form expression. In our case, this includes
            that w(D, y_i) must be optimized with the smoothed version
            of the denoising cost function (smoothing the $L_1$ norm).

        2: using PyTorch Autograd framework, that is used to compute
            \nabla E_D[W,Y] (D). The latter gradient is used  in a
            typical gradient descent to compute D.

    In both the previous cases, D is learnt following the update:
        $ D_{t+1}= D_t - \eta2 * \nabla E_D[W,Y] (D)$
    """

    def __init__(self, wlearner, algo='closed_form', eta2=1e-3, dec_r=None,
                 num_of_iter=2000, batch_size=None, valid_size=None,
                 col0sumconst=False, soft_threshold=0, verbose=False):
        r"""
        Initialize the class that learns D.

        Parameters
        ----------
        wlearner : The donoiser that solves the inner problem.
        algo: The chosen algorithm to learn D with gradient descent.
            pass: algo = 'closed_form' to use the explicit closed_form.
            or algo = 'autograd' to use the gradient computed with autograd.
        eta2 : the step size for the gradient descent method .
            The default is 1e-3.
        dec_r : eta2's decaying rate. The default is None.
        num_of_iter :  The default is 2000.
        batch_size : In case each of the D updates is wanted to be based on
                    a stochastic batch from the dataset, not the whole of it
        col0sumconst: If True, then D is projected on the set of matrices whose
            columns sum equal 0. This projection takes place after each outer
            loop iteration. The default is False.
        soft_threshold: the threshold in the L1's proximal op. If different
            from 0, then D is to be regularized with lasso. The default is 0.
        valid_size : the size of the validation set [integer]. Default is None.
        """
        self.wlearner = wlearner
        self.algo = algo
        self.eta2 = eta2
        self.dec_r = dec_r
        self.num_of_iter = num_of_iter
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.col0sumconst = col0sumconst
        self.soft_threshold = soft_threshold
        self.verbose = verbose

        # Self.loss handles the evolution of E_D(X,Y) w.r.t D-iter index.
        # all will be updated later when computed.
        self.loss = torch.zeros(num_of_iter)
        if self.valid_size is not None:
            self.valid_loss = torch.zeros(num_of_iter)
        self.grad = None
        self.D = None

    def closed_form_gradient(self, w_true, w, D):
        """
        Compute the closed-form gradient of E_D(w_i,y_i) w.r.t D.

        This can be seen as the directional derivative of w(D,y_i)
        w.r.t D in the direction (w(D,y_i)-w_i).

        Parameters
        ----------
        w_true : the original data points ($w_i$) of dimension (px batch_size).
        w : the correspondent $w(D, y_i)$ of dimenstion (px batch_size).
        D : The current value of the dictionary D.

        Returns
        -------
        loss: a float number equal to  avg E_D (w_i, y_i) over the batch.
        dw : the directional derivative of $w(D,y_i)$ w.r.t D (pxm).
        """
        batch_size = w.shape[1]
        p = w.shape[0]
        m = D.shape[1]
        # reshaping w from (p x batch_size) to (batch_size x p x 1)
        w = w.T.reshape(batch_size, p, 1)
        w_true = w_true.T.reshape(batch_size, p, 1)
        # u is of dimension (batch_size x m x 1)
        u = D.T @ w

        # calculating the grandiant of $\Gamma$ term (L1 term) w.r.t u
        den = (u * u + self.wlearner.eps**2 * torch.ones(u.shape))**.5
        # grad_gamma : (batch_size x m x1)
        grad_gamma = self.wlearner.gamma * u / den

        # calculating the hessian matrix of $\Gamma$ term w.r.t u
        den = (u * u + self.wlearner.eps**2 * torch.ones(u.shape))**(1.5)
        hessian_gamma = self.wlearner.eps**2 * torch.ones(u.shape)
        # hessian_gamma : (batch_size x m x 1)
        hessian_gamma = self.wlearner.gamma * (hessian_gamma / den)
        # preparing to make the hessian a batch of square matrices, i.e.
        # (batch_size x m x m). See the next step.
        hessian_gamma = hessian_gamma.repeat(1, 1, m)

        for i in range(batch_size):
            # hessian_gamma[i, :, 0] contains the second derivative of
            # $\Gamma$ term w.r.t the variation vector u_i = D.T * w_i.
            # Still hessian_gamma is of dim: (batch_size x m x m)
            hessian_gamma[i, :, :] = torch.diag(hessian_gamma[i, :, 0])

        # delta : (batch_size x p x p)
        delta = torch.eye(p).reshape(1, p, p).repeat(batch_size, 1, 1) +\
            D @ hessian_gamma @ (D.T)
        delta_inv = torch.inverse(delta)

        # z_bar is of dim (batch_size x p x 1)
        z_bar = delta_inv @ (w-w_true)

        # pay attention to the reshapes, which stands for the batch work.
        # we want grad_gamma to have the shape (batch_size x 1 x m)
        # (hessian_gamma @ D.T @ z_bar) is of shape (batch_size x m x 1)
        # convert it to (batch_size x 1 x m)
        # dw holds batch_size gradients of E_D(w_i,y_i) w.r.t D
        # dw : (batch_size x p x m)
        dw = -z_bar @ grad_gamma.reshape(batch_size, 1, m) - \
            w @ (hessian_gamma @ D.T @ z_bar).reshape(batch_size, 1, m)

        # compute the batch_size different losses E_D(w_i,y_i)
        # loss : (batch_size x 1)
        loss = 1/2 * torch.sum((w-w_true)**2, dim=1)

        # return the average over the batch
        return (loss.mean(dim=0).item(), dw.mean(dim=0))

    def shuffle_dataset(self, w, y):
        """Shuffle in-parrellel both x and y."""
        perm = torch.randperm(w.shape[1])
        return w[:, perm], y[:, perm]

    def optimize_D(self, w, y, D, w_init=None):
        r"""
        Optimize D by passing the dataset.

        it returns the learnt D with a matrix of the training
        loss $E_D$ with respect to iteration num.
        L is number of observations.

        Parameters
        ----------
        w : clean signal without noise(pxL)
        y : observed signal(pxL)
        D : initialization of the dictionary (NxP) (must be given)
        w_init : one initialization to be passed to the denoiser. (px1)

        Returns
        -------
        loss : array of $E_D$ versus the iteration number.

        D : the learnt dictionary.
        """
        # process is used to measure used memory
        # process = psutil.Process(os.getpid())
        # tracemalloc.start()

        # split the dataset into training-validation if specified
        if self.valid_size is not None:
            w_val = w[:, :self.valid_size]
            y_val = y[:, :self.valid_size]
            w = w[:, self.valid_size:]
            y = y[:, self.valid_size:]

        # for decaying the learning rate eta2
        eta2 = self.eta2

        # we define batch_size to keep a trace to the original passed one
        # which is important if it is None and we call this method different
        # times with different dataset size.
        batch_size = self.batch_size

        # we define k to iterate over observations
        k = 0

        # if stochastic learning (batches) is not required by the user.
        if batch_size is None:
            batch_size = w.shape[1]

        for itr in range(self.num_of_iter):
            # memory measuring block
            # current, peak = tracemalloc.get_traced_memory()
            # print(f"Current memory usage is {current / 10**6}MB;",
            #       f"Peak was {peak / 10**6}MB")
            # print('memory :\n', process.memory_info().rss)
            # print('Memory details before iter:', itr,
            #       '\n', psutil.virtual_memory())

            if self.dec_r is not None:
                # decaying step size
                eta2 = self.eta2 / (1. + itr * self.dec_r)

            if self.algo == 'autograd':
                # setting the request-grad in PyTorch
                D.requires_grad_(True)
                # the following is important because setting the .requires_grad
                # doesn't reset the .grad of a tensor.
                D.grad = None

            self.grad = torch.zeros(D.detach().shape)

            # if we finished a pass on the training dataset (epoch)
            # shuffle -> start again
            if k+batch_size > w.shape[1]:
                w, y = self.shuffle_dataset(w, y)
                k = 0

            # Starting processing the batch.
            # Compute w(D,y_k) for the current D, w_est has shape px batch_size
            # Costw_k is the cost of the inner problem (avg over the batch)
            (inner_cost, w_est) = self.wlearner.optimize_w(
                y[:, k:k+batch_size], D, w_init)

            # compute the gradient (avg over the batch). dE is of shape (pxm)
            # E_D is the average loss over the batch ( scalar)
            if self.algo == 'closed_form':
                (E_D, dE_D) = self.closed_form_gradient(
                    w[:, k:k+batch_size], w_est, D)
                self.grad = dE_D
            elif self.algo == 'autograd':
                # remember: retain_graph=0 didnt work with the memory pbm.
                E_D = torch.mean(
                    1/2 * torch.sum((w_est - w[:, k:k+batch_size])**2, dim=0))
                # backward to compute the grad
                E_D.backward()
                # this is important to prevent memory overflow.
                w_est = w_est.detach().clone()
                E_D = E_D.detach().clone()

            # add the k-th loss to the batch overall loss
            self.loss[itr] = E_D

            # updating k
            k = k+batch_size

            # if autograd was used, then assign D.grad to self.grad
            if self.algo == 'autograd':
                self.grad = D.grad

            # D will refer to another tensor after this update (mutable obj).
            # So we reset the grad of what it is refering to now
            # since we don't need it anymore.
            D.requires_grad_(False)
            D.grad = None

            # computing the validation loss if needed
            if self.valid_size is not None:
                (inner_cst, w_est) = self.wlearner.optimize_w(y_val, D, w_init)
                self.valid_loss[itr] = torch.mean(
                    1/2 * torch.sum((w_est - w_val)**2, dim=0))

            # verbose: print the losses of this iteration
            if self.valid_size is not None and self.verbose is True:
                print('D_iter:{:<6} Training_loss={:<23.20f}, Val_loss={:<20.20f}'
                      .format(itr,  self.loss[itr], self.valid_loss[itr]))
            elif self.valid_size is None and self.verbose is True:
                print('D_iter:{:<6} Training_loss={:<23.20f}'.format(
                    itr,  self.loss[itr]))

            # updating D
            D = D - eta2 * self.grad
            # project D on the set of matrices whose cols sum to 0.
            if self.col0sumconst is True:
                D = (D - D.mean(dim=0))
            # Soft thresholding, the "if" line can be deleted.
            # But this way makes the code easier to follow.
            if self.soft_threshold != 0 and itr > 2000 and itr % 50 == 0:
                # The following is not really a soft thresholding, we just try
                # the following to see if it works before modifying the code.
                minn = torch.max(torch.min(D, dim=0).values).item()
                maxx = torch.min(torch.max(D, dim=0).values).item()
                D[torch.logical_and(D < maxx, D > minn)] = 0

        self.D = D
        return (self.loss, D)

    def get_grad(self):
        r"""Get the grad $\nabla w[D,y] (D)$ of the last itr."""
        return self.grad

    def get_validloss(self):
        """Return the validation loss."""
        return self.valid_loss


class DictLearner_2d_signals_inner_alternative:
    r"""This class learns the matrix D.

    Given two clean/noisy images $(W,Y)$ such that:
            $ Y = W + noise $
    where both $ W,Y $ are of dimensions: (pxp), we solve the
    following problem in this class:
        $ D = argmin_D E_D(W,Y) = argmin || W - W(D, Y) ||_F^2$
    such that $W(D,Y)$ is the final solution of the alternative opt:
            $W_{k+1/2}(D,Y)= argmin_W (0.5*||W_k-W||_F^2 ) + \gamma || D^TW ||_1 $
            $W_{k+1}(D,Y)= argmin_W (0.5*||W_{k+1/2}-W||_F^2 ) + \gamma || D^TW^T ||_1$
    where the dimension of D is: (pxm) =(pxp) (refer to denoise.py for more info).

    Till now we use gradient descent, whose gradient is computed by either:
        1: using PyTorch Autograd framework, that is used to compute
            $\nabla E_D[W,Y] (D)$. The gradient is used  in a
            typical gradient descent to compute D.

    D is learnt following the update:
        $ D_{t+1}= D_t - \eta2 * \nabla E_D[W,Y] (D)$
    """

    def __init__(self, wlearner, algo='closed_form', eta2=1e-3, dec_r=None,
                 num_of_iter=2000, batch_size=None, valid_size=None,
                 col0sumconst=False, soft_threshold=0, verbose=False):
        r"""
        Initialize the class that learns D.

        Parameters
        ----------
        wlearner : The donoiser that solves the inner problem.
        algo: The chosen algorithm to learn D with gradient descent.
            pass: algo = 'closed_form' to use the explicit closed_form.
            or algo = 'autograd' to use the gradient computed with autograd.
        eta2 : the step size for the gradient descent method .
            The default is 1e-3.
        dec_r : eta2's decaying rate. The default is None.
        num_of_iter :  The default is 2000.
        batch_size : In case each of the D updates is wanted to be based on
                    a stochastic batch from the dataset, not the whole of it
        col0sumconst: If True, then D is projected on the set of matrices whose
            columns sum equal 0. This projection takes place after each outer
            loop iteration. The default is False.
        soft_threshold: the threshold in the L1's proximal op. If different
            from 0, then D is to be regularized with lasso. The default is 0.
        valid_size : the size of the validation set [integer]. Default is None.
        """
        self.wlearner = wlearner
        self.algo = algo
        self.eta2 = eta2
        self.dec_r = dec_r
        self.num_of_iter = num_of_iter
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.col0sumconst = col0sumconst
        self.soft_threshold = soft_threshold
        self.verbose = verbose

        # Self.loss handles the evolution of E_D(X,Y) w.r.t D-iter index.
        # all will be updated later when computed.
        self.loss = torch.zeros(num_of_iter)
        if self.valid_size is not None:
            self.valid_loss = torch.zeros(num_of_iter)
        self.grad = None
        self.D = None

    def optimize_D(self, w, y, D, w_init=None):
        r"""
        Optimize D by passing the dataset.

        it returns the learnt D with a matrix of the training
        loss $E_D$ with respect to iteration num.

        Parameters
        ----------
        w : clean signal without noise(pxp)
        y : observed signal(pxp)
        D : initialization of the dictionary (NxP) (must be given)
        w_init : one initialization to be passed to the denoiser. (px1)

        Returns
        -------
        loss : array of $E_D$ versus the iteration number.

        D : the learnt dictionary.
        """
        # process is used to measure used memory
        # process = psutil.Process(os.getpid())
        # tracemalloc.start()

        # split the dataset into training-validation if specified
        if self.valid_size is not None:
            w_val = w[:, :self.valid_size]
            y_val = y[:, :self.valid_size]
            w = w[:, self.valid_size:]
            y = y[:, self.valid_size:]

        # for decaying the learning rate eta2
        eta2 = self.eta2

        # we define batch_size to keep a trace to the original passed one
        # which is important if it is None and we call this method different
        # times with different dataset size.
        batch_size = self.batch_size

        # we define k to iterate over observations
        k = 0

        # if stochastic learning (batches) is not required by the user.
        if batch_size is None:
            batch_size = w.shape[1]

        for itr in range(self.num_of_iter):
            # memory measuring block
            # current, peak = tracemalloc.get_traced_memory()
            # print(f"Current memory usage is {current / 10**6}MB;",
            #       f"Peak was {peak / 10**6}MB")
            # print('memory :\n', process.memory_info().rss)
            # print('Memory details before iter:', itr,
            #       '\n', psutil.virtual_memory())

            if self.dec_r is not None:
                # decaying step size
                eta2 = self.eta2 / (1. + itr * self.dec_r)

            if self.algo == 'autograd':
                # setting the request-grad in PyTorch
                D.requires_grad_(True)
                # the following is important because setting the .requires_grad
                # doesn't reset the .grad of a tensor.
                D.grad = None
            self.grad = torch.zeros(D.detach().shape)

            # Starting processing the batch.
            # Compute w(D,y_k) for the current D, w_est has shape px batch_size
            breaking_counter = 0
            while True:
                breaking_counter += 1
                (inner_cost_1, w_est_1) = self.wlearner.optimize_w(y, D, w_init)
                E_D = torch.mean(1/2 * torch.sum((w_est_1 - w)**2, dim=0))
                print(
                    f"inner_cost 1: {inner_cost_1:.3f}, outer_cost: {E_D.item():.3f}")
                (inner_cost_2, w_est_2) = self.wlearner.optimize_w(y.T, D, w_init)
                w_est_2 = w_est_2.T
                E_D = torch.mean(1/2 * torch.sum((w_est_2 - w)**2, dim=0))
                print(
                    f"inner_cost 2: {inner_cost_2:.3f}, outer_cost: {E_D.item():.3f}")
                w_est = (w_est_1 + w_est_2)/2
                if breaking_counter == 1:
                    # if ((w_est-w_old).abs().max() / w_old.abs().max()) < 1e-3:
                    # print('Inner loop itr is: {}'.format(itr))
                    break
            E_D = torch.mean(1/2 * torch.sum((w_est - w)**2, dim=0))
            print(f" Overall outer_cost: {E_D.item():.3f}")

            # while True:
            #     breaking_counter += 1
            # (inner_cost_1, w_est) = self.wlearner.optimize_w(
            #     w_est, D, w_init)
            # print("inner_cost 1", inner_cost_1)
            # w_est = torch.transpose(w_est, 0, 1)
            # (inner_cost_2, w_est) = self.wlearner.optimize_w(
            #     w_est, D, w_init)
            # print("inner_cost 2", inner_cost_2)
            # w_est = torch.transpose(w_est, 0, 1)
            # if breaking_counter == 4:
            #     # if ((w_est-w_old).abs().max() / w_old.abs().max()) < 1e-3:
            #     # print('Inner loop itr is: {}'.format(itr))
            #     break

            # compute the gradient (avg over the batch). dE is of shape (pxp)
            # E_D is the average loss over the batch ( scalar)
            if self.algo == 'autograd':
                # remember: retain_graph=0 didnt work with the memory pbm.
                E_D = torch.mean(
                    1/2 * torch.sum((w_est - w)**2, dim=0))
                # backward to compute the grad
                E_D.backward()
                # this is important to prevent memory overflow.
                w_est = w_est.detach().clone()
                E_D = E_D.detach().clone()

            # add the k-th loss to the batch overall loss
            self.loss[itr] = E_D

            # if autograd was used, then assign D.grad to self.grad
            if self.algo == 'autograd':
                self.grad = D.grad.detach().clone()

            # D will refer to another tensor after this update (mutable obj).
            # So we reset the grad of what it is referring to now
            # since we don't need it anymore.
            D.requires_grad_(False)
            D.grad = None

            # computing the validation loss if needed
            if self.valid_size is not None:
                (inner_cst, w_est) = self.wlearner.optimize_w(y_val, D, w_init)
                self.valid_loss[itr] = torch.mean(
                    1/2 * torch.sum((w_est - w_val)**2, dim=0))

            # verbose: print the losses of this iteration
            if self.valid_size is not None and self.verbose is True:
                print('D_iter:{:<6} Training_loss={:<23.20f}, Val_loss={:<20.20f}'
                      .format(itr,  self.loss[itr], self.valid_loss[itr]))
            elif self.valid_size is None and self.verbose is True:
                print('D_iter:{:<6} Training_loss={:<23.20f}'.format(
                    itr,  self.loss[itr]))

            # updating D
            D = D - eta2 * self.grad
            # project D on the set of matrices whose cols sum to 0.
            if self.col0sumconst is True:
                D = (D - D.mean(dim=0))

        self.D = D
        return (self.loss, D)

    def get_grad(self):
        r"""Get the grad $\nabla w[D,y] (D)$ of the last itr."""
        return self.grad

    def get_validloss(self):
        """Return the validation loss."""
        return self.valid_loss
