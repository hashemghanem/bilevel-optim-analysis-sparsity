#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:31:19 2020.

@author: hashemghanem
"""

import torch
import numpy as np


class Denoiser:
    r"""
    Class of methods that learn $w$ in the inner loop.

    Given a dictionary D and an observation y, This class solves
    the follwing denoising problem:
        $ w(D,y)= argmin J_D(w,y)
                = argmin (0.5*||y-w||_2^2 ) + \gamma * || D'w ||_1 $
        where the dictionary D is of dimension (pxm), w, y : (px batch_size)



    Till now this can be done with one of the following algorithms:
        1: (Smooth) smoothening the $ || . ||_1 $ norm in the former formula
            by fixing a small value $\epsilon$ and replace the norm by:
                $ || w ||_1 -> \sum ( \sqrt( w_i^2 + \epsilon^2 ) )$
            With this version, the cost function is differntiable and convex,
            so we can use gradient descent to solve it.

        2: (Dual_ISTA) using forward-backward split on the dual problem
        3: (Dual_FISTA)  on the dual problem
    """

    def __init__(self, gamma=1., algo=('smooth', 1e-3), lr=.001, dec_r=None,
                 num_of_iter=50, tolerance=None, verbose=False):
        r"""
        Instancialize this class.

        Parameters
        ----------
        gamma :  The default is 1.

        algo :  The algorithm you want to use:
            - for smoothening the L1 norm  pass a tuple:('smooth', epsilon).
            - for dual_ISTA method pass ('dual_ISTA')
            - for dual_FISTA method pass ('dual_FISTA')
            default is ('smooth', 1e-3).

        lr : the step size. If None, then the step size is computed locally:
            $lr = 1.95/L_{grad}$. $L_{grad}$ here is the Lipschitz-constant
            of the gradient of the terms in the cost function that we minize
            with gradient steps, either in the gradient descent algo
            (smooth L1) or in the forward-backward split (Dual_ISTA/FISTA).
            Thus, convergence is guaranteed.
            The default is 0.001.

        dec_r : the step size decaying rate. The default is None.

        num_of_iter : the number of iterations if tolerance is None.
            maximum number of iterations allowed if tolerance is specified,
            i.e. to prevent infinite number of iterations if the tolerance
            is not reached. The default is 50.

        tolerance : the tolerance criteion that stops the inner loop if
            $\| \Delta w\|_{\infty} / \| w \|_{\infty}$ < tolerance.
            The same rule applies on $z$ instead of $w$ when solving
            the dual pbm. The default is None.

        verbose :  The default is False.
        """
        self. gamma = gamma
        self.algo = algo[0]
        if algo[0] == 'smooth':
            self.eps = algo[1]
        self.lr = lr
        self.dec_r = dec_r
        self.num_of_iter = num_of_iter
        self.tolerance = tolerance
        self. verbose = verbose

    def compute_grad_wrt_w(self, y, D, w):
        r"""
        Compute the gradient of the $J_D(y,w)$ w.r.t w in the smoothed-L1 case.

        Parameters
        ----------
        y : observations (batch) of dim (px batch_size)
        D : Dictionary matrix of dim (pxm)
        w : a batch of points w where to evaluate the gradient (px batch_size)

        Returns
        -------
        cost :  $J_D(y,w)$ evaluated at w.
        grad : Gradient of $J_D(y,w)$ at w.
        """
        u = D.T @ w
        with torch.no_grad():
            m = D.shape[1]
        den = (u * u + self.eps**2 * torch.ones((m, y.shape[1])))**.5
        cost = 1./2 * torch.sum((y-w)**2, dim=0) + \
            self.gamma * torch.sum(den, dim=0)
        grad = -(y-w) + self.gamma * D @ (u / den)
        return (cost, grad)

    def optimize_w_gradient(self, y, D, w=None):
        r"""
        Optimize w considering the smoothed L1 norm.

        Args:
            y ([px batch_size]): observation of w (batch).
            D (pxm): the dictionary (incidence matrix).
            w (px batch_size, optional): [an initialization of w that
                will be learnt in this algorithm]. If None,
                then i.i.d init from a normal dist. is considered.
                Defaults to None.

        Returns:
            w [px batch_size]: the learnt signals from y ($w(D,y_i)$).
            cost: the denoising cost J_D(w,y) evaluated at each learnt w
                and averaged over the batch (scalar).
        """
        if w is None:
            w = torch.randn(*y.shape)

        with torch.no_grad():
            # This is necessary to keep a trace to the original self.lr.
            # This is important for example in case self.lr = None and we
            # want to call this method many times with different D's, thus
            # a different lr should be computed each time.
            lr0 = self.lr
            if lr0 is None:
                # A dynamic lr0 with guaranteed convergence
                lr0 = 1.95/(1 + self.gamma *
                            np.linalg.norm(D @ D.T, ord=2)/self.eps)
            # lr is to handle the decaying learning rate.
            lr = lr0

        itr = -1
        while (True):
            itr = itr + 1

            # decaying the learning rate
            if self.dec_r is not None:
                lr = lr0 / (1. + itr * self.dec_r)

            (cost, grad) = self.compute_grad_wrt_w(y, D, w)

            # break-the-loop cases
            with torch.no_grad():
                # this means you arrived the number (or maximum number)
                # of iterations set by the user
                if itr == self.num_of_iter:
                    print("You exceeded inner-loops max_itr_num")
                    break

                if self.tolerance is not None:
                    if ((lr*grad).abs().max(dim=0).values /
                            w.abs().max(dim=0).values).max() < self.tolerance:
                        # print(f"It took in the inner loop: {itr:<20}")
                        break

            w = w - lr * grad
            if self.verbose is True:
                if itr % 10 == 0:
                    print('Iter: {}, Cost= {}'.format(itr, cost.mean().item()))

        (cost, _) = self.compute_grad_wrt_w(y, D, w)
        return (cost.mean().item(), w)

    def optimize_w_dual(self, y, D, z=None):
        """
        Optimize w solving the dual problem.

        Args:
            y ([px batch_size]): the observation of w
            D (pxm): the incidence matrix ()
            z (mx batch_size, optional): [an initialization of z that
            will be learnt in this algorithm]. Defaults to None.

        Returns:
            w [px batch_size]: the learnt signal from y_i, D, i.e. (w(D,y_i))
            cost: the denoising cost J_D(w,y) evaluated at each learnt w,
                then averaged over the batch (scalar).
        """
        with torch.no_grad():
            m = D.shape[1]
            # This is necessary to keep a trace to the original self.lr.
            # This is important for example in case self.lr = None and we
            # want to call this method many times with different D's, thus
            # a different lr should be computed each time.
            lr0 = self.lr
            if lr0 is None:
                # dynamic lr0 with guaranteed convergence
                lr0 = 1./np.linalg.norm(D.T @ D, ord=2)
            lr = lr0

        # init z
        if z is None:
            z = torch.randn(m, y.shape[1])
        # z_prev is the previous z (z in the previous update)
        z_prev = torch.zeros(z.shape)

        # needed in the case of FISTA
        tk, tk_1 = 1., 1.

        itr = -1
        while (True):
            itr = itr + 1
            # decaying the learning rate
            if self.dec_r is not None:
                lr = lr0 / (1. + itr * self.dec_r)

            # Here we go with the updates, first compute the dual cost
            dual_cost = 1/2 * torch.sum((D @ z - y/self.gamma)**2, dim=0)
            # In FISTA case, compute the point you'll update + update tk,tk_1.
            q = z + (tk_1-1)/tk * (z-z_prev)
            tk_1 = tk
            tk = (1 + (1+4*tk**2)**.5)/2
            # Memorize the current z
            z_prev = z
            # if fista is chosen, consider the different point to update
            if self.algo == 'dual_FISTA':
                z = q
            grad = D.T @ (D @ z - y/self.gamma)
            # update z and get the new point
            z = z - lr * grad
            indices = z > 1.0
            z[indices] = 1.
            indices = z < -1.0
            z[indices] = -1.0

            # break-the-loop cases
            with torch.no_grad():
                # this means you arrived the number (or maximum number)
                # of iterations set by the user
                if itr == self.num_of_iter-1:
                    print("You exceeded inner-loops max_itr_num")
                    break
                if self.tolerance is not None:
                    if ((z-z_prev).abs().max(dim=0).values /
                        z_prev.abs().max(dim=0).values).max()\
                            < self.tolerance:
                        # print('Inner loop itr is: {}'.format(itr))
                        break

            if self.verbose is True and itr % 10 == 0:
                print('Iter: {}, Dual_Cost= {}'.format(
                    itr, dual_cost.mean().item()))

        # Finally, we learned z (D,y), compute w(D,y) from z(D,y)
        w = y - self.gamma * D @ z
        cost = 1/2 * torch.sum((y - w)**2, dim=0) + \
            self.gamma * torch.sum(torch.abs(D.T @ w), dim=0)
        return (cost.mean().item(), w)

    def optimize_w(self, y, D, initial_point=None):
        r"""
        Optimize w in the inner loop.

        Parameters
        ----------
        y :  the noisy observations of dim (px batch_size)
        D : Dictionary matrix of dim (pxm)
        initial_point :
            Initialization of w (px batch_size) when using the smooth L1.
            and of $D.T*w$ (mx batch_size) when solving the dual problem.
            if None, we initialize here i.i.d from the Normal Distribution.
            The default is None.

        Returns
        -------
        cost : The denoising cost J_D(w,y) evaluated after the last iteration,
            and averaged over the batch (scalar).
        w : Denoised signals (the solution $w(D,y_i)$) (px batch_size).
        """
        if self.algo == 'smooth':
            return self.optimize_w_gradient(y, D, w=initial_point)
        elif self.algo.split('_')[0] == 'dual':
            return self.optimize_w_dual(y, D, z=initial_point)
