"""Implementation of GP regression (vanilla and hierarchical model for heteroskedascic noise)

Todo: model methods are rather unintuitive.
Should modify at some point to imitate sklearn models (with fit, fit_transform, predict methods)
Should also hide initialize variables method
"""

import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints

import tqdm

def gaussian_kernel(x, lamb, x2=None, scale=1.0):
    """Compute gaussian kernel for 1D tensor"""
    if x2 is None:
        x2 = x.unsqueeze(-1)
    x = x.unsqueeze(-1)
    return scale * torch.exp(-(x - x2.T).pow(2) / lamb**2)

class gpr:
    """Vanilla GPR model

    Attributes:
        y: 1D tensor of time points
        t: 1D tensor of responses
        lamb: kernel bandwidth
        sigma: noise variance
        scale: kernel scale
    """
    def __init__(self, y:torch.tensor, t:torch.tensor, lamb=None, sigma=None, scale=None):
        """Constructor: inputs are y and t
        By default, lambda and sigma are inferred from the data but can also be provided
        """
        assert len(y) == len(t), "Lengths of y and t are not equal"

        # number of observations
        self.n = len(y)
        self.y = y # time points
        self.t = t # response variables

        # if lambda is not given, set it to the median distance between all pairs of points
        if lamb is None:
            y_unsqueezed = self.y.unsqueeze(-1)
            self.lamb = torch.median(((y_unsqueezed - y_unsqueezed.T).pow(2)).pow(0.5))
        else:
            self.lamb = torch.tensor(lamb)

        # if sigma is not given, set it to the squared standard error
        if sigma is None:
            self.sigma = torch.var(t) #/ self.n
        else:
            self.sigma = torch.tensor(sigma)

        if scale is None:
            self.sc = torch.var(t)**0.5
        else:
            self.sc = torch.tensor(scale)

    def initialize_variables(self, jitter=1e-6):
        """Initializes kernel matrix
        """
        # set Gaussian kernel
        self.scale = gaussian_kernel(self.y, self.lamb, scale=self.sc)

        # cholesky decomp
        self.scale_tril = torch.cholesky(self.scale + jitter * torch.eye(self.n))

        # center response variable
        self.t_centered = self.t - torch.mean(self.t)

        # store mean
        self.t_mean = torch.mean(self.t)

    def model(self, jitter=1e-4):
        """Generative model

        Sample f from N(O,K)
        Sample t from N(f, sigma)

        There is really no "learning" because we can use conditional Gaussian
        """
        sigma = pyro.param("sigma", self.sigma, constraint=constraints.positive)

        f = pyro.sample("f", dist.MultivariateNormal(torch.zeros(self.n), scale_tril=self.scale_tril))

        # compute likelihood for observation, using beta as observation precision
        pyro.sample("y", dist.MultivariateNormal(f, 
            covariance_matrix=torch.eye(self.n) * sigma), 
            obs=self.t_centered)

    def guide(self):
        """Guide (not really anything to do here)"""
        pass

    def optimize(self, n_steps:int=1):
        """Run SVI for user-specified number of steps. Usually 1000 is good. 
        Returns (list) of ELBO loss associated with each step"""

        # use Adam optimizer
        optimizer = pyro.optim.Adam({"lr": 0.001})

        # ELBO loss
        loss = pyro.infer.Trace_ELBO()

        # approximate inference with ADVI
        svi = pyro.infer.SVI(self.model, self.guide, optimizer, loss)

        losses = []
        for step in tqdm.tqdm(range(n_steps)):
            loss = svi.step()
            losses.append(loss)
        
        # save noise
        self.sigma = pyro.param("sigma").detach()

        return losses

    def conditional_distribution(self, new_y):
        """Compute conditional distribution of t given observations and some new input points y
        Returns mean and covariance of conditional distribution"""

        # compute kernel for original and new observations
        k = gaussian_kernel(self.y, self.lamb, new_y, scale=self.sc)

        # use noiseless distribution
        c = gaussian_kernel(new_y, self.lamb, scale=self.sc) #+ self.sigma * torch.eye(len(new_y))

        # compute inverse covariance
        precision = torch.inverse(self.scale + self.sigma * torch.eye(self.n))
        # precision = torch.inverse(self.scale + 1e-6 * torch.eye(self.n))

        # compute mean
        mean = k.T @ precision @ (self.t_centered) + self.t_mean

        # compute covariance
        cov = c - k.T @ precision @ k

        #print(k.T @ precision @ k)
        return mean, cov


class dli_gpr:
    """ Implementation of Gaussian Process regression with non-isotropic noise

    Attributes (following notation from Bishop p. 306):
        y: 1D torch tensor of time points
        t: 1D torch tensor of observed responses
        cluster_sizes: 1D torch tensor of cluster sizes for each time point
        lamb (float): initial value of kernel bandwidth
    """
    def __init__(self, y:torch.tensor, t:torch.tensor, 
        cluster_sizes:torch.tensor, lamb=None, gam=None, scale=None):

        # Number of observations
        self.n = len(y)

        # Input points (time points in our case)
        self.y = y

        # Observed response variable (e.g. tumor burden or proportion of cells)
        self.t = t

        # Number of cells in each time point
        self.cluster_sizes = cluster_sizes

        # if None, set to median distance between time points
        if lamb is None:
            y_unsqueezed = self.y.unsqueeze(-1)
            self.lamb = torch.median(((y_unsqueezed - y_unsqueezed.T).pow(2)).pow(0.5))
        else:
            # Lambda: kernel bandwidth
            self.lamb = torch.tensor(lamb)

        # initial value to use as shape parameter for Gamma distribution
        # initialize gamma with the square standard error (doesn't really matter, we optimize this)
        self.gam = 1. / torch.var(t)
        
        # scale of Gaussian kernel (just set to variance of the data)
        if scale is None:
            self.sc = torch.var(t)**0.5
        else:
            self.sc = torch.tensor(scale)

    def initialize_variables(self, jitter=1e-5):
        """Initialize the kernel matrix and noise prior
        """
        # center t so that it has zero mean
        self.t_mean = torch.mean(self.t)
        self.t_centered = self.t - self.t_mean

        # covariance matrix
        self.scale = gaussian_kernel(self.y, self.lamb, scale=self.sc)
        self.scale_tril = torch.cholesky(self.scale + torch.eye(self.n) * jitter)

        # precision
        self.beta = self.cluster_sizes * self.n / torch.sum(self.cluster_sizes)
        self.beta_inverse = 1/self.beta

        # compute inverse cluster sizes for gamma prior
        #inverse_cluster_sizes = 1./self.cluster_sizes

        # rate parameter for Gamma prior (beta inverse ~ variance of observations. Want lower beta_inverse for bigger clusters)
        # initialize beta so that the MEAN of the gamma prior is proportional to the inverse weights of best linear unbiased estimator
        #self.beta_inverse = inverse_cluster_sizes / torch.sum(inverse_cluster_sizes) * self.n

        # beta is proportional to the expected precision 
        #self.beta = 1./self.beta_inverse

    def model(self):
        """Generative process"""

        # sample f from zero mean Gaussian
        f = pyro.sample("f", dist.MultivariateNormal(torch.zeros(self.n), scale_tril=self.scale_tril))

        # sample beta from a gamma distribution (this is a precision, not a variance)
        beta = pyro.sample("beta", dist.Gamma(self.gam, self.beta_inverse))

        # compute likelihood for observation, using beta as observation precision
        pyro.sample("y", dist.MultivariateNormal(f, precision_matrix=torch.diag(beta)), obs=self.t_centered)

    def guide(self):
        """Guide with approximate posterior"""

        # initialize Pyro parameters for shape and rate parameter of Gamma prior (mean = a / b)
        b = pyro.param("b", self.beta_inverse.clone(), constraint=constraints.positive)
        a = pyro.param("a", self.gam, constraint=constraints.positive)

        # sample f from zero mean Gaussian
        f = pyro.sample("f", dist.MultivariateNormal(torch.zeros(self.n), scale_tril=self.scale_tril))

        # sample beta from a gamma distribution
        beta = pyro.sample("beta", dist.Gamma(a, b))

    def optimize(self, n_steps:int=1):
        """Run SVI for user-specified number of steps. Usually 1000 is good. 
        Returns (list) of ELBO loss associated with each step"""

        # use Adam optimizer
        optimizer = pyro.optim.Adam({"lr": 0.001})

        # ELBO loss
        loss = pyro.infer.Trace_ELBO()

        # approximate inference with ADVI
        svi = pyro.infer.SVI(self.model, self.guide, optimizer, loss)

        losses = []
        for step in tqdm.tqdm(range(n_steps)):
            loss = svi.step()
            losses.append(loss)
        
        # save precisions
        self.gam = pyro.param("a").detach()

        # save noise variances
        self.beta_inverse = pyro.param("b").detach()

        return losses

    def conditional_distribution(self, new_y):
        """Compute conditional distribution of t given observations and some new input points y
        Returns mean and covariance of conditional distribution"""

        # compute kernel for original and new observations
        k = gaussian_kernel(self.y, self.lamb, new_y, scale=self.sc)

        # use noiseless distribution
        c = gaussian_kernel(new_y, self.lamb, scale=self.sc) + 1./self.gam * torch.eye(len(new_y))

        # compute inverse covariance
        precision = torch.inverse(self.scale + torch.diag(self.beta_inverse/self.gam))
        # precision = torch.inverse(self.scale + 1e-5 * torch.eye(self.scale.shape[0]))

        # compute mean
        mean = k.T @ precision @ (self.t_centered) + self.t_mean

        # compute covariance
        # empirical_variance = torch.var(mean)**0.5
        # empirical_variance = torch.mean(self.beta_inverse)
        cov = c - k.T @ precision @ k
        #cov = (c) - k.T @ precision @ k

        return mean, cov


