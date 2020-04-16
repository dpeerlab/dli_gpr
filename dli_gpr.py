import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints

import tqdm

def gaussian_kernel(x, lamb, x2=None):
    """Compute gaussian kernel for 1D tensor"""
    if x2 is None:
        x2 = x.unsqueeze(-1)
    x = x.unsqueeze(-1)
    return torch.exp(-(x - x2.T).pow(2) / lamb**2)

class dli_gpr:
    """ Implementation of Gaussian Process regression with non-isotropic noise

    Attributes (following notation from Bishop p. 306):
        y: 1D torch tensor of time points
        t: 1D torch tensor of observed responses
        cluster_sizes: 1D torch tensor of cluster sizes for each time point
        lamb (float): initial value of kernel bandwidth
    """
    def __init__(self, y:torch.tensor, t:torch.tensor, 
        cluster_sizes:torch.tensor, lamb = None):

        # Number of observations
        self.n = len(y)

        # Input points (time points in our case)
        self.y = y

        # Observed response variable (e.g. tumor burden or proportion of cells)
        self.t = t

        # Number of cells in each time point
        self.cluster_sizes = cluster_sizes

        # Lambda: kernel bandwidth
        self.lamb = lamb

        # if None, set to median distance between time points
        if self.lamb is None:
            y_unsqueezed = self.y.unsqueeze(-1)
            self.lamb = torch.median(((y_unsqueezed - y_unsqueezed.T).pow(2)).pow(0.5))

        # initial value to use as shape parameter for Gamma distribution
        self.gam = torch.FloatTensor([1.])

    def initialize_variables(self, jitter=1e-5):
        """Initialize the kernel matrix and noise prior
        """
        # center t so that it has zero mean
        self.t_mean = torch.mean(self.t)
        self.t_centered = self.t - self.t_mean

        # covariance matrix
        self.scale = gaussian_kernel(self.y, self.lamb)
        self.scale_tril = torch.cholesky(self.scale + torch.eye(self.n) * jitter)

        # compute inverse cluster sizes for gamma prior
        inverse_cluster_sizes = 1./self.cluster_sizes

        # rate parameter for Gamma prior (beta inverse ~ variance of observations. Want lower beta_inverse for bigger clusters)
        # initialize beta so that the MEAN of the gamma prior is proportional to the inverse weights of best linear unbiased estimator
        self.beta_inverse = inverse_cluster_sizes / torch.sum(inverse_cluster_sizes) * self.n

        # beta is proportional to the expected precision 
        self.beta = 1./self.beta_inverse

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
        self.beta = pyro.param("a").detach() / pyro.param("b").detach()

        # save noise variances
        self.beta_inverse = 1./ self.beta

        return losses

    def conditional_distribution(self, new_y):
        """Compute conditional distribution of t given observations and some new input points y
        Returns mean and covariance of conditional distribution"""

        # compute kernel for original and new observations
        k = gaussian_kernel(self.y, self.lamb, new_y)
        c = gaussian_kernel(new_y, self.lamb)

        # compute inverse covariance
        precision = torch.inverse(self.scale + torch.diag(self.beta_inverse))

        # compute mean
        mean = k.T @ precision @ (self.t_centered) + self.t_mean

        # compute covariance
        empirical_variance = torch.var(mean)
        # empirical_variance = torch.mean(self.beta_inverse)
        # empirical_variance = self.gam
        cov = (c + empirical_variance * torch.eye(len(new_y))) - k.T @ precision @ k

        return mean, cov


