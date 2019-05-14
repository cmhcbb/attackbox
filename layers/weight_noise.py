import torch
import torch.nn.functional as F
from torch.autograd import Function

class NoiseFn(Function):
    @staticmethod
    def forward(ctx, mu, sigma, eps, sigma_0, N):
        eps.normal_()
        ctx.save_for_backward(mu, sigma, eps)
        ctx.sigma_0 = sigma_0
        ctx.N = N
        return mu + torch.exp(sigma) * eps

    @staticmethod
    def backward(ctx, grad_output):
        mu, sigma, eps = ctx.saved_tensors
        sigma_0, N = ctx.sigma_0, ctx.N
        grad_mu = grad_sigma = grad_eps = grad_sigma_0 = grad_N = None
        tmp = torch.exp(sigma)
        if ctx.needs_input_grad[0]:
            grad_mu = grad_output + mu/(sigma_0*sigma_0*N)
        if ctx.needs_input_grad[1]:
            grad_sigma = grad_output*tmp*eps - 1 / N + tmp*tmp/(sigma_0*sigma_0*N)
        return grad_mu, grad_sigma, grad_eps, grad_sigma_0, grad_N

class IdFn(Function):
    @staticmethod
    def forward(ctx, mu, sigma, eps, sigma_0, N):
        return mu

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None

noise_fn = NoiseFn.apply
