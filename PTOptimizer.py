import math
import torch
from torch.optim import Optimizer

class NativeOptim(Optimizer):
  
  def __init__(self, params, lr=1e-3, eps=1e-8,
               weight_decay=0.0001,
               gradient_scale=1.0,
               gradient_l2_norm=0,
               variance_reduction=False,
               gradient_noise=0.0, gradient_noise_decay=0.55,
               gradient_noise_rel_grad_norm=0.0,
               gradient_clip=0.0, **kwargs
               ):
    if not 0.0 <= lr:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if not 0.0 <= eps:
      raise ValueError("Invalid epsilon value: {}".format(eps))
    if not 0.0 <= weight_decay:
      raise ValueError("Invalid weight decay value: {}".format(weight_decay))
    defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay,
                    gradient_scale=gradient_scale,
                    gradient_l2_norm=gradient_l2_norm, variance_reduction=variance_reduction,
                    gradient_noise=gradient_noise, gradient_noise_decay=gradient_noise_decay,
                    gradient_noise_rel_grad_norm=gradient_noise_rel_grad_norm,
                    gradient_clip=gradient_clip)
    defaults.update(kwargs)
    super(NativeOptim, self).__init__(params, defaults)
    
  @staticmethod
  def _get_average_grad_norm(group):
    total_grad_norm = 0
    n_total_params = 0
    for p in group['params']:
      if p.grad is None:
        continue
      grad = p.grad.data
      total_grad_norm += torch.sum(grad * grad)
      n_total_params += grad.numel()
    return total_grad_norm / n_total_params
    
  def step(self, closure=None):
    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        if grad.is_sparse:
          raise RuntimeError('Nadam does not support sparse gradients, please consider SparseAdam instead')
        # initialize training state dict
        state = self.state[p]
        if 'step' not in state:
          state['step'] = 0
          if group['variance_reduction']:
            eps = 1e-7
            state['taus_x_t'] = torch.empty_like(p.data).fill_(2.1 * (1 + eps))
            state['gamma_nume_sqr'] = torch.empty_like(p.data).fill_(eps)
            state['gamma_deno_sqr'] = torch.empty_like(p.data).fill_(eps)
            state['cov_num_t'] = torch.empty_like(p.data).fill_(eps)
            state['mean_grad'] = torch.empty_like(p.data).fill_(eps)
            state['mean_square_grad'] = torch.empty_like(p.data).fill_(eps)
            state['mean_dx'] = torch.zeros_like(p.data)
            state['mean_square_dx'] = torch.zeros_like(p.data)
            state['old_grad'] = torch.empty_like(p.data).fill_(eps)
            state['old_plain_grad'] = torch.empty_like(p.data).fill_(eps)
            state['mean_curvature'] = torch.empty_like(p.data).fill_(eps)
            state['mean_curvature_sqr'] = torch.empty_like(p.data).fill_(eps)
        state['step'] += 1
        step = state['step']
      
        # multiple optional normalizations
        if group['gradient_scale']:
          gradient_scale = group['gradient_scale']
          if isinstance(gradient_scale, list):
            gradient_scale = gradient_scale[0] if (step - 1) % gradient_scale[1] == 0 else 0
          grad *= gradient_scale
      
        if group['gradient_l2_norm']:
          grad /= torch.norm(grad, p=2) + 1e-10
      
        if group['gradient_noise'] > 0:
          nu = group['gradient_noise']
          gamma = group['gradient_noise_decay']
          sigma = nu / (1 + step) ** gamma
          grad += torch.empty_like(grad).normal_(0, sigma)
      
        if group['gradient_noise_rel_grad_norm'] > 0.0:
          avg_grad_norm = self._get_average_grad_norm(group)
          rel_elem_grad_norm = grad * grad - avg_grad_norm
          sigma = group['gradient_noise_rel_grad_norm']
          noise = torch.empty_like(grad).normal_(0, sigma)
          noise *= torch.clamp(-rel_elem_grad_norm, min=1)
          grad += noise
      
        if group['gradient_clip'] > 0.0:
          grad.clamp(min=-group['gradient_clip'], max=group['gradient_clip'])
      
        if group['variance_reduction']:
          svr, decay, gamma_clip, eps = 0, 0.9, 2.5, 1e-7
          grad /= torch.norm(grad, p=2) + eps
          cond = (step == 1)
          learning_rate = group['lr']
          taus_x_t, cov_num_t = state['taus_x_t'], state['cov_num_t']
          mean_grad, mean_square_grad = state['mean_grad'], state['mean_square_grad']
          gamma_nume_sqr, gamma_deno_sqr = state['gamma_nume_sqr'], state['gamma_deno_sqr']
          mean_dx, mean_square_dx = state['mean_dx'], state['mean_square_dx']
          old_grad, old_plain_grad = state['old_grad'], state['old_plain_grad']
          mean_curvature, mean_curvature_sqr = state['mean_curvature'], state['mean_curvature_sqr']
        
          msdx = cond * (grad ** 2) + (1 - cond) * mean_square_dx
          mdx = cond * grad + (1 - cond) * mean_dx
        
          new_mean_squared_grad = mean_square_grad * decay + (grad ** 2) * (1 - decay)
          new_mean_grad = mean_grad * decay + grad * (1 - decay)
          new_gamma_nume_sqr = gamma_nume_sqr * (1 - 1 / taus_x_t)
          new_gamma_nume_sqr += ((grad - old_grad) * (old_grad - new_mean_grad)) ** 2 / taus_x_t
          new_gamma_deno_sqr = gamma_deno_sqr * (1 - 1 / taus_x_t)
          new_gamma_deno_sqr += ((new_mean_grad - grad) * (old_grad - new_mean_grad)) ** 2 / taus_x_t
        
          gamma = gamma_nume_sqr.sqrt() / torch.sqrt(gamma_deno_sqr + eps)
          gamma = torch.clamp(gamma, min=gamma_clip)
          momentum_step = gamma * new_mean_grad
          corrected_grad_cand = (grad + momentum_step) / (1 + gamma)
          corrected_grad = corrected_grad_cand if svr <= step else grad
          cur_curvature = grad - old_plain_grad
          new_curvature_ave = mean_curvature * (1 - 1 / taus_x_t)
          new_curvature_ave += cur_curvature / taus_x_t
          nc_ave = new_curvature_ave
          new_curvature_sqr_ave = mean_curvature_sqr * (1 - 1 / taus_x_t)
          new_curvature_sqr_ave += cur_curvature ** 2 / taus_x_t
          nc_sq_ave = new_curvature_sqr_ave
          rms_dx_tm1 = torch.sqrt(msdx + group['lr'])
          rms_curve_t = torch.sqrt(new_curvature_sqr_ave + learning_rate)
          delta_x_t = - rms_dx_tm1 / rms_curve_t + cov_num_t / (new_curvature_sqr_ave + learning_rate)
          delta_x_t *= corrected_grad
          new_taus_t = (1 - mdx.sqrt() / (msdx + eps)) * taus_x_t + 1 + eps
          new_mean_square_dx = msdx * (1 - 1 / taus_x_t) + delta_x_t ** 2 / taus_x_t
          new_mean_dx = mean_dx * (1 - 1 / taus_x_t) + delta_x_t / taus_x_t
        
          cond1_l = torch.abs(grad - new_mean_grad)
          cond1_r = 2 * (new_mean_squared_grad - new_mean_grad ** 2).sqrt()
          cond2_l = torch.abs(cur_curvature - nc_ave)
          cond2_r = 2 * (nc_sq_ave - nc_ave ** 2).sqrt()
          cond = (cond1_l.gt(cond1_r) | cond2_l.gt(cond2_r)).float()
          new_taus_t = (1 - cond) * new_taus_t + cond * torch.empty_like(new_taus_t).fill_(2.2)
          new_taus_t = torch.clamp(new_taus_t, min=1.5, max=1e8)
        
          new_cov_num_t = cov_num_t * (1 - 1 / taus_x_t) + delta_x_t * cur_curvature * (1 / taus_x_t)
          grad = -delta_x_t
        
          state['mean_grad'], state['mean_square_grad'] = new_mean_grad, new_mean_squared_grad
          state['mean_dx'], state['mean_square_dx'] = new_mean_dx, new_mean_square_dx
          state['gamma_nume_sqr'], state['gamma_deno_sqr'] = new_gamma_nume_sqr, new_gamma_deno_sqr
          state['taus_x_t'], state['cov_num_t'] = new_taus_t, new_cov_num_t
          state['old_grad'], state['old_plain_grad'] = corrected_grad, grad
          state['mean_curvature'], state['mean_curvature_sqr'] = new_curvature_ave, new_curvature_sqr_ave
      
        if group['weight_decay'] != 0:
          grad.add_(group['weight_decay'], p.data)
          
        p.grad.data = grad
          

class Nadam(NativeOptim):
  
  def __init__(self, params, betas=(0.9,0.999), nadam_decay=0.004, **kwargs):
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
    super(Nadam, self).__init__(params, betas=betas, nadam_decay=nadam_decay, **kwargs)
    
  def step(self, closure=None):
    super(Nadam, self).step(closure)
    
    loss = None
    if closure is not None:
      loss = closure()
    
    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        state = self.state[p]
        if 'mt_pre' not in state:
          state['mt_pre'] = torch.zeros_like(p.data)
          state['nt_pre'] = torch.zeros_like(p.data)
          state['mul_mut_pre'] = 1.0
        step = state['step']
        mt_pre = state['mt_pre']
        nt_pre = state['nt_pre']
        mul_mu_t = state['mul_mut_pre']
        beta1, beta2 = group['betas']

        adam_eps = 1e-8
        nadam_decay = group['nadam_decay']

        mu = lambda x: 1 - 0.5 * 0.96 ** (x * nadam_decay)
        mu_t = beta1 * mu(step)
        mu_tn = beta1 * mu(step + 1)
        mul_mu_t *= mu_t
        mul_mu_tn = mul_mu_t * mu_tn

        _grad = grad / (1.0 - mu_tn)
        mt = beta1 * (grad if step == 1 else mt_pre) + (1 - beta1) * grad
        _mt = mt / (1 - mul_mu_tn)
        nt = beta2 * (grad ** 2 if step == 1 else nt_pre) + (1 - beta2) * (grad ** 2)
        _nt = nt / (1 - beta2 ** step)
        __mt = (1 - mu_t) * _grad + (1 - mul_mu_tn) * _mt
        diff = - group['lr'] * __mt / (_nt.sqrt() + adam_eps)

        p.data.add_(diff)

        state['mul_mut_pre'] = mul_mu_t
        state['mt_pre'] = mt
        state['nt_pre'] = nt
      
    return loss
