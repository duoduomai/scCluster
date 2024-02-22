import torch
import torch.nn as nn

class Instance_C3(nn.Module):
    def __init__(self, batch_size, zeta):
        super(Instance_C3, self).__init__()
        self.batch_size = batch_size
        self.zeta = zeta

    def forward(self, z_i, z_j):
        z_i = (z_i - z_i.mean(dim=0)) / z_i.std(dim=0)
        z_j = (z_j - z_j.mean(dim=0)) / z_j.std(dim=0)
        z = torch.cat((z_i, z_j), dim=0)

        multiply = torch.matmul(z, z.T)

        multiply = torch.clamp(multiply, min=-3, max=9)

        a = torch.ones([self.batch_size])
        mask = 2 * (torch.diag(a, -self.batch_size) + torch.diag(a, self.batch_size) + torch.eye(2 * self.batch_size))
        mask = mask.cuda()
        
        exp_mul = torch.exp(multiply)

        numerator = torch.sum(torch.where((multiply + mask) > self.zeta, exp_mul, torch.zeros(multiply.shape).cuda()), dim=1)

        den = torch.sum(exp_mul, dim=1)

        return -torch.sum(torch.log(torch.div(numerator, den))) / self.batch_size
