import torch.optim as optim
from model import *
import util
import torch
class trainer():
    def __init__(self, scaler, in_dim, seq_length, nhid, dropout, lrate, wdecay, device,
                 supports, gcn_bool, addaptadj, fuse_init=None):
        self.model = MultiGWNet(device, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj,
                           in_dim=in_dim,  residual_channels=nhid,
                           dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16, alpha=fuse_init)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)

        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, x_recent, x_daily, x_weekly, real_val):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(x_recent, x_daily, x_weekly)
        output = output.transpose(1, 3)          # [B, T, N, 1]
        real = torch.unsqueeze(real_val, dim=1)  # [B, 1, N, T]


        predict = output

        loss = self.loss(predict, real, 0.0)

        if hasattr(self.model, "last_alpha_bt") and (self.model.last_alpha_bt is not None):
            alpha_bt = self.model.last_alpha_bt   # [B, T, 3]

            ent = -(alpha_bt * (alpha_bt + 1e-8).log()).sum(dim=-1).mean()

            lam = 1e-4  
            loss = loss + lam * (-ent)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, x_recent, x_daily, x_weekly, real_val):
        self.model.eval()

        output = self.model(x_recent, x_daily, x_weekly)
        output = output.transpose(1, 3)          # [B, T, N, 1]
        real = torch.unsqueeze(real_val, dim=1)  # [B, 1, N, T]


        predict = output

        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse

