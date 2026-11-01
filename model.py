import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A, dims):
        if dims == 2:
            x = torch.einsum('ncvl,vw->ncwl', (x, A))
        elif dims == 3:
            x = torch.einsum('ncvl,nvw->ncwl', (x, A))
        else:
            raise NotImplementedError('not implemented for A of dimension ' + str(dims))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a, a.dim())
            out.append(x1)

            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a, a.dim())
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(self, device, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True,
                 in_dim=7,  out_dim=12,residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2,rhythm_cfg=None, granularity='recent',return_feature=False):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports
        self.return_feature = return_feature
        receptive_field = 1
        rk, rd = None, None
        if rhythm_cfg is not None:
            target_layer = int(rhythm_cfg.get('target_layer', -1))
            if granularity == 'recent':
                rk, rd = rhythm_cfg.get('recent', (kernel_size, 1))
            elif granularity == 'daily':
                rk, rd = rhythm_cfg.get('daily',  (kernel_size, 1))
            else:
                rk, rd = rhythm_cfg.get('weekly', (kernel_size, 1))
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if supports is None:
                self.supports = []
            self.gate = nn.Parameter(torch.randn(1, 435,435).to(device), requires_grad=True)
            self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                lid = b * layers + i
                use_k = kernel_size
                use_d = new_dilation
                if rhythm_cfg is not None and lid == target_layer:
                    use_k, use_d = rk, rd
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, use_k), dilation=(1,use_d)))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, use_k), dilation=(1,use_d)))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += (use_k - 1) * use_d
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1, 1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

        self.use_era5_gate = True
        self.era5_lambda = 0.5   
        self.use_era5_hard_gate = False
        self.era5_mu = 0.5       
        self.era5_idx = [1, 2, 3, 4, 5]



    def forward(self, input):
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
    
            raw_seq = input[:, 0, :, -12:]

            diff_seq = torch.diff(raw_seq, dim=-1)
            diff_seq = F.pad(diff_seq, (1, 0), value=0.0)

            xn = F.normalize(diff_seq, p=2, dim=-1, eps=1e-6)
            hebb = torch.bmm(xn, xn.transpose(1, 2))

            S_rs = None
            need_S = (getattr(self, "use_era5_gate", False) or getattr(self, "use_era5_hard_gate", False))
            if need_S and input.size(1) >= 7:

                idx = getattr(self, "era5_idx", [1,2,3,4,5])
                era5 = input[:, idx, :, -1]   # [B,len(idx),N]

                era5 = era5.permute(0, 2, 1).contiguous() # [B, N, 5]

                era5 = era5 / (era5.norm(dim=-1, keepdim=True) + 1e-6)
                S_rs = torch.bmm(era5, era5.transpose(1, 2))   # [B,N,N], in [-1,1]
                S_rs = torch.clamp(S_rs, -1.0, 1.0)
                S_rs = (S_rs + 1.0) / 2.0                      # -> [0,1]

            gate = F.softplus(self.gate)
            adp_logits = hebb * gate  

            if S_rs is not None:
                if getattr(self, "use_era5_gate", False):
                    adp_logits = adp_logits * (1.0 + self.era5_lambda * S_rs)
                if getattr(self, "use_era5_hard_gate", False):
                    adp_logits = adp_logits - self.era5_mu * (1.0 - S_rs)


            adp = F.softmax(F.relu(adp_logits), dim=-1)
            new_supports = self.supports + [adp]



        for i in range(self.blocks * self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)

        feature = F.relu(self.end_conv_1(x))

        if self.return_feature:
            return feature

        x = self.end_conv_2(feature)

        return x


class MultiGWNet(nn.Module):
    def __init__(self, device, dropout, supports, gcn_bool, addaptadj,
                 in_dim, residual_channels, dilation_channels, skip_channels, end_channels,
                 alpha=None):
        super().__init__()

        w0 = np.array(alpha, dtype=np.float32)
        w0 = np.clip(w0 / w0.sum(), 1e-6, 1.0)   
        logit0 = np.log(w0)                      
        self.alpha = nn.Parameter(torch.tensor(logit0, dtype=torch.float32), requires_grad=True)

        rhythm_cfg = {
            'target_layer': 0,
            'recent': (5, 1),   
            'daily':  (5, 2),   
            'weekly': (5, 3),   
        }

        self.g_r = gwnet(device, dropout, supports=supports,
                         gcn_bool=gcn_bool, addaptadj=addaptadj,
                         in_dim=in_dim, residual_channels=residual_channels,
                         dilation_channels=dilation_channels,
                         skip_channels=skip_channels, end_channels=end_channels,
                         rhythm_cfg=rhythm_cfg, granularity='recent',return_feature=True)

        self.g_d = gwnet(device, dropout, supports=supports,
                         gcn_bool=gcn_bool, addaptadj=addaptadj,
                         in_dim=in_dim, residual_channels=residual_channels,
                         dilation_channels=dilation_channels,
                         skip_channels=skip_channels, end_channels=end_channels,
                         rhythm_cfg=rhythm_cfg, granularity='daily',return_feature=True)

        self.g_w = gwnet(device, dropout, supports=supports,
                         gcn_bool=gcn_bool, addaptadj=addaptadj,
                         in_dim=in_dim, residual_channels=residual_channels,
                         dilation_channels=dilation_channels,
                         skip_channels=skip_channels, end_channels=end_channels,
                         rhythm_cfg=rhythm_cfg, granularity='weekly',return_feature=True)
        self.end_conv_1 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=end_channels,
            kernel_size=(1,1)
        )

        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=12,  
            kernel_size=(1,1)
        )
        self.g_r.era5_idx = [2, 3, 5]        
        self.g_d.era5_idx = [1, 2, 3, 5]     
        self.g_w.era5_idx = [4, 1]          

        self.df_proj = nn.Linear(3, 3)
        nn.init.zeros_(self.df_proj.weight)
        nn.init.zeros_(self.df_proj.bias)

        self.last_alpha_bt = None

    @property
    def fuse_weights(self):

        w = torch.softmax(self.alpha, dim=0).detach().cpu().numpy()
        return [float(wi) for wi in w]

    @property
    def df_weights_mean(self):
  
        if self.last_alpha_bt is None:
            return None
        w = self.last_alpha_bt.mean(dim=(0, 1))   # [3]
        return [float(x) for x in w.detach().cpu().numpy()]

    def forward(self, x_recent, x_daily, x_weekly):

        hr = self.g_r(x_recent)
        hd = self.g_d(x_daily)
        hw = self.g_w(x_weekly)

        hr_mean = hr.mean(dim=2).mean(dim=1)
        hd_mean = hd.mean(dim=2).mean(dim=1)
        hw_mean = hw.mean(dim=2).mean(dim=1)

        Y_cat = torch.stack([hr_mean, hd_mean, hw_mean], dim=-1)

   
        Y_cat = Y_cat - Y_cat.mean(dim=-1, keepdim=True)         
        std = Y_cat.std(dim=-1, keepdim=True) + 1e-6
        Y_cat = Y_cat / std                                      

        delta = self.df_proj(Y_cat)                               # [B, T, 3]

        logits_static = self.alpha.view(1, 1, 3)                  # [1,1,3]
        logits = logits_static + delta                            # [B, T, 3]

        logits[:,0,0] += 0.3   
        logits[:,0,1] += 0.1  
        alpha_bt = torch.softmax(logits, dim=-1) 
        self.last_alpha_bt = alpha_bt                            

        H_stack = torch.stack([hr, hd, hw], dim=1)

        alpha_perm = alpha_bt.permute(0, 2, 1)                    # [B, 3, T]
        alpha_exp = alpha_perm.unsqueeze(2).unsqueeze(2)          # [B, 3, 1, 1, T]
        H = (alpha_exp * H_stack).sum(dim=1)
        H = F.relu(self.end_conv_1(H))
        y = self.end_conv_2(H)

        return y





