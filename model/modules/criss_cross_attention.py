import torch
import torch.nn as nn


def inf_mask(B, H, W, device):
    return -torch.diag(torch.full((H,), float("inf"), device=device)).unsqueeze(0).repeat(B * W, 1, 1)


class CCA_Module(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.init_proj = nn.Conv2d(1, embed_dim, kernel_size=1)
        self.query_conv = nn.Conv2d(embed_dim, embed_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(embed_dim, embed_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(embed_dim, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, T = x.size()
        x_in = x.unsqueeze(1)
        x = self.init_proj(x_in)

        _, E, F, T = x.size()

        Q = self.query_conv(x)
        Q_H = Q.permute(0, 3, 1, 2).contiguous().view(B * T, -1, F).permute(0, 2, 1)
        Q_W = Q.permute(0, 2, 1, 3).contiguous().view(B * F, -1, T).permute(0, 2, 1)

        K = self.key_conv(x)
        K_H = K.permute(0, 3, 1, 2).contiguous().view(B * T, -1, F)
        K_W = K.permute(0, 2, 1, 3).contiguous().view(B * F, -1, T)

        V = self.value_conv(x)
        V_H = V.permute(0, 3, 1, 2).contiguous().view(B * T, -1, F)
        V_W = V.permute(0, 2, 1, 3).contiguous().view(B * F, -1, T)

        energy_H = (torch.bmm(Q_H, K_H) + inf_mask(B, F, T, x.device)).view(B, T, F, F).permute(0, 2, 1, 3)
        energy_W = torch.bmm(Q_W, K_W).view(B, F, T, T)

        attn = self.softmax(torch.cat([energy_H, energy_W], dim=3))
        attn_H = attn[:, :, :, 0:F].permute(0, 2, 1, 3).contiguous().view(B * T, F, F)
        attn_W = attn[:, :, :, F:F + T].contiguous().view(B * F, T, T)

        out_H = torch.bmm(V_H, attn_H.permute(0, 2, 1)).view(B, T, -1, F).permute(0, 2, 3, 1)
        out_W = torch.bmm(V_W, attn_W.permute(0, 2, 1)).view(B, F, -1, T).permute(0, 2, 1, 3)

        out = self.gamma * (out_H + out_W) + x
        out = self.out_proj(out)          
        return out.squeeze(1) + x_in.squeeze(1) 