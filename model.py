# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# -----------------------------
# Capsule Layers
# -----------------------------
class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_capsules, capsule_dim):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)

    def squash(self, tensor, dim=-1):
        s2 = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = s2 / (1.0 + s2)
        return scale * tensor / torch.sqrt(s2 + 1e-8)

    def forward(self, x):
        b = x.size(0)
        o = self.relu(self.conv(x))
        o = o.view(b, self.num_capsules, self.capsule_dim, -1)
        o = o.permute(0, 3, 1, 2).contiguous()
        o = o.view(b, -1, self.capsule_dim)
        return self.squash(o)


class DigitCapsules(nn.Module):
    def __init__(self, num_capsules, in_capsules, in_dim, out_dim, num_iterations=3):
        super().__init__()
        self.num_capsules = num_capsules
        self.num_iterations = num_iterations
        self.W = nn.Parameter(0.01 * torch.randn(1, in_capsules, num_capsules, out_dim, in_dim))

    def squash(self, tensor, dim=-1):
        s2 = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = s2 / (1.0 + s2)
        return scale * tensor / torch.sqrt(s2 + 1e-8)

    def forward(self, x):
        B = x.size(0)
        x = x.unsqueeze(2).unsqueeze(4)
        W = self.W.repeat(B, 1, 1, 1, 1)
        u_hat = torch.matmul(W, x).squeeze(-1)
        b_ij = torch.zeros(B, u_hat.size(1), u_hat.size(2), 1, device=x.device)

        for _ in range(self.num_iterations):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j, dim=-1)
            b_ij = b_ij + (u_hat * v_j).sum(dim=-1, keepdim=True)

        return v_j.squeeze(1)


# -----------------------------
# Capsule Network Backbone
# -----------------------------
class CapsuleNetwork(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])  # [B, 512, 7, 7]

        self.primary_capsules = PrimaryCapsules(512, 256, 3, 1, 1, 32, 8)
        self.digit_capsules   = DigitCapsules(num_classes, 32 * 7 * 7, 8, 16)

    def forward(self, x):
        x = self.cnn(x)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        return x.norm(dim=-1)


class SiameseCapsuleNetwork(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.capsule_net = CapsuleNetwork(num_classes)

    def forward(self, x1, x2):
        e1 = self.capsule_net(x1)
        e2 = self.capsule_net(x2)
        return torch.norm(e1 - e2, dim=1, keepdim=True)


# -----------------------------
# Loaders
# -----------------------------
def load_model(path: str, device: torch.device) -> SiameseCapsuleNetwork:
    model = SiameseCapsuleNetwork(num_classes=4).to(device)
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"])
    else:
        model.load_state_dict(obj)
    model.eval()
    return model


def load_references(path: str, device: torch.device, model: SiameseCapsuleNetwork):
    refs = torch.load(path, map_location=device)
    out = {}
    with torch.no_grad():
        for cls, items in refs.items():
            vecs = []
            for t in items:
                t = t.to(device)
                if t.dim() == 4 and t.size(-1) == 224 and t.size(-2) == 224:
                    emb = model.capsule_net(t).squeeze(0)
                else:
                    emb = t.squeeze()
                    if emb.dim() == 2 and emb.size(0) == 1:
                        emb = emb.squeeze(0)
                vecs.append(emb.float().view(-1))
            out[int(cls)] = vecs
    return out
