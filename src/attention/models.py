import math

import torch


def freeze(m: torch.nn.Module) -> torch.nn.Module:
    for p in m.parameters():
        p.requires_grad = False


def unfreeze(m: torch.nn.Module) -> torch.nn.Module:
    for p in m.parameters():
        p.requires_grad = True


def freeze_bn(m):
    for (name, child) in m.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            for param in child.parameters():
                param.requires_grad = False
            child.eval()
        else:
            freeze_bn(child)


def unfreeze_bn(m):
    for (name, child) in m.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            for param in child.parameters():
                param.requires_grad = True
            child.train()
        else:
            unfreeze_bn(child)


# ------------------------------------------------------------------------


# Loss function
class EntropyLoss(torch.nn.Module):

    def __init__(self, mu: float = None, sigma: float = None):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x: torch.Tensor):
        if self.mu is not None:
            mu = self.mu
        else:
            mu = torch.mean(x)

        if self.sigma is not None:
            sigma = self.sigma
        else:
            sigma = torch.std(x, unbiased=True) + 1e-6

        px = -0.5 * ((x - mu) / sigma)**2
        px = torch.exp(px) / math.sqrt(2 * math.pi)

        retval = -torch.mean(px * torch.log(px + 1e-6))
        return retval


# ------------------------------------------------------------------------


class CheaplabSegmenter(torch.nn.Module):

    def __init__(self, clss: int = 4, num_heads=1):
        super().__init__()

        self.cheaplabs = torch.nn.ModuleList([
            torch.hub.load(
                'jamesmcclain/CheapLab:21a2cd4cfe02ca48c7fdd58f47e121236c09e657',
                'make_cheaplab_model',
                num_channels=12,
                out_channels=1,
                trust_repo=True,
                skip_validation=True,
            ) for _ in range(clss)
        ])

        self.attn = torch.nn.ModuleList([
            torch.hub.load(
                'jamesmcclain/CheapLab:21a2cd4cfe02ca48c7fdd58f47e121236c09e657',
                'make_cheaplab_model',
                num_channels=12,
                out_channels=1,
                trust_repo=True,
                skip_validation=True,
            ) for _ in range(num_heads)
        ])

    def freeze_resnet(self):
        freeze(self.cheaplabs)

    def unfreeze_resnet(self):
        unfreeze(self.cheaplabs)

    def forward(self, x, pos=None):
        bs, ss, cs, xs, ys = x.shape

        def compute_weights(x,i):
            xs2 = max(xs//(4**i), 1)
            ys2 = max(ys//(4**i), 1)
            x = x.reshape(-1, cs, xs, ys)
            x = torch.nn.functional.interpolate(x, mode='bilinear', size=(xs2, ys2))
            x = self.attn[i](x)
            x = torch.nn.functional.interpolate(x, size=(xs, ys))
            x = x.reshape(bs, ss, 1, xs, ys)
            x = x - torch.mean(x)
            return x

        weights = [
            torch.nn.functional.softmax(compute_weights(x, i), dim=1)
            for i in range(len(self.attn))
        ]

        y = []
        for cheaplab in self.cheaplabs:
            yi = cheaplab(x.reshape(-1, cs, xs, ys)).reshape(bs, ss, 1, xs, ys)
            # yapf: disable
            yi = torch.sum(torch.stack([(wj * yi) for wj in weights], dim=1), dim=(1, 2))  # weighted combination
            # yapf: enable
            y.append(yi)
        y = torch.cat(y, dim=1)
        y = torch.nn.functional.normalize(y, p=1.0, dim=1)
        return y


# ------------------------------------------------------------------------


class AttentionSegmenter(torch.nn.Module):

    def __init__(self,
                 arch,
                 state,
                 size,
                 clss: int = 4,
                 num_heads: int = 1,
                 dropout: float = 0.0):
        super().__init__()

        self.resnet = torch.hub.load(
            'jamesmcclain/pytorch-fpn:02eb7d4a3b47db22ec30804a92713a08acff6af8',
            'make_fpn_resnet',
            name=arch,
            fpn_type='panoptic',
            num_classes=6,
            fpn_channels=256,
            in_channels=12,
            out_size=(size, size),
            trust_repo=True,
            skip_validation=True,
        )
        if state is not None:
            self.resnet.load_state_dict(torch.load(
                state, map_location=torch.device('cpu')),
                                        strict=True)
        self.embed = self.resnet[0]
        self.fpn = self.resnet[1:]
        self.fpn[0][-1] = torch.nn.Conv2d(128,
                                          clss,
                                          kernel_size=(1, 1),
                                          stride=(1, 1))

        self.arch = arch
        self.size = size
        if self.arch in {'resnet18', 'resnet34'}:
            self.shapes = [
                [64, size // 4, size // 4],
                [64, size // 4, size // 4],
                [128, size // 8, size // 8],
                [256, size // 16, size // 16],
                [512, size // 32, size // 32],
                [512, size // 32, size // 32],
            ]
        else:
            raise Exception(f'Not prepared for {self.arch}')

        self.attn = torch.nn.ModuleList()
        for shape in self.shapes:
            dim = shape[0]
            layer = torch.nn.ModuleList(
                [torch.nn.Linear(dim, 1) for _ in range(num_heads)])
            self.attn.append(layer)

    def freeze_resnet(self):
        freeze(self.embed)

    def unfreeze_resnet(self):
        unfreeze(self.embed)

    def forward(self, x, pos=None):
        # yapf: disable
        bs, ss, cs, xs, ys = x.shape
        x = x.reshape(-1, cs, xs, ys)  # reshape for resnet
        x = self.embed(x)  # embed

        y = []
        for shape in self.shapes:
            y.append(
                torch.zeros(bs,
                            *shape,
                            dtype=torch.float32,
                            device=x[0].device))

        for i in range(len(x)):
            shape = self.shapes[i]
            xi = x[i]
            xi = xi.reshape(bs, ss, *shape)  # Restore "original" shape post resnet
            xi = xi.transpose(-3, -1)  # move embeddings to end
            xi = torch.stack([
                torch.nn.functional.softmax(head(xi), dim=-1) * xi
                for head in self.attn[i]
            ], dim=1)  # apply attention
            xi = torch.sum(xi, dim=(1, 2))  # weighted combination
            xi = xi.transpose(-3, -1)  # restore embeddings to original dimension
            y[i] = torch.nn.functional.normalize(xi, p=1.0, dim=1)  # normalize

        y = self.fpn(tuple(y))  # pass through fpn
        return y
        # yapf: enable
