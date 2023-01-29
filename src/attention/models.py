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


class ResnetTransformerClassifier(torch.nn.Module):

    def __init__(self,
                 arch,
                 state,
                 size,
                 d_model,
                 nhead,
                 num_layers,
                 clss: int = 1):
        super().__init__()
        self.embed = torch.hub.load(
            'jamesmcclain/pytorch-fpn:02eb7d4a3b47db22ec30804a92713a08acff6af8',
            'make_fpn_resnet',
            name=arch,
            fpn_type='panoptic',
            num_classes=6,
            fpn_channels=256,
            in_channels=12,
            out_size=(size, size))
        self.embed.load_state_dict(torch.load(state), strict=True)
        self.embed = self.embed[0]

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.fc = torch.nn.Linear(d_model, clss)

    def forward(self, x, pos):
        bs, ss, cs, xs, ys = x.shape
        x = self.embed(x.reshape(-1, cs, xs, ys))  # embed
        x = x[-1]  # get last output from resnet backbone
        x = torch.mean(x, dim=(2, 3))  # pool the ~8x8 embeddings
        x = x.reshape(bs, ss, -1)  # shape: batch x seq x embeddings
        # x = torch.cat([x, pos], dim=2)  # XXX concat positional embeddings
        x = x + pos
        bs, ss, ds = x.shape
        cls = (torch.ones(bs, 1, ds) / (bs * ds)).to(
            x.device)  # generate cls tokens
        x = torch.cat([cls, x], axis=1)  # staple cls tokens to the front
        x = self.transformer_encoder(x)  # pass through transformer encoder
        x = self.fc(x[:, 0, :])  # pass through fully-connected layer
        return x


class BaselineClassifier(torch.nn.Module):

    def __init__(self, arch, state, size, d_model: int = 512, clss: int = 1):
        super().__init__()
        self.embed = torch.hub.load(
            'jamesmcclain/pytorch-fpn:02eb7d4a3b47db22ec30804a92713a08acff6af8',
            'make_fpn_resnet',
            name=arch,
            fpn_type='panoptic',
            num_classes=6,
            fpn_channels=256,
            in_channels=12,
            out_size=(size, size))
        self.embed.load_state_dict(torch.load(state), strict=True)
        self.embed = self.embed[0]
        self.fc = torch.nn.Linear(d_model, clss)

    def forward(self, x):
        bs, ss, cs, xs, ys = x.shape
        x = self.embed(x.reshape(-1, cs, xs, ys))  # embed
        x = x[-1]  # get last output from resnet backbone
        _, ds, xs, ys = x.shape
        x = x.reshape(bs, ss, ds, xs, ys)
        x = torch.mean(x, dim=(1, 3, 4))  # average embeddings
        x = self.fc(x)  # pass through fully-connected layer
        return x


class AttentionClassifier(BaselineClassifier):

    def __init__(self, arch, state, size, d_model: int = 512, clss: int = 1):
        super().__init__(arch, state, size, d_model, clss)
        self.poor_mans_attention = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
        )

    def forward(self, x, pos):
        bs, ss, cs, xs, ys = x.shape
        x = self.embed(x.reshape(-1, cs, xs, ys))  # embed
        x = x[-1]  # get last output from resnet backbone
        _, ds, xs, ys = x.shape
        x = x.reshape(bs, ss, ds, xs, ys)
        x = torch.mean(x, dim=(3, 4))  # average embeddings
        weights = self.poor_mans_attention(pos)
        x = torch.sum(x * weights, dim=1)
        x = torch.nn.functional.normalize(x, p=2.0, dim=1)
        x = self.fc(x)  # pass through fully-connected layer
        return x


class AttentionSegmenter(torch.nn.Module):

    def __init__(self, arch, state, size, d_model: int = 512, clss: int = 1):
        super().__init__()

        self.resnet = torch.hub.load(
            'jamesmcclain/pytorch-fpn:02eb7d4a3b47db22ec30804a92713a08acff6af8',
            'make_fpn_resnet',
            name=arch,
            fpn_type='panoptic',
            num_classes=6,
            fpn_channels=256,
            in_channels=12,
            out_size=(size, size))
        self.resnet.load_state_dict(torch.load(state), strict=True)
        self.embed = self.resnet[0]
        self.fpn = self.resnet[1:]
        self.fpn[0][-1] = torch.nn.Conv2d(128,
                                          clss,
                                          kernel_size=(1, 1),
                                          stride=(1, 1))

        if arch in {'resnet18', 'resnet34'}:
            self.dims = [64, 64, 128, 256, 512, 512]
        else:
            raise Exception()

        self.poor_mans_attention = torch.nn.ModuleList()
        for dim in self.dims:
            self.poor_mans_attention.append(
                torch.nn.Sequential(
                    torch.nn.Linear(d_model, dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(dim, dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(dim, dim),
                    torch.nn.ReLU(),
                ))

    def freeze_embed(self):
        freeze(self.embed)

    def unfreeze_embed(self):
        unfreeze(self.embed)

    def forward(self, x, pos):
        # yapf: disable
        bs, ss, cs, xs, ys = x.shape
        x = x.reshape(-1, cs, xs, ys)  # reshape for resnet
        x = self.embed(x)  # embed
        y = []
        for i in range(len(x)):
            xi = x[i]
            _, ds, xs, ys = xi.shape
            xi = xi.reshape(bs, ss, ds, xs, ys)  # restore "original" shape
            wi = self.poor_mans_attention[i](pos)  # compute attention weights
            wi = wi.reshape(bs, ss, ds, 1, 1)  # reshape for element-wise mult
            xi = torch.sum(xi * wi, dim=1)  # apply weights to create composite embedding
            xi = torch.nn.functional.normalize(xi, p=2.0, dim=1)
            y.append(xi)
        y = tuple(y)
        y = self.fpn(y)  # pass through fpn
        return y
        # yapf: enable

class AttentionSegmenter2(AttentionSegmenter):

    def __init__(self, arch, state, size, d_model: int = 512, clss: int = 1):
        super().__init__(arch, state, size, d_model, clss)
        del self.poor_mans_attention
        self.poor_mans_attention = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model // 2, clss),
            torch.nn.ReLU(),
        )
        self.clss = clss

    def forward(self, x, pos):
        bs, ss, cs, xs, ys = x.shape
        x = x.reshape(-1, cs, xs, ys)  # reshape for resnet
        x = self.resnet(x)  # segmentation
        x = x.reshape(bs, ss, self.clss, xs, ys)
        w = self.poor_mans_attention(pos)  # compute attention weights
        w = w.reshape(bs, ss, self.clss, 1, 1)  # reshape for element-wise mult
        x = torch.sum(x * w, dim=1)  # apply weights to create composite result
        x = torch.nn.functional.normalize(x, p=1.0, dim=1)
        return x
