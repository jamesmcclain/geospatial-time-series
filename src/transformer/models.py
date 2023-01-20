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

    def __init__(self, arch, state, size, d_model, nhead, num_layers, clss: int = 1):
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
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model // 2, 1),
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
        if False:
            x = torch.sum(x * weights, dim=1)
            x = torch.nn.functional.normalize(x, p=1.0, dim=1)
        elif False:
            x = torch.mean(x * weights, dim=1)
        else:
            x = torch.sum(x * weights, dim=1)
            x = torch.nn.functional.normalize(x, p=2.0, dim=1)
        x = self.fc(x)  # pass through fully-connected layer
        return x
