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


class AttentionSegmenter(torch.nn.Module):

    def __init__(self, arch, state, size, d_model: int = 512, clss: int = 4, num_heads=1):
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
            raise Exception(f'Not prepared for {arch}')

        self.self_attn = torch.nn.ModuleList()
        self.q_fcns = torch.nn.ModuleList()
        self.k_fcns = torch.nn.ModuleList()
        for dim in self.dims:
            self.q_fcns.append(torch.nn.Linear(dim, dim))
            self.k_fcns.append(torch.nn.Linear(dim, dim))
            self.self_attn.append(torch.nn.MultiheadAttention(dim, num_heads, batch_first=True))

    def freeze_resnet(self):
        freeze(self.embed)

    def unfreeze_resnet(self):
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
            xi = xi.reshape(bs, ss, ds, xs, ys)  # Restore "original" shape
            stash = []
            for j in range(xs):  # For every spatial nugget ...
                for k in range(ys):  # ...
                    # Compute Q
                    qew = torch.mean(self.q_fcns[i](xi[:, :, :, j, k]), dim=1, keepdim=True)
                    # Compute K
                    kay = self.k_fcns[i](xi[:, :, :, j, k])
                    # Get V
                    vee = xi[:, :, :, j, k]
                    # Stash results
                    result, _ = self.self_attn[i](qew, kay, vee)
                    stash.append(result)
            # Retrieve and save results
            stash = torch.stack(stash, dim=3).reshape(bs, ds, xs, ys)
            y.append(stash)
        y = tuple(y)
        y = self.fpn(y)  # pass through fpn
        return y
        # yapf: enable


class AttentionSegmenterIn(AttentionSegmenter):

    def __init__(self, arch, state, size, d_model: int = 512, clss: int = 1):
        raise NotImplementedError()
        super().__init__(arch, state, size, d_model, clss)
        del self.poor_mans_attention
        # self.poor_mans_attention = torch.nn.Sequential(
        #     torch.nn.Linear(d_model, d_model // 2),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(d_model // 2, 12),
        #     torch.nn.ReLU(),
        # )
        self.poor_mans_attention = torch.nn.Linear(d_model, 12)


    def forward(self, x, pos):
        bs, ss, cs, xs, ys = x.shape
        w = self.poor_mans_attention(pos)  # compute attention weights
        w = w.reshape(bs, ss, cs, 1, 1)  # reshape for element-wise mult
        x = torch.sum(x * w, dim=1)  # produce composite input
        x = x.reshape(-1, cs, xs, ys)  # reshape for resnet
        x = self.resnet(x)  # segmentation
        return x


class AttentionSegmenterOut(AttentionSegmenter):

    def __init__(self, arch, state, size, d_model: int = 512, clss: int = 1):
        raise NotImplementedError()
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
