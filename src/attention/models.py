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

    def __init__(self,
                 arch,
                 state,
                 size,
                 d_model: int = 512,
                 clss: int = 4,
                 num_heads=1,
                 dropout=1.0):
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

        self.arch = arch
        if self.arch in {'resnet18', 'resnet34'}:
            self.shapes = [
                [64, 64, 64],
                [64, 64, 64],
                [128, 32, 32],
                [256, 16, 16],
                [512, 8, 8],
                [512, 8, 8],
            ]
        else:
            raise Exception(f'Not prepared for {self.arch}')

        self.self_attn = torch.nn.ModuleList()
        self.q_fcns = torch.nn.ModuleList()
        self.k_fcns = torch.nn.ModuleList()
        for shape in self.shapes:
            dim = shape[0]
            self.q_fcns.append(torch.nn.Linear(dim, dim))
            self.k_fcns.append(torch.nn.Linear(dim, dim))
            self.self_attn.append(
                torch.nn.MultiheadAttention(dim,
                                            num_heads,
                                            batch_first=True,
                                            dropout=dropout))

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
        for shape in self.shapes:
            y.append(torch.zeros(bs, *shape, dtype=torch.float32, device=x[0].device))

        for i in range(len(x)):
            xi = x[i]
            shape = self.shapes[i]

            xi = xi.reshape(bs, ss, *shape)  # Restore "original" shape
            xi = torch.transpose(xi, 2, 4)  # move embeddngs to end
            xi = torch.transpose(xi, 1, 2)  # move spatial dimenson up
            xi = torch.transpose(xi, 2, 3)  # move other spatial dimension up
            # V
            vee = xi = xi.reshape(-1, ss, shape[0])  # put batch and spatial dimensions together
            # Q
            qew = torch.mean(self.q_fcns[i](xi), dim=1, keepdim=True)
            # K
            kay = self.k_fcns[i](xi)

            # Use MultiheadAttention block
            result, _ = self.self_attn[i](qew, kay, vee)
            result = result.reshape(bs, shape[1], shape[2], shape[0])
            result = torch.transpose(result, 2, 3)
            result = torch.transpose(result, 1, 2)
            y[i] = result

        y = self.fpn(tuple(y))  # pass through fpn
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
        self.stash = None

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
