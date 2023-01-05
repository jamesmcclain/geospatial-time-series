#!/usr/bin/env python3

import argparse
import datetime
import logging
import math
import random
import sys

import numpy as np
import torch
import torchvision as tv
import tqdm
from PIL import Image

from dataset import NpzSeriesDataset


def worker_init_fn(i):
    seed = i + int(round(datetime.datetime.now().timestamp()))
    random.seed(seed)


dataloader_cfg = {
    'batch_size': None,
    'num_workers': None,
    'worker_init_fn': worker_init_fn,
    'shuffle': False,
}


def cli_parser():
    # yapf: disable
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--dimensions', required=False, type=int, default=512)
    parser.add_argument('--num-heads', required=False, type=int, default=1)
    parser.add_argument('--encoder-layers', required=False, type=int, default=1)
    parser.add_argument('--gamma', required=False, type=float, default=0.7)
    parser.add_argument('--lr', required=False, type=float, default=3e-4)
    parser.add_argument('--entropy', dest='entropy', default=False, action='store_true')
    parser.add_argument('--bce', dest='bce', default=False, action='store_true')

    # Other hyperparameters
    parser.add_argument('--batch-size', required=False, type=int, default=16)
    parser.add_argument('--epochs', required=False, type=int, default=2**7)

    # Other
    parser.add_argument('--num-workers', required=False, type=int, default=8)
    parser.add_argument('--input-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=False, type=str, default=None)
    parser.add_argument('--tiles', dest='tiles', default=False, action='store_true')
    parser.add_argument('--png', dest='png', default=False, action='store_true')

    return parser
    # yapf: enable


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


class TransformerModel(torch.nn.Module):

    def __init__(self, dimensions=512, num_heads=1, encoder_layers=1):
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=dimensions,
                                                         nhead=num_heads,
                                                         batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=encoder_layers)

    def forward(self, source_seq):
        out = self.transformer_encoder(source_seq)
        return out


if __name__ == '__main__':
    args = cli_parser().parse_args()
    logging.basicConfig(stream=sys.stderr,
                        level=logging.INFO,
                        format='%(asctime)-15s %(message)s')
    log = logging.getLogger()
    if args.output_dir is not None:
        fh = logging.FileHandler(f'{args.output_dir}/output.log')
        log.addHandler(fh)
    dataloader_cfg['batch_size'] = args.batch_size
    dataloader_cfg['num_workers'] = args.num_workers

    if not args.tiles:
        train_dl = torch.utils.data.DataLoader(
            NpzSeriesDataset(f"{args.input_dir}/train",
                             narrow=True,
                             tiles=False),
            **dataloader_cfg,
        )
        eval_dl = torch.utils.data.DataLoader(
            NpzSeriesDataset(f"{args.input_dir}/eval",
                             narrow=True,
                             tiles=False),
            **dataloader_cfg,
        )
    else:
        train_dl = torch.utils.data.DataLoader(
            NpzSeriesDataset(f"{args.input_dir}/train",
                             narrow=False,
                             tiles=True),
            **dataloader_cfg,
        )
        eval_dl = torch.utils.data.DataLoader(
            NpzSeriesDataset(f"{args.input_dir}/eval",
                             narrow=False,
                             tiles=True),
            **dataloader_cfg,
        )

    if not args.tiles or args.output_dir is None:
        args.png = False

    if args.png:
        tile_count = len(eval_dl) * args.batch_size
        tile_count = math.ceil(math.sqrt(tile_count))
        tile_pixels = int(math.sqrt(args.dimensions))
        current = 0
        s = (tile_count * tile_pixels, tile_count * tile_pixels)
        all_tiles = np.zeros(s, dtype=np.uint8)
        for batch in eval_dl:
            for tile in batch[1].detach().cpu().numpy():
                y = (current // tile_count)*tile_pixels
                x = (current % tile_count)* tile_pixels
                current = current + 1
                tile = tile * 0xff
                tile = tile.astype(np.uint8).reshape(tile_pixels, tile_pixels)
                all_tiles[x:x+tile_pixels, y:y+tile_pixels] = tile
        filename = f'{args.output_dir}/0000.png'
        Image.fromarray(all_tiles).save(filename)
        # log.info(f'{filename} saved')

    device = torch.device('cuda')
    model = TransformerModel(dimensions=args.dimensions,
                             num_heads=args.num_heads,
                             encoder_layers=args.encoder_layers).to(device)

    if args.bce:
        obj1 = torch.nn.BCEWithLogitsLoss().to(device)
    else:
        obj1 = torch.nn.L1Loss().to(device)
    obj2 = EntropyLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=args.gamma)

    log.info(args.__dict__)

    best = math.inf
    for epoch in range(1, args.epochs + 1):

        current = 0
        for mode in ['train', 'eval']:
            loss_float = 0.0
            if mode == 'train':
                model.train()
                dl = train_dl
            elif mode == 'eval':
                model.eval()
                dl = eval_dl
            for batch in tqdm.tqdm(dl):
                out = model(batch[0].to(device))
                target = batch[1].to(device)

                # _, S, _ = out.shape
                loss = obj1(out[:, 0, :], target)
                # loss = obj1(out[:, 0, :], batch[0][:, 0, :].to(device))
                # for s in range(1, S):
                #     loss += obj1(out[:, s, :], target)
                # loss /= float(S)
                if args.entropy:
                    if args.bce:
                        loss = loss - (float(epoch)/args.epochs) * torch.mean(obj2(torch.sigmoid(out)))
                    else:
                        loss = loss - (float(epoch)/args.epochs) * torch.mean(obj2(out))
                loss_float += loss.item()

                opt.zero_grad()
                loss.backward()
                opt.step()

                if args.png and mode == 'eval':
                    for tile in out[:, 0, :]:
                        if args.bce:
                            tile = torch.sigmoid(tile)
                        tile = tile.detach().cpu().numpy()
                        y = (current // tile_count)*tile_pixels
                        x = (current % tile_count)* tile_pixels
                        current = current + 1
                        tile = tile * 0xff
                        tile[tile < 0] = 0
                        tile[tile > 0xff] = 0xff
                        tile = tile.astype(np.uint8).reshape(tile_pixels, tile_pixels)
                        all_tiles[x:x+tile_pixels, y:y+tile_pixels] = tile

            loss_float /= float(len(dl))

            if mode == 'train':
                loss_t = loss_float
            elif mode == 'eval':
                loss_e = loss_float

            if args.png and mode == 'eval':
                filename = f'{args.output_dir}/{epoch:04}.png'
                Image.fromarray(all_tiles).save(filename)
                # log.info(f'{filename} saved')

        if loss_e < best:
            best = loss_e
            log.info(f'✓ Epoch={epoch} train={loss_t} eval={loss_e}')
            # yapf: disable
            if args.output_dir:
                torch.save(model.state_dict(), f'{args.output_dir}/transformer-best.pth')
            # yapf: enable
        else:
            log.info(f'✗ Epoch={epoch} train={loss_t} eval={loss_e}')

    # yapf: disable
    if args.output_dir:
        torch.save(model.state_dict(), f'{args.output_dir}/transformer-last.pth')
    # yapf: enable
