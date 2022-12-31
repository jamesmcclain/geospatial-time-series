#!/usr/bin/env python3

import argparse
import datetime
import logging
import math
import random
import sys

import torch
import torchvision as tv
import tqdm

from dataset import NpzSeriesDataset


def worker_init_fn(i):
    seed = i + int(round(datetime.datetime.now().timestamp()))
    random.seed(seed)


dataloader_cfg = {
    'batch_size': None,
    'num_workers': None,
    'worker_init_fn': worker_init_fn,
    'shuffle': True,
}


def cli_parser():
    # yapf: disable
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--dimensions', required=False, type=int, default=512)
    parser.add_argument('--num-heads', required=False, type=int, default=1)
    parser.add_argument('--encoder-layers', required=False, type=int, default=1)

    # Other hyperparameters
    parser.add_argument('--batch-size', required=False, type=int, default=16)
    parser.add_argument('--epochs', required=False, type=int, default=2**7)

    # Other
    parser.add_argument('--num-workers', required=False, type=int, default=8)
    parser.add_argument('--input-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=False, type=str, default=None)

    return parser
    # yapf: enable


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

    train_dl = torch.utils.data.DataLoader(
        NpzSeriesDataset(f"{args.input_dir}/train", narrow=True),
        **dataloader_cfg,
    )
    eval_dl = torch.utils.data.DataLoader(
        NpzSeriesDataset(f"{args.input_dir}/eval", narrow=True),
        **dataloader_cfg,
    )

    device = torch.device('cuda')
    model = TransformerModel(dimensions=args.dimensions,
                             num_heads=args.num_heads,
                             encoder_layers=args.encoder_layers).to(device)

    obj = torch.nn.L1Loss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    log.info(args.__dict__)

    best = math.inf
    for epoch in range(1, args.epochs + 1):

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

                loss = obj(out[:, 0, :], target)
                # _, S, _ = out.shape
                # for s in range(0, S):
                #     loss += obj(out[:, s, :], target)
                # loss /= float(S)
                loss_float += loss.item()

                opt.step()
                opt.zero_grad()
            loss_float /= float(len(dl))

            if mode == 'train':
                loss_t = loss_float
            elif mode == 'eval':
                loss_e = loss_float

        log.info(f'Epoch={epoch} train={loss_t} eval={loss_e}')
        if loss_e < best:
            best = loss_e
            # yapf: disable
            if args.output_dir:
                torch.save(model.state_dict(), f'{args.output_dir}/transformer-best.pth')
            # yapf: enable

    # yapf: disable
    if args.output_dir:
        torch.save(model.state_dict(), f'{args.output_dir}/reconstruct-last.pth')
    # yapf: enable
