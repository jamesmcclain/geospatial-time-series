#!/usr/bin/env python3

from PIL import Image
import logging
import argparse
import torch
import tqdm
import random
import sys
import numpy as np
import math


def cli_parser():
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--batch-size', required=False, type=int, default=48)
    parser.add_argument('--dimensions', required=False, type=int, default=512)
    parser.add_argument('--epochs', required=False, type=int, default=2**7)
    parser.add_argument('--eval-batches',
                        required=False,
                        type=int,
                        default=2**8)
    parser.add_argument('--train-batches',
                        required=False,
                        type=int,
                        default=2**9)

    # More hyperparameters
    parser.add_argument('--decoder-layers',
                        required=False,
                        type=int,
                        default=1)
    parser.add_argument('--dropout', required=False, type=float, default=0.5)
    parser.add_argument('--encoder-layers',
                        required=False,
                        type=int,
                        default=1)
    parser.add_argument('--num-heads', required=False, type=int, default=1)

    # Experiment parameters
    parser.add_argument('--noise', required=False, type=float, default=0.5)
    parser.add_argument('--sequence-length',
                        required=False,
                        type=int,
                        default=20)

    # Other
    parser.add_argument('--output-dir', required=False, type=str, default=None)
    parser.add_argument('--png', dest='png', action='store_true')
    parser.set_defaults(png=False)

    return parser


class TransformerModel(torch.nn.Module):

    def __init__(self,
                 dimensions=512,
                 num_heads=1,
                 encoder_layers=1,
                 decoder_layers=1,
                 dropout=0.5):
        super().__init__()
        self.transformer = torch.nn.Transformer(
            d_model=dimensions,
            nhead=num_heads,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, source_seq, target_seq):
        out = self.transformer(source_seq, target_seq)
        return out


def generate_seqs(seq_length, batch_size, dimensions, noise, device):
    target = torch.rand(batch_size,
                        1,
                        dimensions,
                        dtype=torch.float32,
                        device=device)
    target_seq = torch.cat([target for _ in range(0, seq_length)], dim=1)
    mask = torch.rand(batch_size,
                      seq_length,
                      dimensions,
                      dtype=torch.float32,
                      device=device)
    source_seq = target_seq * (mask <= noise)
    target_seq = torch.zeros(batch_size,
                             1,
                             dimensions,
                             dtype=torch.float32,
                             device=device)
    target_mask = torch.zeros(1, 1, dtype=torch.float32, device=device)
    target_mask = target_mask.fill_(-math.inf)
    return source_seq, target_seq, target.reshape(batch_size, dimensions)

if __name__ == '__main__':
    args = cli_parser().parse_args()
    logging.basicConfig(stream=sys.stderr,
                        level=logging.INFO,
                        format='%(asctime)-15s %(message)s')
    log = logging.getLogger()
    if args.output_dir is not None:
        fh = logging.FileHandler(f'{args.output_dir}/reconstruct.log')
        log.addHandler(fh)

    device = torch.device('cuda')
    model = TransformerModel(dimensions=args.dimensions,
                             num_heads=args.num_heads,
                             encoder_layers=args.encoder_layers,
                             decoder_layers=args.decoder_layers,
                             dropout=args.dropout).to(device)

    obj = torch.nn.L1Loss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    log.info(args.__dict__)

    best = math.inf
    for epoch in range(1, args.epochs+1):

        model.train()
        loss_t = 0.0
        for _ in tqdm.tqdm(range(0, args.train_batches)):
            source_seq, target_seq, target = generate_seqs(args.sequence_length,
                                                           args.batch_size,
                                                           args.dimensions, args.noise,
                                                           device)
            out = model(source_seq, target_seq)
            loss = obj(out[:,0,:], target)
            loss_t += loss.item()
            loss.backward()
            opt.step()
            opt.zero_grad()
        loss_t /= float(args.train_batches)

        model.eval()
        loss_e = 0.0
        for _ in tqdm.tqdm(range(0, args.eval_batches)):
            source_seq, target_seq, target = generate_seqs(args.sequence_length,
                                                           args.batch_size,
                                                           args.dimensions, args.noise,
                                                           device)
            out = model(source_seq, target_seq)
            loss_e += obj(out[:,0,:], target).item()
        loss_e /= float(args.eval_batches)

        log.info(f'Epoch={epoch} train={loss_t} eval={loss_e}')
        if loss_e < best:
            best = loss_e
            if args.output_dir:
                torch.save(model.state_dict(), f'{args.output_dir}/reconstruct-best.pth')

        if args.output_dir and args.png:
            encoder_layer_0 = model.transformer.encoder.layers[0].self_attn
            # encoder_layer_0 = model.transformer.decoder.layers[0].multihead_attn
            encoder_qkv = encoder_layer_0.in_proj_weight.cpu().detach().numpy()
            encoder_out = encoder_layer_0.out_proj.weight.cpu().detach().numpy()
            encoder_data = np.concatenate([encoder_qkv, encoder_out], axis=0).T
            encoder_data -= encoder_data.min()
            hi = encoder_data.max()
            encoder_data = (255 * (encoder_data / hi)).astype(np.uint8)
            filename = f'{args.output_dir}/encoder-{epoch:04}.png'
            Image.fromarray(encoder_data).save(filename)

    if args.output_dir:
        torch.save(model.state_dict(), f'{args.output_dir}/reconstruct-last.pth')
