#!/usr/bin/env python3

import rasterio as rio
import torch
import tqdm


def cli_parser():
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', required=False, type=int, default=16)
    parser.add_argument('--dimensions', required=False, type=int, default=512)
    parser.add_argument('--encoder-layers', required=False, type=int, default=1)
    parser.add_argument('--num-heads', required=False, type=int, default=1)
    parser.add_argument('--series', required=True, type=str, nargs='+')
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
