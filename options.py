#reference: https://github.com/sujoyp/wtalc-pytorch
import argparse

parser = argparse.ArgumentParser(description='Two-Stream-Hierarchical Network')
parser.add_argument('--lr', type=float, default=0.001,help='learning rate (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=10, help='number of instances in a batch of data (default: 10)')
parser.add_argument('--dataset-name', default='Breakfast Dataset', help='dataset to train on (default: )')
parser.add_argument('--max-seqlen', type=int, default=7500, help='maximum sequence length during training (default: 750*10)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--max-iter', type=int, default=15000, help='maximum iteration to train (default: 20000)')
parser.add_argument('--feature-type', type=str, default='i3d', help='type of feature to be used i3d or idt (default: i3d)')
parser.add_argument('--conv-len', type=int, default=15, help='kernel size (L) of the temporal conv model (default: 15)')
parser.add_argument('--split', type=int, default=1,help='Split number (default: 1)')
parser.add_argument('--dataset-path', default='./data/', help='root path to the dataset (default: ')
parser.add_argument('--save-path', default='./Visualization/', help='path to save visualizations (default: ./Visualization/)')

