#reference: https://github.com/sujoyp/wtalc-pytorch

from __future__ import print_function
import argparse
import os
import torch
import numpy as np
from TwoSH_model import TwoSH_Model
from video_dataset import Dataset
from test import test
from train import train
import options
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import torch.optim as optim
import utils

#pytorch 1.1
if __name__ == '__main__':
    n_nodes = [64]
    args = options.parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    print("The max length for the network is {} frames".format(args.max_seqlen))
    dataset = Dataset(args)
    W_tfidf,W=utils.get_tfidf(dataset,normalized='linear')
    model = TwoSH_Model(n_nodes, args.conv_len,dataset.feature_size, dataset.num_attr, dataset.num_dish, W_tfidf, W,dataset.classlist).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    print(" ")
    print("-----------------------------------------Training Starts---------------------------------------------- ")
    print(" ")
    for itr in range(args.max_iter):
       train(itr, dataset, args, model,optimizer, device,s=4)
       if itr == 1 or (itr % 500 == 0 and not itr == 1):
          model.eval()
          test(itr, dataset, args, model, device,s=4)
          model.train()
    
