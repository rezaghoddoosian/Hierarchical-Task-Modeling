import torch
import torch.nn.functional as F
import torch.optim as optim
from video_dataset import Dataset
from tensorboard_logger import log_value
import utils
import numpy as np
from torch.autograd import Variable
import time
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def compute_loss(itr,W,SHS_dish_logits_list,THS_dish_logits_list,final_logits,labels,dish_labels):
    ################
    dish_logits1,dish_logits2,dish_logits3,dish_logits_total=THS_dish_logits_list
    psi_a_logits,psi_c_logits=SHS_dish_logits_list
    attribute_loss=-torch.mean(torch.sum(torch.Tensor(W) * (Variable(labels) * F.log_softmax(psi_a_logits, dim=1)), dim=1),dim=0)
    dish_loss=-torch.mean(torch.sum(Variable(dish_labels) * F.log_softmax(psi_c_logits, dim=1), dim=1), dim=0)
    L_sh=(0.9*attribute_loss+0.1*dish_loss)
    ################
    dish_stage1 = -torch.mean(torch.sum(Variable(dish_labels) * F.log_softmax(dish_logits1, dim=1), dim=1), dim=0)
    dish_stage2 = -torch.mean(torch.sum(Variable(dish_labels) * F.log_softmax(dish_logits2, dim=1), dim=1), dim=0)
    dish_stage3 = -torch.mean(torch.sum(Variable(dish_labels) * F.log_softmax(dish_logits3, dim=1), dim=1), dim=0)
    dish_milloss_total = -torch.mean(torch.sum(Variable(dish_labels) * F.log_softmax(dish_logits_total, dim=1), dim=1), dim=0)
    L_th=dish_stage1 + dish_stage2 + dish_stage3 + dish_milloss_total
    ################
    L_fused = -torch.mean(torch.sum(Variable(dish_labels) * F.log_softmax(final_logits, dim=1), dim=1),dim=0)

    return L_sh+0.25*L_th+L_fused




def train(itr, dataset, args, model, optimizer, device,s=8):

    features, labels,dish_labels,W_tfidf,filenames = dataset.load_data(is_training=True,Wiftdf=model.W)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:,:np.max(seq_len),:]
    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)
    dish_labels = torch.from_numpy(dish_labels).float().to(device)
    SHS_logits,THS_logits,final_logits = model(itr,filenames,Variable(features),device,s,seq_len)
    total_loss = compute_loss(itr, W_tfidf, SHS_logits,THS_logits,final_logits, labels,dish_labels)
    if itr%100==0:
        print('Iteration: %d, Loss: %.3f' %(itr, total_loss.data.cpu().numpy()))
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()




