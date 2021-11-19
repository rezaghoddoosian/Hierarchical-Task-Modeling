import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import utils
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1 or classname.find('TC_Model') != -1:
        if classname.find('TC_Model') == -1:
            torch_init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        else:
            for c in m.Conv1d:
                torch_init.xavier_uniform_(c.weight)




class TwoSH_Model(torch.nn.Module):
    def __init__(self,n_nodes,conv_len,n_feature, n_class,n_dish,W_tfidf,W,classlist):
        super(TwoSH_Model, self).__init__()
        self.classlist=classlist
        self.n_dish=n_dish
        self.W_tfidf = W_tfidf
        self.W=W
        self.n_layers=len(n_nodes)
        self.n_nodes=n_nodes
        self.conv_len=conv_len
        self.n_feature=n_feature
        self.build_TC()
        self.fc_1 = nn.Linear(n_feature, n_feature)
        self.nStages = 3
        self.attr2dish_fc_list=[]
        self.x2dish_fc1 = nn.Linear(self.n_nodes[-1], n_dish);self.x2dish_fc2 = nn.Linear(self.n_nodes[-1], n_dish);self.x2dish_fc3 = nn.Linear(self.n_nodes[-1], n_dish)
        self.tfidf_fc = nn.Linear(n_class, n_dish)
        self.tfidf_fc.weight.requires_grad = True
        self.tfidf_fc.bias.requires_grad = False
        self.classifier_dish = nn.Linear(self.nStages*n_dish, n_dish)
        self.classifier_dish.bias.requires_grad = False
        self.classifier = nn.Linear(self.n_nodes[-1], n_class)
        self.dropout = nn.Dropout(0.7)
        self.w_alpha = nn.Parameter(torch.randn((1, n_dish), requires_grad=True))
        self.apply(weights_init)
        self.attr2dish_fc_list.append(self.x2dish_fc1)
        self.attr2dish_fc_list.append(self.x2dish_fc2)
        self.attr2dish_fc_list.append(self.x2dish_fc3)



    def build_TC(self):
        self.Conv1d=[]
        self.BNorm=[]
        self.Drop_out=[]
        self.Relu=[]
        self.Max_pool=[]
        input_n_nodes=self.n_feature
        # ---- Encoder ----
        for i in range(self.n_layers):
            p=int((self.conv_len-1)/2)
            self.Conv1d.append(nn.Conv1d(in_channels=input_n_nodes,out_channels=self.n_nodes[i],kernel_size=self.conv_len,padding=p,bias=False))
            self.BNorm.append(nn.BatchNorm1d(self.n_nodes[i]))
            self.Drop_out.append(nn.Dropout2d(0.1)) # Input should be [B,C,1,T]
            self.Relu.append(nn.ReLU())
            self.Max_pool.append(nn.MaxPool1d(2,stride=2, padding=0))
            input_n_nodes=self.n_nodes[i]


    def SHS(self,phi,seq_len,s,device,is_training):

        psi_a_logits = torch.zeros(0).to(device)
        Psi = self.classifier(phi)
        for i in range(Psi.size()[0]):
            out_t = int(seq_len[i])
            k = np.ceil(out_t / s).astype('int32')
            tmp, _ = torch.topk(Psi[i][:out_t], k=int(k), dim=0)
            psi_a_logits = torch.cat([psi_a_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
        ####
        # the mask is initialized with W_tfidf, however, after a few itertions turns into a binary mask.
        self.W_tfidf[self.W_tfidf != 0] = np.sqrt(self.W_tfidf[self.W_tfidf != 0])
        M_tfidf = torch.Tensor(np.transpose(self.W_tfidf))
        self.tfidf_fc.weight.data = self.tfidf_fc.weight.data * M_tfidf
        ####
        if is_training:
            psi_a_logits_do = self.dropout(psi_a_logits)
        else:
            psi_a_logits_do = psi_a_logits
        psi_c_logits = self.tfidf_fc(F.relu(psi_a_logits_do))

        return psi_a_logits,psi_c_logits

    def THS(self,phi,seq_len,device,is_training):
        stage_video_features_list = []
        for kappa in range(self.nStages):
            stage_video_features_list.append(torch.zeros(0).to(device))
        for i in range(phi.size()[0]):
            out_t = int(seq_len[i])
            for kappa in range(self.nStages):
                d = out_t // self.nStages
                begin=kappa*d;end=(kappa+1)*d
                h=torch.mean(phi[i][begin:end,:],dim=0,keepdim=True)
                stage_video_features_list[kappa] = torch.cat([stage_video_features_list[kappa], h], dim=0)

        if is_training:
            for kappa in range(self.nStages):
                stage_video_features_list[kappa]=self.dropout(stage_video_features_list[kappa])
        else:
            for kappa in range(self.nStages):
                stage_video_features_list[kappa]=stage_video_features_list[kappa]

        stage_dish_logits_list = []
        dish_logits_bundle = torch.zeros(0).to(device)
        for kappa in range(self.nStages):
            stage_dish_logits_list.append(self.attr2dish_fc_list[kappa](stage_video_features_list[kappa]))
            dish_logits_bundle = torch.cat([dish_logits_bundle, F.relu(stage_dish_logits_list[kappa])], dim=1)
        ####The Stage Aggregation Function#####
        if is_training:
            for item in range(phi.size()[0]):
                rand_sampleid = int(np.random.choice(self.nStages, size=1))
                mask = torch.ones((1, self.nStages * self.n_dish))
                mask[0, rand_sampleid * self.n_dish:(rand_sampleid + 1) * self.n_dish] = 0
                dish_logits_bundle[item] = dish_logits_bundle[item, :] * mask
            dish_logits_bundle = dish_logits_bundle * self.nStages / (self.nStages - 1)
        dish_logits_total = self.classifier_dish(dish_logits_bundle)
        #######################################
        return [stage_dish_logits_list[0],stage_dish_logits_list[1],stage_dish_logits_list[2],dish_logits_total]


    def Stream_Fusion(self,SHS_logits,THS_logits,is_training):

        alpha = torch.sigmoid(self.w_alpha)
        H_alpha = (torch.sign(alpha.data - 0.5) + 1) / 2
        if is_training:
            dish_logits_final = (1 - alpha) * SHS_logits.detach() + alpha * THS_logits.detach()
        else:
            dish_logits_final = (1 - H_alpha) * SHS_logits.detach() + H_alpha * THS_logits.detach()

        return dish_logits_final

    def forward(self,itr,filenames,inputs,device,s,seq_len,is_training=True):

        g_x = F.relu(self.fc_1(inputs))
        if is_training:
            g_x = self.dropout(g_x)
        ####TC##### forward
        g_x = g_x.permute(0, 2, 1)
        for i in range(self.n_layers):
            phi=self.Conv1d[i](g_x)
            phi=self.Relu[i](phi)
        ####TC#####
        phi = phi.permute(0, 2, 1)

        SHS_logits_list=self.SHS(phi,seq_len,s,device,is_training)
        THS_logits_list=self.THS(phi,seq_len,device,is_training)
        dish_logits_final=self.Stream_Fusion(SHS_logits_list[-1],THS_logits_list[-1],is_training)


        return SHS_logits_list,THS_logits_list,dish_logits_final
