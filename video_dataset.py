#reference: https://github.com/sujoyp/wtalc-pytorch
import numpy as np
import utils
import random
from utils_folder.read_breakfast_dataset import Breakfast_Dataset

class Dataset():
    def __init__(self, args):

        self.feature_type = args.feature_type
        self.split = args.split
        self.pca=True
        self.extended_attr=False
        if self.extended_attr:
            print("Attributes Extended")
        self.path_to_data = args.dataset_path
        self.read_breakfast_from_disk(self.path_to_data,pca_used=self.pca, feature_type=self.feature_type)
        self.batch_size = args.batch_size
        self.t_max = args.max_seqlen
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.dishwiseidx = []
        self.currenttestidx = 0
        self.labels_multihot = [utils.strlist2multihot(labs, self.classlist) for labs in self.labels]
        self.dish_labels_onehot = [utils.strlist2multihot(labs, self.dish_classlist) for labs in self.dish_labels]
        self.train_test_idx()
        self.classwise_feature_mapping()
        self.dishwise_feature_mapping()
        self.dish_W=self.get_dish_weights()
        print(" ")
        print ("Data Read!,Split : {} ".format(self.split))



    def read_breakfast_from_disk(self,base_path,pca_used=False,feature_type='i3d'):
        self.filenames=[]
        self.features=[]
        self.labels=[]
        self.dish_labels = []
        self.frame_level_gt=[]
        self.attribute_transcript=[]
        self.segment_interval=[]
        self.dish_index2label = {}
        breakfast_data = Breakfast_Dataset(base_path,self.split,self.extended_attr,pca_used=pca_used, feature_type=feature_type)
        self.index2label = breakfast_data.index2label
        self.num_attr = breakfast_data.nAttributes
        self.classlist = breakfast_data.attr_classlist
        self.dish_classlist = breakfast_data.dish_classlist;self.dish_index2label=self.dish_classlist
        self.subset = breakfast_data.subset
        self.num_dish = breakfast_data.nDishes
        for video in breakfast_data.final_annot:
            name = video[0]
            self.filenames.append(name)
            if self.feature_type == "i3d":
                self.features.append(np.transpose(np.concatenate([video[3], video[4]], axis=0)))
                self.labels.append(video[5])
                self.dish_labels.append([video[6]])
                self.attribute_transcript.append(video[7])
                self.segment_interval.append(video[8])
            elif self.feature_type == "idt":
                self.features.append(np.transpose(video[3]))
                self.labels.append(video[4])
                self.dish_labels.append([video[5]])
                self.attribute_transcript.append(video[6])
                self.segment_interval.append(video[7])

            self.frame_level_gt.append(video[1])

        self.feature_size =self.features[0].shape[1]
        print("feature size is {}".format(self.feature_size) )



    def train_test_idx(self):
        self.filenames_train=[]
        self.filenames_test=[]
        for i, s in enumerate(self.subset):
            if s.decode('utf-8') == 'training':
                self.trainidx.append(i)
                self.filenames_train.append(self.filenames[i])
            elif s.decode('utf-8') == 'test':
                self.testidx.append(i)
                self.filenames_test.append(self.filenames[i])



    def classwise_feature_mapping(self):
        for c in range(len(self.classlist)):
            category=self.classlist[c]
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category:
                        idx.append(i); break;
            self.classwiseidx.append(idx)

    def dishwise_feature_mapping(self):
        for d in range(len(self.dish_classlist)):
            dish = self.dish_classlist[d]
            idx = []
            for i in self.trainidx:
                for label in self.dish_labels[i]:
                    if label == dish:
                        idx.append(i); break;
            self.dishwiseidx.append(idx)


    def get_dish_weights(self):
        count=np.zeros([self.num_dish])
        for i in self.trainidx:
            count[np.argmax(self.dish_labels_onehot[i])]+=1
        assert np.sum(count)==len(self.trainidx)
        dish_fact=np.sum(count)/count
        return dish_fact/np.sum(dish_fact)



    def load_data(self, is_training=True,Wiftdf=None):

        if is_training==True:
            idx = []
            W = np.ones([self.batch_size,self.num_attr])
            labels=[]
            for r_id in range(self.batch_size):
                rand_sampleid = np.random.choice(len(self.trainidx), size=1)
                while self.trainidx[rand_sampleid[0]] in idx:
                    rand_sampleid = np.random.choice(len(self.trainidx), size=1)
                r=rand_sampleid[0]
                idx.append(self.trainidx[r])
                labels.append(np.argmax(self.dish_labels_onehot[self.trainidx[r]]))

            assert len(np.unique(idx)) == self.batch_size ,"all videos should be unique in one batch"
            # weights for the cross entropy loss based on the tf*idf weights
            for j,i in enumerate(idx):
                W[j, :]=Wiftdf[self.filenames[i]]



            feat=np.array([utils.process_feat(self.features[i], self.t_max) for i in idx])
            lab=np.array([self.labels_multihot[i] for i in idx])
            d_lab=np.array([self.dish_labels_onehot[i] for i in idx])
            names=np.array([self.filenames[i] for i in idx])

            return feat, lab,d_lab,W,names

        else:
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            dish_labs = self.dish_labels_onehot[self.testidx[self.currenttestidx]]
            feat = self.features[self.testidx[self.currenttestidx]]
            names = self.filenames[self.testidx[self.currenttestidx]]

            if self.currenttestidx == len(self.testidx)-1:
                done = True; self.currenttestidx = 0
            else:
                done = False; self.currenttestidx += 1

            return np.array([feat]), labs,dish_labs ,names,done


