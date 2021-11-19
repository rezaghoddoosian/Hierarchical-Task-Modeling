from scipy.io import loadmat
import numpy as np
import glob
import re
import pickle
from sklearn.decomposition import PCA

class Breakfast_Dataset(object):

    def save_annotation(self,file):
        with open(file, "wb") as fp:
            pickle.dump(self.final_annot, fp)
        print("annotations saved to the disk")
    def load_annotation(self,file):
        with open(file, "rb") as fp:  # Pickling
            self.final_annot=pickle.load(fp)
        print("annotations loaded from the disk")

    def calculate_pca(self,final_annot, Ncomponents, mode):
        for i, video in enumerate(final_annot):
            if mode == 'rgb':
                feat = video[3]
            if mode == 'flow':
                feat = video[4]
            if i == 0:
                all_feat = feat
            else:
                all_feat = np.concatenate([all_feat, feat], axis=-1)
        print("size is: {}".format(all_feat.shape))
        all_feat = np.transpose(all_feat)
        pca = PCA(n_components=Ncomponents)
        pca.fit(all_feat)
        return pca

    def apply_pca(self,samples, pca_rgb, pca_flow):
        for i, video in enumerate(samples):
            samples[i][3] = np.transpose(pca_rgb.transform(np.transpose(samples[i][3])))
            samples[i][4] = np.transpose(pca_flow.transform(np.transpose(samples[i][4])))

    def ignore_repetition(self,actions):
        actions = np.asarray(actions)
        temp = actions[1:] - actions[0:-1]
        idx = np.where(temp != 0)
        if np.sum(idx)==0:
            u_action=np.asarray([actions[-1]])
            return u_action
        u_action = actions[idx]
        if len(actions != 0):
            u_action = np.append(u_action, actions[-1])

        return u_action



    def pack_data(self,v,data):
        [video_name, gt, transcripts, features, attributes, dish, attribute_transcript, intervals]=data
        self.final_annot.append([video_name])
        if self.feature_type == "i3d":
            self.final_annot[v].append(gt)
            self.final_annot[v].append(transcripts)
            self.final_annot[v].append(features[0])
            self.final_annot[v].append(features[1])
        elif self.feature_type == 'idt':
            self.final_annot[v].append(gt)
            self.final_annot[v].append(transcripts)
            self.final_annot[v].append(features[0])

        self.final_annot[v].append(list(attributes))
        self.final_annot[v].append(dish)
        self.final_annot[v].append(attribute_transcript)
        self.final_annot[v].append(intervals)

    def read_data(self,path,extended=False):
        self.attr_classlist={}
        attr_ind=-1
        if extended:
            index2attr={}
            with open(path + 'attributes2.txt') as f:
                lines=f.readlines()
            for i in range(len(lines)):
                pieces=lines[i].split(' ')
                index2attr[i]=[pieces[1],pieces[2][:-1]]

        features = {}
        transcripts={}
        gt={}
        intervals={}
        attributes={}
        dish={}
        attribute_transcript={}
        dataset_attributes=[]
        for v,video in enumerate(self.all_vids): #iterate through videos
            # video features
            if self.feature_type == 'idt':

                features[video] = [np.load(path +'/idt/features/' +video + '.npy')]
            elif self.feature_type=='i3d':
                all_feat=np.load(path + '/i3d/features/' + video + '.npy')
                features[video] = [all_feat[:1024],all_feat[1024:]]

            # read ground truth
            intervals[video] = []
            gt_file = path + '/groundTruth/' + video + '.txt'
            with open(gt_file, 'r') as f:
                groundtruth_str = f.read().split('\n')[0:-1]
            gt_frames = np.repeat(0, len(groundtruth_str))
            for i, lbl in enumerate(groundtruth_str):  # iterate through frames
                gt_frames[i] = self.label2index[lbl]
                if i == 0:
                    start = 0
                else:
                    if gt_frames[i - 1] != gt_frames[i]:
                        end = i - 1
                        if groundtruth_str[i - 1] != 'SIL': intervals[video].append([start, end])
                        start = i
            end = i
            if lbl != 'SIL': intervals[video].append([start, end])
            gt[video] = gt_frames


            # transcript
            transcripts[video]=[]
            with open(path + '/transcripts/' + video + '.txt') as f:
                n_segments_wo_bg=len(intervals[video])
                t_lines=f.read().split('\n')[0:-1]
                for l,line in enumerate(t_lines):
                    if l != 0:
                        if t_lines[l - 1] == t_lines[l]: continue  # skip consecutive repeated action labels
                    if len(transcripts[video])>n_segments_wo_bg and line!='SIL': # for the cases where the frame level annotations and transcript do not match
                        break
                    transcripts[video].append(self.label2index[line])



            # attributes
            attribute_set = set()
            attribute_segments=[]
            for j,ind in enumerate(transcripts[video]):
                if ind==0:continue
                if extended:
                    if index2attr[ind][0] not in dataset_attributes:
                        attr_ind+=1
                        dataset_attributes.append(index2attr[ind][0])
                        self.attr_classlist[attr_ind]=index2attr[ind][0]
                    if index2attr[ind][1] not in dataset_attributes:
                        attr_ind+=1
                        dataset_attributes.append(index2attr[ind][1])
                        self.attr_classlist[attr_ind]=index2attr[ind][1]
                    attribute_segments.append([index2attr[ind][0],index2attr[ind][1]])
                    attribute_set.add(index2attr[ind][0]); attribute_set.add(index2attr[ind][1])
                else:
                    if self.index2label[ind] not in dataset_attributes:
                        attr_ind+=1
                        self.attr_classlist[attr_ind]=self.index2label[ind]
                        dataset_attributes.append(self.index2label[ind])
                    attribute_segments.append([self.index2label[ind]])
                    attribute_set.add(self.index2label[ind])

            attribute_transcript[video]=attribute_segments
            attributes[video]=attribute_set
            assert len(intervals[video])==len(attribute_transcript[video]),video
            assert features[video][0].shape[1]==len(gt[video]),video

            #dish
            if video.split('_')[-1] == "scrambledegg":
                dish_str = "friedegg"
            else:
                dish_str = video.split('_')[-1]
            dish[video]=dish_str

            data=[video,gt[video],transcripts[video],features[video],attributes[video],dish[video],attribute_transcript[video],intervals[video]]
            self.pack_data(v,data)


    def split_data(self,training_names,test_names):
        test=[]
        training=[]
        subset_=[]
        for x,vid in enumerate(self.final_annot):
            query=vid[0]
            if query in training_names:
                subset_.append("training")
                training.append(vid)
            elif query in test_names:
                subset_.append("test")
                test.append(vid)

        subset=np.asarray(subset_, dtype=np.bytes_)
        return training,test,subset


    def __init__(self,path,split,extended=False,pca_used=False,feature_type="i3d"):
        print("feature type chosen is: {}".format(feature_type))
        print
        if feature_type=='i3d':
            if pca_used:
                print("PCA is on")
            else:
                print("PCA is off")
        print
        self.feature_type=feature_type
        self.path=path
        self.pca_used=pca_used
        self.final_annot=[]

        with open(path+'all_videos', 'r') as f:
            self.all_vids = f.read().split('\n')[0:-1]

        self.dish_classlist={}
        self.dish_label2index = {}
        with open(path+'CompositeActions.txt','r') as f:
            lines=f.readlines()
        for line in lines:
            self.dish_classlist[int(line.split(' ')[0])]=line.split(' ')[1][:-1]
            self.dish_label2index[line.split(' ')[1][:-1]]=int(line.split(' ')[0])

        self.all_vids.sort()
        self.index2label =dict()
        self.label2index =dict()
        with open(path+'mapping.txt', 'r') as f:
            content = f.read().split('\n')[0:-1]
            for line in content:
                self.label2index[line.split()[1]] = int(line.split()[0])
                self.index2label[int(line.split()[0])] = line.split()[1]

        self.read_data(path,extended=extended)
        self.nActions_wo_bg=len(self.index2label)-1
        self.nDishes = len(self.dish_classlist)
        self.nAttributes = len(self.attr_classlist)
        print("Total number of fine-grained actions w/o bg is {}".format(self.nActions_wo_bg))
        print("Total number of dishes is {}".format(self.nDishes))
        print("Total number of attributes is {}".format(self.nAttributes))

        with open(path+'split{}.train'.format(split), 'r') as f:
            training_files = f.read().split('\n')[0:-1]
        with open(path+'split{}.test'.format(split), 'r') as f:
            test_files = f.read().split('\n')[0:-1]
        print("There are {} training and {} test videos".format(len(training_files),len(test_files)))
        self.train, self.test,self.subset = self.split_data(training_files,test_files)
        if self.pca_used and self.feature_type == "i3d":
            #self.load_annotation(path+"i3d/final_annot_i3d_{}.txt".format(split))
            pca_rgb = self.calculate_pca(self.train, 128, 'rgb')
            pca_flow = self.calculate_pca(self.train, 128, 'flow')
            self.apply_pca(self.final_annot, pca_rgb, pca_flow)
            # self.save_annotation(path+"i3d/final_annot_i3d_{}.txt".format(split))
        print
