from __future__ import division
import numpy as np


class ComputeMetrics:
    metric_types = ["accuracy", "acc_wo_bg", "IoU", "IoD","dish_acc_indirect","dish_acc_direct","video_level_mAP"]
    trials = []

    def __init__(self,ind2label,dish_ind2label,filenames,vl_predicted,vl_gt,predicted_dish=None,gt_dish=None,gt=None,predicted=None,bg_class=0):
        self.filenames=filenames
        self.gt=gt
        self.predicted=predicted
        self.bg_class=bg_class
        self.predicted_dish=predicted_dish
        self.gt_dish=gt_dish
        self.vl_predicted=vl_predicted
        self.vl_gt=vl_gt
        self.index2label=ind2label ##including the bg as label 0
        self.dish_ind2label=dish_ind2label


    def acc(self):
        def acc_(p, y):
            return np.mean(p == y) * 100
        self.vid_acc = []
        self.frame_acc={}
        if type(self.predicted) == list:

            for i in range(len(self.predicted)):
                # assert len(self.gt[i])==len(self.predicted[i]),"video index {}".format(i)
                self.vid_acc.append(np.mean(self.predicted[i] == self.gt[i]))
                self.frame_acc[self.filenames[i]]=self.gt[i]==self.predicted[i]
            return np.mean(self.vid_acc)*100
        else:
            assert len(self.gt) == len(self.predicted)
            self.vid_acc.append(acc_(self.predicted, self.gt))
            return acc_(self.predicted, self.gt)


    def dish_acc_indirect(self):

        def ignore_repetition(actions):
            actions = np.asarray(actions)
            temp = actions[1:] - actions[0:-1]
            idx = np.where(temp != 0)
            if np.sum(idx) == 0:
                u_action = np.asarray([actions[-1]])
                return u_action
            u_action = actions[idx]
            if len(actions != 0):
                u_action = np.append(u_action, actions[-1])

            return u_action
        def dish_acc_(p,y,bg):
            correct=0
            total=0
            recognized_video_level=ignore_repetition(p)
            ground_truth_video_level = ignore_repetition(y)
            for i in range(len(recognized_video_level)):
                if (recognized_video_level[i] in ground_truth_video_level) and recognized_video_level[i]!=bg:
                    correct += 1
                if (recognized_video_level[i]!=bg):
                    total+=1

            if correct/total<0.50:
                return 0
            else:
                return 1



        if type(self.predicted) == list:
            return np.mean([dish_acc_(self.predicted[i], self.gt[i], self.bg_class) for i in range(len(self.predicted))])
        else:
            return dish_acc_(self.predicted, self.gt, self.bg_class)




    def dish_acc_direct(self):

        def dish_acc_(p, y):
            if p==y:
                # print("+ "+self.dish_ind2label[p])
                return 1
            else:
                # print("- " + self.dish_ind2label[p]+" / true: "+self.dish_ind2label[y])
                return 0

        print("{} different dishes predicted".format(len(set(self.predicted_dish))))
        if type(self.predicted_dish) == list:
            return np.mean([dish_acc_(self.predicted_dish[i], self.gt_dish[i]) for i in range(len(self.predicted_dish))])
        else:
            return dish_acc_(self.predicted_dish, self.gt_dish)


    def per_dish_acc(self,predicted_dish):
        print(" ")
        tp=np.zeros([len(self.dish_ind2label)])
        fp = np.zeros([len(self.dish_ind2label)])
        for v in range(len(predicted_dish)):
            if predicted_dish[v]==self.gt_dish[v]:
                tp[self.gt_dish[v]]+=1
            else:
                fp[self.gt_dish[v]]+=1
        for c in range(len(self.dish_ind2label)):
            print("{} : {}".format(self.dish_ind2label[c],tp[c]/(tp[c]+fp[c])))
        print(" ")





    def acc_wo_bg(self):
        def acc_w(p, y, bg_class):
            ind = y != bg_class
            return np.mean(p[ind] == y[ind]) * 100

        self.vid_acc_wo_bg = []
        if type(self.predicted) == list:
            for i in range(len(self.predicted)):
                # assert len(self.gt[i]) == len(self.predicted[i]),"video index {}".format(i)
                self.vid_acc_wo_bg.append(acc_w(self.predicted[i], self.gt[i], self.bg_class))
            return np.mean(self.vid_acc_wo_bg)

        else:
            assert len(self.gt) == len(self.predicted)
            self.vid_acc_wo_bg.append(acc_w(self.predicted,self.gt,self.bg_class))
            return acc_w(self.predicted,self.gt,self.bg_class)



    def IoU(self):
        # From ICRA paper:
        # Learning Convolutional Action Primitives for Fine-grained Action Recognition
        # Colin Lea, Rene Vidal, Greg Hager
        # ICRA 2016
        def segment_intervals(Yi):
            idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
            intervals = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
            return intervals

        def segment_labels(Yi):
            idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
            Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
            return Yi_split

        def overlap_(p, y, bg_class):
            true_intervals = np.array(segment_intervals(y))
            true_labels = segment_labels(y)
            pred_intervals = np.array(segment_intervals(p))
            pred_labels = segment_labels(p)

            if bg_class is not None:
                true_intervals = np.array([t for t, l in zip(true_intervals, true_labels) if l != bg_class])
                true_labels = np.array([l for l in true_labels if l != bg_class])
                pred_intervals = np.array([t for t, l in zip(pred_intervals, pred_labels) if l != bg_class])
                pred_labels = np.array([l for l in pred_labels if l != bg_class])

            n_true_segs = true_labels.shape[0]
            n_pred_segs = pred_labels.shape[0]
            seg_scores = np.zeros(n_true_segs, np.float)

            for i in range(n_true_segs):
                for j in range(n_pred_segs):
                    if true_labels[i] == pred_labels[j]:
                        intersection = min(pred_intervals[j][1], true_intervals[i][1]) - max(pred_intervals[j][0],
                                                                                             true_intervals[i][0])
                        union = max(pred_intervals[j][1], true_intervals[i][1]) - min(pred_intervals[j][0],
                                                                                      true_intervals[i][0])
                        score_ = float(intersection) / union
                        seg_scores[i] = max(seg_scores[i], score_)

            return seg_scores.mean() * 100

        if type(self.predicted) == list:
            return np.mean([overlap_(self.predicted[i], self.gt[i], self.bg_class) for i in range(len(self.predicted))])
        else:
            return overlap_(self.predicted, self.gt, self.bg_class)


    def IoD(self):
        # From ICRA paper:
        # Learning Convolutional Action Primitives for Fine-grained Action Recognition
        # Colin Lea, Rene Vidal, Greg Hager
        # ICRA 2016
        def segment_intervals(Yi):
            idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
            intervals = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
            return intervals

        def segment_labels(Yi):
            idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
            Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
            return Yi_split

        def overlap_d(p, y, bg_class):
            true_intervals = np.array(segment_intervals(y))
            true_labels = segment_labels(y)
            pred_intervals = np.array(segment_intervals(p))
            pred_labels = segment_labels(p)

            if bg_class is not None:
                true_intervals = np.array([t for t, l in zip(true_intervals, true_labels) if l != bg_class])
                true_labels = np.array([l for l in true_labels if l != bg_class])
                pred_intervals = np.array([t for t, l in zip(pred_intervals, pred_labels) if l != bg_class])
                pred_labels = np.array([l for l in pred_labels if l != bg_class])

            n_true_segs = true_labels.shape[0]
            n_pred_segs = pred_labels.shape[0]
            seg_scores = np.zeros(n_true_segs, np.float)

            for i in range(n_true_segs):
                for j in range(n_pred_segs):
                    if true_labels[i] == pred_labels[j]:
                        intersection = min(pred_intervals[j][1], true_intervals[i][1]) - max(pred_intervals[j][0],
                                                                                             true_intervals[i][0])
                        union = pred_intervals[j][1] - pred_intervals[j][0]
                        score_ = float(intersection) / union
                        seg_scores[i] = max(seg_scores[i], score_)

            return seg_scores.mean() * 100

        if type(self.predicted) == list:
            return np.mean([overlap_d(self.predicted[i], self.gt[i], self.bg_class) for i in range(len(self.predicted))])
        else:
            return overlap_d(self.predicted, self.gt, self.bg_class)



    def video_level_mAP(self):
        #AP: From all the videos with the true underlying given action, what is the average precision if their(true vids)
        # predicted scores are chosen as thresholds
        def getAP(conf, labels):
            assert len(conf) == len(labels)
            sortind = np.argsort(-conf)
            tp = labels[sortind] == 1
            fp = labels[sortind] != 1
            npos = np.sum(labels)
            fp = np.cumsum(fp).astype('float32');
            tp = np.cumsum(tp).astype('float32')
            rec = tp / npos;
            prec = tp / (fp + tp)
            tmp = (labels[sortind] == 1).astype('float32')

            return np.sum(tmp * prec) / npos


        AP = []
        for i in range(np.shape(self.vl_gt)[1]):
            if  np.sum(self.vl_gt[:, i])==0:
                # print(" {}-{} not available in test".format( i+1,self.index2label[i + 1]))
                continue
            if np.sum(self.vl_gt[:, i]) == len(self.vl_gt[:, i]):
                # print(" {}-{} available in all test videos".format( i+1,self.index2label[i + 1]))
                continue
            AP.append(getAP(self.vl_predicted[:, i], self.vl_gt[:, i]))
            # print("mAP for {} is {:.4f} with {} videos".format(self.index2label[i+1], AP[-1],np.sum(self.vl_gt[:, i]) ))


        return 100 * sum(AP) / len(AP)





    def dish_mAP(self,dish_predicted,dish_gt):
        #AP: From all the videos with the true underlying given action, what is the average precision if their(true vids)
        # predicted scores are chosen as thresholds
        def getAP(conf, labels):
            assert len(conf) == len(labels)
            sortind = np.argsort(-conf)
            tp = labels[sortind] == 1
            fp = labels[sortind] != 1
            npos = np.sum(labels)
            fp = np.cumsum(fp).astype('float32');
            tp = np.cumsum(tp).astype('float32')
            rec = tp / npos;
            prec = tp / (fp + tp)
            tmp = (labels[sortind] == 1).astype('float32')

            return np.sum(tmp * prec) / npos


        AP = []
        for i in range(np.shape(dish_gt)[1]):
            if  np.sum(dish_gt[:, i])==0:
                # print(" {}-{} not available in test".format( i+1,self.index2label[i + 1]))
                continue
            if np.sum(dish_gt[:, i]) == len(dish_gt[:, i]):
                # print(" {}-{} available in all test videos".format( i+1,self.index2label[i + 1]))
                continue
            AP.append(getAP(dish_predicted[:, i], dish_gt[:, i]))
            # print("mAP for {} is {:.4f} with {} videos".format(self.index2label[i+1], AP[-1],np.sum(self.vl_gt[:, i]) ))


        return 100 * sum(AP) / len(AP)

    def three_logit_dish_report(self,filenames,logits1,logits2,logits3,logits_total):

        def dish_acc_(filename,p1,p2,p3,pT,y):
            acc1=0;acc2=0;acc3=0
            if p1==y:
                acc1=1
            if p2==y:
                acc2=1
            if p3==y:
                acc3=1
            if p1 == y or p2==y or p3==y:
                print("+ stage1: {}, stage2: {}, stage3: {}".format(self.dish_ind2label[p1],self.dish_ind2label[p2],self.dish_ind2label[p3]) + "/ total: {}".format(self.dish_ind2label[pT]) +"/ video: " + filename)
                return 1
            else:
                print("- stage1: {}, stage2: {}, stage3: {}".format(self.dish_ind2label[p1], self.dish_ind2label[p2],self.dish_ind2label[p3]) + "/ total: {}".format(self.dish_ind2label[pT]) + "/ video: " + filename)
                return 0



        return np.mean([dish_acc_(filenames[i],logits1[i],logits2[i],logits3[i],logits_total[i],self.gt_dish[i]) for i in range(len(filenames))])

    def two_logit_dish_report(self,filenames,logits1,logits2,logits_total):

        def dish_acc_(filename,p1,p2,pT,y):
            acc1=0;acc2=0;acc3=0
            if p1==y:
                acc1=1
            if p2==y:
                acc2=1
            if p1 == y or p2==y:
                print("+ stage1: {}, stage2: {}".format(self.dish_ind2label[p1],self.dish_ind2label[p2]) + "/ total: {}".format(self.dish_ind2label[pT]) +"/ video: " + filename)
                return 1
            else:
                print("- total: {}".format(self.dish_ind2label[pT]) + "/ video: " + filename)
                return 0



        return np.mean([dish_acc_(filenames[i],logits1[i],logits2[i],logits_total[i],self.gt_dish[i]) for i in range(len(filenames))])


    def one_logit_dish_report(self,filenames,logits_total):

        def dish_acc_(filename,pT,y):
            if pT == y :
                print("+ total: {}".format(self.dish_ind2label[pT]) +"/ video: " + filename)
                return 1
            else:
                print("- total: {}".format(self.dish_ind2label[pT]) + "/ video: " + filename)
                return 0



        return np.mean([dish_acc_(filenames[i],logits_total[i],self.gt_dish[i]) for i in range(len(filenames))])


    def per_dish_stage_acc(self,filenames,logits1,logits2,logits3,logits_total,n_dishes):
        def class_wise_acc(class_lbl,P,Y):
            def acc(p,y):
                if p==y:
                    return 1
                else:
                    return 0

            results=[]
            for i in range(len(P)):
                if Y[i] == class_lbl:
                    results.append(acc(P[i],Y[i]))
            return np.mean(np.asarray(results))

        for c in range(n_dishes):
            stage1_acc=class_wise_acc(c,logits1,self.gt_dish)
            stage2_acc = class_wise_acc(c, logits2, self.gt_dish)
            stage3_acc = class_wise_acc(c, logits3, self.gt_dish)
            print("{} =====> Stage1: {:.2f}   | Stage2: {:.2f}   | Stage3: {:.2f}".format(self.dish_ind2label[c],stage1_acc,stage2_acc,stage3_acc))





