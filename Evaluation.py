from utils_folder.metrics import ComputeMetrics
import numpy as np

class Evaluation:
   def __init__(self,index2label,dish_index2label):
       self.index2label=index2label
       self.dish_index2label = dish_index2label
   def eval_fn(self,filenames,predictions,gt,predicted_dish,gt_dish):
      self.metrics=ComputeMetrics(self.index2label,self.dish_index2label,filenames,predictions,gt,predicted_dish,gt_dish)
      self.mAP=self.metrics.video_level_mAP()
      self.dish_acc = self.metrics.dish_acc_direct()
      print('Dish accuracy is : %f' % self.dish_acc)

   def dish_mAP(self,predicted_dish_vec,gt_dish_vec):
      self.dish_mAP = self.metrics.dish_mAP(np.asarray(predicted_dish_vec), np.asarray(gt_dish_vec))
      print('Dish mAP is : %f' % self.dish_mAP)

   def multi_logit_dish_report(self,filenames,logits1,logits2,logits3,logits_total,N):
      if N==1:
         self.best_case_dish_acc = self.metrics.one_logit_dish_report(filenames, logits_total)
      if N==2:
         self.best_case_dish_acc = self.metrics.two_logit_dish_report(filenames, logits1, logits2,logits_total)
      if N==3:
         self.best_case_dish_acc=self.metrics.three_logit_dish_report(filenames, logits1, logits2, logits3, logits_total)
      print('Best case dish accuracy : %f' % self.best_case_dish_acc)

   def per_dish_acc(self,predicted_dish):
      self.metrics.per_dish_acc(predicted_dish)

   def per_dish_stage_acc(self, filenames, logits1, logits2, logits3, logits_total, n_dishes):
      self.metrics.per_dish_stage_acc(filenames, logits1, logits2, logits3, logits_total, n_dishes)