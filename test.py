import torch
import torch.nn.functional as F
import torch.optim as optim
import utils
import numpy as np
from torch.autograd import Variable
from Evaluation import Evaluation
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def test(itr, dataset, args, model, device,s=8):
    print("Test Iteration: {}".format(itr))
    print(" ")
    done = False
    attribute_logits_stack= []
    dish_vec_score_stack=[]
    dish_vec_label_stack = []
    dish_logits_SHS_stack=[]
    dish_logits_stack1=[]
    dish_logits_stack2=[]
    dish_logits_stack3=[]
    dish_logits_total_stack=[]
    dish_final_stack=[]
    labels_stack = []
    dish_labels_stack = []
    while not done:
        features, labels,dish_labels,filename,done = dataset.load_data(is_training=False)
        features = torch.from_numpy(features).float().to(device)
        with torch.no_grad():
            SHS_logits_list, THS_logits_list, dish_logits_final=model(itr,itr,Variable(features),device,s,np.asarray([len(features[0])]), is_training=False)

        attribute_vector_probability_np=F.softmax(SHS_logits_list[0],dim=-1).cpu().data.numpy()
        dish_logits_SHS_np = SHS_logits_list[1].cpu().data.numpy()

        dish_logits1_np = THS_logits_list[0].cpu().data.numpy()
        dish_logits2_np= THS_logits_list[1].cpu().data.numpy()
        dish_logits3_np = THS_logits_list[2].cpu().data.numpy()
        dish_logits_total_np = THS_logits_list[3].cpu().data.numpy()

        dish_score_final_np = F.softmax(dish_logits_final,dim=-1).cpu().data.numpy()

        attribute_logits_stack.append(attribute_vector_probability_np[0])
        dish_logits_SHS_stack.append(int(np.argmax(dish_logits_SHS_np)))
        dish_logits_stack1.append(int(np.argmax(dish_logits1_np)))
        dish_logits_stack2.append(int(np.argmax(dish_logits2_np)))
        dish_logits_stack3.append(int(np.argmax(dish_logits3_np)))
        dish_logits_total_stack.append(int(np.argmax(dish_logits_total_np)))
        dish_final_stack.append(int(np.argmax(dish_score_final_np)))
        dish_vec_label_stack.append(dish_labels)
        dish_vec_score_stack.append(dish_score_final_np[0])
        labels_stack.append(labels)
        dish_labels_stack.append(int(np.argmax(dish_labels)))

    attribute_logits_stack=np.array(attribute_logits_stack)
    labels_stack = np.array(labels_stack)
    eval=Evaluation(dataset.index2label,dataset.dish_index2label)

    print("Semantic Hierarchy Stream:")
    eval.eval_fn(dataset.filenames_test,attribute_logits_stack,labels_stack,dish_logits_SHS_stack,dish_labels_stack)
    print("------------------------------------------------")
    print("Stage #1:")
    eval.eval_fn(dataset.filenames_test, attribute_logits_stack, labels_stack, dish_logits_stack1, dish_labels_stack)
    print("Stage #2:")
    eval.eval_fn(dataset.filenames_test, attribute_logits_stack, labels_stack, dish_logits_stack2, dish_labels_stack)
    print("Stage #3:")
    eval.eval_fn(dataset.filenames_test, attribute_logits_stack, labels_stack, dish_logits_stack3, dish_labels_stack)
    print("------------------------------------------------")
    print("Stage-Aggregated:")
    eval.eval_fn(dataset.filenames_test, attribute_logits_stack, labels_stack, dish_logits_total_stack, dish_labels_stack)
    print("------------------------------------------------")
    print("Final:")
    eval.eval_fn(dataset.filenames_test, attribute_logits_stack, labels_stack, dish_final_stack, dish_labels_stack)
    eval.dish_mAP(dish_vec_score_stack, dish_vec_label_stack)

    if itr % 10000 == 0:
        eval.multi_logit_dish_report(dataset.filenames_test,dish_logits_stack1,dish_logits_stack2,dish_logits_stack3,dish_logits_total_stack,3)

    eval.per_dish_acc(dish_final_stack)
    #utils.save_predictions(dataset.filenames_test,dish_final_stack,dish_labels_stack,dataset.dish_index2label,'./logs/split{}/predictions_{}.txt'.format(dataset.split,itr))


