#reference: https://github.com/sujoyp/wtalc-pytorch
import numpy as np
from scipy.special import factorial
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F




def str2ind(categoryname,classlist):
   return [i for i in range(len(classlist)) if categoryname==classlist[i]][0]

def strlist2indlist(strlist, classlist):
	return [str2ind(s,classlist) for s in strlist]

def strlist2multihot(strlist, classlist):
	return np.sum(np.eye(len(classlist))[strlist2indlist(strlist,classlist)], axis=0)

def idx2multihot(id_list,num_class):
   return np.sum(np.eye(num_class)[id_list], axis=0)

def random_extract(feat, t_max):
   r = np.random.randint(len(feat)-t_max)
   return feat[r:r+t_max]

def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
       return np.pad(feat, ((0,min_len-np.shape(feat)[0]), (0,0)), mode='constant', constant_values=0)
    else:
       return feat

def process_feat(feat, length):
    if len(feat) > length:
        return random_extract(feat, length)
    else:
        return pad(feat, length)

def write_to_file(dname, cmap, itr):
    fid = open(dname + '-results.log', 'a+')
    string_to_write = str(itr)
    string_to_write += ' ' + '%.2f' %cmap
    fid.write(string_to_write + '\n')
    fid.close()

def modify_labs(lab_wght, state_attr,A_):
    a1,a2=state_attr[0],state_attr[1]
    fg1=a1 in A_;fg2=a2 in A_ and a2!=a1
    lbl=1/(float(fg1) + float(fg2)+1e-10)
    if a1 in A_:
        if a1 not in lab_wght:lab_wght[a1]=[]
        lab_wght[a1].append(lbl)
    if a2 in A_ and a2!=a1:
        if a2 not in lab_wght:lab_wght[a2]=[]
        lab_wght[a2].append(lbl)
    return lab_wght

def update_training_labels(prev_labels,A_attr_labels,trainidx):
    for i in trainidx:
        assert len(A_attr_labels[i])!=0
        for j in range(len(prev_labels[i])):
            if j in A_attr_labels[i]:
                prev_labels[i][j]=np.mean( A_attr_labels[i][j])
            else:
                prev_labels[i][j]=0
    return prev_labels


def get_tSNE(X,labels,dish_classlist,save_path,itr):
    nclass=len(np.unique(labels))
    X_embedded = TSNE(n_components=2).fit_transform(X)
    # Create data
    data_x=dict();data_y=dict()
    colors = cm.rainbow(np.linspace(0, 1, nclass))
    edgecolors = itertools.cycle(["r","b","k","m","g"])
    marks = itertools.cycle(["o","v","s", "P", "X"])
    groups=[]
    for i in range(len(labels)):
        group=dish_classlist[labels[i]]
        if group not in data_x:
            data_x[group]=[];data_y[group]=[];groups.append(group)
        x=X_embedded[i,0];y=X_embedded[i,1]
        data_x[group].append(x)
        data_y[group].append(y)
    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i,group in enumerate(groups):
        x=np.asarray(data_x[group]);y=np.asarray(data_y[group])
        ax.scatter(x, y, alpha=0.8, color=colors[i], edgecolors=next(edgecolors), marker=next(marks),s=30, label=group)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),ncol=1)
    plt.title('TSNE plot-itr: {}'.format(itr))
    # plt.show()
    plt.savefig(save_path + "{}.png".format(itr))
    plt.close()

def get_tfidf(dataset,normalized="none"):
    TFIDF=np.zeros([dataset.num_attr,dataset.num_dish])
    Docs={}
    for dish_id,dish in enumerate(dataset.dish_classlist):
        words =np.zeros([dataset.num_attr])
        list_of_vids=dataset.dishwiseidx[dish_id]
        for v in list_of_vids:
            tmp=dataset.labels_multihot[v]
            words=tmp+words
        if len(list_of_vids)!=0:
            words=words/len(list_of_vids)
        Docs[dish]=words
        TFIDF[:, dish_id] =np.transpose(words)
    for attr_id, attr in enumerate(dataset.classlist):
        d_clms=np.where(TFIDF[attr_id,:]>0)[0]
        idf=np.log((dataset.num_dish+1)/(len(d_clms)+1))
        TFIDF[attr_id, :]=TFIDF[attr_id,:]*idf

    W={} # This blockis for the weights used for the attr recognition loss
    for i in range(len(dataset.filenames_train)):
        ds_id=dataset.trainidx[i]
        dish_lb=np.argmax(dataset.dish_labels_onehot[ds_id])
        W[dataset.filenames_train[i]]=TFIDF[:, dish_lb]*dataset.labels_multihot[ds_id]
        W[dataset.filenames_train[i]] = W[dataset.filenames_train[i]] / (np.sum(W[dataset.filenames_train[i]]) + 0.000000001)

    if normalized == "linear":
        TFIDF=TFIDF / (np.sum(TFIDF, axis=0)+0.000000001)
    if normalized == 'none':
        for col in range(TFIDF.shape[1]):
            best=np.max(TFIDF[:,col])
            coeff=1/(best+0.000000001)
            TFIDF[:, col]*=coeff
    if normalized == 'non-linear':
        k = np.linspace(1, 20, TFIDF.shape[0])
        non_lin = (np.exp(-1)) / factorial(k)
        for col in range(TFIDF.shape[1]):
            idx=np.argsort(-TFIDF[:, col])
            TFIDF[idx, col]=TFIDF[idx, col]*non_lin
        TFIDF=TFIDF / (np.sum(TFIDF, axis=0)+0.000000001)

    return TFIDF,W


def get_anchor_positive_triplet_mask(labels,PK):
    # reference: https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: int `list` with shape [batch_size]
        PK: int scalar referring to the size of the anchor pool
    Returns:
        mask: float `array` with shape [PK, batch_size]
    """
    # Check that i and j are distinct
    indices_equal=np.eye(PK,len(labels),dtype=bool)
    indices_not_equal = np.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = np.equal(np.expand_dims(labels, 0), np.expand_dims(labels[:PK], 1))
    # Combine the two masks
    mask = np.logical_and(indices_not_equal, labels_equal)
    return mask.astype(float)

def get_anchor_negative_triplet_mask(labels,PK):
    # reference: https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
    labels_equal = np.equal(np.expand_dims(labels, 0), np.expand_dims(labels[:PK], 1))
    mask = np.logical_not(labels_equal)
    return mask.astype(float)

def find_pos_neg_idx(embeddings,PK,dataset,Similarity=True):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: array of shape (batch_size, embed_dim), values should be >=0
        PK: int scalar referring to the size of the anchor pool
    Returns:
        pairwise_cos_similarity: array of shape (PK, batch_size)
    """
    if Similarity:
        # Get the dot product between embeddings
        dot_product = np.matmul(embeddings[:PK], np.transpose(embeddings))
        a=1/(np.linalg.norm(embeddings[:PK], 2, axis=1)+1e-10) # (PK,)
        b= 1 / (np.linalg.norm(embeddings, 2, axis=1) + 1e-10)  # (batch,)
        c=np.expand_dims(a,1)*np.expand_dims(b,0) #(Pk, batch)
        similarity =c*dot_product #cosine similarity btw anchor embeddings and all other embeddings
        anchor_positive_sim = (similarity-1.1) * (dataset.mask_anchor_positive)
        p=np.argmin(anchor_positive_sim,axis=1)
        anchor_negative_sim = similarity * dataset.mask_anchor_negative
        n = np.argsort(-anchor_negative_sim, axis=1)

    else:
        augmentedA = np.expand_dims(embeddings[:PK], 2)  # (PK,d,1)
        augmentedB = np.expand_dims(np.transpose(embeddings), 0)  # (1,d,batch)
        diff_mat = augmentedA - augmentedB
        dist_mat = np.linalg.norm(diff_mat,2,axis=1)  # (PK,batch)
        masked_p_dist=dist_mat*dataset.mask_anchor_positive
        p = np.argmax(masked_p_dist, axis=1)
        masked_p_dist = (dist_mat-np.amax(dist_mat,axis=1,keepdims=True))*(dataset.mask_anchor_negative)
        n = np.argsort(masked_p_dist, axis=1)
    return p,n


def find_pos_neg_idx_test(embeddings,dataset,dish_labels,Similarity):
    """

    """
    # Get the dot product between embeddings
    if Similarity:
        dot_product = np.matmul(embeddings, np.transpose(embeddings))
        a=1/(np.linalg.norm(embeddings, 2, axis=1)+1e-10) # (batch,)
        b= 1 / (np.linalg.norm(embeddings, 2, axis=1) + 1e-10)  # (batch,)
        c=np.expand_dims(a,1)*np.expand_dims(b,0) #(batch, batch)
        similarity =c*dot_product #cosine similarity btw anchor embeddings and all other embeddings
    else:
        augmentedA = np.expand_dims(embeddings, 2)  # (batch,d,1)
        augmentedB = np.expand_dims(np.transpose(embeddings), 0)  # (1,d,batch)
        diff_mat = augmentedA - augmentedB
        similarity = np.linalg.norm(diff_mat,2,axis=1)  # (batch,batch)
    print("Here is how close attr predictions of test samples are to each other from closest to furthest in order:")
    if Similarity:
        idx=np.argsort(-similarity,axis=1)
        print("Results based on cosine similarity")
    else:
        idx = np.argsort(similarity, axis=1)
        print("Results based on L2 distance")
    total_proximity=0
    ranks=[];counter=0;dish_wise_rank={}
    for a in range(len(embeddings)):
        if np.sum(np.array(dish_labels) == dish_labels[a]) == 1: #only videos that have another video of the same dish
            continue
        proximity=0
        print("# For "+str(dataset.dish_classlist[dish_labels[a]])+"->",end=' ')
        for p in range(1,len(similarity[a,:])):
            v_id=idx[a, p]
            if p<6: print(dataset.dish_classlist[dish_labels[v_id]],end=',')
            if dish_labels[v_id]==dish_labels[a]:
                if dish_labels[a] not in dish_wise_rank:
                    dish_wise_rank[dish_labels[a]] = [p]
                else:
                    dish_wise_rank[dish_labels[a]].append(p)
                proximity = proximity + similarity[a, v_id];ranks.append(p) #;print(similarity[a, v_id],end=',')
                if p<4: counter=counter+1
                if p==1:
                    proximity = proximity - similarity[a, idx[a, 2]]#;print(similarity[a, idx[a, 2]],end=',')
                print("#{}".format(p), end=',')
                break
            elif p==1:
                proximity = proximity - similarity[a, v_id]#;print(similarity[a, v_id],end=',')
        total_proximity+=proximity
        print("proximity is {:.5f}".format(proximity))

    print("Total proximity is {:.5f} and average rank is {}, and {} vids are in top-three".format(total_proximity,np.mean(np.asarray(ranks)),counter))
    print(" ")
    for dish_id in dish_wise_rank:
        print("{}: {:.2f}".format(dataset.dish_classlist[dish_id],np.mean(dish_wise_rank[dish_id])))

    print(" ")
    return


def torch_cos_similarity_matrix(embeddingsA,embeddingsB):
    """
        embeddingsA=    (T_a,d)
        embeddingsB=    (T_b,d)
    """
    dot_product = torch.matmul(embeddingsA, torch.transpose(embeddingsB, 1, 0)) #(Ta,Tb)
    norm_a = 1 / (torch.norm(embeddingsA, 2, dim=1, keepdim=True) + 1e-10)  # (T_a,1)
    norm_p = 1 / (torch.norm(embeddingsB, 2, dim=1, keepdim=True) + 1e-10)  # (T_b,1)
    sim_mat = dot_product * norm_a * torch.transpose(norm_p, 1, 0)  # (T_a,T_b)
    assert torch.max(sim_mat) <= 1.01 and torch.min(sim_mat) >= -1.01, str(torch.max(sim_mat)) + " " + str(torch.min(sim_mat))
    return sim_mat



def torch_L2_distance_matrix(embeddingsA,embeddingsB):
    """
        embeddingsA=    (T_a,d)
        embeddingsB=    (T_b,d)
    """
    augmentedA = torch.unsqueeze(embeddingsA, 2) #(Ta,d,1)
    augmentedB=torch.unsqueeze(torch.transpose(embeddingsB, 1, 0), 0) #(1,d,Tb)
    diff_mat=augmentedA-augmentedB
    dist_mat=torch.norm(diff_mat,2,dim=1) # (Ta,Tb)
    assert dist_mat.shape[0]==embeddingsA.shape[0] and dist_mat.shape[1]==embeddingsB.shape[0]
    return dist_mat

def soft_topk(s,a,k,device):
    # z:[T], s=[T]
    T=s.shape[0]
    z = torch.linspace(0, T - 1, T).to(device)
    def peaked_softmax(s,z, a):
        exp_s = F.softmax(a * s, dim=0)
        return torch.matmul(z, exp_s)
    m,hard_topk_argmax=torch.topk(s, k=int(k));hard_topk_argmax_np=hard_topk_argmax.cpu().data.numpy()
    Z_ = torch.zeros(0).to(device)
    for i in range(k):
        z_=peaked_softmax(-(s-m[i])**2,z,a)
        Z_=torch.cat([Z_, torch.unsqueeze(z_,0)],dim=0)
    return Z_,Z_.cpu().data.numpy(),hard_topk_argmax_np

def topk_visual(itr,out_t,indices,all_values,kvalues,filename,W_tfidf,attr_idx,classlist,A_,save_path):
    indices,idx=torch.sort(indices,dim=0)
    indices=indices.cpu().data.numpy();idx=idx.cpu().data.numpy()
    colors = cm.rainbow(np.linspace(0, 1, len(attr_idx)))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    indices_sorted=np.sort(indices,0)
    k=indices.shape[0]
    selected_labels=np.argsort(-W_tfidf)
    for i, a in enumerate(attr_idx):
        # if selected_labels[i] in A_:continue
        # label=classlist[selected_labels[i]]
        label = classlist[a]
        x=indices[i * (k // 3):min((i + 1) * (k // 3), k),0]
        y = kvalues[idx[i * (k // 3):min((i + 1) * (k // 3), k),0],0]
        # x=np.linspace(0,out_t-1,out_t)
        # y =all_values[:,a]
        ax.plot(x, y,'.', alpha=0.8, color=colors[i], label=label)
    plt.xlabel('Frame Number')
    plt.ylabel('Full Video Logits')
    plt.xlim((0, out_t))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),ncol=1)
    plt.savefig(save_path+"{}_{}.png".format(filename,itr))
    plt.close()

def HMM_visual(itr,out_t,indices,values,filename,A_,classlist,save_path):
    colors = cm.rainbow(np.linspace(0, 1, len(A_)))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i,a in enumerate(A_):
        label = classlist[a]
        x = np.asarray(indices[a])
        y = values[a].cpu().data.numpy()
        ax.plot(x, y,alpha=0.8, color=colors[i], label=label)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),ncol=1)
    plt.xlabel('Frame Number')
    plt.ylabel('HMM_decoding')
    plt.xlim((0, out_t))
    plt.savefig(save_path+"{}_{}.png".format(filename,itr))
    plt.close()

def attention_visual(itr,T,indices,values,filename,save_path):
    x=indices[:,0]
    y = values[:,0]
    plt.plot(x, y,'.')
    plt.xlabel('Frame Number')
    plt.ylabel('Selected (1/0)')
    plt.savefig(save_path+"{}_{}_Attention.png".format(filename,itr))
    plt.close()


def save_to_file(attr_k,classlist,filename):
    file1 = open(filename, "w+")
    for a in attr_k:
        file1.write(str(classlist[a])+" : "+str(np.mean(attr_k[a]))+'\n')

    file1.close()




def topk_visual_test(itr,out_t,attr_scores,all_values,attr_label,filename,classlist,save_path):
    I=int(np.sum(attr_label))
    colors = cm.rainbow(np.linspace(0, 1, I))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    i=-1
    for a in range(len(attr_label)):
        if attr_label[a]==0:continue
        i=i+1
        label = classlist[a]
        x=np.linspace(0,out_t-1,out_t)
        y =all_values[:,a]
        ax.plot(x, y,'.', alpha=0.8, color=colors[i], label=label)
    plt.xlabel('Frame Number')
    plt.ylabel('Full Video Logits')
    plt.xlim((0, out_t))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),ncol=1)
    plt.savefig(save_path+"{}_{}.png".format(filename,itr))
    plt.close()

def video_stage_visual(itr,indices,kvalues,attr_idx,n_stages,out_t,filename,save_path):
    indices, idx = torch.sort(indices, dim=0)
    indices = indices.cpu().data.numpy();
    idx = idx.cpu().data.numpy()
    colors = cm.rainbow(np.linspace(0, 1, n_stages))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    k = indices.shape[0]
    for i in range(n_stages):
        label = "stage {}".format(i+1)
        x = indices[i * (k // 3):min((i + 1) * (k // 3), k), attr_idx[0]]
        y = kvalues[idx[i * (k // 3):min((i + 1) * (k // 3), k), 0], attr_idx[0]]
        ax.plot(x, y, '.', alpha=0.8, color=colors[i], label=label)
    plt.xlabel('Frame Number')
    plt.ylabel('Full Video Logits')
    plt.xlim((0, out_t))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.savefig(save_path + "{}_{}.png".format(filename, itr))
    plt.close()

def attr_softmax_visual(itr,data,filename,classlist,I,save_path):
    # plot the scores for the top I attrs
    colors = cm.rainbow(np.linspace(0, 1,I))
    fig=plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    labels_sorted=np.argsort(-data.cpu().data.numpy())[0:I]
    for i, label in enumerate(labels_sorted):
        ax.bar(i, data[label], color=colors[i], width=0.25,label=classlist[label])

    plt.xlabel('Attribute')
    plt.ylabel('Softmax Value')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),ncol=1)
    plt.savefig(save_path+"{}_{}.png".format(filename,itr))
    plt.close()


def save_predictions(filenames,dish_predicted,dish_labels,dish_index2label,save_path):
    file1 = open(save_path, "w+")
    for i in range(len(filenames)):
        f=filenames[i];p=dish_predicted[i];y=dish_labels[i]
        file1.write(f+"***"+dish_index2label[p]+"***"+dish_index2label[y]+'\n')
    file1.close()


def find_attr_stage_relation(attr,dish_label,dataset,n_stages=3):
    dishes, frame_level_gt, classlist, testidx=dataset.dish_labels, dataset.frame_level_gt, dataset.index2label, dataset.testidx
    attr_label=classlist[attr]
    attr_freq_per_stage=np.zeros([n_stages])
    relavent_vid_count=0
    for v in testidx:
        out_t=len(frame_level_gt[v])
        attr_total_frames=np.sum(frame_level_gt[v]==attr)
        if attr_total_frames==0:continue
        if dishes[v][0]!=dish_label:continue
        relavent_vid_count+=1
        for q in range(n_stages):
            d = out_t // n_stages
            begin = q * d;
            end = (q + 1) * d
            if 0.50<np.sum(attr==frame_level_gt[v][begin:end])/attr_total_frames:
                attr_freq_per_stage[q]=attr_freq_per_stage[q]+1

    # assert np.sum(attr_freq_per_stage)==relavent_vid_count
    attr_freq_per_stage=attr_freq_per_stage/np.sum(attr_freq_per_stage)
    print("Frequency per stage for attribute {} :".format(attr_label))
    print(attr_freq_per_stage)



