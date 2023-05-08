import math
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
from torch import nn
from torch import optim
import os
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, recall_score
from tqdm import tqdm
import wandb

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.cuda.empty_cache()
torch.backends.cudnn.enabled = True
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
wandb.init(project="big_data_final_ass2")

######################
#Parameter Setting
######################
#Set number of epochs
epoch_num = 50

#Set the batch size
batch_size = 16

#Set the percent of training set within dataset, we set 80% of data as training set
train_percent = 0.8
val_percent = 1 -train_percent
random_seed = 912240

#Choose what model are used in this experiment
model = models.resnet50(pretrained=True).to(device)
#Modify the last layer of model to match the size of output
classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(in_features=model.fc.in_features,out_features=4)
)

model.fc = classifier.cuda()

#Set the loss function
loss_fn = nn.CrossEntropyLoss().cuda()

#Set the optimizer
optimizer = optim.SGD(model.fc.parameters(), lr=0.0001)

#Use cuda to accelerate the training process if available

print(torch.cuda.is_available())

lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma= 0.5)

transform_norm = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

#Define a customized Dataset Class
class Mydata(Dataset):
    def __init__(self, annotations_file, img_dir, train_flag=True):
        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.img_size = 512
        self.train_flag = train_flag
        self.train_tf = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transform_norm
        ])
        self.val_tf = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transform_norm
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[index][0]) + ".jpg")
        image = Image.open(img_path)
        image = image.convert('RGB')
        label = self.img_labels.iloc[index][2] #use the transformed categorical label
        if self.train_flag:
            image = self.train_tf(image)
        else:
            image = self.val_tf(image)
        return image, label


def gettraindata(size,filepath,labelname):
    full_label = pd.read_csv(filepath,encoding="ISO-8859-1")
    label_df = full_label[["image_id",labelname]].drop_duplicates(subset=['image_id'],keep="first") #delete the duplicates
    label_df = label_df.dropna() #delete missing value
    cate_num = len(set(label_df[labelname].tolist()))
    samplesize_pergroup = math.floor(size/cate_num)
    group_sizes = label_df.groupby(labelname).size()
    samples = []
    for label_cate in group_sizes.index:
        groupdata = label_df[label_df[labelname]==label_cate]
        sample = groupdata.sample(min(group_sizes[label_cate],samplesize_pergroup),random_state=912240)
        samples.append(sample)
    sampled_label_df = pd.concat(samples,axis=0)
    le = LabelEncoder()
    sampled_label_df['cate_encode'] = le.fit_transform(sampled_label_df[labelname])
    print('Train Data Ready')
    return sampled_label_df

def train_one_batch(images, labels, modelname, optimizer, lossfun, currentepoch, batchidx):
    images = images.to(device)
    labels = labels.to(device)

    outputs = modelname(images)
    loss = lossfun(outputs,labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    loss = loss.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    log_train_perbatch = {}
    log_train_perbatch['epoch'] = currentepoch
    log_train_perbatch['batch'] = batchidx
    log_train_perbatch['train_loss'] = loss
    log_train_perbatch['train_accuracy'] = accuracy_score(labels,preds)
    return log_train_perbatch

def eval(dataloader,modelname,lossfun, currentepoch):
    loss_list = []
    labels_list = []
    preds_list = []
    modelname.eval()
    with torch.no_grad():
        for image, label in dataloader:
            image = image.to(device)
            label = label.to(device)
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            loss = lossfun(outputs, label)
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
        loss_list.append(loss)
        labels_list.extend(label)
        preds_list.extend(preds)
    print(preds_list)
    print(label)
    log_test = {}
    log_test['epoch'] = currentepoch
    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')
    return log_test



if __name__ == '__main__':
    #toy data set
    #img_dir = '/tmp/pycharm_project_444/assignment2/images_toy'
    #full image pool path
    img_dir = '/root/autodl-tmp/images'
    label_path = '/tmp/pycharm_project_444/assignment2/toy_label.csv'
    full_label_path = '/tmp/pycharm_project_444/assignment2/images_cuisines_price.csv'

    print(model)

    #img_label = pd.read_csv(label_path)
    #print(img_label)
    img_label = gettraindata(10000,full_label_path,'expensiveness_category').reset_index(drop=True)
    print(img_label['expensiveness_category'].value_counts())
    train_label, val_label = train_test_split(img_label,train_size=train_percent,random_state=random_seed)

    train_dataset = Mydata(train_label,img_dir,train_flag=True)
    val_dataset = Mydata(val_label,img_dir,train_flag=False)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    train_log_df = pd.DataFrame()
    val_log_df = pd.DataFrame()
    best_val_accuracy = 0

    checkpoint_path = '/tmp/pycharm_project_444/assignment2/model_checkpoint_q2/'
    training_logpath = '/tmp/pycharm_project_444/assignment2/training_log_q2/'

    if len(os.listdir(checkpoint_path)) != 0:
        for file_name in os.listdir(checkpoint_path):
            file_path = os.path.join(checkpoint_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"{file_name} has been deleted.")
            except Exception as e:
                print(e)

    ########################
    #Model Training
    #######################
    cum_batch_idx = 1
    for epoch in range(1, epoch_num+1):
        print(f'Epoch {epoch}/{epoch_num}')
        model.train()
        batch_idx = 1
        for (image,label) in tqdm(train_loader):
            log_train = train_one_batch(image, label, model, optimizer, loss_fn, epoch, batch_idx)
            train_log_df = pd.concat([train_log_df,pd.DataFrame(log_train,index=[0])], axis=0)
            batch_idx = batch_idx + 1
            cum_batch_idx = cum_batch_idx +1
            #print(log_train['train_loss'])
            wandb.log({"Loss curve":log_train['train_loss']},step=cum_batch_idx)

        lr_scheduler.step()
        model.eval()
        log_val = eval(val_loader,model,loss_fn,epoch)
        print("val accuracy",log_val['test_accuracy'])
        val_log_df = pd.concat([val_log_df,pd.DataFrame(log_val,index=[0])],axis=0)

        ##########################
        #Model Saving
        ##########################

        if log_val['test_accuracy'] > best_val_accuracy:
            old_best_checkpoint_path = checkpoint_path + 'best-{:.3f}.pth'.format(best_val_accuracy)
            if os.path.exists(old_best_checkpoint_path):
                os.remove(old_best_checkpoint_path)
            best_val_accuracy = log_val['test_accuracy']
            new_best_checkpoint_path = checkpoint_path + 'best-{:.3f}.pth'.format(best_val_accuracy)
            torch.save(model,new_best_checkpoint_path)
            print('Best Model -{:.3f} Saved'.format(best_val_accuracy))
    val_log_df.to_csv(training_logpath+'val_log.csv',index=False)
    train_log_df.to_csv(training_logpath+'train_log.csv',index=False)

    wandb.finish()







    # for batch, (image, label) in enumerate(train_loader):
    #     log = train_one_batch(image, label, model, optimizer, loss_fn, 1, batch)
    #     train_log_df = pd.concat([train_log_df,pd.DataFrame(log,index=[0])], axis=0)
    # print(train_log_df)
    # print(eval(val_loader,model,loss_fn,1))

