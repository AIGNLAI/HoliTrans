import copy
import logging
import numpy as np
import os
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from inc_net import ResNetCosineIncrementalNet,SimpleVitNet
from utils.toolkit import target2onehot, tensor2numpy, accuracy

num_workers = 8

class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments=[]
        self._network = None

        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        acc_total,grouped = self._evaluate(y_pred, y_true)
        return acc_total,grouped,y_pred[:,0],y_true

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[1] 
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)  
    
    def _evaluate(self, y_pred, y_true):
        ret = {}
        acc_total,grouped = accuracy(y_pred.T[0], y_true, self._known_classes,self.class_increments)
        return acc_total,grouped 
    
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if args["model_name"]!='ncm':
            if args["model_name"]=='adapter' and '_adapter' not in args["convnet_type"]:
                raise NotImplementedError('Adapter requires Adapter backbone')
            if args["model_name"]=='ssf' and '_ssf' not in args["convnet_type"]:
                raise NotImplementedError('SSF requires SSF backbone')
            if args["model_name"]=='vpt' and '_vpt' not in args["convnet_type"]:
                raise NotImplementedError('VPT requires VPT backbone')

            if 'resnet' in args['convnet_type']:
                self._network = ResNetCosineIncrementalNet(args, True)
                self._batch_size=128
            else:
                self._network = SimpleVitNet(args, True)
                self._batch_size= args["batch_size"]
            
            self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
            self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        else:
            self._network = SimpleVitNet(args, True)
            self._batch_size= args["batch_size"]
        self.args=args

    def after_task(self):
        self._known_classes = self._classes_seen_so_far
    
    def replace_fc(self,trainloader):
        self._network = self._network.eval()

        if self.args['use_RP']:
            self._network.fc.use_RP=True
            if self.args['M']>0:
                self._network.fc.W_rand=self.W_rand
            else:
                self._network.fc.W_rand=None

        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.cuda()
                label=label.cuda()
                embedding = self._network.convnet(data)
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        Y=target2onehot(label_list,self.total_classnum)
        if self.args['use_RP']:
            if self.args['M']>0:
                Features_h=torch.nn.functional.relu(Features_f @ self._network.fc.W_rand.cpu())
            else:
                Features_h=Features_f
            for class_index in np.unique(label_list):
                data_index = (label_list==class_index).nonzero().squeeze(-1)
                Features_class = Features_f[data_index]
                class_prototype = Features_class.mean(0)
                if class_index not in self.cp:
                    self.cp[class_index] = 0
                    self.num[class_index] = 0
                self.cp[class_index] = class_prototype
                self.var.append(Features_class.var(0))
            self.Q=self.Q+Features_h.T @ Y 
            self.G=self.G+Features_h.T @ Features_h
            ridge=self.optimise_ridge_parameter(Features_h,Y)
            Wo=torch.linalg.solve(self.G+ridge*torch.eye(self.G.size(dim=0)),self.Q).T
            self._network.fc.weight.data=Wo[0:self._network.fc.weight.shape[0],:].to(device='cuda')
        else:
            for class_index in np.unique(self.train_dataset.labels):
                data_index=(label_list==class_index).nonzero().squeeze(-1)
                if self.is_dil:
                    class_prototype=Features_f[data_index].sum(0)
                    self._network.fc.weight.data[class_index]+=class_prototype.to(device='cuda')
                else:
                    class_prototype=Features_f[data_index].mean(0)
                    self._network.fc.weight.data[class_index]=class_prototype

    def learn_distance(self):
        score = []
        for i, (_, inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[0]
            for scores in predicts:
                score.append(scores)
        return sum(score)/len(score),max(score),min(score)

    def compute_accuracy(self,lmda, threshold, score, y_true):
        from sklearn.metrics import accuracy_score
        open_predict = [0 if s < threshold * lmda else 1 for s in score]
        return accuracy_score(y_true, open_predict)

    def ternary_search(self,lower_bound, upper_bound, threshold, score, y_true):
        epsilon = 1e-7
        while upper_bound - lower_bound > epsilon:
            mid1 = lower_bound + (upper_bound - lower_bound) / 3
            mid2 = upper_bound - (upper_bound - lower_bound) / 3
            acc1 = self.compute_accuracy(mid1, threshold, score, y_true)
            acc2 = self.compute_accuracy(mid2, threshold, score, y_true)
            if acc1 < acc2:
                lower_bound = mid1
            else:
                upper_bound = mid2
        return self.compute_accuracy((lower_bound + upper_bound) / 2, threshold, score, y_true), (lower_bound + upper_bound) / 2


    def train_r(self,thereold,lower_bound,upper_bound,num_points = 10,cent_num = 50,domain_range = 2.5):
        print(f"Domain_std {domain_range}")
        random_points = []
        score = []
        y_true = []
        for label1, prototype1 in self.cp.items():
            for label2, prototype2 in self.cp.items():
                if label1 != label2:
                    midpoint = (prototype1 + prototype2) / 2
                    for _ in range(num_points):
                        random_point = midpoint + np.random.uniform(-domain_range, domain_range, size=prototype1.shape)
                        random_points.append(random_point)
                        y_true.append(0)
        for label,prototype in self.cp.items():
            for _ in range(cent_num):
                random_point = prototype + np.random.uniform(-domain_range, domain_range, size=prototype.shape)
                random_points.append(random_point)
                y_true.append(1)
        random_points = torch.stack(random_points).float()
        random_points = random_points.to(self._device)
        outputs = self._network.fc(random_points)['logits']
        predicts = torch.max(outputs, dim=1)[0]
        for scores in predicts:
            score.append(scores)
        y_true = np.array(y_true)
        max_lmda = 0
        best_acc = 0
        from sklearn.metrics import roc_auc_score, roc_curve,accuracy_score
        best_acc, max_lmda = self.ternary_search(lower_bound, upper_bound, thereold, score, y_true)
        for lmda in np.arange(lower_bound, upper_bound + 0.1, 0.05):
            open_predict = []
            for s in score:
                if s < thereold*lmda:
                    open_predict.append(0)
                else:
                    open_predict.append(1)
            open_predict = np.array(open_predict)
            y_true = np.array(y_true)
            acc = accuracy_score(y_true,open_predict)
            if acc > best_acc:
                max_lmda = lmda
                best_acc = acc
        logging.info(f'Select lmda {max_lmda}')
        return max_lmda

    def open_detection_with_prototype(self):
        y_true = []
        thereold,maxx,minn = self.learn_distance()
        score = []
        label_true = []
        label_pred = []
        for i, (_, inputs, targets) in enumerate(self.test_loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts, plabel = torch.max(outputs, dim=1)
            y_true.extend([1]*len(predicts))
            for scores in predicts:
                score.append(scores.item())
            for pl in plabel:
                label_pred.append(pl.item())
            for tar in targets:
                label_true.append(tar.item())

        
        for i, (_, inputs, targets) in enumerate(self.open_test_loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[0]
            predicts, plabel = torch.max(outputs, dim=1)
            y_true.extend([0]*len(predicts))
            for scores in predicts:
                score.append(scores.item())
            for pl in plabel:
                label_pred.append(pl.item())
            for tar in targets:
                label_true.append(-1)
        from sklearn.metrics import roc_auc_score, roc_curve,accuracy_score
        lmda = self.train_r(thereold.item(),minn.item()/thereold.item(),maxx.item()/thereold.item(),domain_range = sum(self.var)/len(self.var))
        open_predict = []
        for s in score:
            if s < thereold*lmda:
                open_predict.append(0)
            else:
                open_predict.append(1)
        open_predict = np.array(open_predict)
        y_true = np.array(y_true)
        acc = accuracy_score(y_true,open_predict)
        auc = roc_auc_score(y_true, open_predict)
        fpr, _, _ = roc_curve(y_true, open_predict)
        logging.info(f'Open ACC {acc}; AUC {auc}; FPR {fpr[1]}')
        right = 0
        for i in range(len(label_pred)):
            if label_true[i] == -1:
                if open_predict[i] == 0:
                    right += 1
            else:
                if open_predict[i] == 1 and label_pred[i] == label_true[i]:
                    right += 1
        logging.info(f'Test ACC {right/len(label_pred)}')

    def optimise_ridge_parameter(self,Features,Y):
        ridges=10.0**np.arange(-8,9)
        num_val_samples=int(Features.shape[0]*0.8)
        losses=[]
        Q_val=Features[0:num_val_samples,:].T @ Y[0:num_val_samples,:]
        G_val=Features[0:num_val_samples,:].T @ Features[0:num_val_samples,:]
        for ridge in ridges:
            Wo=torch.linalg.solve(G_val+ridge*torch.eye(G_val.size(dim=0)),Q_val).T
            Y_train_pred=Features[num_val_samples::,:] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred,Y[num_val_samples::,:]))
        ridge=ridges[np.argmin(np.array(losses))]
        logging.info("Optimal lambda: "+str(ridge))
        return ridge
    
    def incremental_train(self, data_manager):
        self.total_classnum = data_manager.get_total_classnum()
        self._cur_task += 1
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(self._cur_task)
        if self.args['use_RP']:
            del self._network.fc
            self._network.fc=None
        self._network.update_fc(self._classes_seen_so_far)
        if self.is_dil == False:
            logging.info("Starting CIL Task {}".format(self._cur_task+1))
        logging.info("Learning on classes {}-{}".format(self._known_classes, self._classes_seen_so_far-1))
        self.class_increments.append([self._known_classes, self._classes_seen_so_far-1])
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far),source="train", mode="train", )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=num_workers)
        train_dataset_for_CPs = data_manager.get_dataset(np.arange(self._known_classes, self._classes_seen_so_far),source="train", mode="test", )
        self.train_loader_for_CPs = DataLoader(train_dataset_for_CPs, batch_size=self._batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._classes_seen_so_far), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=num_workers)
        open_test_dataset = data_manager.get_dataset(np.arange(self._classes_seen_so_far,100), source="test", mode="test" )
        self.open_test_loader = DataLoader(open_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=num_workers)
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_CPs)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def freeze_backbone(self,is_first_session=False):
        if 'vit' in self.args['convnet_type']:
            if isinstance(self._network.convnet, nn.Module):
                for name, param in self._network.convnet.named_parameters():
                    if is_first_session:
                        if "head." not in name and "ssf_scale" not in name and "ssf_shift_" not in name: 
                            param.requires_grad = False
                    else:
                        param.requires_grad = False
        else:
            if isinstance(self._network.convnet, nn.Module):
                for name, param in self._network.convnet.named_parameters():
                    if is_first_session:
                        if "ssf_scale" not in name and "ssf_shift_" not in name: 
                            param.requires_grad = False
                    else:
                        param.requires_grad = False

    def show_num_params(self,verbose=False):
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} training parameters.')
        if total_params != total_trainable_params and verbose:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())

    def _train(self, train_loader, test_loader, train_loader_for_CPs):
        self._network.to(self._device)
        if self._cur_task == 0 and self.args["model_name"] in ['ncm','joint_linear']:
             self.freeze_backbone()
        if self.args["model_name"] in ['joint_linear','joint_full']: 
            if self.args["model_name"] =='joint_linear':
                assert self.args['body_lr']==0.0
            self.show_num_params()
            optimizer = optim.SGD([{'params':self._network.convnet.parameters()},{'params':self._network.fc.parameters(),'lr':self.args['head_lr']}], momentum=0.9, lr=self.args['body_lr'],weight_decay=self.weight_decay)
            scheduler=optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100000])
            logging.info("Starting joint training on all data using "+self.args["model_name"]+" method")
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            self.show_num_params()
        else:
            if self._cur_task == 0 and self.dil_init==False:
                if 'ssf' in self.args['convnet_type']:
                    self.freeze_backbone(is_first_session=True)
                if self.args["model_name"] != 'ncm':
                    self.show_num_params()
                    optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.args['body_lr'],weight_decay=self.weight_decay)
                    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
                    logging.info("Starting PETL training on first task using "+self.args["model_name"]+" method")
                    self._init_train(train_loader, test_loader, optimizer, scheduler)
                    self.freeze_backbone()
                if self.args['use_RP'] and self.dil_init==False:
                    self.setup_RP()
            if self.is_dil and self.dil_init==False:
                self.dil_init=True
                self._network.fc.weight.data.fill_(0.0)
            self.replace_fc(train_loader_for_CPs)
            self.show_num_params()
        
    
    def setup_RP(self):
        self.cp = {}
        self.num = {}
        self.var = []
        self.score = 10
        self.train_data = []
        self.initiated_G=False
        self._network.fc.use_RP=True
        if self.args['M']>0:
            M=self.args['M']
            self._network.fc.weight = nn.Parameter(torch.Tensor(self._network.fc.out_features, M).to(device='cuda'))
            self._network.fc.reset_parameters()
            self._network.fc.W_rand=torch.randn(self._network.fc.in_features,M).to(device='cuda')
            self.W_rand=copy.deepcopy(self._network.fc.W_rand)
        else:
            M=self._network.fc.in_features
        self.Q=torch.zeros(M,self.total_classnum)
        self.G=torch.zeros(M,M)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)
        
    

   