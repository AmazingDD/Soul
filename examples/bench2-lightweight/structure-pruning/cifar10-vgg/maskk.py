import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from statistics import mean
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

import random
import copy
'xiugai'

class PruningNetworkManager:
    def __init__(self, model):
        self.pruning_layers = self.get_pruning_layers(model)
        self.layers = self.printgra(model)
        self.all_num = 0.0
        self.grow_num = 0.0
        self.prune_num = 0.0
        self.masks = []
        self.mask_grows = []
       
        self.grads=[]
        self.count=1
        


    def get_pruning_layers(self, model):
        #model_list=list(model.modules())
        pruning_layers = []
        #print(model)
        for module in model.modules():
        
        #for i in range(len(model_list)):
            #print(i)
            #print(model_list[i])
            if type(module).__name__ == 'PruningLayer':
            #if isinstance(model_list[i], PruningLayer):
                #print(model_list[i])
                #if type(module) is type(PruningLayer):
                #if isinstance(module, PruningLayer):
                print('aaa')
                #print(module)
                pruning_layers.append(module)
        #print(pruning_layers)
        return pruning_layers

    def evaling(self):
        for pruning_layer in self.pruning_layers:
            pruning_layer.seteval()

    def training(self):
        for pruning_layer in self.pruning_layers:
            #print('11111111')
            pruning_layer.settrain()
            #print('22222222222')
    def reset_zeros(self):
        
        for pruning_layer in self.pruning_layers:
            pruning_layer.reset_zero()

    def update_masks(self,model,a,b):
        old=copy.deepcopy(self.masks)
        acts = []
        #print(self.pruning_layers)
        for pruning_layer in self.pruning_layers:
            #print('aaa')
            #print(pruning_layer)
            activation = pruning_layer.get_actt()
            #print(activation)
            acts.append(activation)
        #print(acts)
        
        
     
            
       
        #print(len(acts))
        
        sorted_indices = torch.argsort(torch.cat(acts), descending=True)
        print(sorted_indices.shape)
        num_elements = int(len(sorted_indices) * a) #a是要留的比例
        #print(num_elements)
        threshold_indice = sorted_indices[num_elements]
        print(num_elements)
        print(threshold_indice)
        threshold=torch.cat(acts)[threshold_indice]
        print(len(acts))
        print(threshold)
        i = 0
        #print(len(acts))
        for i in range(len(acts)):
        
            if len(self.masks) != 13:
                mask=(acts[i] > threshold).float().detach()
                self.masks.append(mask)
            else:
                self.masks[i]=(acts[i] > threshold).float().detach()
            i += 1
        i = 0
        prune_num =0

        for i in range(len(self.masks)):
            prune_num += torch.sum(self.masks[i] == 0).item()
            i+=1
        print(prune_num)
        
        
        
        
        i=0
        channel_gradients_grows=[]
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                gradients =module.weight.grad
                
                gradients_grow=torch.where(self.masks[i] == 0,torch.abs(module.weight.grad),
                                                             torch.zeros_like(module.weight.grad))
                channel_gradients_grows.append(gradients_grow)
                i += 1
        sorted_indices_grow = torch.argsort(torch.cat(channel_gradients_grows), descending=True)
        print(sorted_indices_grow.shape)
        num_elements_grow = int(len(sorted_indices_grow) * b)
        print(num_elements_grow)
        print(num_elements_grow)
        threshold_indice_grow = sorted_indices_grow[num_elements_grow]
        #print(threshold_indice)
        threshold_grow=torch.cat(channel_gradients_grows)[threshold_indice_grow]
        print(threshold_grow)
        for i in range(len(self.masks)):
            self.masks[i][channel_gradients_grows[i] > threshold_grow] = 1
        i = 0
        prune_num =0

        for i in range(len(self.masks)):
            prune_num += torch.sum(self.masks[i] == 0).item()
            i+=1
        print(prune_num)
        
        
        num=0
        new=copy.deepcopy(self.masks)
        if len(old) == 13:
            for o,n in zip(old,new):
                numl=torch.logical_xor(o,n)
                numl=torch.sum(numl==1).item()
                num=num+numl
        print(num)
        
        df = pd.DataFrame([num/2])
        print(df)
        df.to_csv('01.csv', mode='a', index=False)
        
    
  
        
      
                    
                    


    def do_masks(self,model):

        i = 0
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                prune_indices = (self.masks[i] == 0).nonzero().view(-1)
                #print(module.weight.data.shape)
                mask_l = torch.ones_like(module.weight.data)
                mask_l[prune_indices,:, :, :] = 0
                module.weight.data.mul_(mask_l)
                i += 1
        i = 0
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                prune_indices = (self.masks[i] == 0).nonzero().view(-1)
                mask_l = torch.ones_like(module.weight.data)
                mask_l[prune_indices] = 0
                module.weight.data.mul_(mask_l)
                module.bias.data.mul_(mask_l)
                i += 1

    def compute_prune(self):
        i = 0
        self.prune_num =0

        for i in range(len(self.masks)):
            self.prune_num += torch.sum(self.masks[i] == 0).item()
            i+=1
        j=0
        self.reserve_num=0
        for j in range(len(self.masks)):
            self.grow_num += torch.sum(self.masks[j] == 1).item()
            j += 1
        s=0
        self.all_num=0
        for s in range(len(self.masks)):
            self.all_num +=self.masks[s].numel()
            s += 1



    


    def save_csv(self):
        data = {'Zero Percentage': [self.prune_num/self.all_num],'reserve_num': [self.reserve_num]}
        df = pd.DataFrame(data)
        df.to_csv('aaa.csv', mode='a', index=False)
        # print(mask[0])
       
        i = 0

        for i in range(len(self.masks)):
            self.prune_num_l = torch.sum(self.masks[i] == 0).item()         
            l = {'Zero Percentage_l': [self.prune_num_l]}
            print(l)
            df = pd.DataFrame(l)
            df.to_csv('layerprune.csv', mode='a', index=False)
            i+=1
        
        


    def save_csv_max(self):
        data = {'Zero Percentage': [self.prune_num/self.all_num],'grow Percentage': [self.grow_num/self.all_num]}
        df = pd.DataFrame(data)
        df.to_csv('aaaa.csv', mode='a', index=False)
        # print(mask[0])
        
        
        
        i = 0

        for i in range(len(self.masks)):
            self.prune_num_l = torch.sum(self.masks[i] == 0).item()         
            l = {'Zero Percentage_l': [self.prune_num_l]}
            print(l)
            df = pd.DataFrame(l)
            df.to_csv('layerprunee.csv', mode='a', index=False)
            i+=1
        mask=[]
        for i in range(len(self.masks)):
            mask.append(self.masks[i].cpu().numpy())
        df=pd.DataFrame(mask)
        df.to_csv("mask.csv",mode="a",index=False)
        
        


    def printgra(self, model):
        layers = []
        for module in model.modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
                layers.append(module)
        return layers


    def prints(self):
        masks = []
        for pruning_layer in self.pruning_layers:
            mask = pruning_layer.get_mask()
            # print(mask.shape)
            masks.append(mask)
        i = 0
        '''for layer in self.layers:
            #print(layer)
            weight_gradient = layer.weight
            print(weight_gradient.shape)
            pruned_indices = (masks[i] == 0).nonzero().squeeze(dim=1)
            print(masks[0].shape)
            print(pruned_indices.shape)
            if pruned_indices.nelement() == 0:
                #print('111')
                continue
            else:
                pruned_gradients = weight_gradient[pruned_indices]
            i=i+1
            print(f"Gradients of pruned Conv2d layer weights: {pruned_gradients}")
        '''
        # print(layer)
        print(len(masks))
        print(len(self.layers))
        weight_gradient = self.layers[1].weight
        print(weight_gradient.shape)
        pruned_indices = (masks[0] == 0).nonzero()
        # pruned_indices = np.where(masks[0] == 0)
        print(masks[0].shape)
        print(pruned_indices.shape)
        if pruned_indices.nelement() == 0:
            print('111')

        else:
            pruned_gradients = weight_gradient[pruned_indices]
            print(pruned_gradients.shape)

            print(f"Gradients of pruned Conv2d layer weights: {pruned_gradients[0]}")


class PruningLayer(nn.Module):
    def __init__(self):
        super(PruningLayer, self).__init__()
        self.activation_means = None
        self.mask = None
       
        
        self.v_accumulated = None
        self.prune_count = 0  # 记录被剪去的神经元个数
        self.restore_count = 0
        self.p1 = 0
        self.p2 = 0
        self.a = 0
        self.aa = 0
        
        self.count1 = 1
        self.trainingstate = True
        self.spikes = 0
        self.membrance=0

    def seteval(self):
        self.trainingstate = False

    def settrain(self):
        #print('bbbbbbb')
        self.trainingstate = True

    def forward(self, x,v):
        #print('aaaaaa')
        if self.trainingstate:
            #print('aaaaaaaaaa')

            if self.aa==0 or self.v_accumulated == None:
                print('111111111111111111111111111')
                self.spikes = x.detach()
                self.membrance=v.abs().detach()
                
                v_temp = (self.spikes + self.membrance).detach()
                self.v_accumulated=torch.mean(v_temp, dim=(0, 1, 3, 4))

            else:
                self.spikes = x.detach()
                self.membrance = v.abs().detach()
                #print(self.spikes[0][0][0])
                #print(self.membrance[0][0][0])
                v_temp = (self.spikes + self.membrance).detach()
                self.v_accumulated = self.v_accumulated * self.count1+torch.mean(v_temp, dim=(0, 1, 3, 4))
                #self.activation_means = self.decay_rate * self.activation_means + (1 - self.decay_rate) * torch.mean(x,dim=(0,1,3,4)).detach()
                self.count1 += 1
                self.v_accumulated/= self.count1
                # self.activation_means = (self.activation_means * self.count1 + torch.mean(x,dim=(0, 1,3,4))).detach()
                # self.count1 += 1
                #print(self.count1)
                # self.activation_means /= self.count1

        return x  # ???????????????????????????????????????????????????????????

    
    
    
    
    '''def get_activation_means(self):
    
        return self.activation_means'''
        

    

    

    def reset_zero(self):
        self.v_accumulated = None
        self.count1 = 1
        self.aa=1
       
   
    def get_prunenum(self):
        num = torch.sum(self.mask == 0).item()
        return num

    def get_allnum(self):
        num = self.mask.numel()
        return num

    def get_actt(self):
        #print("Calling get_actt")

        return self.v_accumulated

    '''
    def restore_neurons(self, membrane_potentials, threshold):
        mask_indices = (self.mask == 0).nonzero()  # 获取掩码为0的位置的索引
        for index in mask_indices:
            if membrane_potentials[index] > threshold:
                self.mask[index] = 1.0
    '''

    def save_csv(self):
        self.mask_zero_percentage = torch.sum(self.mask == 0).item() / self.mask.numel() * 100
        data = {'Zero Percentage': [self.mask_zero_percentage]}
        df = pd.DataFrame(data)
        df.to_csv('aaa.csv', mode='a', index=False)
        # print(mask[0])
        datathre = {'thre': [self.threshold], 'threres': self.restore_threshold}
        dfthre = pd.DataFrame(datathre)
        dfthre.to_csv('bbb.csv', mode='a', index=False)

        # 最后总的mask中0和1的个数
        prune_indices = (self.mask == 0).nonzero()
        self.prune_count = len(prune_indices)

        restore_indices = (self.mask == 1).nonzero()
        self.restore_count = len(restore_indices)
        print('Pruned neurons:', self.prune_count)
        print('Restored neurons:', self.restore_count)
        print('P neurons:', self.p1)
        print('P neurons:', self.p2)
        print('A neurons:', self.a)
        print('total:', self.mask.numel())
        print(self.threshold)
        print(self.restore_threshold)

    def get_mask(self):
        return self.mask

