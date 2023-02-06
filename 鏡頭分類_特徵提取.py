import cv2
import numpy as np
import glob as gb
#from __future__ import division
import torch 
import torch.nn as nn
from torch.autograd import Variable
#from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import pickle as pkl
import itertools
''''''
def get_blur_degree(image_file, sv_num=10):
    img = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
    u, s, v = np.linalg.svd(img)
    top_sv = np.sum(s[0:sv_num])
    total_sv = np.sum(s)
    return top_sv/total_sv
''''''

class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers= num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5,5) for x in range(num_layers)])
        self.output = nn.Linear(5,2)
    
    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)
        
def get_test_input(input_dim, CUDA):
    img = cv2.imread("1_0.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    num_classes
    return img_

def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    
    return parser.parse_args()
if __name__ ==  '__main__':
    params = []
    img_path=[]#//*.png
    params.append(cv2.IMWRITE_PNG_COMPRESSION)
    number=550
    while number<560:
        img_path = gb.glob("I://code//S//t5_2//"+str(number)+"//*.png")#S//t2
        img_path.sort(key=len)
        while len(img_path)==0 and (number<560):
            print(str(number)+"空")
            number=number+1
            img_path = gb.glob("I://code//S//t5_2//"+str(number)+"//*.png")#S//t2
            img_path.sort(key=len)            
        args = arg_parse()
        important=[]
        area_=[]
        area_std=[]
        distance=[]
        scales = args.scales
        quantity=[]
        images = args.images
        batch_size = int(args.bs)
        confidence = float(args.confidence)
        nms_thesh = float(args.nms_thresh)

        CUDA = torch.cuda.is_available()

        num_classes = 40
        classes = load_classes('data/coco.names') 

        #Set up the neural network
        print("Loading network.....")
        model = Darknet(args.cfgfile)
        model.load_weights(args.weightsfile)
        print("Network successfully loaded")
    
        model.net_info["height"] = args.reso
        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 == 0 
        assert inp_dim > 32
        if CUDA:
            model.cuda()
    
    
    #Set the model in evaluation mode
        model.eval()

    #Detection phase
        try:
            imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
        except NotADirectoryError:
            imlist = []
            imlist.append(osp.join(osp.realpath('.'), images))
        except FileNotFoundError:
            print ("No file or directory with the name {}".format(images))
            exit()
        imlist = img_path  
        if not os.path.exists(args.det):
            os.makedirs(args.det)
        batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
        im_batches = [x[0] for x in batches]
        orig_ims = [x[1] for x in batches]
        im_dim_list = [x[2] for x in batches]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)    
        if CUDA:
            im_dim_list = im_dim_list.cuda()
    
        leftover = 0
    
        if (len(im_dim_list) % batch_size):
            leftover = 1
        
        
        if batch_size != 1:
            num_batches = len(imlist) // batch_size + leftover            
            im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                            len(im_batches))]))  for i in range(num_batches)]        
        i = 0
        write =False
        objs = {}
        area_list_len=[0]
        jj=0
        for batch in im_batches:
            #load the image 
            
            if CUDA:
                batch = batch.cuda()
            with torch.no_grad():
                prediction = model(Variable(batch), CUDA)
            prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            if type(prediction) == int:
                i += 1
                aa=[jj]
                area_list_len.extend(aa)
                continue
            prediction[:,0] += i*batch_size         
            if not write:
                output = prediction
                write = 1
            else:
                output = torch.cat((output,prediction))
            
            for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
                im_id = i*batch_size + im_num
                objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            jj=jj+len(prediction)
            aa=[jj]
            area_list_len.extend(aa)
            i += 1       
        #*****
        im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())     
        scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)    
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2   
        output[:,1:5] /= scaling_factor
        for ii in range(0,len(area_list_len)-1):
            area_list=[]
            area=0
            aaa=0
            disx=[]
            disy=[]
            for iii in range(area_list_len[ii],area_list_len[ii+1]):
                if output[iii,-1]==0 and output[iii,-2]>0.5 :
                    if output[iii,-3]>0.985:
                        aaa=aaa+1
                    if output[iii,-3]>0.9:
                        area=(int)(abs(output[iii,1]-output[iii,3])*abs(output[iii,2]-output[iii,4]))
                        xc=[int((output[iii,1]+output[iii,3])/2)]
                        disx.extend(xc)
                        yc=[int((output[iii,2]+output[iii,4])/2)]
                        disy.extend(yc)
                        a=[area]
                        area_list.extend(a)
            a=[aaa]
            important.extend(a)
            if np.std(area_list)>0:
                std=[np.std(area_list)]
            else:
                std=[0]
            area_std.extend(std)
            if len(area_list)>2:            
                area_list.remove(np.min(area_list))
                area_list.remove(np.max(area_list))
            if len(area_list)!=0:
                mean=np.mean(area_list)
                med=np.median(area_list)
                a=[(int)((mean+med)/2)]
                area_.extend(a)
            else:
                a=[(int)(0)]
                area_.extend(a)
            a=[len(area_list)]
            quantity.extend(a)  
            ddd=[]
            for iii in range(0,len(disx)-1):
                for iiii in range(iii+1,len(disx)):
                    dd=[int(((disx[iii]-disx[iiii])**2+(disy[iii]-disy[iiii])**2)**0.5)]
                    ddd.extend(dd)
            tt=(np.mean(ddd)+np.median(ddd))/2
            if tt>0:
                dd=[tt]
            else:
                dd=[0]
            distance.extend(dd)
        if CUDA:
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        print("---完成面積:"+str(number)+"---")
       #*************
        wi=[]
        ne=[]
        x100=[]
        answer=[]
        n=0
        zero=np.zeros(1280)
        for path in img_path:    
            img  = cv2.imread(path)
            blur_degree=get_blur_degree(path)#
            pp=path.split('\\')
            pp=pp[-1].split('_')
            pp=pp[-1].split('.')
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, w = img.shape[:2]
            Binary1 = cv2.inRange(hsv, (20,0,63),(101,255,255))#去除觀眾
            kernel = np.ones((5,5), np.uint8)#18/18
            result = cv2.morphologyEx(Binary1,cv2.MORPH_OPEN, kernel)
            for i in range(0,720):
                if  (result[i,0:1280].sum()/w<30 ) :#or test[i,0:1280].sum()/1280<30
                     result[i,0:1280]=zero
            grass= result.sum()/255
            total=[grass,quantity[(int)(pp[0])],important[(int)(pp[0])],(int)(distance[(int)(pp[0])]),blur_degree]
            x100.append(total)
            answer.append(number)
        npz_file = np.load('I:\\code\\S\\data5_2\\'+str(number)+'.npz')
        answer = npz_file['answer']
        #**************
        np.savez_compressed('I:\\code\\S\\dection_tree_data5_2\\'+str(number)+'.npz', x=x100, y=answer)#)
        print("---完成:"+str(number)+"---")
        number=number+1
#%%
import cv2
import numpy as np
import glob as gb
#from __future__ import division
import torch 
import torch.nn as nn
from torch.autograd import Variable
#from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import pickle as pkl
import itertools
''''''
def get_blur_degree(image_file, sv_num=10):
    img = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
    u, s, v = np.linalg.svd(img)
    top_sv = np.sum(s[0:sv_num])
    total_sv = np.sum(s)
    return top_sv/total_sv
''''''

class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers= num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5,5) for x in range(num_layers)])
        self.output = nn.Linear(5,2)
    
    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)
        
def get_test_input(input_dim, CUDA):
    img = cv2.imread("1_0.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    num_classes
    return img_

def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    
    return parser.parse_args()
if __name__ ==  '__main__':
    params = []
    img_path=[]#//*.png
    params.append(cv2.IMWRITE_PNG_COMPRESSION)
    number=1
    while number<6:
        img_path = gb.glob("I://code//S//tt//124//"+str(number)+"//*.png")#S//t2
        img_path.sort(key=len)
        while len(img_path)==0 and (number<6):
            print(str(number)+"空")
            number=number+1
            img_path = gb.glob("I://code//S//tt//124//"+str(number)+"//*.png")#S//t2
            img_path.sort(key=len)            
        args = arg_parse()
        important=[]
        area_=[]
        area_std=[]
        distance=[]
        scales = args.scales
        quantity=[]
        images = args.images
        batch_size = int(args.bs)
        confidence = float(args.confidence)
        nms_thesh = float(args.nms_thresh)

        CUDA = torch.cuda.is_available()

        num_classes = 40
        classes = load_classes('data/coco.names') 

        #Set up the neural network
        print("Loading network.....")
        model = Darknet(args.cfgfile)
        model.load_weights(args.weightsfile)
        print("Network successfully loaded")
    
        model.net_info["height"] = args.reso
        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 == 0 
        assert inp_dim > 32
        if CUDA:
            model.cuda()
    
    
    #Set the model in evaluation mode
        model.eval()

    #Detection phase
        try:
            imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
        except NotADirectoryError:
            imlist = []
            imlist.append(osp.join(osp.realpath('.'), images))
        except FileNotFoundError:
            print ("No file or directory with the name {}".format(images))
            exit()
        imlist = img_path  
        if not os.path.exists(args.det):
            os.makedirs(args.det)
        batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
        im_batches = [x[0] for x in batches]
        orig_ims = [x[1] for x in batches]
        im_dim_list = [x[2] for x in batches]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)    
        if CUDA:
            im_dim_list = im_dim_list.cuda()
    
        leftover = 0
    
        if (len(im_dim_list) % batch_size):
            leftover = 1
        
        
        if batch_size != 1:
            num_batches = len(imlist) // batch_size + leftover            
            im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                            len(im_batches))]))  for i in range(num_batches)]        
        i = 0
        write =False
        objs = {}
        area_list_len=[0]
        jj=0
        for batch in im_batches:
            #load the image 
            
            if CUDA:
                batch = batch.cuda()
            with torch.no_grad():
                prediction = model(Variable(batch), CUDA)
            prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            if type(prediction) == int:
                i += 1
                aa=[jj]
                area_list_len.extend(aa)
                continue
            prediction[:,0] += i*batch_size         
            if not write:
                output = prediction
                write = 1
            else:
                output = torch.cat((output,prediction))
            
            for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
                im_id = i*batch_size + im_num
                objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            jj=jj+len(prediction)
            aa=[jj]
            area_list_len.extend(aa)
            i += 1       
        #*****
        im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())     
        scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)    
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2   
        output[:,1:5] /= scaling_factor
        for ii in range(0,len(area_list_len)-1):
            area_list=[]
            area=0
            aaa=0
            disx=[]
            disy=[]
            for iii in range(area_list_len[ii],area_list_len[ii+1]):
                if output[iii,-1]==0 and output[iii,-2]>0.5 :
                    if output[iii,-3]>0.985:
                        aaa=aaa+1
                    if output[iii,-3]>0.9:
                        area=(int)(abs(output[iii,1]-output[iii,3])*abs(output[iii,2]-output[iii,4]))
                        xc=[int((output[iii,1]+output[iii,3])/2)]
                        disx.extend(xc)
                        yc=[int((output[iii,2]+output[iii,4])/2)]
                        disy.extend(yc)
                        a=[area]
                        area_list.extend(a)
            a=[aaa]
            important.extend(a)
            if np.std(area_list)>0:
                std=[np.std(area_list)]
            else:
                std=[0]
            area_std.extend(std)
            if len(area_list)>2:            
                area_list.remove(np.min(area_list))
                area_list.remove(np.max(area_list))
            if len(area_list)!=0:
                mean=np.mean(area_list)
                med=np.median(area_list)
                a=[(int)((mean+med)/2)]
                area_.extend(a)
            else:
                a=[(int)(0)]
                area_.extend(a)
            a=[len(area_list)]
            quantity.extend(a)  
            ddd=[]
            for iii in range(0,len(disx)-1):
                for iiii in range(iii+1,len(disx)):
                    dd=[int(((disx[iii]-disx[iiii])**2+(disy[iii]-disy[iiii])**2)**0.5)]
                    ddd.extend(dd)
            tt=(np.mean(ddd)+np.median(ddd))/2
            if tt>0:
                dd=[tt]
            else:
                dd=[0]
            distance.extend(dd)
        if CUDA:
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        print("---完成面積:"+str(number)+"---")
       #*************
        wi=[]
        ne=[]
        x100=[]
        answer=[]
        n=0
        zero=np.zeros(1280)
        for path in img_path:    
            img  = cv2.imread(path)
            blur_degree=get_blur_degree(path)#
            pp=path.split('\\')
            pp=pp[-1].split('_')
            pp=pp[-1].split('.')
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, w = img.shape[:2]
            Binary1 = cv2.inRange(hsv, (20,0,63),(101,255,255))#去除觀眾
            kernel = np.ones((5,5), np.uint8)#18/18
            result = cv2.morphologyEx(Binary1,cv2.MORPH_OPEN, kernel)
            for i in range(0,720):
                if  (result[i,0:1280].sum()/w<30 ) :#or test[i,0:1280].sum()/1280<30
                     result[i,0:1280]=zero
            grass= result.sum()/255
            total=[grass,quantity[(int)(pp[0])],important[(int)(pp[0])],(int)(distance[(int)(pp[0])]),blur_degree]
            x100.append(total)
            answer.append(number)
        #npz_file = np.load('I:\\code\\S\\dection_tree_data tt\\'+str(number)+'.npz')
        
        #**************
        np.savez_compressed('I:\\code\\S\\tt\\124\\'+str(number)+'.npz', x=x100)#, y=answer)
        print("---完成:"+str(number)+"---")
        number=number+1
#%%
#5feature
import cv2
import numpy as np
import glob as gb
#from __future__ import division
import torch 
import torch.nn as nn
from torch.autograd import Variable
#from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import pickle as pkl
import itertools
import datetime
''''''
def get_blur_degree(image_file, sv_num=10):
    img = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
    u, s, v = np.linalg.svd(img)
    top_sv = np.sum(s[0:sv_num])
    total_sv = np.sum(s)
    return top_sv/total_sv
''''''

class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers= num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5,5) for x in range(num_layers)])
        self.output = nn.Linear(5,2)
    
    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)
        
def get_test_input(input_dim, CUDA):
    img = cv2.imread("1_0.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    num_classes
    return img_

def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    
    return parser.parse_args()
if __name__ ==  '__main__':
    start_time=datetime.datetime.now()
    img_path=[]#//*.png
    number=1
    while number<2:
        img_path = gb.glob("I://code//all_image//1_1//"+str(number)+"//*.png")#S//t2
        img_path.sort(key=len)
        while len(img_path)==0 and (number<6):
            print(str(number)+"空")
            number=number+1
            img_path = gb.glob("I://code//all_image//1_1//"+str(number)+"//*.png")#S//t2
            img_path.sort(key=len)            
        args = arg_parse()
        important=[]
        area_=[]
        area_std=[]
        distance=[]
        scales = args.scales
        quantity=[]
        images = args.images
        batch_size = int(args.bs)
        confidence = float(args.confidence)
        nms_thesh = float(args.nms_thresh)

        CUDA = torch.cuda.is_available()

        num_classes = 40
        classes = load_classes('data/coco.names') 

        #Set up the neural network
        print("Loading network.....")
        model = Darknet(args.cfgfile)
        model.load_weights(args.weightsfile)
        print("Network successfully loaded")
    
        model.net_info["height"] = args.reso
        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 == 0 
        assert inp_dim > 32
        if CUDA:
            model.cuda()
    
    
    #Set the model in evaluation mode
        model.eval()

    #Detection phase
        try:
            imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
        except NotADirectoryError:
            imlist = []
            imlist.append(osp.join(osp.realpath('.'), images))
        except FileNotFoundError:
            print ("No file or directory with the name {}".format(images))
            exit()
        imlist = img_path  
        if not os.path.exists(args.det):
            os.makedirs(args.det)
        batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
        im_batches = [x[0] for x in batches]
        orig_ims = [x[1] for x in batches]
        im_dim_list = [x[2] for x in batches]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)    
        if CUDA:
            im_dim_list = im_dim_list.cuda()
    
        leftover = 0
    
        if (len(im_dim_list) % batch_size):
            leftover = 1
        
        
        if batch_size != 1:
            num_batches = len(imlist) // batch_size + leftover            
            im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                            len(im_batches))]))  for i in range(num_batches)]        
        i = 0
        write =False
        objs = {}
        area_list_len=[0]
        jj=0
        for batch in im_batches:
            #load the image 
            
            if CUDA:
                batch = batch.cuda()
            with torch.no_grad():
                prediction = model(Variable(batch), CUDA)
            prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            if type(prediction) == int:
                i += 1
                aa=[jj]
                area_list_len.extend(aa)
                continue
            prediction[:,0] += i*batch_size         
            if not write:
                output = prediction
                write = 1
            else:
                output = torch.cat((output,prediction))
            
            for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
                im_id = i*batch_size + im_num
                objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            jj=jj+len(prediction)
            aa=[jj]
            area_list_len.extend(aa)
            i += 1       
        #*****
        im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())     
        scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)    
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2   
        output[:,1:5] /= scaling_factor
        for ii in range(0,len(area_list_len)-1):
            area_list=[]
            area=0
            aaa=0
            disx=[]
            disy=[]
            for iii in range(area_list_len[ii],area_list_len[ii+1]):
                if output[iii,-1]==0 and output[iii,-2]>0.5 :
                    if output[iii,-3]>0.985:
                        aaa=aaa+1
                    if output[iii,-3]>0.9:
                        area=(int)(abs(output[iii,1]-output[iii,3])*abs(output[iii,2]-output[iii,4]))
                        xc=[int((output[iii,1]+output[iii,3])/2)]
                        disx.extend(xc)
                        yc=[int((output[iii,2]+output[iii,4])/2)]
                        disy.extend(yc)
                        a=[area]
                        area_list.extend(a)
            a=[aaa]
            important.extend(a)
            if np.std(area_list)>0:
                std=[np.std(area_list)]
            else:
                std=[0]
            area_std.extend(std)
            if len(area_list)>2:            
                area_list.remove(np.min(area_list))
                area_list.remove(np.max(area_list))
            if len(area_list)!=0:
                mean=np.mean(area_list)
                med=np.median(area_list)
                a=[(int)((mean+med)/2)]
                area_.extend(a)
            else:
                a=[(int)(0)]
                area_.extend(a)
            a=[len(area_list)]
            quantity.extend(a)  
            ddd=[]
            for iii in range(0,len(disx)-1):
                for iiii in range(iii+1,len(disx)):
                    dd=[int(((disx[iii]-disx[iiii])**2+(disy[iii]-disy[iiii])**2)**0.5)]
                    ddd.extend(dd)
            tt=(np.mean(ddd)+np.median(ddd))/2
            if tt>0:
                dd=[tt]
            else:
                dd=[0]
            distance.extend(dd)
        if CUDA:
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        print("---完成面積:"+str(number)+"---")
       #*************
        wi=[]
        ne=[]
        x100=[]
        n=0
        zero=np.zeros(1280)
        for path in img_path:    
            img  = cv2.imread(path)
            blur_degree=get_blur_degree(path)#
            pp=path.split('\\')
            pp=pp[-1].split('_')
            pp=pp[-1].split('.')
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, w = img.shape[:2]
            Binary1 = cv2.inRange(hsv, (20,0,63),(101,255,255))#去除觀眾
            kernel = np.ones((5,5), np.uint8)#18/18
            result = cv2.morphologyEx(Binary1,cv2.MORPH_OPEN, kernel)
            for i in range(0,720):
                if  (result[i,0:1280].sum()/w<30 ) :#or test[i,0:1280].sum()/1280<30
                     result[i,0:1280]=zero
            grass= result.sum()/255
            total=[grass,quantity[(int)(pp[0])],important[(int)(pp[0])],(int)(distance[(int)(pp[0])]),blur_degree]
            x100.append(total)
        #npz_file = np.load('I:\\code\\S\\dection_tree_data tt\\'+str(number)+'.npz')
        
        #**************
        np.savez_compressed("I://code//all_image//1_1//"+str(number)+'.npz', x=x100)#把特徵壓縮
        print("---完成:"+str(number)+"---")
        number=number+1
    end_time=datetime.datetime.now()
    print(end_time-start_time)