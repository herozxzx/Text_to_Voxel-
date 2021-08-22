import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import pptk
import kaolin as kal

import csv
import datetime
import random
import os
import sys
import gc
from  memory_profiler import profile

import model
import shapenetdataset as ds


pth2 = "../datas/dataset/"


def prvox(vox, count, pl):
	vox = vox.squeeze()
	vox_size = vox.shape[0]
	buf = np.empty((vox_size,vox_size,vox_size))
	for i in range(0,vox_size):
		print("\rCalculating "+str(i+1)+"/"+str(vox_size), end='')
		for j in range(0,vox_size):
			for k in range(0,vox_size):
				buf[i,k,j] = bool(vox[i,j,k])
	figx = plt.figure()
	ax = figx.gca(projection='3d')
	ax.voxels(buf,edgecolor='k')
	
	#dt_now = datetime.datetime.now()
	#a = str(dt_now.hour) + ':' + str(dt_now.minute) + ':' + str(dt_now.second) 
	place = ""
	if(pl==0):
		place = "voxel/"
	elif(pl==1):
		place = "train_voxels/"
	else:
		place = "test_voxels/"
	
	b = True
	try:
		plt.savefig('../results/'+place+'voxel'+str(count)+'.jpg', dpi=140)
		print('\rSaved voxel'+str(count)+'.jpg')
	except:
		print("error plotting")
		b = False
	plt.close(figx)
	vox, vox_size, buf, ax = None,None,None,None
		
	return b

def minmaxsc(nl):
	min1 = nl.min()
	max1 = nl.max()
	nn = (nl - min1) / (max1 - min1)

	return min1,max1,nn

#@profile
def main():
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	device = torch.device('cpu')
	if (torch.cuda.is_available()):
		device = torch.device('cuda')
		print("CUDA is available")
		print()
	
	voxsize = 32
	wvecsize = 200
	ws = 150
	batch_size = 50
	epoch = 1000
	lr_g = 0.0025
	lr_d = 1e-5
	data_pc = 1000
	pr = 15
	
	ds_l = ds.load_data(voxsize, True, data_pc)
	print('Datas: ' + str(ds_l))
	n_train = int(ds_l * 0.9)
	n_test = ds_l - n_train
	n_train = n_train - (n_train%batch_size)
	n_test = n_test - (n_test%batch_size)
	print("Datasets:", n_train, n_test)
	print()
	
	l = []
	for i in range(1, n_train + n_test + 1):
		l.append(i)
	#shuffle
	random.seed(10)
	train = random.sample(l, n_train)
	test = [x for x in set(l + train) if (l + train).count(x) == 1]
	random.seed()
	
	#not shuffle
	#train = l[0:n_train]
	#test = l[n_train:]
	
	train_b = []
	for i in range(0, int(n_train / batch_size)):
		train_b.append(train[i*batch_size : (i+1)*batch_size])
	test_b = []
	for i in range(0, int(n_test / batch_size)):
		test_b.append(test[i*batch_size : (i+1)*batch_size])	

	history = {
		'Dtrain_loss': [],
		'Gtrain_loss': [],
		'DRtrain_acc': [],
		'DF1train_acc': [],
		'DF2train_acc': [],
		'Gtrain_acc': [],
		'test_loss': [],
		'test_acc': [],
		'lr': []
	}

	netG: nn.Module = model.Generator()
	netD: nn.Module = model.Discriminator()
	try:
		netG.load_state_dict(torch.load('../datas/saved_model/Gen.pth'))
		netD.load_state_dict(torch.load('../datas/saved_model/Dis.pth'))
		print("Load model")
	except:
		print("New model")
	netG = netG.to(device)
	netD = netD.to(device)
	
	optimizerG = torch.optim.Adam(params=netG.parameters(), lr=lr_g,betas=[0.5,0.999])
	#lr -> lr * lr_scheduler_func
	optimizerD = torch.optim.Adam(params=netD.parameters(), lr=1.0,betas=[0.5,0.999])
	lr_scheduler_func = LearningRateScheduler(lr_d, epoch, power=pr)
	lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=lr_scheduler_func)
	

	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	netD.train(True)
	netG.train(True)
	data, target ,fdata = None, None, None
	for e in range(epoch):
		Dloss = 0.
		Gloss = 0.
		count = 0
		gl = ""
		dl = ""
		dlr = 0.
		criterion = nn.BCELoss().to(device)
		criterion2 = nn.L1Loss().to(device)
		lastg = 0.
		for b in train_b:
			data, target ,fdata = ds.get_data(b, ds_l, device)	
			b_s = target.shape[0]
			zero_l = torch.zeros(b_s,1).to(device)
			one_l = torch.ones(b_s,1).to(device)
			
			optimizerD.zero_grad()
			optimizerG.zero_grad()
			
			
			#Discriminator_train
			tf = ""
			for param in netD.parameters():
				if(lastg<0.4):
					param.requires_grad_(False)
					tf = "#"
				else:
					param.requires_grad_(True)
					tf = "*"
			print("\r"+tf,end="")
			
			target = target.unsqueeze(1).float()
			output_real = netD(target, data[:,:ws])
			loss_real = criterion(output_real.unsqueeze(1), one_l)
			
			ftarget = netG(data).reshape(b_s,1,voxsize,voxsize,voxsize)
			output_fake1 = netD(ftarget, data[:,:ws])
			loss_fake1 = criterion(output_fake1.unsqueeze(1), zero_l)
			output_fake2 = netD(target, fdata[:,:ws])
			loss_fake2 = criterion(output_fake2.unsqueeze(1), zero_l)
			loss_netD = loss_real + (loss_fake1 + loss_fake2)/2
			
			loss_netD.backward()
			optimizerD.step()
			Dloss += loss_netD.mean().item()
			
			acc = 0
			for n in range(0, b_s):
				acc += output_real[n].mean().item()
			history['DRtrain_acc'].append(acc/b_s)
			acc = 0
			for n in range(0, b_s):
				acc += output_fake1[n].mean().item()
			history['DF1train_acc'].append(acc/b_s)
			acc = 0
			for n in range(0, b_s):
				acc += output_fake2[n].mean().item()
			history['DF2train_acc'].append(acc/b_s)
			
			
			#Generator_train
			for param in netD.parameters():
				param.requires_grad_(False)
			
			ftarget = netG(data).reshape(b_s,1,voxsize,voxsize,voxsize)
			output_fake = netD(ftarget, data[:,:ws])
			loss_netG1 = criterion(output_fake.unsqueeze(1), one_l)
			o11 = netD.layer_x4(netD.layer_x3(netD.layer_x2(ftarget)))
			o22 = netD.layer_x4(netD.layer_x3(netD.layer_x2(target)))
			loss_netG2 = criterion2(o11, o22)
			loss_netG = loss_netG1 + 0.1*loss_netG2
			
			loss_netG.backward()
			optimizerG.step()
			Gloss += loss_netG.mean().item()
			
			acc = 0
			for n in range(0, b_s):
				acc += output_fake[n].mean().item()
			history['Gtrain_acc'].append(acc/b_s)
			lastg = acc/b_s
			
			#learning rate
			for p in optimizerD.param_groups:
				c_lrd = p['lr']
			for p in optimizerG.param_groups:
				c_lrg = p['lr']
			dlr = c_lrd
		
			
			count += 1
			print('Training log: {} epoch ({} data) lr/DLoss:{} lr/GLoss:{}'.format(e, count*batch_size, (str(c_lrd*100000)[0:5]+"e-5", str(loss_netD.item())[0:6]), (str(c_lrg)[0:6], str(loss_netG.item())[0:6])),end='  ')
			
			data, target ,fdata = None, None, None
			gc.collect()
		
		print("\rEpoch: "+str(e)+" trained score is " + "Dloss:" + str(Dloss/count) + " Gloss:" + str(Gloss/count) + "                         ")
		history['Dtrain_loss'].append(Dloss/count)
		history['Gtrain_loss'].append(Gloss/count)
		history['lr'].append(c_lrd)
		
		#lr step
		lr_scheduler.step()
		
		##############################saving##############################
		fig = plt.figure()
		plt.plot(range(1, len(history['Dtrain_loss'])+1), history['Dtrain_loss'], label='Dtrain_loss')
		plt.plot(range(1, len(history['Gtrain_loss'])+1), history['Gtrain_loss'], label='Gtrain_loss')
		plt.xlabel('epoch')
		plt.legend()
		plt.savefig('../results/trainloss.png')
		plt.close(fig)
		
		fig2 = plt.figure()
		plt.plot(range(1, len(history['DRtrain_acc'])+1), history['DRtrain_acc'], label='DRtrain_acc')
		plt.plot(range(1, len(history['DF1train_acc'])+1), history['DF1train_acc'], label='DF1train_acc')
		plt.plot(range(1, len(history['DF2train_acc'])+1), history['DF2train_acc'], label='DF2train_acc')
		plt.plot(range(1, len(history['Gtrain_acc'])+1), history['Gtrain_acc'], label='Gtrain_acc')
		plt.xlabel('epoch*tr_data')
		plt.legend()
		plt.savefig('../results/trainacc.png')
		plt.close(fig2)

		fig5 = plt.figure()
		plt.plot(range(1, len(history['lr'])+1), history['lr'], label='lr')
		plt.xlabel('epoch')
		plt.legend()
		plt.savefig('../results/Dis_lr.png')
		plt.close(fig5)
		
		
		#per 10 epoch
		if(e%10 == 0):
			print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
			x, y, _ = ds.get_data([1], ds_l, device)
			out0 = (netG(x.reshape(1,1,wvecsize)))
			prvox((torch.round(out0)),str(e)+"_g",0)
			if(e==0):
				prvox(y.squeeze(),"_d",0)
			
			rd = random.choice(train)
			x, y, _ = ds.get_data([rd], ds_l, device)
			out1 = (netG(x.reshape(1,1,wvecsize)))
			prvox((torch.round(out1)),str(e)+"_1_g",1)
			prvox(y.squeeze(),str(e)+"_1_d",1)
			
			rd = random.choice(test)
			x, y, _ = ds.get_data([rd], ds_l, device)
			out2 = (netG(x.reshape(1,1,wvecsize)))
			prvox((torch.round(out2)),str(e)+"_2_g",2)
			prvox(y.squeeze(),str(e)+"_2_d",2)
			
			
			torch.save(netG.state_dict(), "../datas/saved_model/Gen.pth")
			torch.save(netD.state_dict(), "../datas/saved_model/Dis.pth")
			rd,x,y,out0,out1,out2 = None,None,None,None,None,None
			print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	try:
		x, y, _ = ds.get_data([1], ds_l, device)
		out0 = (netG(x.reshape(1,1,wvecsize)))
		prvox((torch.round(out0)), "lastg",0)
		
		rd = random.choice(train)
		x, y, _ = ds.get_data([rd], ds_l, device)
		out1 = (netG(x.reshape(1,1,wvecsize)))
		prvox((torch.round(out1)), "last1_g",1)
		prvox(y.squeeze(), "last1_d",1)
		
		
		rd = random.choice(test)
		x, y, _ = ds.get_data([rd], ds_l, device)
		out2 = (netG(x.reshape(1,1,wvecsize)))
		prvox((torch.round(out2)), "last2_g",2)
		prvox(y.squeeze(), "last2_d",2)
		
		
		torch.save(netG.state_dict(), "../datas/saved_model/Gen.pth")
		torch.save(netD.state_dict(), "../datas/saved_model/Dis.pth")
	except:
		print("error")


class LearningRateScheduler:
    def __init__(self, base_lr: float, max_epoch: int, power=5):
        self._max_epoch = max_epoch
        self._power = power
        self._base_lr = base_lr

    def __call__(self, epoch: int):
        out = self._base_lr * (1 - max(epoch - 1, 1) / self._max_epoch) ** self._power + 5e-7

        return out


main()
