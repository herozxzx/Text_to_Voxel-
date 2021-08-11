import numpy as np
import torch
from torch import float32
import torchvision

import kaolin as kal
import pptk

import nltk
from nltk.corpus import stopwords
import torchtext
from torchtext import data,datasets
from tensorboardX import SummaryWriter
import unicodedata

import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import os
import random


pth = "/mnt/3Dmodels/ShapeNetCore.v1/"
pth2 = "../datas/dataset/"

def mkvec(nn, nns, vn, oh, ohs, vo):
	if (torch.cuda.is_available()):
		device = torch.device('cuda')
	
	wv = torch.zeros(100)
	c = 0
	for i in range(0, len(nn)):
		for j in range(len(nns)):
			if(nns[j]==nn[i] and vn[j].max().item()>0.):
				wv += vn[j]
				c+=1
	if(c==0):
		print("noun is 0")
		wv = torch.load(pth2 + "cache/nbuf.pt")
	else:
		wv /= c
	nbuf = torch.save(wv, pth2 + "cache/nbuf.pt")
	
	c = 0
	wv2 = torch.zeros(50)
	for i in range(0, len(oh)):
		for j in range(len(ohs)):
			if(ohs[j]==oh[i] and vo[j].max().item()>0.):
				wv2 += vo[j]
				c+=1
	if(c!=0):
		wv2 /= c
	
	out = torch.cat([wv, wv2, torch.randn(50)], dim=0)
	
	return out


def prvox(vox, count):
	vox.squeeze()
	vox_size = vox.shape[0]
	buf = np.empty((vox_size,vox_size,vox_size))
	for i in range(0,vox_size):
		print("\rCalculating "+str(i+1)+"/"+str(vox_size), end='')
		for j in range(0,vox_size):
			for k in range(0,vox_size):
				buf[i,j,k] = bool(vox[i,j,k])
	figx = plt.figure()
	ax = figx.gca(projection='3d')
	ax.voxels(buf,edgecolor='k')
	
	place = "test/"
	
	b = True
	try:
		plt.savefig('../results/'+place+'voxel'+str(count)+'.jpg', dpi=140)
		print('\nSaved voxel'+str(count)+'.jpg')
	except:
		print("error plotting")
		b = False
	plt.close(figx)
		
	return b


def load_data(vox_size, con, data_num):
	if(con):
		#geting csvlist from root dataset
		files = os.listdir(pth)
		file = [f for f in files if os.path.isfile(os.path.join(pth, f))]
		file = [f for f in files if os.path.splitext(f)[1]==".csv"]
		
		p_w = []
		ratio = []
		d_class = 0
		t_data = 0
		for buf_csv in file:
			
			buf_l = []
			with open(pth + buf_csv) as f:
				rr = csv.reader(f)
				buf_l = [row for row in rr]
			print(buf_l[1][2])
			if(buf_csv=="02858304.csv"):
				ps == "p"
			else:
				ps = input()
			if(ps=="p"):
				print("passing")
			
			else:
				for i in range(1, len(buf_l)):
					ptt = pth + buf_csv.replace('.csv','') + "/" + buf_l[i][0].split('.')[1]
					cls = buf_csv.replace('.csv','')
					if(len(buf_l) < data_num):
						p_w.append([ptt,buf_l[i][2],buf_l[i][5],cls])
						t_data+=1
					elif(i%int(len(buf_l)/data_num) == 0):
						p_w.append([ptt,buf_l[i][2],buf_l[i][5],cls])
						t_data+=1
				ratio.append(len(buf_l))
				d_class+=1
				print(d_class, t_data)
		
		fl = len(p_w)
		
		with open(pth2+"p_w.csv", mode='w') as f:
				for p in p_w:
					f.write(str(p[3]) + str(','))
					f.write(str(p[0]) + str(','))
					f.write(str(p[1]) + str(','))
					f.write(str(p[2]) + str('\n'))


		noun = []
		oth = []
		for [ph, wds, ns, cls] in p_w:
			wds_l = wds.split(',')
			for wds in wds_l:
				wd_l = nltk.word_tokenize(wds)
				p_wd = nltk.pos_tag(wd_l)
				for (wd, ps) in p_wd:
					if ps[:2]=='JJ' or ps[:2]=='VB':
						oth.append(wd)
					elif ps[:2]=='NN':
						noun.append(wd)
			ns = ns.replace('-', ' ')
			ns = ns.replace('/', ' ')
			ns_l = nltk.word_tokenize(ns)
			p_ns = nltk.pos_tag(ns_l)
			for (wd, ps) in p_ns:
				if ps[:2]=='JJ' or ps[:2]=='VB':
					oth.append(wd)
		
		noun = [word for word in list(set(noun)) if unicodedata.east_asian_width(word[0])=='Na']
		oth = [word for word in list(set(oth)) if unicodedata.east_asian_width(word[0])=='Na']
		
		with open(pth2+"noun.csv", mode='w') as f:
			for p in noun:
				f.write(p+str('\n'))
		with open(pth2+"oth.csv", mode='w') as f:
			for p in oth:
				f.write(p+str('\n'))
		
		device = torch.device('cpu')
		if (torch.cuda.is_available()):
			device = torch.device('cuda')
		
		#
		glove100 = torchtext.vocab.GloVe(name='6B',dim=100)
		TEXT1 = data.Field(sequential=True, use_vocab=True, lower=True)
		pos1 = data.TabularDataset(
			pth2+"noun.csv",format='csv',
			fields=[('text', TEXT1)])
		TEXT1.build_vocab(pos1, vectors=glove100)
		vocab_ = TEXT1.vocab
		v_noun = vocab_.vectors
		w_noun = vocab_.itos
		
		#
		glove50 = torchtext.vocab.GloVe(name='6B',dim=50)
		TEXT2 = data.Field(sequential=True, use_vocab=True, lower=True)
		pos2 = data.TabularDataset(
			pth2+"oth.csv",format='csv',
			fields=[('text', TEXT2)])
		TEXT2.build_vocab(pos2, vectors=glove50)
		vocab_ = TEXT2.vocab
		v_oth = vocab_.vectors
		w_oth = vocab_.itos

		
		torch.save(v_noun[2:].clone().detach(), "../datas/embvec/v_noun.pt")
		torch.save(v_oth[2:].clone().detach(), "../datas/embvec/v_oth.pt")
		
		#cmd --> tensorboard --logdir=runs
		writer = SummaryWriter()
		writer.add_embedding(torch.FloatTensor(v_noun[2:]), metadata=w_noun[2:], global_step=1)
		writer.add_embedding(torch.FloatTensor(v_oth[2:]),  metadata=w_oth[2:], global_step=2)
		
		x = []
		x_l = []
		y = []
		count=0
		for [ph, wds, ns, cls] in p_w:
			wds_l = wds.split(',')
			w_n = []
			w_o =[]
			for wds in wds_l:
				wd_l = nltk.word_tokenize(wds.lower())
				p_wd = nltk.pos_tag(wd_l)
				for (wd, ps) in p_wd:
					if ps[:2]=='JJ' or ps[:2]=='VB':
						w_o.append(wd)
					elif ps[:2]=='NN':
						w_n.append(wd)
			ns = ns.replace('-', ' ')
			ns = ns.replace('/', ' ')
			ns_l = nltk.word_tokenize(ns.lower())
			p_ns = nltk.pos_tag(ns_l)
			for (wd, ps) in p_ns:
				if ps[:2]=='JJ' or ps[:2]=='VB':
					w_o.append(wd)
			print(w_o)
			buf = mkvec(w_n, w_noun, v_noun, w_o, w_oth, v_oth)
			x.append(buf.cpu().numpy())
			print(cls)
			x_l.append(int(cls))
			y.append(ph)
			count+=1
		
		writer.add_embedding(torch.FloatTensor(x), metadata=x_l, global_step=3)
		
		with open(pth2+"y.csv", mode='w') as f:
			for p in y:
				f.write(p+str('\n'))
		torch.save(torch.FloatTensor(x), pth2+"x.pt")
		x, y, x_l, v_noun, w_nown, v_oth, w_oth = None, None, None, None, None, None, None
		
		torch.cuda.empty_cache()
	else:
		p_w = []
		with open(pth2+"p_w.csv") as f:
			for row in csv.reader(f):
				p_w.append(row)
		fl = len(p_w)
	
	
	return fl


def get_data(batch, l, device):
	x,y,x2 = None,None,None
	try:
		x = torch.load(pth2+"cache/x_"+str(batch[0])+"_"+str(len(batch)))
		y = torch.load(pth2+"cache/y_"+str(batch[0])+"_"+str(len(batch)))
		x2 = torch.load(pth2+"cache/x2_"+str(batch[0])+"_"+str(len(batch)))
	except:
		x = torch.load(pth2+"x.pt").to(device)
		y = []
		with open(pth2+"y.csv") as f:
			for row in csv.reader(f):
				y.append(row)
		l1 = []
		l2 = []
		l3 = []
		for n in batch:
			num = n-1
			
			buf1 = x[num]
			try:
				buf = kal.io.obj.import_mesh(str(y[num][0])+'/model.obj')
				torch.save(buf, pth2 + "cache/vbuf.pt")
			except:
				print("No file " + str(y[num][0]) + '/model.obj')
				buf = torch.load(pth2 + "cache/vbuf.pt")
			try:
				buf2 = kal.ops.conversions.trianglemeshes_to_voxelgrids(buf.vertices.unsqueeze(0).to(device), buf.faces.to(device), 32)
				torch.save(buf2, pth2 + "cache/vbuf2.pt")
			except:
				print("Convertion Error")
				buf2 = torch.load(pth2 + "cache/vbuf2.pt")
			r = random.randint(0, l-1)
			buf3 = x[r]
			while (buf1[:100]-buf3[:100]).max().item()==0:
				r = random.randint(0, l-1)
				buf3 = x[r]

			l1.append(buf1)
			l2.append(buf2)
			l3.append(buf3)
		x = torch.stack(l1, dim=0).to(device)
		y = torch.stack(l2, dim=0).to(device).squeeze()
		x2 = torch.stack(l3, dim=0).to(device)
		
		torch.save(x, pth2+"cache/x_"+str(batch[0])+"_"+str(len(batch)))
		torch.save(y, pth2+"cache/y_"+str(batch[0])+"_"+str(len(batch)))
		torch.save(x2, pth2+"cache/x2_"+str(batch[0])+"_"+str(len(batch)))

	return x, y, x2
