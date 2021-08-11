import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Generator(nn.Module):
	def __init__(self, c_size = 1, v_size = 32, w_size = 200):
		super().__init__()
		self.main = nn.Sequential(
		nn.ConvTranspose3d(v_size*8,v_size*4,4,stride=2,padding=1,bias=False),
		nn.BatchNorm3d(v_size*4),
		nn.ReLU(inplace=True),
		nn.ConvTranspose3d(v_size*4,v_size*2,4,stride=2,padding=1,bias=False),
		nn.BatchNorm3d(v_size*2),
		nn.ReLU(inplace=True),
		nn.ConvTranspose3d(v_size*2,v_size*1,4,stride=2,padding=1,bias=False),
		nn.BatchNorm3d(v_size*1),
		nn.ReLU(inplace=True),
		nn.ConvTranspose3d(v_size*1,1,4,stride=2,padding=1,bias=False),
		nn.Sigmoid()
		)
		self.fc = nn.Sequential(
		nn.Linear(w_size, 2048),
		nn.ReLU(inplace=True),
		)


	def forward(self, x):
		b_size = x.shape[0]
		r = torch.randn(b_size,50)
		z = x.reshape(b_size,200)
		z[:,150:] = r

		z = self.fc(z)
		z2 = torch.zeros(b_size, 256, 2, 2, 2).cuda()
		for i in range(0, b_size):
			z2[i] = z[i].reshape(256, 2, 2, 2)
		output = self.main(z2).squeeze()

		return output


class Discriminator(nn.Module):
	def __init__(self, c_size = 1, v_size = 32, w_size = 200):
		super().__init__()
		self.layer_x2 = nn.Sequential(
		nn.Conv3d(1,v_size*1,4,stride=2,padding=1,bias=False),
		nn.BatchNorm3d(v_size*1),
		nn.LeakyReLU(0.2, inplace=True),
		nn.Conv3d(v_size*1,v_size*2,4,stride=2,padding=1,bias=False),
		nn.BatchNorm3d(v_size*2),
		nn.LeakyReLU(0.2, inplace=True),
		)
		self.layer_x3 = nn.Sequential(
		nn.Conv3d(v_size*2,v_size*4,4,stride=2,padding=1,bias=False),
		nn.BatchNorm3d(v_size*4),
		nn.LeakyReLU(0.2, inplace=True),
		)
		self.layer_x4 = nn.Sequential(
		nn.Conv3d(v_size*4,v_size*8,4,stride=2,padding=1,bias=False),
		nn.BatchNorm3d(v_size*8),
		nn.LeakyReLU(0.2, inplace=True),
		)
		self.layer_x = nn.Sequential(
		nn.Linear(2048, 150),
		nn.LeakyReLU(0.2, inplace=True)
		)
		self.main = nn.Sequential(
		nn.Linear(300, 150),
		nn.LeakyReLU(0.2, inplace=True),
		nn.Linear(150, 10),
		nn.LeakyReLU(0.2, inplace=True),
		nn.Linear(10, 1),
		nn.Sigmoid()
		)


	def forward(self, x, y):
		x = self.layer_x2(x)
		x = self.layer_x3(x)
		x = self.layer_x4(x)

		b_size = y.shape[0]
		y_size = y.shape[1]
		x_size = x.shape[2]

		x2 = torch.zeros(b_size, 2048).cuda()
		for i in range(0, b_size):
			x2[i] = x[i].reshape(2048)
		x2 = self.layer_x(x2)

		z = torch.cat([x2, y], dim=1)
		output = self.main(z).squeeze()

		return output

