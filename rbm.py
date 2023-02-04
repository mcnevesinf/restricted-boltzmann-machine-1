
import torch

#Create the RBM
class RBM():
	def __init__(self, nv=1000, nh=50):
		self.W = torch.randn(nh, nv)
		self.a = torch.randn(1, nh)
		self.b = torch.randn(1, nv)
		
	def sample_h(self, x):
		wx = torch.mm(x, self.W.t())
		activation = wx + self.a.expand_as(wx)
		p_h_given_v = torch.sigmoid(activation)
		return p_h_given_v, torch.bernoulli(p_h_given_v)
		
	def sample_v(self, y):
		wy = torch.mm(y, self.W)
		activation = wy + self.b.expand_as(wy)
		p_v_given_h = torch.sigmoid(activation)
		return p_v_given_h, torch.bernoulli(p_v_given_h)	

	def train(self, v0, vk, ph0, phk):
		self.W = torch.mm(v0.t(), ph0).t() - torch.mm(vk.t(), phk).t()
		self.b += torch.sum((v0 - vk), 0)
		self.a += torch.sum((ph0 - phk), 0)
		
	def save_rbm(self):
		model = {'W': self.W, 'a': self.a, 'b':self.b}
		torch.save(model, "trained_rbm.pt")
		
	def load_rbm(self):
		loaded = torch.load("trained_rbm.pt")
		self.W = loaded['W']
		self.a = loaded['a']
		self.b = loaded['b']
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
