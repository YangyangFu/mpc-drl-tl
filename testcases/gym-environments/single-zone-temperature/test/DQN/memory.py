# class of memory for experience replay
import numpy as np
import random

class Memory(object):
	def __init__(self,m_size,re_policy='FIFO'):
	# m_size: capacity of memory
	# re_policy: experience replacement policy, default as FIFO
		self.capacity = m_size
		self.policy = re_policy
		self.size = 0
		self.experience = [] # n by 4

	def add(self,observation):
	# observation: new observation (s,a,r,s')
		if(self.size==self.capacity):
			self.experience.pop(0)
			self.experience.append(observation)
		else:
			self.experience.append(observation)
			self.size = self.size + 1

	def backup(self,path,Mname):
		np.save(path+'/'+Mname,[self.experience,self.size])

	def recover(self,path,Mname):
		recall = np.load(path+'/'+Mname)
		self.experience = recall[0]
		self.size = recall[1]

	def sampling(self,num_batch):
	# num: number of mini-batch
		# there is no enough experience in memory, return all available experience
		if self.size<=num_batch:
			return self.experience
		else:
			# index = np.random.choice(self.size,num_batch,replace=False)
			index = random.sample(range(self.size),num_batch)
			mini_batch = []
			for i in range(0,num_batch):
				mini_batch.append(self.experience[index[i]])
			return mini_batch

	def clear(self):
	# clear memory
		self.experience = []
		self.size = 0

	def show(self):
	# display memory information
		print('Experience Replay Memory')
		print('----------------------------------------')
		print('Memory Capacity:    '+str(self.capacity))
		print('Replacement Policy: '+str(self.policy))

