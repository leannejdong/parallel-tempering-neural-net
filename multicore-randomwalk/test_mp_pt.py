""" Feed Forward Network with Parallel Tempering for Multi-Core Systems"""

from __future__ import print_function, division
import multiprocessing

import numpy as np
import random
import time
import operator
import math
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.stats import multivariate_normal
from scipy.stats import norm

np.random.seed(1)

#REGRESSION FNN Randomwalk (Taken from R. Chandra, L. Azizi, S. Cripps, 'Bayesian neural learning via Langevin dynamicsfor chaotic time series prediction', ICONIP 2017.)

class Network:

	def __init__(self, Topo, Train, Test, learn_rate):
		self.Top = Topo  # NN topology [input, hidden, output]
		self.TrainData = Train
		self.TestData = Test
		self.lrate = learn_rate

		self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
		self.B1 = np.random.randn(1, self.Top[1]) / np.sqrt(self.Top[1])  # bias first layer
		self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
		self.B2 = np.random.randn(1, self.Top[2]) / np.sqrt(self.Top[1])  # bias second layer

		self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
		self.out = np.zeros((1, self.Top[2]))  # output last layer

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def sampleEr(self, actualout):
		error = np.subtract(self.out, actualout)
		sqerror = np.sum(np.square(error)) / self.Top[2]
		return sqerror

	def ForwardPass(self, X):
		z1 = X.dot(self.W1) - self.B1
		self.hidout = self.sigmoid(z1)  # output of first hidden layer
		z2 = self.hidout.dot(self.W2) - self.B2
		self.out = self.sigmoid(z2)  # output second hidden layer

	def BackwardPass(self, Input, desired):
		out_delta = (desired - self.out) * (self.out * (1 - self.out))
		hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

		#self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
		#self.B2 += (-1 * self.lrate * out_delta)
		#self.W1 += (Input.T.dot(hid_delta) * self.lrate)
		#self.B1 += (-1 * self.lrate * hid_delta)

		layer = 1  # hidden to output
		for x in range(0, self.Top[layer]):
			for y in range(0, self.Top[layer + 1]):
				self.W2[x, y] += self.lrate * out_delta[y] * self.hidout[x]
		for y in range(0, self.Top[layer + 1]):
			self.B2[y] += -1 * self.lrate * out_delta[y]

		layer = 0  # Input to Hidden
		for x in range(0, self.Top[layer]):
			for y in range(0, self.Top[layer + 1]):
				self.W1[x, y] += self.lrate * hid_delta[y] * Input[x]
		for y in range(0, self.Top[layer + 1]):
			self.B1[y] += -1 * self.lrate * hid_delta[y]

	def decode(self, w):
		w_layer1size = self.Top[0] * self.Top[1]
		w_layer2size = self.Top[1] * self.Top[2]

		w_layer1 = w[0:w_layer1size]
		self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

		w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
		self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
		self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]]
		self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]]


	def encode(self):
		w1 = self.W1.ravel()
		w2 = self.W2.ravel()
		w = np.concatenate([w1, w2, self.B1, self.B2])
		return w

	def langevin_gradient(self, data, w, depth):  # BP with SGD (Stocastic BP)

		self.decode(w)  # method to decode w into W1, W2, B1, B2.
		size = data.shape[0]

		Input = np.zeros((1, self.Top[0]))  # temp hold input
		Desired = np.zeros((1, self.Top[2]))
		fx = np.zeros(size)

		for i in range(0, depth):
			for i in range(0, size):
				pat = i
				Input = data[pat, 0:self.Top[0]]
				Desired = data[pat, self.Top[0]:]
				self.ForwardPass(Input)
				self.BackwardPass(Input, Desired)

		w_updated = self.encode()

		return  w_updated

	def evaluate_proposal(self, data, w ):  # BP with SGD (Stocastic BP)

		self.decode(w)  # method to decode w into W1, W2, B1, B2.
		size = data.shape[0]

		Input = np.zeros((1, self.Top[0]))  # temp hold input
		Desired = np.zeros((1, self.Top[2]))
		fx = np.zeros(size)

		for i in range(0, size):  # to see what fx is produced by your current weight update
			Input = data[i, 0:self.Top[0]]
			self.ForwardPass(Input)
			fx[i] = self.out

		return fx

class ptReplica(multiprocessing.Process):

	def __init__(self, w, samples, traindata, testdata, topology, burn_in, temperature, swap_interval, path, parameter_queue, main_process):
		#MULTIPROCESSING VARIABLES
		multiprocessing.Process.__init__(self)
		self.processID = temperature
		self.parameter_queue = parameter_queue
		self.signal_main = main_process
		#PARALLEL TEMPERING VARIABLES
		self.temperature = temperature
		self.swap_interval = swap_interval
		self.path = path
		self.burn_in = burn_in
		#FNN CHAIN VARIABLES (MCMC)
		self.samples = samples
		self.topology = topology
		self.traindata = traindata
		self.testdata = testdata
		self.w = w

	def rmse(self, pred, actual):
		return np.sqrt(((pred-actual)**2).mean())

	def likelihood_func(self, fnn, data, w, tau_sq):
		y = data[:, self.topology[0]]
		fx = fnn.evaluate_proposal(data,w)
		rmse = self.rmse(fx, y)
		loss = np.sum(-0.5*np.log(2*math.pi*tau_sq) - 0.5*np.square(y-fx)/tau_sq)
		return [np.sum(loss)/self.temperature, fx, rmse]

	def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq):
		h = self.topology[1]  # number hidden neurons
		d = self.topology[0]  # number input neurons
		part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
		part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
		log_loss = part1 - part2  - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
		return log_loss

	def run(self):
		#INITIALISING FOR FNN
		testsize = self.testdata.shape[0]
		trainsize = self.traindata.shape[0]
		samples = self.samples
		self.sgd_depth = 1
		x_test = np.linspace(0,1,num=testsize)
		x_train = np.linspace(0,1,num=trainsize)
		netw = self.topology
		y_test = self.testdata[:,netw[0]]
		y_train = self.traindata[:,netw[0]]
		
		w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  # num of weights and bias
		pos_w = np.ones((samples, w_size)) #Posterior for all weights
		pos_tau = np.ones((samples,1)) #Tau is the variance of difference in predicted and actual values
		
		fxtrain_samples = np.ones((samples, trainsize)) #Output of regression FNN for training samples
		fxtest_samples = np.ones((samples, testsize)) #Output of regression FNN for testing samples
		rmse_train  = np.zeros(samples)
		rmse_test = np.zeros(samples)
		learn_rate = 0.5

		naccept = 0
		#Random Initialisation of weights
		w = self.w
		#print(w,self.temperature)
		w_proposal = np.random.randn(w_size)
		#Randomwalk Steps
		step_w = 0.025
		step_eta = 0.2
		#Declare FNN
		fnn = Network(self.topology, self.traindata, self.testdata, learn_rate)
		#Evaluate Proposals
		pred_train = fnn.evaluate_proposal(self.traindata,w)
		pred_test = fnn.evaluate_proposal(self.testdata, w)
		#Check Variance of Proposal
		eta = np.log(np.var(pred_train - y_train))
		print(np.asarray([eta]).shape)
		tau_pro = np.exp(eta)
		sigma_squared = 25
		nu_1 = 0
		nu_2 = 0
		sigma_diagmat = np.zeros((w_size, w_size))  # for Equation 9 in Ref [Chandra_ICONIP2017]
		np.fill_diagonal(sigma_diagmat, step_w)

		delta_likelihood = 0.5 # an arbitrary position
		prior_current = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro)  # takes care of the gradients
		#Evaluate Likelihoods
		[likelihood, pred_train, rmsetrain] = self.likelihood_func(fnn, self.traindata, w, tau_pro)
		[_, pred_test, rmsetest] = self.likelihood_func(fnn, self.testdata, w, tau_pro)
		#Beginning Sampling using MCMC RANDOMWALK
		plt.plot(x_train, y_train)

		accept_list = open(self.path+'/acceptlist_'+str(self.temperature)+'.txt', "a+")


		for i in range(samples - 1):
			#GENERATING SAMPLE
			#w_gd = fnn.langevin_gradient(self.traindata, w.copy(), self.sgd_depth) # Eq 8
			w_proposal = np.random.normal(w, step_w, w_size) # Eq 7
			#w_prop_gd = fnn.langevin_gradient(self.traindata, w_proposal.copy(), self.sgd_depth)
			
			#diff_prop =  np.log(multivariate_normal.pdf(w, w_prop_gd, sigma_diagmat)  - np.log(multivariate_normal.pdf(w_proposal, w_gd, sigma_diagmat)))

			eta_pro = eta + np.random.normal(0, step_eta, 1)
			tau_pro = math.exp(eta_pro)

			[likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(fnn, self.traindata, w_proposal,tau_pro)
			
			[_, pred_test, rmsetest] = self.likelihood_func(fnn, self.testdata, w_proposal,tau_pro)
			prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,tau_pro)  # takes care of the gradients
			diff_prior = prior_prop - prior_current
			diff_likelihood = likelihood_proposal - likelihood
			#ACCEPTANCE OF SAMPLE
	
			try:
				mh_prob = min(1, math.exp(diff_likelihood + diff_prior  ))
			except OverflowError:
				mh_prob = 1

			u = random.uniform(0, 1)
		

			if u < mh_prob:
				naccept  =  naccept + 1
				likelihood = likelihood_proposal
				prior_current = prior_prop
				w = w_proposal
				eta = eta_pro
				#print (i,'accepted')
				accept_list.write('{} {} {} {} {} {} {}\n'.format(self.temperature,naccept, i, rmsetrain, rmsetest, likelihood, diff_likelihood + diff_prior))
				pos_w[i + 1,] = w_proposal
				pos_tau[i + 1,] = tau_pro
				fxtrain_samples[i + 1,] = pred_train
				fxtest_samples[i + 1,] = pred_test
				rmse_train[i + 1,] = rmsetrain
				rmse_test[i + 1,] = rmsetest
				plt.plot(x_train, pred_train)
			else:
				accept_list.write('{} x {} {} {} {} {}\n'.format(self.temperature, i, rmsetrain, rmsetest, likelihood, diff_likelihood + diff_prior))
				pos_w[i + 1,] = pos_w[i,]
				pos_tau[i + 1,] = pos_tau[i,]
				fxtrain_samples[i + 1,] = fxtrain_samples[i,]
				fxtest_samples[i + 1,] = fxtest_samples[i,]
				rmse_train[i + 1,] = rmse_train[i,]
				rmse_test[i + 1,] = rmse_test[i,]
			#print('INITIAL W(PROP) BEFORE SWAP',self.temperature,w_proposal,i,rmsetrain)
			#print('INITIAL W BEFORE SWAP',self.temperature,i,w)
			#SWAPPING PREP
			if (i%self.swap_interval == 0):
				param = np.concatenate([w, np.asarray([eta]).reshape(1), np.asarray([likelihood]),np.asarray([self.temperature])])
				self.parameter_queue.put(param)
				self.signal_main.set()
				# retrieve parameters fom queues if it has been swapped
				if not self.parameter_queue.empty() : 
					try:
						result =  self.parameter_queue.get()
						#print(self.temperature, w, 'param after swap')
						w= result[0:w.size]     
						eta = result[w.size]
						likelihood = result[w.size+1]
					except:
						print ('error')
		param = np.concatenate([w, np.asarray([eta]).reshape(1), np.asarray([likelihood]),np.asarray([self.temperature])])
		#print('SWAPPED PARAM',self.temperature,param)
		self.parameter_queue.put(param)
		

		print ((naccept*100 / (samples * 1.0)), '% was accepted')
		accept_ratio = naccept / (samples * 1.0) * 100
		plt.title("Plot of Accepted Proposals")
		plt.savefig(self.path+'/results/proposals.png')
		plt.savefig(self.path+'/results/proposals.svg', format='svg', dpi=600)
		plt.clf()
		#SAVING PARAMETERS
		file_name = self.path+'/posterior/pos_w_chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name,pos_w ) 
		file_name = self.path+'/posterior/fxtrain_samples_chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, fxtrain_samples, fmt='%1.2f')
		file_name = self.path+'/posterior/fxtest_samples_chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, fxtest_samples, fmt='%1.2f')		
		file_name = self.path+'/posterior/rmse_test_chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, rmse_test, fmt='%1.2f')		
		file_name = self.path+'/posterior/rmse_train_chain_'+ str(self.temperature)+ '.txt'
		np.savetxt(file_name, rmse_train, fmt='%1.2f')		
		file_name = self.path + '/posterior/accept_list_chain_' + str(self.temperature) + '_accept.txt'
		np.savetxt(file_name, [accept_ratio], fmt='%1.2f')

		self.signal_main.set()

class ParallelTempering:

	def __init__(self, traindata, testdata, topology, num_chains, maxtemp, NumSample, swap_interval, path):
		#FNN Chain variables
		self.traindata = traindata
		self.testdata = testdata
		self.topology = topology
		self.num_param = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
		#Parallel Tempering variables
		self.swap_interval = swap_interval
		self.path = path
		self.maxtemp = maxtemp
		self.num_swap = 0
		self.num_chains = num_chains
		self.chains = []
		self.temperatures = []
		self.NumSamples = int(NumSample/self.num_chains)
		self.sub_sample_size = max(1, int( 0.05* self.NumSamples))
		# create queues for transfer of parameters between process chain
		self.parameter_queue = [multiprocessing.Queue() for i in range(num_chains)]
		self.chain_queue = multiprocessing.JoinableQueue()	
		self.wait_chain = [multiprocessing.Event() for i in range (self.num_chains)]	

	def assign_temperatures(self):
		#Linear Spacing
		temp = 2
		for i in range(0,self.num_chains):
			self.temperatures.append(temp)
			temp += (self.maxtemp/self.num_chains)
			print (self.temperatures[i])
		#Geometric Spacing
		#### TBD - Konark

	def initialize_chains(self, burn_in):
		self.burn_in = burn_in
		self.assign_temperatures()
		w = np.random.randn(self.num_param)
		
		for i in range(0, self.num_chains):
			self.chains.append(ptReplica(w,self.NumSamples,self.traindata,self.testdata,self.topology,self.burn_in,self.temperatures[i],self.swap_interval,self.path,self.parameter_queue[i],self.wait_chain[i]))
	
	def swap_procedure(self, parameter_queue_1, parameter_queue_2):
		if parameter_queue_2.empty() is False and parameter_queue_1.empty() is False:
			param1 = parameter_queue_1.get()
			param2 = parameter_queue_2.get()
			w1 = param1[0:self.num_param]
			eta1 = param1[self.num_param]
			lhood1 = param1[self.num_param+1]
			T1 = param1[self.num_param+2]
			w2 = param2[0:self.num_param]
			eta2 = param2[self.num_param]
			lhood2 = param2[self.num_param+1]
			T2 = param2[self.num_param+2]
			#SWAPPING PROBABILITIES
			swap_proposal =  (lhood1/[1 if lhood2 == 0 else lhood2])*(1/T1 * 1/T2)
			u = np.random.uniform(0,1)
			if u < 1:
				self.num_swap += 1
				param_temp =  param1
				param1 = param2
				param2 = param_temp
			return param1, param2
		else:
			return
		

	def run_chains(self):
		x_test = np.linspace(0,1,num=self.testdata.shape[0])
		x_train = np.linspace(0,1,num=self.traindata.shape[0])
		# only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
		swap_proposal = np.ones(self.num_chains-1) 
		# create parameter holders for paramaters that will be swapped
		replica_param = np.zeros((self.num_chains, self.num_param))  
		lhood = np.zeros(self.num_chains)
		eta = np.zeros(self.num_chains)
		# Define the starting and ending of MCMC Chains
		start = 0
		end = self.NumSamples-1
		number_exchange = np.zeros(self.num_chains)
		filen = open(self.path + '/num_exchange.txt', 'a')
		#RUN MCMC CHAINS
		for l in range(0,self.num_chains):
			self.chains[l].start_chain = start
			self.chains[l].end = end
		for j in range(0,self.num_chains):        
			self.chains[j].start()
		#SWAP PROCEDURE
		chain_num = 0
		while True:
			for k in range(0,self.num_chains-1):
				self.wait_chain[j].wait()
				chain_num += 1
				#print(chain_num)

			for k in range(0,self.num_chains-1):
				#print('starting swap')
				self.chain_queue.put(self.swap_procedure(self.parameter_queue[k],self.parameter_queue[k+1])) 
				while True:
					if self.chain_queue.empty():
						self.chain_queue.task_done()
						break
					swap_process = self.chain_queue.get()
					if swap_process is None:
						self.chain_queue.task_done()
						break
					param1, param2 = swap_process
					self.chain_queue.task_done()
					self.parameter_queue[k].put(param1)
					self.parameter_queue[k+1].put(param2)
			print(chain_num)
			if chain_num == self.num_chains -1  :
				print(chain_num)
				break
			
		
		#JOIN THEM TO MAIN PROCESS
		for j in range(0,self.num_chains):
			self.chains[j].join()
		self.chain_queue.join()
		#GETTING DATA
		burnin = int(self.NumSamples*self.burn_in)
		pos_w = np.zeros((self.num_chains,self.NumSamples - burnin, self.num_param))
		fxtrain_samples = np.zeros((self.num_chains,self.NumSamples - burnin, self.traindata.shape[0]))
		rmse_train = np.zeros((self.num_chains,self.NumSamples - burnin))
		fxtest_samples = np.zeros((self.num_chains,self.NumSamples - burnin, self.testdata.shape[0]))
		rmse_test = np.zeros((self.num_chains,self.NumSamples - burnin))
		accept_ratio = np.zeros((self.num_chains,1))

		for i in range(self.num_chains):
			file_name = self.path+'/posterior/pos_w_chain_'+ str(self.temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			pos_w[i,:,:] = dat[burnin:,:] 
			file_name = self.path+'/posterior/fxtrain_samples_chain_'+ str(self.temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			fxtrain_samples[i,:,:] = dat[burnin:,:]
			file_name = self.path+'/posterior/fxtest_samples_chain_'+ str(self.temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			fxtest_samples[i,:,:] = dat[burnin:,:]	
			file_name = self.path+'/posterior/rmse_test_chain_'+ str(self.temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			rmse_test[i,:] = dat[burnin:]		
			file_name = self.path+'/posterior/rmse_train_chain_'+ str(self.temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			rmse_train[i,:] = dat[burnin:]
			file_name = self.path + '/posterior/accept_list_chain_' + str(self.temperatures[i]) + '_accept.txt'
			dat = np.loadtxt(file_name)
			accept_ratio[i,:] = dat

		pos_w = pos_w.transpose(2,0,1).reshape(self.num_param,-1)
		accept_total = np.sum(accept_ratio)/self.num_chains
		fx_train = fxtrain_samples.reshape(self.num_chains*(self.NumSamples - burnin), self.traindata.shape[0])
		rmse_train = rmse_train.reshape(self.num_chains*(self.NumSamples - burnin), 1)
		fx_test = fxtest_samples.reshape(self.num_chains*(self.NumSamples - burnin), self.testdata.shape[0])
		rmse_test = rmse_test.reshape(self.num_chains*(self.NumSamples - burnin), 1)

		print("NUMBER OF SWAPS =", self.num_swap)
		return (pos_w, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_total)

def main():

	for problem in range(2,3):
		#l	= "/home/konark/parallel-tempering-neural-net-master/LDMCMC_timeseries-master/"
		path = "results"
		problem =	2
		if problem ==	1:
			traindata = np.loadtxt(l	+ "Data_OneStepAhead\\Lazer\\train.txt")
			testdata	= np.loadtxt(l + "Data_OneStepAhead\\Lazer\\test.txt")	#
			name	= "Lazer"
		if problem ==	2:
			traindata = np.loadtxt(  "Data_OneStepAhead/Sunspot/train.txt")
			testdata	= np.loadtxt( "Data_OneStepAhead/Sunspot/test.txt")	#
			name	= "Sunspot"
		if problem ==	3:
			traindata = np.loadtxt(l	+ "Data_OneStepAhead/Mackey/train.txt")
			testdata	= np.loadtxt(l + "Data_OneStepAhead/Mackey/test.txt")  #
			name	= "Mackey"
		if problem ==	4:
			traindata = np.loadtxt(l	+ "Data_OneStepAhead/Lorenz/train.txt")
			testdata	= np.loadtxt(l + "Data_OneStepAhead/Lorenz/test.txt")  #
			name	= "Lorenz"
		if problem ==	5:
			traindata = np.loadtxt(l	+ "Data_OneStepAhead/Rossler/train.txt")
			testdata	= np.loadtxt(l + "Data_OneStepAhead/Rossler/test.txt")	#
			name	= "Rossler"
		if problem ==	6:
			traindata = np.loadtxt(l	+ "Data_OneStepAhead/Henon/train.txt")
			testdata	= np.loadtxt(l+"Data_OneStepAhead/Henon/test.txt")	#
			name	= "Henon"
		if problem ==	7:
			traindata = np.loadtxt(l+"Data_OneStepAhead/ACFinance/train.txt") 
			testdata	= np.loadtxt(l+"Data_OneStepAhead/ACFinance/test.txt")	#
			name	= "ACFinance"
		
		hidden = 5
		ip = 4 #input
		output = 1
		topology = [ip, hidden, output]

		NumSample = 4000
		maxtemp = 10
		swap_ratio = 0.1
		num_chains = 4
		burn_in = 0.3
		swap_interval =   int(swap_ratio * (NumSample/num_chains)) #how ofen you swap neighbours
		timer = time.time()

		pt = ParallelTempering(traindata, testdata, topology, num_chains, maxtemp, NumSample, swap_interval, path)
		pt.initialize_chains(burn_in)

		pos_w, fx_train, fx_test, x_train, x_test, rmse_train, rmse_test, accept_total = pt.run_chains()

		print ('Successfully Regressed')
		print (accept_total, '% total accepted')

		timer2 = time.time()
		print ((timer2 - timer), 'sec time taken')

		#PLOTS
		fx_mu = fx_test.mean(axis=0)
		fx_high = np.percentile(fx_test, 95, axis=0)
		fx_low = np.percentile(fx_test, 5, axis=0)

		fx_mu_tr = fx_train.mean(axis=0)
		fx_high_tr = np.percentile(fx_train, 95, axis=0)
		fx_low_tr = np.percentile(fx_train, 5, axis=0)

		rmse_tr = np.mean(rmse_train[:])
		rmsetr_std = np.std(rmse_train[:])
		rmse_tes = np.mean(rmse_test[:])
		rmsetest_std = np.std(rmse_test[:])
		outres = open(path+'/result.txt', "a+")
		np.savetxt(outres, (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std, accept_total), fmt='%1.5f')
		print (rmse_tr, rmsetr_std, rmse_tes, rmsetest_std)
		
		ytestdata = testdata[:, ip]
		ytraindata = traindata[:, ip]

		plt.plot(x_test, ytestdata, label='actual')
		plt.plot(x_test, fx_mu, label='pred. (mean)')
		plt.plot(x_test, fx_low, label='pred.(5th percen.)')
		plt.plot(x_test, fx_high, label='pred.(95th percen.)')
		plt.fill_between(x_test, fx_low, fx_high, facecolor='g', alpha=0.4)
		plt.legend(loc='upper right')

		plt.title("Plot of Test Data vs MCMC Uncertainty ")
		plt.savefig(path+'/restest.png')
		plt.savefig(path+'/restest.svg', format='svg', dpi=600)
		plt.clf()
		# -----------------------------------------
		plt.plot(x_train, ytraindata, label='actual')
		plt.plot(x_train, fx_mu_tr, label='pred. (mean)')
		plt.plot(x_train, fx_low_tr, label='pred.(5th percen.)')
		plt.plot(x_train, fx_high_tr, label='pred.(95th percen.)')
		plt.fill_between(x_train, fx_low_tr, fx_high_tr, facecolor='g', alpha=0.4)
		plt.legend(loc='upper right')

		plt.title("Plot of Train Data vs MCMC Uncertainty ")
		plt.savefig(path+'/restrain.png')
		plt.savefig(path+'/restrain.svg', format='svg', dpi=600)
		plt.clf()

		mpl_fig = plt.figure()
		ax = mpl_fig.add_subplot(111)

		# ax.boxplot(pos_w)

		# ax.set_xlabel('[W1] [B1] [W2] [B2]')
		# ax.set_ylabel('Posterior')

		# plt.legend(loc='upper right')

		# plt.title("Boxplot of Posterior W (weights and biases)")
		# plt.savefig(path+'/w_pos.png')
		# plt.savefig(path+'/w_pos.svg', format='svg', dpi=600)

		# plt.clf()

if __name__ == "__main__": main()
