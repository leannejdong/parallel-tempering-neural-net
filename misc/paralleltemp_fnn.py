import matplotlib.pyplot as	plt
import numpy as	np
import random
import time
import math
import pickle
#import	tensorflow as tf

class Network:
	def __init__(self,	Topo, Train, Test):
		self.Top = Topo  # NN	topology [input, hidden, output]
		self.TrainData = Train
		self.TestData	= Test
		np.random.seed()

		self.W1 =	np.random.randn(self.Top[0], self.Top[1]) /	np.sqrt(self.Top[0])
		self.B1 =	np.random.randn(1, self.Top[1])	/ np.sqrt(self.Top[1])	# bias first	layer
		self.W2 =	np.random.randn(self.Top[1], self.Top[2]) /	np.sqrt(self.Top[1])
		self.B2 =	np.random.randn(1, self.Top[2])	/ np.sqrt(self.Top[1])	# bias second layer
		self.lrate = 0.05
		self.hidout =	np.zeros((1, self.Top[1]))	# output	of first hidden	layer
		self.out = np.zeros((1, self.Top[2]))	 # output last layer

	def sigmoid(self, x):
		return 1 / (1	+ np.exp(-x))

	def sampleEr(self,	actualout):
		error	= np.subtract(self.out,	actualout)
		sqerror =	np.sum(np.square(error)) / self.Top[2]
		return sqerror

	def ForwardPass(self, X):
		z1 = X.dot(self.W1) -	self.B1
		self.hidout =	self.sigmoid(z1)  #	 Foutput of	first hidden layer
		z2 = self.hidout.dot(self.W2)	- self.B2
		self.out = self.sigmoid(z2)  # output	second hidden layer

	def BackwardPass(self,	Input, desired):
		out_delta	= (desired - self.out) * (self.out * (1	- self.out))
		hid_delta	= out_delta.dot(self.W2.T) * (self.hidout *	(1 - self.hidout))
		self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
		self.B2 += (-1 * self.lrate *	out_delta)
		self.W1 += (Input.T.dot(hid_delta) * self.lrate)
		self.B1 += (-1 * self.lrate *	hid_delta)

	def decode(self, w):
		w_layer1size = self.Top[0] * self.Top[1]
		w_layer2size = self.Top[1] * self.Top[2]

		w_layer1 = w[0:w_layer1size]
		self.W1 =	np.reshape(w_layer1, (self.Top[0], self.Top[1]))

		w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
		self.W2 =	np.reshape(w_layer2, (self.Top[1], self.Top[2]))
		self.B1 =	w[w_layer1size + w_layer2size:w_layer1size + w_layer2size +	self.Top[1]].reshape(1,	self.Top[1])
		self.B2 =	w[w_layer1size + w_layer2size +	self.Top[1]:w_layer1size + w_layer2size	+ self.Top[1] +	self.Top[2]].reshape(1,	self.Top[2])

	def encode(self):
		w1 = self.W1.ravel()
		w2 = self.W2.ravel()
		w	= np.concatenate([w1, w2, self.B1.reshape((-1,)), self.B2.reshape((-1,))])
		return w
		
	def evaluate_proposal(self, data, w):	# BP	with SGD (Stocastic	BP)

		self.decode(w)  #	method to decode w into	W1,	W2,	B1,	B2.

		size = data.shape[0]

		Input	= np.zeros((1, self.Top[0]))  #	temp hold input
		Desired =	np.zeros((1, self.Top[2]))
		fx = np.zeros(size)

		for pat in range(0, size):
			Input[:]	= data[pat,	0:self.Top[0]]
			Desired[:] =	data[pat, self.Top[0]:]

			self.ForwardPass(Input)
			fx[pat] = self.out
			#self.BackwardPass(Input, Desired)
			w = self.encode()
		#print(fx.shape)
		return [fx, w]

class MCMC:
	def __init__(self,	samples, traindata,	testdata, topology,	tempr):
		self.samples = samples  #	NN topology	[input,	hidden,	output]
		self.topology	= topology	# max epocs
		self.traindata = traindata  #
		self.testdata	= testdata
		self.temprature =	tempr
		w_size = (self.topology[0] * self.topology[1]) + (self.topology[1] * self.topology[2]) + self.topology[1]	+ self.topology[2]
		self.pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over	all	samples
		self.pos_tau = np.ones((samples, 1))
		testsize = self.testdata.shape[0]
		trainsize	= self.traindata.shape[0]
		self.fxtrain_samples = np.ones((samples, trainsize))	# fx	of train data over all samples
		self.fxtest_samples =	np.ones((samples, testsize))  #	fx of test data	over all samples
		self.rmse_train =	np.zeros(samples)
		self.rmse_test = np.zeros(samples)
		#	----------------

	def rmse(self,	predictions, targets):
		return np.sqrt(((predictions - targets) ** 2).mean())

	def likelihood_func(self, neuralnet, data,	w, tausq):
		y	= data[:, self.topology[0]]
		[fx,_] = neuralnet.evaluate_proposal(data, w)
		rmse = self.rmse(fx, y)
		loss = -0.5 *	np.log(2 * math.pi * tausq)	- 0.5 *	np.square(y	- fx) /	tausq
		return [(np.sum(loss))* (1.0/self.temprature), fx, rmse]

	def prior_likelihood(self,	sigma_squared, nu_1, nu_2, w, tausq):
		h	= self.topology[1]	# number	hidden neurons
		d	= self.topology[0]	# number	input neurons
		part1	= -1 * ((d * h + h + 2)	/ 2) * np.log(sigma_squared)
		part2	= 1	/ (2 * sigma_squared) *	(sum(np.square(w)))
		log_loss = part1 - part2 - (1	+ nu_1)	* np.log(tausq)	- (nu_2	/ tausq)
		return log_loss
		
	def sampler(self, beta):

		#	-------------------	initialize MCMC
		testsize = self.testdata.shape[0]
		trainsize	= self.traindata.shape[0]
		samples =	self.samples

		x_test = np.linspace(0, 1, num=testsize)
		x_train =	np.linspace(0, 1, num=trainsize)

		netw = self.topology	# [input, hidden, output]
		y_test = self.testdata[:,	netw[0]]
		y_train =	self.traindata[:, netw[0]]
		print	(y_train.size)
		print	(y_test.size)

		w_size = (netw[0]	* netw[1]) + (netw[1] *	netw[2]) + netw[1] + netw[2]  #	num	of weights and bias

		pos_w	= self.pos_w  #	posterior of all weights and bias over all samples
		pos_tau =	self.pos_tau

		fxtrain_samples =	self.fxtrain_samples  #	fx of train	data over all samples
		fxtest_samples = self.fxtest_samples	# fx	of test	data over all samples
		rmse_train = self.rmse_train
		rmse_test	= self.rmse_test
		w	= np.random.randn(w_size)
		w_proposal = np.random.randn(w_size)

		step_w = 0.02;  #	defines	how	much variation you need	in changes to w
		step_eta = 0.01;
		#	--------------------- Declare FNN and initialize

		neuralnet	= Network(self.topology, self.traindata, self.testdata)
		print	('evaluate Initial w')
		[pred_train, w] =	neuralnet.evaluate_proposal(self.traindata,	w)
		[pred_test, _] = neuralnet.evaluate_proposal(self.testdata, w)

		eta =	np.log(np.var(pred_train - y_train))
		tau_pro =	np.exp(eta)

		sigma_squared	= 25
		nu_1 = 0
		nu_2 = 0

		prior_likelihood = self.prior_likelihood(sigma_squared, nu_1,	nu_2, w, tau_pro)  # takes care	of the gradients

		[likelihood, pred_train, rmsetrain] =	self.likelihood_func(neuralnet,	self.traindata,	w, tau_pro)
		[likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata,	w, tau_pro)

		print	(likelihood)

		naccept =	0
		print	('begin	sampling using mcmc	random walk')
		plt.plot(x_train,	y_train)
		plt.plot(x_train,	pred_train)
		plt.title("Plot of Data vs Initial Fx")
		plt.savefig('ptresults/result0_40k_mcmc/begin.png')
		plt.clf()

		plt.plot(x_train,	y_train)

		for i	in range(samples - 1):
			
			w_proposal =	w +	np.random.normal(0,	step_w,	w_size)

			eta_pro = eta + np.random.normal(0, step_eta, 1)
			tau_pro = math.exp(eta_pro)

			[likelihood_proposal, pred_train, rmsetrain]	= self.likelihood_func(neuralnet, self.traindata, w_proposal,
																				tau_pro)
			[likelihood_ignore, pred_test, rmsetest]	= self.likelihood_func(neuralnet, self.testdata, w_proposal,
																			tau_pro)

			# likelihood_ignore	refers to parameter that	will not be	used in	the	alg.

			prior_prop =	self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,
											   tau_pro)	# takes care	of the gradients

			diff_likelihood = likelihood_proposal - likelihood
			diff_priorliklihood = prior_prop	- prior_likelihood

			try:
				mh_prob	= min(1, math.exp(diff_likelihood +	diff_priorliklihood))
			except OverflowError:
				mh_prob	= 1
			u = random.uniform(0, 1)

			if u	< mh_prob:
				# Update position
				#print	  i,	' is accepted sample'
				naccept	+= 1
				likelihood = likelihood_proposal
				prior_likelihood = prior_prop
				w =	w_proposal
				eta	= eta_pro

				#print	likelihood, prior_likelihood, rmsetrain,	rmsetest, w, 'accepted'

				pos_w[i	+ 1,] =	w_proposal
				pos_tau[i +	1,]	= tau_pro
				fxtrain_samples[i +	1,]	= pred_train
				fxtest_samples[i + 1,] = pred_test
				rmse_train[i + 1,] = rmsetrain
				rmse_test[i	+ 1,] =	rmsetest
				lhood_current =	likelihood
				plt.plot(x_train, pred_train)


			else:
				pos_w[i	+ 1,] =	pos_w[i,]
				pos_tau[i +	1,]	= pos_tau[i,]
				fxtrain_samples[i +	1,]	= fxtrain_samples[i,]
				fxtest_samples[i + 1,] = fxtest_samples[i,]
				rmse_train[i + 1,] = rmse_train[i,]
				rmse_test[i	+ 1,] =	rmse_test[i,]

				# print	i, 'rejected and retained'

		print	(naccept, '	num	accepted')
		print	(naccept*100.0 / (samples),	'% was accepted')
		accept_ratio = naccept / (samples	* 1.0) * 100
		lhood	= lhood_current
		plt.title("Plot of Accepted Proposals")
		plt.savefig('ptresults/result0_40k_mcmc/proposals'+str(beta)+'.png')
		plt.savefig('ptresults/result0_40k_mcmc/proposals.svg', format='svg',	dpi=600)
		plt.clf()

		return (pos_w, pos_tau, fxtrain_samples, fxtest_samples, x_train,	x_test,	rmse_train,	rmse_test, accept_ratio, lhood)

class ParallelTempering:

	def __init__(self,	num_chains,	maxtemp,NumSample,traindata,testdata,topology):
		
		self.maxtemp = maxtemp
		self.num_chains =	num_chains
		self.chains =	[]
		self.tempratures = []
		self.NumSamples =	int(NumSample/num_chains)
		self.sub_sample_size = int(0.05* self.NumSamples)
		self.traindata = traindata
		self.testdata	= testdata
		self.topology	= topology
		w_size = (self.topology[0] * self.topology[1]) + (self.topology[1] * self.topology[2]) + self.topology[1]	+ self.topology[2]
		#self.sub_sample_size	=  100
		

		self.fxtrain_samples = np.ones((num_chains,self.NumSamples, self.traindata.shape[0]))	 # fx of train data	over all samples
		self.fxtest_samples =	np.ones((num_chains,self.NumSamples, self.testdata.shape[0]))  # fx	of test	data over all samples
		self.rmse_train =	np.zeros((num_chains,self.NumSamples))
		self.rmse_test = np.zeros((num_chains,self.NumSamples))
		self.pos_w = np.ones((num_chains,self.NumSamples,	w_size))
		self.pos_tau = np.ones((num_chains,self.NumSamples,  1))
		  
	def default_beta_ladder(self, ndim, ntemps, Tmax):	#https://github.com/konqr/ptemcee/blob/master/ptemcee/sampler.py
		"""
		Returns a	ladder of :math:`\beta \equiv 1/T` under a geometric spacing that is determined	by the
		arguments	``ntemps`` and ``Tmax``.  The temperature selection	algorithm works	as follows:
		Ideally, ``Tmax``	should be specified	such that the tempered posterior looks like	the	prior at
		this temperature.	 If	using adaptive parallel	tempering, per `arXiv:1501.05823
		<http://arxiv.org/abs/1501.05823>`_, choosing	``Tmax = inf`` is a	safe bet, so long as
		``ntemps`` is	also specified.
		:param ndim:
			The number of dimensions	in the parameter space.
		:param ntemps: (optional)
			If set, the number of temperatures to generate.
		:param Tmax: (optional)
			If set, the maximum temperature for the ladder.
		Temperatures are chosen according	to the following algorithm:
		*	If neither ``ntemps`` nor ``Tmax`` is specified, raise an exception	(insufficient
		  information).
		*	If ``ntemps`` is specified but not ``Tmax``, return	a ladder spaced	so that	a Gaussian
		  posterior would	have a 25% temperature swap	acceptance ratio.
		*	If ``Tmax``	is specified but not ``ntemps``:
		  * If ``Tmax	= inf``, raise an exception	(insufficient information).
		  * Else,	space chains geometrically as above	(for 25% acceptance) until ``Tmax``	is reached.
		*	If ``Tmax``	and	``ntemps`` are specified:
		  * If ``Tmax	= inf``, place one chain at	``inf``	and	``ntemps-1`` in	a 25% geometric	spacing.
		  * Else,	use	the	unique geometric spacing defined by	``ntemps`` and ``Tmax``.
		"""

		if type(ndim)	!= int or ndim < 1:
			raise ValueError('Invalid number	of dimensions specified.')
		if ntemps	is None	and	Tmax is	None:
			raise ValueError('Must specify one of ``ntemps``	and	``Tmax``.')
		if Tmax is not None and Tmax <= 1:
			raise ValueError('``Tmax`` must be greater than 1.')
		if ntemps	is not None	and	(type(ntemps) != int or	ntemps < 1):
			raise ValueError('Invalid number	of temperatures	specified.')

		tstep	= np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
						  2.71225, 2.49879, 2.34226, 2.22198,	2.12628,
						  2.04807, 1.98276, 1.92728, 1.87946,	1.83774,
						  1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
						  1.66657, 1.64647, 1.62795, 1.61083,	1.59494,
						  1.58014, 1.56632, 1.55338, 1.54123,	1.5298,
						  1.51901, 1.50881, 1.49916, 1.49, 1.4813,
						  1.47302, 1.46512, 1.45759, 1.45039,	1.4435,
						  1.4369,	1.43056, 1.42448, 1.41864, 1.41302,
						  1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
						  1.38327, 1.37888, 1.37463, 1.37051,	1.36652,
						  1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
						  1.3449,	1.34164, 1.33847, 1.33538, 1.33236,
						  1.32943, 1.32656, 1.32377, 1.32104,	1.31838,
						  1.31578, 1.31325, 1.31076, 1.30834,	1.30596,
						  1.30364, 1.30137, 1.29915, 1.29697,	1.29484,
						  1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
						  1.28291, 1.28106, 1.27923, 1.27745,	1.27569,
						  1.27397, 1.27227, 1.27061, 1.26898,	1.26737,
						  1.26579, 1.26424, 1.26271, 1.26121,
						  1.25973])

		if ndim >	tstep.shape[0]:
			# An	approximation to the temperature step at large
			# dimension
			tstep = 1.0 + 2.0*np.sqrt(np.log(4.0))/np.sqrt(ndim)
		else:
			tstep = tstep[ndim-1]

		appendInf	= False
		if Tmax == np.inf:
			appendInf = True
			Tmax	= None
			ntemps =	ntemps - 1

		if ntemps	is not None:
			if Tmax is None:
				# Determine	Tmax from ntemps.
				Tmax = tstep **	(ntemps	- 1)
		else:
			if Tmax is None:
				raise ValueError('Must specify at least	one	of ``ntemps'' and '
								 'finite ``Tmax``.')

			# Determine ntemps from Tmax.
			ntemps =	int(np.log(Tmax) / np.log(tstep) + 2)

		betas	= np.logspace(0, -np.log10(Tmax), ntemps)
		if appendInf:
			# Use a geometric spacing, but replace the top-most temperature with
			# infinity.
			betas = np.concatenate((betas, [0]))

		return betas
		
	# assigin tempratures dynamically	 
	def assign_temptarures(self):
# for geometric	spacing	use	this
#		 betas =	self.default_beta_ladder(2,	ntemps=self.num_chains,	Tmax=self.maxtemp)		
#		 for	i in range(0, self.num_chains):			   
#			 self.tempratures.append(1.0/betas[i])
#			 print (self.tempratures[i])
		""" for linear spacing use this """
		self.tempratures.append(1.0)
		for i	in range(1,	self.num_chains):			 
			self.tempratures.append(self.tempratures[i-1]+(self.maxtemp/self.num_chains))
			print (self.tempratures[i])
			
	
	# Create the chains.. Each	chain gets its own temprature
	def initialize_chains (self):
		self.assign_temptarures()
		for i	in range(0,	self.num_chains):
			self.chains.append(MCMC(self.NumSamples,self.traindata,self.testdata,self.topology, self.tempratures[i]))
			
	# Propose swapping	between	adajacent chains		
	def propose_swap (self, swap_proposal): 
		 for l in	range( self.num_chains-1, 0, -1):			 
				u =	random.uniform(0, 1) 
				swap_prob =	min(1, swap_proposal[l-1])
				if u < swap_prob : 
					self.swap_info(self.chains[l],self.chains[-1])
					print ('chains	swapped')	  
			
			
	# Swap	configuration of two chains	   
	def swap_info(self, chain_cooler, chain_warmer):  
		
		temp_chain = chain_cooler;
		
		chain_cooler.fxtrain_samples = chain_warmer.fxtrain_samples
		chain_cooler.fxtest_samples =	chain_warmer.fxtest_samples
		chain_cooler.rmse_train =	chain_warmer.rmse_train
		chain_cooler.rmse_test = chain_warmer.rmse_test
		chain_cooler.pos_w = chain_warmer.pos_w
		chain_cooler.pos_tau = chain_warmer.pos_tau
		
		chain_warmer.fxtrain_samples = temp_chain.fxtrain_samples
		chain_warmer.fxtest_samples =	temp_chain.fxtest_samples
		chain_cooler.rmse_train =	temp_chain.rmse_train
		chain_cooler.rmse_test = temp_chain.rmse_test
		chain_cooler.pos_w = temp_chain.pos_w
		chain_cooler.pos_tau = temp_chain.pos_tau
		
	# Merge different MCMC	chains y stacking them on top of each other		  
	def merge_chain (self,	chain):
		comb_chain = []
		for i	in range(0,	self.num_chains):
			for j in	range(0, self.NumSamples):
				comb_chain.append(chain[i][j].tolist())		
		return np.asarray(comb_chain)
		

	def run_chains	(self):
		self.initialize_chains()
		swap_proposal	= np.ones(self.num_chains-1) # only	adjacent chains	can	be swapped therefore, the number of	proposals is ONE less num_chains
		
		print	(self.NumSamples,self.sub_sample_size,self.NumSamples/self.sub_sample_size)
		#input ()
		start	= 0
		end =	 start + self.sub_sample_size
		#for i in	range(0, int(self.NumSamples/self.sub_sample_size)):
		while	(end < self.NumSamples):
			
			
			print (start, end)
			print ('--------------------------------------\n\n')
			x_test =	np.ones((self.num_chains,self.testdata.shape[0]))
			x_train = np.ones((self.num_chains,self.traindata.shape[0]))
			lhood = np.zeros(self.num_chains)
			accept_ratio	= np.zeros(self.num_chains)
			g = open("ptresults/result0_40k_mcmc/resultspriors_acceptedpercent_80k.txt",	"a+")
			g.write("{} {} {} ".format(self.num_chains,self.maxtemp,(end/self.sub_sample_size)))
			#run	each chain for a fixed number of SAMPLING Period along the MCMC	Chain
			for j in	range(0,self.num_chains):		 
				self.pos_w[j,],	self.pos_tau[j,], self.fxtrain_samples[j,],	self.fxtest_samples[j,], x_train[j,], x_test[j,], self.rmse_train[j,], self.rmse_test[j,], accept_ratio[j],	lhood[j] = self.chains[j].sampler(j)
				print (j)
				print (lhood[j])
				g.write("{}	".format(accept_ratio[j]))
			g.write("{} \n".format(np.mean(accept_ratio)))
			g.close()
			#calculate the swap acceptance rate for parallel	chains	  
			for k in	range(0, self.num_chains-1): 
				 swap_proposal[k]=	(lhood[k]/lhood[k+1])*(1/self.tempratures[k]	* 1/self.tempratures[k+1])	
				
			
			#propose	swapping
			self.propose_swap(swap_proposal)
			
			#update the starting	and	ending positon within one chain
			start =	end
			end =  start	+ self.sub_sample_size
		  
  
		#concatenate all chains into one complete	chain by stacking them on each other 
		chain_fxtrain	= self.merge_chain(self.fxtrain_samples)
		chain_fxtest = self.merge_chain(self.fxtest_samples)
		chain_w =	self.merge_chain(self.pos_w)
		chain_tau	= self.merge_chain(self.pos_tau)
		chain_rmse_train = self.merge_chain(self.rmse_train)
		chain_rmse_test =	self.merge_chain(self.rmse_test)
			
		return chain_fxtrain,chain_fxtest,chain_w,chain_tau,chain_rmse_train,	chain_rmse_test, x_train, x_test

def	main():
   
	hidden	= 5
	input = 4	#
	output	= 1

	topology =	[input,	hidden,	output]

	MinCriteria = 0.005  #	stop when RMSE reaches MinCriteria ( problem dependent)

	#random.seed(time.time())

	NumSample = 40000	# need to pick yourself
	for i in range(6,7): 
		l	= "/home/konark/parallel-tempering-neural-net-master/LDMCMC_timeseries-master/"
		problem =	2
		if problem ==	1:
			traindata = np.loadtxt(l	+ "Data_OneStepAhead\\Lazer\\train.txt")
			testdata	= np.loadtxt(l + "Data_OneStepAhead\\Lazer\\test.txt")	#
			name	= "Lazer"
		if problem ==	2:
			traindata = np.loadtxt(l	+ "Data_OneStepAhead/Sunspot/train.txt")
			testdata	= np.loadtxt(l + "Data_OneStepAhead/Sunspot/test.txt")	#
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
		outres = open("ptresults/result0_40k_mcmc/resultspriors_40k_10_geometric.txt", "a+")
		#Number of chains	of MCMC	required to	be run
		num_chains =	10
		t	= time.time()
		#Maximum tempreature of hottest chain	 
		maxtemp =	50
		
		#	Create A a Patratellel Tempring	object instance	
		pt = ParallelTempering(num_chains, maxtemp,NumSample,traindata,testdata,topology)

		#run the chains in a sequence	in ascending order
		fx_train,fx_test,pos_w,pos_tau,rmse_train, rmse_test,	x_train, x_test	= pt.run_chains()
		print	('sucessfully sampled')
		burnin = 0.05* NumSample	# use post burn in samples

		pos_w	= pos_w[int(burnin):, ]
		pos_tau =	pos_tau[int(burnin):, ]
		print('#############')	
		print(fx_train.shape)
		fx_mu	= fx_test.mean(axis=0)
		fx_high =	np.percentile(fx_test, 95, axis=0)
		fx_low = np.percentile(fx_test, 5, axis=0)

		fx_mu_tr = fx_train.mean(axis=0)
		fx_high_tr = np.percentile(fx_train, 95, axis=0)
		fx_low_tr	= np.percentile(fx_train, 5, axis=0)

		rmse_tr =	np.mean(rmse_train[int(burnin):])
		rmsetr_std = np.std(rmse_train[int(burnin):])
		rmse_tes = np.mean(rmse_test[int(burnin):])
		rmsetest_std = np.std(rmse_test[int(burnin):])
		print	(rmse_tr, rmsetr_std, rmse_tes,	rmsetest_std)
		outres.write("{} {} {} {}	{} {} {} \n".format(name, num_chains, maxtemp, rmse_tr,	rmsetr_std,	rmse_tes, rmsetest_std))
		print(len(fx_mu), fx_high.shape, fx_low.shape, x_test[-1,:].shape)
		ytestdata	= testdata[:, input]
		ytraindata = traindata[:,	input]

		plt.plot(x_test[-1,:], ytestdata,	label='actual')
		plt.plot(x_test[-1,:], fx_mu,	label='pred. (mean)')
		plt.plot(x_test[-1,:], fx_low, label='pred.(5th percen.)')
		plt.plot(x_test[-1,:], fx_high, label='pred.(95th	percen.)')
		plt.fill_between(x_test[-1,:], fx_low, fx_high, facecolor='g', alpha=0.4)
		plt.legend(loc='upper	right')

		plt.title("Plot of Test Data vs MCMC Uncertainty ")
		plt.savefig('ptresults/result0_40k_mcmc/mcmcrestest_'+name+'_40k_10_geometric.png')
		plt.savefig('ptresults/result0_40k_mcmc/mcmcrestest_'+str(maxtemp)+'_ldpt_40k.svg', format='svg',	dpi=600)
		plt.clf()
		#	-----------------------------------------
		plt.plot(x_train[-1,:], ytraindata, label='actual')
		plt.plot(x_train[-1,:], fx_mu_tr,	label='pred. (mean)')
		plt.plot(x_train[-1,:], fx_low_tr, label='pred.(5th percen.)')
		plt.plot(x_train[-1,:], fx_high_tr, label='pred.(95th	percen.)')
		plt.fill_between(x_train[-1,:], fx_low_tr, fx_high_tr, facecolor='g',	alpha=0.4)
		plt.legend(loc='upper	right')

		plt.title("Plot of Train Data	vs MCMC	Uncertainty	")
		plt.savefig('ptresults/result0_40k_mcmc/train'+name+'_40k_10_geometric.png')
		plt.savefig('ptresults/result0_40k_mcmc/train_'+str(maxtemp)+'_ldpt_40k.svg',	format='svg', dpi=600)
		plt.clf()

		mpl_fig =	plt.figure()
		ax = mpl_fig.add_subplot(111)

		ax.boxplot(pos_w)
		ax.set_xlabel('[W1] [B1] [W2]	[B2]')
		ax.set_ylabel('Posterior')
		f	= open('ptresults/result0_40k_mcmc/ldpt.pckl','wb')
		pickle.dump(pos_w, f,	protocol = 2)
		f.close()
		plt.legend(loc='upper	right')

		plt.title("Boxplot of	Posterior W	(weights and biases)")
		plt.savefig('ptresults/result0_40k_mcmc/w_pos_'+name+'_40k_10_geometric.png')
		plt.savefig('ptresults/result0_40k_mcmc/w_pos_'+str(maxtemp)+'_ldpt_40k.svg',	format='svg', dpi=600)

		plt.clf()
		outres.close()
		elapsed =	time.time()	- t
		print("BP	used \nTime	elapsed: %.2f seconds" %elapsed)
if __name__	== "__main__": main()

