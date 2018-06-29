import numpy as np 
import  matplotlib.pyplot as plt
# def make_patch_spines_invisible(ax):
#     ax.set_frame_on(True)
#     ax.patch.set_visible(False)
#     for sp in ax.spines.values():
#         sp.set_visible(False)
# train = np.asarray([0.05752,0.0483,0.03891,0.03372,0.03203,0.03274,0.02687,0.02724,0.02751,0.03115])
# trainerr = np.asarray([0.05329,0.04695,0.03012,0.01367,0.02398,0.01585,0.01331,0.02493,0.01844,0.025])
# test = np.asarray([0.05036,0.04195,0.03589,0.03335,0.02908,0.02904,0.02544,0.0238,0.02548,0.02915])
# testerr = np.asarray([0.04607,0.03506,0.02484,0.01302,0.01914,0.01335,0.01121,0.02285,0.01866,0.02204])
# timetaken = np.asarray([324.18886,356.21861,339.69956,340.98859,391.81021,366.33081,376.67785,378.48699,388.73609,410.7246])
# acceptance = np.asarray([12.967,19.8475,15.671,17.8875,22.321,23.284,23.903,27.2335,27.1645,29.1905])
# x = np.linspace(1000,10000,10)
# print(x)
# fig, host = plt.subplots()
# fig.subplots_adjust(right=0.75)
# second = host.twinx()
# third = host.twinx()
# # Offset the right spine of par2.  The ticks and label have already been
# # placed on the right by twinx above.
# third.spines["right"].set_position(("axes", 1.2))
# # Having been created by twinx, par2 has its frame off, so the line of its
# # detached spine is invisible.  First, activate the frame but make the patch
# # and spines invisible.
# make_patch_spines_invisible(third)
# # Second, show the right spine.
# third.spines["right"].set_visible(True)
# host.set_xlabel("Swap Interval")
# host.set_ylabel("RMSE Values")
# host.set_ylim(0,0.06)
# second.set_ylabel("Time in Seconds")
# second.set_ylim(100,1000)
# third.set_ylabel("Percentage Acceptance")
# third.set_ylim(0,100)
# p2 = host.errorbar(x, test, yerr=testerr/20, label = "Test")
# p1 = host.errorbar(x, train, yerr=trainerr/20, label="Train")
# p3 = second.plot(x, timetaken, 'r-.', label="Time")
# p4 = third.plot(x,acceptance,'b--', label="Percentage Acceptance")
# host.legend(loc=1)
# second.legend(loc=4)
# third.legend(loc=3)
# second.yaxis.label.set_color('r')
# third.yaxis.label.set_color('b')
# fig.savefig('plot.pdf')
temperatures = [1.0,1.3949507939624208,1.945887717576389,2.7144176165949068,3.7864790094146477,5.281951900505004,7.368062997280774,10.278085328021955,14.337423288737734,20.0]
pos_w = np.zeros((10,2000,99))
for i in range(10):
			file_name ='./multicore-pt-classification/RESULTS/iris_results_20000_20_10_0.125/posterior/pos_w_chain_'+ str(temperatures[i])+ '.txt'
			dat = np.loadtxt(file_name)
			pos_w[i,:,:] = dat

pos_w = pos_w.transpose(2,0,1).reshape(dat.shape[1],-1)
for s in [1,20,40,60,80]:
	list_points = pos_w[s,:]
	title = str(s)
	fname = "./multicore-pt-classification/RESULTS/Figs"
	width = 9 

	font = 12

	fig = plt.figure(figsize=(5, 6))
	ax = fig.add_subplot(111)


	slen = np.arange(0,len(list_points),1) 
	 
	fig = plt.figure(figsize=(5,6))
	ax = fig.add_subplot(111)
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
	ax.set_title(' Posterior distribution', fontsize=  font+2)#, y=1.02)

	ax1 = fig.add_subplot(211) 

	n, rainbins, patches = ax1.hist(list_points,  bins = 20,  alpha=0.5, facecolor='sandybrown', normed=False)	


	color = ['blue','red', 'pink', 'green', 'purple', 'cyan', 'orange','olive', 'brown', 'black']

	ax1.grid(True)
	ax1.set_ylabel('Frequency',size= font+1)
	ax1.set_xlabel('Parameter values', size= font+1)

	ax2 = fig.add_subplot(212)

	list_points = np.asarray(np.split(list_points, 10))




	ax2.set_facecolor('#f2f2f3') 
	ax2.plot( list_points.T , label=None)
	ax2.set_title(r'Trace plot',size= font+2)
	ax2.set_xlabel('Samples',size= font+1)
	ax2.set_ylabel('Parameter values', size= font+1) 

	fig.tight_layout()
	fig.subplots_adjust(top=0.88)
	 

	plt.savefig(fname + '/' + title  + '_pos_.png', bbox_inches='tight', dpi=300, transparent=False)
	plt.clf()