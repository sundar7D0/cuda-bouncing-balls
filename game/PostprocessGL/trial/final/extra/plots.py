import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.preprocessing import  normalize


#12790729,1615459,654912,961952,743135

latency=[[48,60,150,300,500,1000,1500,2500,5000,15000]
,[308,525,680,830,1500,1888,6225,10040,10040,10040]]
colors=["-b","-r","-g","-c","-k","-m"]
x_labels=['CPUv','GPUv']
y_labels = ['10','20','40','60','80','100','150','250','400','700']
for i in range(2):
	plt.plot(latency[i],colors[i],marker="o",label=x_labels[i])
plt.xticks([0,1,2,3,4,5,6,7,8,9],y_labels)
plt.xlabel('# balls (n)')
plt.ylabel('unhidden processKernel+over-head latency in micro-secs.')
plt.title('Latency variations over (h/w, n) with (kx,ky,repeat,r)=(2,2,1,0.025f).')
plt.legend(loc="upper right")
plt.show()

'''
latency=[[1,1.31,1.28,5.36]
,[0.28,0.46,0.45,1.3]]
colors=["-b","-r","-g","-c","-k","-m"]
x_labels=['0.05f','0.025f']
y_labels = ['2x2','2x8','8x2','8x8']
for i in range(2):
	plt.plot(latency[i],colors[i],marker="o",label=x_labels[i])
plt.xticks([0,1,2,3,4],y_labels)
plt.xlabel('kx * ky')
plt.ylabel('processKernel latency in ms (0.25ms launch overhead)')
plt.title('Latency variations over (kx,ky,r) for n=60, 480*640 window (8.5*11.5cm2).')
plt.legend(loc="upper right")
plt.show()
'''

'''
latency=[[125,250,650]
,[105,230,573]]
colors=["-b","-r","-g","-c","-k","-m"]
x_labels=['#repeat=1','#repeat=3']
y_labels = ['20','40','60']
for i in range(2):
	plt.plot(latency[i],colors[i],marker="o",label=x_labels[i])
plt.xticks([0,1,2],y_labels)
plt.xlabel('# balls (n)')
plt.ylabel('Overlap errors at 3000+ rounds (~1 for CPUv)')
plt.title('Ball-overlapping-error variations over (n, repeat) for 480*640 window.')
plt.legend(loc="upper right")
plt.show()
'''

'''
latency=[[29680838,24730367,24030269,23282811,23129731,22127139],[24537059,24657581,22469388,22121962,21869129,21710809],[23978101,22533747,22263317,22010492,21760335,21638339],
[23287481,22241119,22039481,21819690,21554697,21559181],[23435655,22310787,22086204,21848906,21583351,21585073],[22111213,21807171,21693793,21574931,21282910,21476695]]
util=[[0.06,0.04,0.04,0.03,0.03,0.03],[0.04,0.04,0.03,0.03,0.03,0.03],[0.04,0.03,0.03,0.03,0.03,0.03],[0.03,0.03,0.03,0.03,0.03],[0.03,0.03,0.03,0.03,0.03,0.03],[0.03,0.03,0.03,0.03,]]
_access=[[28898,9698,7298,4898,4898,2498],[9698,9698,2498,1698,1698,898],[7298,2498,1898,1298,1298,698],[4898,1698,1298,898,898,498],[4898,1698,1298,898,898,498],[2498,898,698,498,498,298]]
latency=np.array(latency)
print(latency)
x_labels = [10,35,55,75,100,112]
colors=["-b","-r","-g","-c","-k","-m"]
y_labels = [10,35,55,75,100,112]

for i in range(6):
	plt.plot(latency[i],colors[i],marker="o",label="F_size="+str(x_labels[i]))
plt.xticks([0,1,2,3,4,5],y_labels)
plt.xlabel('E_size')
plt.ylabel('Total #cycles')
plt.title('Latency variations over (E_size, F_size) for Layer1 of ResNet50.')
plt.legend(loc="upper right")
plt.show()

for i in range(6):
	plt.plot(util[i],colors[i],marker="o",label="F_size="+str(x_labels[i]))
plt.xticks([0,1,2,3,4,5],y_labels)
plt.xlabel('E_size')
plt.ylabel('GEMM-utilization (# active cycles of GEMM / total #cycles)')
plt.title('GEMM-utilization variations over (E_size, F_size) for Layer1 of ResNet50.')
plt.legend(loc="upper right")
plt.show()

for i in range(6):
	plt.plot(_access[i],colors[i],marker="o",label="F_size="+str(x_labels[i]))
plt.xticks([0,1,2,3,4,5],y_labels)
plt.xlabel('E_size')
plt.ylabel('Total #memory (SRAM+DRAM) access by all modules')
plt.title('#Memory-access variations over (E_size, F_size) for Layer1 of ResNet50.')
plt.legend(loc="upper right")
plt.show()
'''

'''
latency=[[47870623,45036623,43885702,43804729,42952583],
[23940337,22582689,21941811,21902535,21476695],
[62854699,47881137,47757645,43885811,42952695],
[48983814,28608943,23892103,22527209,21911121],
[31501150,79533835,31778339,24027941,22094806]]  #315011527

util=[[0.03,0.03,0.03,0.03,0.03],
[0.04,0.03,0.03,0.03,0.03],
[0.07,0.04,0.04,0.03,0.03],
[0.28,0.13,0.04,0.05,0.03],
[1.00,0.67,0.64,0.23,0.07]]
_access=[[19976,6632,4668,1796,596],[9992,4792,2340,898,298],[79316,19928,14632,3796,596],[158652,36246,9062,2308,898],[1091296,158652,39674,9980,1898]]  #1091296

latency=np.array(latency)
#latency=normalize(latency[0],norm='max',axis=1)
#latency=(latency-latency.min())/(latency.max()-latency.min())
print(latency)

x_labels = ['32x32','32x128','128x32','256x256','1024x1024']
colors=["-b","-r","-g","-c","-k"]
y_labels = ['32x32x64','8x128x256','256x256x512','1024x1024x2048','10240x10240x20480']


for i in range(5):
	if i!=4:
		plt.plot(_access[i],colors[i],marker="o",label=x_labels[i])
	else:
		plt.plot([1,2,3,4],_access[i][1:],colors[i],marker="o",label=x_labels[i])
plt.xticks([0,1,2,3,4],y_labels)
plt.xlabel('SRAM buffer sizes (weight x input x output) in KB')
plt.ylabel('Total #memory (SRAM+DRAM) access by all modules')
plt.title('#Memory-access variations over (array, buffer) sizes for Layer1 of ResNet50.')
plt.legend(loc="upper right")
plt.show()



for i in range(5):
	plt.plot(util[i],colors[i],marker="o",label=x_labels[i])
plt.xticks([0,1,2,3,4],y_labels)
plt.xlabel('SRAM buffer sizes (weight x input x output) in KB')
plt.ylabel('GEMM-utilization (# active cycles of GEMM / total #cycles)')
plt.title('GEMM-utilization variations over (array, buffer) sizes for Layer1 of ResNet50.')
plt.legend(loc="upper right")
plt.show()



for i in range(5):
	plt.plot(latency[i],colors[i],marker="o",label=x_labels[i])
plt.xticks([0,1,2,3,4],y_labels)
plt.xlabel('SRAM buffer sizes (weight x input x output) in KB')
plt.ylabel('Total #cycles')
plt.title('Latency variations over (array, buffer) sizes for Layer1 of ResNet50.')
plt.legend(loc="upper right")
plt.show()
'''

'''
cmap = plt.cm.rainbow
xxx=input('hel')
min_,max_=0,1

#for i in range(len(latency)):
#	for j in range(len(latency[0])):
#		temp_lat=latency[i][j]
#		if temp_lat<min_:
#			min_=temp_lat
#		if temp_lat>max_:
#			max_=temp_lat
norm = colors.BoundaryNorm(np.linspace(0,1,1000),cmap.N)  #np.arange(min_,max_,500)

x=[[0]*5,[1]*5,[2]*5,[3]*5,[4]*5]
y=[[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4]]
x_labels = ['32x32','32x128','128x32','256x256','1024x1024']  #[['32x32']*5,['32x128']*5,['128x32']*5,['256x256']*5,['1024x1024']*5]
y_labels = ['32x32x64','8x128x256','256x256x512','1024x1024x2048','10240x10240x20480']  #[['32x32x64','8x128x256','256x256x512','1024x1024x2048','10240x10240x20480'],['32x32x64','8x128x256','256x256x512','1024x1024x2048','10240x10240x20480'],['32x32x64','8x128x256','256x256x512','1024x1024x2048','10240x10240x20480'],['32x32x64','8x128x256','256x256x512','1024x1024x2048','10240x10240x20480'],['32x32x64','8x128x256','256x256x512','1024x1024x2048','10240x10240x20480']]
plt.scatter(x, y, c=latency,cmap=cmap, norm=norm, s=75, edgecolor='none')
plt.yticks([0,1,2,3,4],y_labels)
plt.xticks([0,1,2,3,4],x_labels)
plt.colorbar()
#plt.colorbar(ticks=np.linspace(min_,max_,100))
plt.show()
'''