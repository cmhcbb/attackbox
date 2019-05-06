import sys
import numpy as np

query_list = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 30000, 40000, 60000, 80000, 100000]

lines = open(sys.argv[1]).readlines()
#distortion_list = [0]*len(query_list)
distortion_list = [[] for _ in range(len(query_list))]

count = 0
nowq = 0
for line in lines: 
	if line[0:8]=="========":
		sp = line.strip().split()
		delta = float(sp[4])
		nowq = 0
		count += 1
	if line[0:9]=="Iteration":
		sp = line.strip().split()
		nquery = int(sp[5])
		if nowq < len(query_list):
			while nquery > query_list[nowq]:
				distortion_list[nowq].append(delta)
				nowq += 1
				if nowq == len(query_list):
					break 
		delta = float(sp[3])
	if line[0:11] =="Adversarial":
		while nowq < len(query_list):
			distortion_list[nowq].append(delta)
			nowq += 1

print("meidum distortion over iterations:")
for ii in range(len(distortion_list)):
	print("Query %d AvgDistortion %lf"%(query_list[ii], np.median(distortion_list[ii])))

epsilon = 1.5
print("attack success rate when epsilon=%lf"%(epsilon))
for ii in range(len(distortion_list)):
	success_rate = np.sum(np.array(distortion_list[ii])<=epsilon)/float(len(distortion_list[ii]))
	print("Query %d Success_rate %lf"%(query_list[ii], success_rate))
