import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

# Network
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net('src/siamese/siamese_mbk.prototxt', 'src/siamese/siamese_mbk/_iter_50000.caffemodel', caffe.TEST);

test1 = open("Caffe_Files/test1.txt","r")
test2 = open("Caffe_Files/test2.txt","r")

#After the while loop
# Contains the x,y coordinates of the first image
out1=[]
# Contains the x,y coordinates of the second image
out2=[]
# Contains the computed energy
energies=[]
# expected[k] contains 1 if the images in line k of test1.txt and test2.txt represent the same person and 0 else.
expected=[]

while 1:
	txt1 = test1.readline()
	txt2 = test2.readline()
	if txt1='' or txt2='':
		break
	source1,_,label1=txt1.rpartition(' ')
	source2,_,label2=txt2.rpartition(' ')
	im1 = np.array(Image.open(source1))
	im1 = im1.reshape(3,256,256)
	im2 = np.array(Image.open(source2))
	im2 = im2.reshape(3,256,256)
	net.blobs['data'].data[...] = im1
	output1 = net.forward()
	out1.append(output1['feat'])
	net.blobs['data'].data[...] = im2
	output2 = net.forward()
	out2.append(output2['feat'])
	print output1['feat']
	print output2['feat']
	expected.append(int(label1))
	energy = sqrt((output1*output1+output2*output2))
	energies.append(energy)

test1.close()
test2.close()
# If the energy is above the threshold then the two images are considered as different. If the energy is under the threshol, they are identic.

threshold = 0.

# We consider the threshold as the barycenter of the barycenter of the set of the energies of two images of the same person and the barycenter of the set of the energies of two distinct people.

barycenter1 = 0.
barycenter2 = 0.
size1 = 0

for i in range(0,len(energies)-1):
	if expected[k] = 0:
		barycenter1 += energies[k]
		size1+=1
	else:
		barycenter2 += energies[k]

barycenter1 = barycenter1/size1
barycenter2 = barycenter2/(size(energies)-size1)

threshold = (barycenter1+barycenter2)/20

# Now that the threshold is computed we can compute the accuracy of the system and also the percentage of true positives, false positives and so on.

true_pos = 0
true_neg = 0
false_pos = 0
falseneg = 0

for i in range(0,len(energies)-1):
	if (energy[i]>threshold and expected[i]==0)
		true_neg+=1
	else if (energy[i]>threshold and expected[i]==1):
		false_neg+=1
	else if (energy[i]<=threshold and expected[i]==1):
		true_pos+=1
	else if (energy[i]<=threshold and expected[i]==0):
		false_pos+=1

# We finally give the result
length = len(energies)
print "False positive = "+str(false_pos/float(length))
print "False Negative = "+str(false_neg/float(length))
print "True positive = "+str(true_pos/float(length))
print "True negative = "+str(true_neg/float(length))		
print "Accuracy = "+str((true_pos+true_neg)/float(length))
