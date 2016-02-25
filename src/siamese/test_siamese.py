import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

#load the model
net = caffe.Net('src/siamese/siamese_mbk.prototxt',
                'src/siamese/siamese_mbk/_iter_50000.caffemodel',
                caffe.TEST)

# load input and configure preprocessing

test1= open("Database/Caffe_Files/test1.txt", "r")
test2= open("Database/Caffe_Files/test2.txt", "r")

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1)) # height*width*channel -> channel*height*width

# Arrays where the data is saved

im1 = []
im2= []
label_theoretical = []
out1 = []
out2 = []
k=0

while 1:
	txt1=test1.readline()
	txt2=test2.readline()
	if (txt1=='' or k>20):
		break
	source1,_,label1 = txt1.rpartition(' ')
	source2,_,label2 = txt2.rpartition(' ')
	
	im1.append(caffe.io.load_image(source1))
	im2.append(caffe.io.load_image(source2))
	
	label_theoretical.append(label1)
	
	net.blobs['data'].data[...] = transformer.preprocess('data', im1[k])
	out = net.forward()
	out1.append(out['feat'])
	
	net.blobs['data'].data[...] = transformer.preprocess('data', im2[k])
	outbis = net.forward()
	out2.append(out['feat'])

	k+=1

# Now we got the labels, the values. We just need to define the threshold and improve the work. 

# We could create a loss function, like sum(Y-information(E_w>t)0, depending on the threshold and we could try to find a threshold minimizing it.
#I have to think a bit about it, but we can probably fine even better or simpler methods.
