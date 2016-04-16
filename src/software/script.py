import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
from collections import OrderedDict

############ HERE I ASK FOR THE IMAGES AND I PREPROCESS THEM
######## THE FINAL RESULT IS CONTAINED IN THE ARRAY ADDRESS_OF_IMAGES_OF_CRIMINAL

address_of_images_of_criminal = []

##### THESE LINES WILL BE USED IN FINAL VERSION
#print "How many pictures of the criminal do you want to provide?"
#number_pictures = int(raw_input())
#for i in range(0, number_pictures):
#	print "Address of image number "+str(i+1)
#	address_of_images_of_criminal.append(raw_input())
#####

##### AT THE MOMENT WE USE THESE IMAGES FOR TESTING
address_of_images_of_criminal.append("criminal_pictures/1.jpg")
address_of_images_of_criminal.append("criminal_pictures/2.jpg")
address_of_images_of_criminal.append("criminal_pictures/3.jpg")

########### THEN I HAVE TO COMPUTE THE FEATURES OF THESE IMAGES

###### I DEFINE A FUNCTION EXTRACTED FROM CAFE_FTR.PY WHICH WILL SAVE EVERYTHING IN 

def extract_feature(network_proto_path,
                    network_model_path,
                    image_list, layer_name, image_as_grey = False):
    """
    Extracts features for given model and image list.

    Input
    network_proto_path: network definition file, in prototxt format.
    network_model_path: trainded network model file
    image_list: A list contains paths of all images, which will be fed into the
                network and their features would be saved.
    layer_name: The name of layer whose output would be extracted.
    save_path: The file path of extracted features to be saved.
    """
    #network_proto_path, network_model_path = network_path
    print "hello"
    net = caffe.Classifier(network_proto_path, network_model_path)
    # net.set_phase_test()

    caffe.set_mode_cpu()

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    #net.set_mean('data', caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')  # ImageNet mean
    
    #if data_mean is None:
        #data_mean = np.zeros(1)
    #net.transformer.set_mean('data', data_mean)
    #if not image_as_grey:
    #    net.transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    #net.set_input_scale('data', 256)  # the reference model operates on images in [0,255] range instead of [0,1]
    net.transformer.set_input_scale('data', 1)

    #img_list = [caffe.io.load_image(p) for p in image_file_list]

    #----- test

    blobs = OrderedDict( [(k, v.data) for k, v in net.blobs.items()])

    #blobs = OrderedDict( [(k, v.data) for k, v in net.blobs.items()])
    shp = blobs[layer_name].shape
    print blobs['data'].shape

    batch_size = blobs['data'].shape[0]
    print blobs[layer_name].shape
    #print 'debug-------\nexit'
    #exit()

    #params = OrderedDict( [(k, (v[0].data,v[1].data)) for k, v in net.params.items()])
    if len(shp) is 2:
        features_shape = (len(image_list), shp[1])
    elif len(shp) is 3:
        features_shape = (len(image_list), shp[1], shp[2])
    elif len(shp) is 4:
        features_shape = (len(image_list), shp[1], shp[2], shp[3])

    features = np.empty(features_shape, dtype='float32', order='C')
    img_batch = []
    for cnt, path in zip(range(features_shape[0]), image_list):
        img = caffe.io.load_image(path, color = not image_as_grey)
        if image_as_grey and img.shape[2] != 1:
            img = skimage.color.rgb2gray(img)
            img = img[:, :, np.newaxis]
        if cnt == 0:
            print 'image shape: ', img.shape
        #print img[0:10,0:10,:]
        #exit()
        img_batch.append(img)
        #print 'image shape: ', img.shape
        #print path, type(img), img.mean()
        if (len(img_batch) == batch_size) or cnt==features_shape[0]-1:
            scores = net.predict(img_batch, oversample=False)
            '''
            print 'blobs[%s].shape' % (layer_name,)
            tmp =  blobs[layer_name]
            print tmp.shape, type(tmp)
            tmp2 = tmp.copy()
            print tmp2.shape, type(tmp2)
            print blobs[layer_name].copy().shape
            print cnt, len(img_batch)
            print batch_size
            #exit()

            #print img_batch[0:10]
            #print blobs[layer_name][:,:,0,0]
            #exit()
            '''

            # must call blobs_data(v) again, because it invokes (mutable_)cpu_data() which
            # syncs the memory between GPU and CPU
            blobs = OrderedDict( [(k, v.data) for k, v in net.blobs.items()])

            print '%d images processed' % (cnt+1,)

            #print blobs[layer_name][0,:,:,:]
            # items of blobs are references, must make copy!
            features[cnt-len(img_batch)+1:cnt+1] = blobs[layer_name][0:len(img_batch)].copy()
            img_batch = []

        #features.append(blobs[layer_name][0,:,:,:].copy())
    features = np.asarray(features, dtype='float32')
    return features

features = extract_feature('../face_id/face_verification_experiment-master/proto/LightenedCNN_A_deploy.prototxt', '../face_id/face_verification_experiment-master/model/LightenedCNN_A.caffemodel', address_of_images_of_criminal, 'prob', 1)


cascPath = "../others/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)
'''
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
'''