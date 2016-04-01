import os
import sys
import random
import shutil

train = open("Caffe_Files/train.txt",'r')
train1 = open("Caffe_Files/train1.txt",'w+')
train2 = open("Caffe_Files/train2.txt",'w+')


#new

count_number_of_label[]
labeled[][] # An array of array of strings

while 1:
    txt = train.readline()
    if txt =='':
    source,_,label = txt.rpartition(' ')
    count_number_of_label[label]+=1
    labeled[label].append(source)

for k in range(0, len(count_number_of_label)-1)
    for i in range(0, count_number_of_label[k]-1):
        for j in range(i+1, count_number_of_label[k]-1):
            train1.write(labeled[k][i]+' 1\n')
            train2.write(labeled[k][j]+' 1\n')
    for i in range(0, count_number_of_label[k]-1):
        for j in range(i+1, count_number_of_label[k]-1):
            train1.write(labeled[k][i]+' 0\n')
            nbrs = range(1,i-1) + range(i+1,len(count_number_of_label)))
            rd = random.choice(nbrs)
            train2.write(labeled[rd][random.randint(0,count_number_of_label[rd]-1)]+' 0\n')

train.close()
train1.close()
train2.close()

# Same for the test file.


test = open("Caffe_Files/test.txt",'r')
test1 = open("Caffe_Files/test1.txt",'w+')
test2 = open("Caffe_Files/test2.txt",'w+')

count_number_of_label[]
labeled[][] # An array of array of strings

while 1:
    txt = test.readline()
    if txt =='':
    source,_,label = txt.rpartition(' ')
    count_number_of_label[label]+=1
    labeled[label].append(source)

for k in range(0, len(count_number_of_label)-1)
    for i in range(0, count_number_of_label[k]-1):
        for j in range(i+1, count_number_of_label[k]-1):
            test1.write(labeled[k][i]+' 1\n')
            test2.write(labeled[k][j]+' 1\n')
    for i in range(0, count_number_of_label[k]-1):
        for j in range(i+1, count_number_of_label[k]-1):
            test1.write(labeled[k][i]+' 0\n')
            nbrs = range(1,i-1) + range(i+1,len(count_number_of_label)))
            rd = random.choice(nbrs)
            test2.write(labeled[rd][random.randint(0,count_number_of_label[rd]-1)]+' 0\n')


test.close()
test1.close()
test2.close()