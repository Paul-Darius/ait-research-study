import os
import sys
import random
import shutil

train = open("Database/Caffe_Files/train.txt",'r')
train1 = open("Database/Caffe_Files/train1.txt",'w+')
train2 = open("Database/Caffe_Files/train2.txt",'w+')

count_number_of_label_1 = 0
count_number_of_label_0 = 0
count_number_of_label_2 = 0
count_number_of_label_3 = 0

labeled3=[]
labeled2=[]
labeled1=[]
labeled0=[]

while 1:
	txt = train.readline()
	if txt =='':
		break
	source,_,label = txt.rpartition(' ')
	if label[0]=='1':
		count_number_of_label_1+=1
		labeled1.append(source)
	if label[0]=='0':
		count_number_of_label_0+=1
		labeled0.append(source)
	if label[0]=='2':
		count_number_of_label_2+=1
		labeled2.append(source)
	if label[0]=='3':
		count_number_of_label_3+=1
		labeled3.append(source)

for i in range(0, count_number_of_label_1-1):
	for j in range(i+1, count_number_of_label_1-1):
		train1.write(labeled1[i]+' 1\n')
		train2.write(labeled1[j]+' 1\n')
for i in range(0, count_number_of_label_1-1):
	for j in range(i+1, count_number_of_label_1-1):
		train1.write(labeled1[i]+' 0\n')
		train2.write(labeled0[random.randint(0,count_number_of_label_0-1)]+' 0\n')


for i in range(0, count_number_of_label_2-1):
	for j in range(i+1, count_number_of_label_2-1):
		train1.write(labeled2[i]+' 1\n')
		train2.write(labeled2[j]+' 1\n')
for i in range(0, count_number_of_label_2-1):
	for j in range(i+1, count_number_of_label_2-1):
		train1.write(labeled2[i]+' 0\n')
		train2.write(labeled0[random.randint(0,count_number_of_label_0-1)]+' 0\n')


for i in range(0, count_number_of_label_3-1):
	for j in range(i+1, count_number_of_label_3-1):
		train1.write(labeled3[i]+' 1\n')
		train2.write(labeled3[j]+' 1\n')
for i in range(0, count_number_of_label_3-1):
	for j in range(i+1, count_number_of_label_3-1):
		train1.write(labeled3[i]+' 0\n')
		train2.write(labeled0[random.randint(0,count_number_of_label_0-1)]+' 0\n')


train.close()
train1.close()
train2.close()

# Same for the test file.


test = open("Database/Caffe_Files/test.txt",'r')
test1 = open("Database/Caffe_Files/test1.txt",'w+')
test2 = open("Database/Caffe_Files/test2.txt",'w+')

count_number_of_label_1 = 0
count_number_of_label_0 = 0
count_number_of_label_2 = 0
count_number_of_label_3 = 0

labeled3=[]
labeled2=[]
labeled1=[]
labeled0=[]

while 1:
    txt = test.readline()
    if txt =='':
        break
    source,_,label = txt.rpartition(' ')
    if label[0]=='1':
        count_number_of_label_1+=1
        labeled1.append(source)
    if label[0]=='0':
        count_number_of_label_0+=1
        labeled0.append(source)
    if label[0]=='2':
        count_number_of_label_2+=1
        labeled2.append(source)
	if label[0]=='3':
		count_number_of_label_3+=1
		labeled3.append(source)

for i in range(0, count_number_of_label_1-1):
    for j in range(i+1, count_number_of_label_1-1):
        test1.write(labeled1[i]+' 1\n')
        test2.write(labeled1[j]+' 1\n')
for i in range(0, count_number_of_label_1-1):
    for j in range(i+1, count_number_of_label_1-1):
        test1.write(labeled1[i]+' 0\n')
        test2.write(labeled0[random.randint(0,count_number_of_label_0-1)]+' 0\n')


for i in range(0, count_number_of_label_2-1):
    for j in range(i+1, count_number_of_label_2-1):
        test1.write(labeled2[i]+' 1\n')
        test2.write(labeled2[j]+' 1\n')
for i in range(0, count_number_of_label_2-1):
    for j in range(i+1, count_number_of_label_2-1):
        test1.write(labeled2[i]+' 0\n')
        test2.write(labeled0[random.randint(0,count_number_of_label_0-1)]+' 0\n')


for i in range(0, count_number_of_label_3-1):
    for j in range(i+1, count_number_of_label_3-1):
        test1.write(labeled3[i]+' 1\n')
        test2.write(labeled3[j]+' 1\n')
for i in range(0, count_number_of_label_3-1):
    for j in range(i+1, count_number_of_label_3-1):
        test1.write(labeled3[i]+' 0\n')
        test2.write(labeled0[random.randint(0,count_number_of_label_0-1)]+' 0\n')


test.close()
test1.close()
test2.close()