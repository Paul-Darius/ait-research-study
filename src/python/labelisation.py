import os
import sys
import random
import shutil

full_db = open("Database/Caffe_Files/full_database.txt", "r")
labelised_full_db = open("Database/Caffe_Files/labelised_full_database.txt",'w+')

while 1:
	txt = full_db.readline()
	if txt =='':
		break
	source,_,label = txt.rpartition(' ')
	name = source.rsplit('/', 1)[-1]
	if name[0]=='1':
		labelised_full_db.write(source+' 1\n')
	elif name[0]=='2':
		labelised_full_db.write(source+' 2\n')
	elif name[0]=='3':
		labelised_full_db.write(source+' 3\n')
	else:
		labelised_full_db.write(source+' 0\n')

full_db.close()
labelised_full_db.close()