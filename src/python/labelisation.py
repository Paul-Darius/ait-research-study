import os
import sys
import random
import shutil

full_db = open("Caffe_Files/full_database.txt", "r")
labelised_full_db = open("Caffe_Files/labelised_full_database.txt",'w+')

while 1:
	txt = full_db.readline()
	if txt =='':
		break
	source,_,label = txt.rpartition(' ')
	name = source.rsplit('/', 1)[-1]
	if name[0].isdigit():
		labelised_full_db.write(source+' '+name[0]+'\n')
	else:
		labelised_full_db.write(source+' '+label+'\n')
full_db.close()
labelised_full_db.close()