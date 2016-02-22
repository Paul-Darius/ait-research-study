import os
import random
import shutil

number_of_input_per_label = dict() # is a dictionnary

full_db = open("Database/Caffe_Files/full_database.txt", "r")

while 1:
	txt = full_db.readline()
	if txt =='':
		break
	key = txt.split(" ")[1:][0]
	if key in number_of_input_per_label:
		number_of_input_per_label[key] += 1
	else:
		number_of_input_per_label[key] = 1

full_db.close()

# Random permutation of the lines of the file.

with open("Database/Caffe_Files/full_database.txt", "r") as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('Database/Caffe_Files/full_database_tmpcpy.txt','w+') as target:
    for _, line in data:
        target.write( line )

# Now we can generate train.txt, valid.txt and test.txt using the permutated full database.

full_db_permutated = open("Database/Caffe_Files/full_database_tmpcpy.txt", "r")

train = open('Database/Caffe_Files/train.txt','w+')
valid = open('Database/Caffe_Files/valid.txt','w+') 
test = open('Database/Caffe_Files/test.txt','w+') 

train_label_number = dict() # is a dictionnary
valid_label_number = dict() # is a dictionnary
test_label_number = dict() # is a dictionnary

while 1:
	txt = full_db_permutated.readline()
	if txt =='':
		break
	key = txt.split(" ")[1:][0]
	if key in train_label_number:
		if number_of_input_per_label[key]/3 >= train_label_number[key]:
			train_label_number[key] += 1
			train.write(txt) # copy txt in train.txt
		else:
			if key in valid_label_number:
				if number_of_input_per_label[key]/3 >= valid_label_number[key]:
					valid_label_number[key] += 1
					valid.write(txt) # copy txt in valid.txt
				else:
					test.write(txt) # copy txt in test.txt
					if key in test_label_number:
						test_label_number[key] += 1
					else:
							test_label_number[key] = 1
			else:
				valid_label_number[key] = 1
				valid.write(txt)
		
	else:
		train_label_number[key] = 1
		train.write(txt)
	
full_db_permutated.close()
os.remove('Database/Caffe_Files/full_database_tmpcpy.txt')

train.close()
valid.close()
test.close()

# Generate Database_Final here until I get why the won't generate my lmdb the way I want it. Big waste of time because the documentation on caffe is really poor.

if not os.path.exists("Database_Final"):
    os.makedirs("Database_Final")

if not os.path.exists("Database_Final/train"):
    os.makedirs("Database_Final/train")

if not os.path.exists("Database_Final/test"):
    os.makedirs("Database_Final/test")

if not os.path.exists("Database_Final/valid"):
    os.makedirs("Database_Final/valid")

if not os.path.exists("Database_Final/files"):
    os.makedirs("Database_Final/files")
    
train_f = open('Database_Final/files/train.txt','w+')
valid_f = open('Database_Final/files/valid.txt','w+') 
test_f = open('Database_Final/files/test.txt','w+')

train = open('Database/Caffe_Files/train.txt','r+')
valid = open('Database/Caffe_Files/valid.txt','r+') 
test = open('Database/Caffe_Files/test.txt','r+')


while 1:
	txt = train.readline()
	if txt =='':
		break
	source,_,label = txt.partition(' ')
	
	destination = "Database_Final/train/"
	shutil.copy(source, os.getcwd()+"/"+destination)
	new_name = source.rsplit('/', 1)[-1]
	new_name = new_name.rsplit('.', 1)[0]+".JPEG"
	os.rename(os.getcwd()+"/"+destination+"/"+source.rsplit('/', 1)[-1], os.getcwd()+"/"+destination+"/"+new_name)
	train_f.write(new_name+" "+label)
	# os.getcwd()+"/"+destination+ just above in the first part of the comments

while 1:
	txt = test.readline()
	if txt =='':
		break
	source,_,label = txt.partition(' ')
	destination = "Database_Final/test/"
	shutil.copy(source, os.getcwd()+"/"+destination)
	new_name = source.rsplit('/', 1)[-1]
	new_name = new_name.rsplit('.', 1)[0]+".JPEG"
	os.rename(os.getcwd()+"/"+destination+"/"+source.rsplit('/', 1)[-1], os.getcwd()+"/"+destination+"/"+new_name)
	test_f.write(new_name+" "+label)
	
while 1:
	txt = valid.readline()
	if txt =='':
		break
	source,_,label = txt.partition(' ')
	destination = "Database_Final/valid/"
	shutil.copy(source, os.getcwd()+"/"+destination)
	new_name = source.rsplit('/', 1)[-1]
	new_name = new_name.rsplit('.', 1)[0]+".JPEG"
	os.rename(os.getcwd()+"/"+destination+"/"+source.rsplit('/', 1)[-1], os.getcwd()+"/"+destination+"/"+new_name)
	valid_f.write(new_name+" "+label)


train.close()
valid.close()
test.close()


train_f.close()
valid_f.close()
test_f.close()

