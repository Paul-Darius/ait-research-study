import os

first_level_in_database = []
second_level = []
k=0

for (dirpath, dirnames, filenames) in os.walk("../../Database"):
    first_level_in_database.extend(dirnames)
    break

for i in range(0,len(first_level_in_database)):
	for root, dirs, files in os.walk(os.path.abspath("../../Database/"+first_level_in_database[i])):
		for file in files:
			second_level.append(os.path.join(root, file))

if not os.path.exists("../../Database/Caffe_Files"):
    os.makedirs("../../Database/Caffe_Files")
    
full_database_file= open("../../Database/Caffe_Files/full_database.txt", "w+")
for i in range(0,len(second_level)):
	full_database_file.write(second_level[i]+" 0\n")

full_database_file.close()
