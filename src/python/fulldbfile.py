import os
import re

first_level_in_database = []
second_level = []
k=0

for (dirpath, dirnames, filenames) in os.walk("Database"):
    first_level_in_database.extend(dirnames)
    break

for i in range(0,len(first_level_in_database)):
	for root, dirs, files in os.walk(os.path.abspath("Database/"+first_level_in_database[i])):
		for file in files:
			second_level.append(str(i)+os.path.join(root, file))

if not os.path.exists("Caffe_Files"):
    os.makedirs("Caffe_Files")

try:
    os.remove("Caffe_Files/full_database.txt")
except OSError:
    pass

final_label = -1
tmp_label = 0
full_database_file= open("Caffe_Files/full_database.txt", "w+")
for i in range(0,len(second_level)):
        name = second_level[i].rsplit('/', 1)[-1]
        try:
            label = re.search('Label(.+?)Frame', name).group(1)
        except AttributeError:
        # Label and Frame not found in the original string
            label = '0' # apply your error handling
        label2 = int(label)
        if label2 != tmp_label:
            tmp_label=label2
            final_label+=1
        full_database_file.write(second_level[i][1:]+" "+str(final_label)+"\n")

full_database_file.close()
