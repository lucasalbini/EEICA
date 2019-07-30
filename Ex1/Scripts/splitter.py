import os

path="/home/users/lucas/EEICA/Ex1/Dataset/UTFPR-BOP-splits-species/train/"

classe=os.listdir(path)
for c in classe:
	path2=os.listdir(str(path)+str(classe)+"/")
	files = str(path2)+str(f)
print files
