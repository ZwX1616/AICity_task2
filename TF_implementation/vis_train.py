import numpy as np
import cv2
import csv

n = 200

train_index = []
with open('./data/train_index.txt',encoding='utf-8') as cfile:
	reader = csv.reader(cfile)
	readeritem=[]
	readeritem.extend([row for row in reader])
for _, row in enumerate(readeritem):
	train_index.append([int(row[0]),int(row[1])])
del reader
del readeritem

# load the filelist
train_filelist = []
with open('./data/train_label.csv',encoding='utf-8') as cfile:
	reader = csv.reader(cfile)
	readeritem=[]
	readeritem.extend([row for row in reader])
for _, row in enumerate(readeritem):
	train_filelist.append(row[1])
del reader
del readeritem

output=[]
current_row=[]
i=train_index[n-1][0]
while(i<=train_index[n-1][1]):
	if (i-train_index[n-1][0])>0 and (i-train_index[n-1][0])%10==0:
		current_row=np.hstack(current_row)
		output.append(current_row)
		current_row=[]
	img = cv2.imread('./data/image_train/'+train_filelist[i])
	img = cv2.resize(img,(128,128))
	current_row.append(img)
	i+=1
output = np.vstack(output)
cv2.imshow(str('train '+str(n)),output)
cv2.waitKey(120000)