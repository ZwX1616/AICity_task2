import numpy as np
import cv2

n = 2

i = 0
with open('./output/submission_hey.txt','r+') as f:
	while(i<n):
		pred = f.readline()
		i+=1
pred = pred.split()

output=[]
for r in range(10):
	current_row=[]
	for c in range(10):
		img = cv2.imread('./data/image_train/'+str(format(int(pred[r*10+c]),'06d'))+'.jpg')
		img = cv2.resize(img,(128,128))
		current_row.append(img)
	current_row = np.hstack(current_row)
	output.append(current_row)
output = np.vstack(output)
cv2.imshow('haha',output)
cv2.waitKey(120000)