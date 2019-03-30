# read train_label.csv (sorted) and find start and end index for 
#  each of the 333 identities

if __name__ == "__main__":
	import csv

	with open('../data/train_label.csv',encoding='utf-8') as cfile:
		reader = csv.reader(cfile)
		readeritem=[]
		readeritem.extend([row for row in reader])

	wf=open('../data/train_index.txt','w+',newline='') # format: start, end
	writer=csv.writer(wf)

	for i,row in enumerate(readeritem):
		if i>0:
			if int(row[0])!=last:
				writer.writerow([start, i-1])
				start = i
		else:
			start = i
		index = i
		last = int(row[0])
	writer.writerow([start, i-1])