import csv
row_list = [[]]
f = open('jdx/NIST/Sulfur hexafluoride.txt')
count = 0
for line in f:
    count = 0
    for word in line.split():
        if count != 0:
            row =[word]
            row_list.append(row)
        count=count+1

with open('Sulfur Hexafluoride.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(row_list)