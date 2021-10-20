import pandas as pd
import numpy as np
from scipy.linalg import svd
import csv
import math


Start = 450
End = 3966
gap = 4
A = 0.9
C = 50

## filter for svd
with open("tao's project/Code and data(TAO)/filter/var/filter_rec.csv", 'r', encoding='utf-8-sig') as f:
#with open('var/filter_rec_a.csv', 'r', encoding='utf-8-sig') as f:
    filter_rec = np.genfromtxt(f, dtype=float, delimiter=',', skip_header=False)

num_starting=5
num_ending=20
for temp in range(num_starting,num_ending+1):
    center=filter_rec[0:temp]
    filter = []
    x=Start
    while x <=End:
        trans = []
        for b in center:
            #trans.append(A*math.exp((-pow((10000000/x-10000000/b),2))/(2*pow(C,2))))
            trans.append(A * math.exp((-pow((  x -  b), 2)) / (2 * pow(C, 2))))
        filter.append(trans)
        x += 0.3#4
    #with open('filter/var/setA/c=50/'+str(temp)+'.csv', 'w', newline='') as file:
    with open("tao's project/Code and data(TAO)/filter/var/c=1/" + str(temp) + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(filter)

##filter for equally spaced
# center_5 = [802,1506,2210,2914,3618]
# center_6 = [742,1326,1910,2494,3078,3662]
# center_7 = [702,1206,1710,2214,2718,3222,3726]
# center_8 = [670,1110,1550,1990,2430,2870,3310,3750]
# center_9 = [646,1038,1430,1822,2214,2606,2998,3390,3782]
# center_10 = [626,978,1330,1682,2034,2386,2738,3090,3442,3794]
# center_11 = [610,930,1250,1570,1890,2210,2530,2850,3170,3490,3810]
# center_12 = [598,890,1182,1474,1766,2058,2350,2642,2934,3226,3518,3810]
# center_13 = [586,858,1130,1402,1674,1946,2218,2490,2762,3034,3306,3578,3850]
# center_14 = [574,826,1078,1330,1582,1834,2086,2338,2590,2842,3094,3346,3598,3850]
# center_15 = [566,802,1038,1274,1510,1746,1982,2218,2454,2690,2926,3162,3398,3634,3870]
# center_16 = [562,782,1002,1222,1442,1662,1882,2102,2322,2542,2762,2982,3202,3422,3642,3862]
# center_17 = [554,762,970,1178,1386,1594,1802,2010,2218,2426,2634,2842,3050,3258,3466,3674,3882]
# center_18 = [550,746,942,1138,1334,1530,1726,1922,2118,2314,2510,2706,2902,3098,3294,3490,3686,3882]
# center_19 = [542,726,910,1094,1278,1462,1646,1830,2014,2198,2382,2566,2750,2934,3118,3302,3486,3670,3854]
# center_20 = [538,714,890,1066,1242,1418,1594,1770,1946,2122,2298,2474,2650,2826,3002,3178,3354,3530,3706,3882]
#
#
# num_starting=5
# num_ending=20
#
# filter = []
# x=Start
# while x <=End:
#     trans = []
#     for b in center_20:
#         trans.append(A*math.exp((-pow((10000000/x-10000000/b),2))/(2*pow(C,2))))
#     filter.append(trans)
#     x += 4
# with open('filter/equallySpaced/c=200/'+str(20)+'.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(filter)
#

