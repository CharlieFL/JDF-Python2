'''
Created on Jan 2, 2017

@author: Gargs
'''
import glob
from decimal import Decimal
from scipy import stats
import numpy as np

def calc_score(lst, wgt):
    #return the calculated score, or return -1 if errors
    score = -1 #error val
    if isinstance(lst, list):
        if len(lst) == len(wgt):
            score = 0
            for x in range (len(lst)):
                score += (lst[x] * wgt[x])
            score = score/len(lst)
            print "mean of scores: " + str(score) + "\n"
            
            score = int(Decimal(score).quantize(Decimal(1)))
            print "rounded score: " + str(score)
 
        else:
            print "length of list != length of weight list"
    else:
        print "arg 0 requires list, " + str(type(lst)) + " given instead"
    
    return score

dataDir = "F:/Research data/subm/"

#weights for the 6 models
##percentage of total = 33, .71329
# w1 = 0.180
# w2 = 0.159
# w3 = 0.165
# w4 = 0.175
# w6 = 0.138
# w7 = 0.182

#arbitrary weights= 33, .71430
# w1=1
# w2=0.63
# w3=0.75
# w4=0.88
# w6=0.38
# w7=1
# #

#log base/ log n  33, 0.71305
# w1=1
# w2=0.71
# w3=0.77
# w4=0.88
# w6=0.55
# w7=1

#arb 33, 0.71484
# w1=0.95
# w2=0.63
# w3=0.75
# w4=0.88
# w6=0.38
# w7=1


#arb 33, 0.71498
# w1=0.94
# w2=0.63
# w3=0.75
# w4=0.88
# w6=0.38
# w7=0.97

#33, 0.71553
# w1=0.90
# w2=0.63
# w3=0.75
# w4=0.88
# w6=0.2
# w7=0.93

#33, 0.71684
# w1=0.89
# w2=0.59
# w3=0.71
# w4=0.82
# w6=0.2
# w7=0.93

#33, 0.71834
#w1=0.89
#w2=0.70
#w3=0.75
#w4=0.85 
#w6=0.0
#w7=0.95

#31, 0.72208
# w1=0.89
# w2=0.0
# w3=0.75
# w4=0.85
# w6=0.0
# w7=0.95

#31, 0.72681
# w1=0.89
# w2=0.0
# w3=0.0
# w4=0.85
# w6=0.0
# w7=0.95


#35, 0.69610
#w1=0.0
#w2=0.5
#w3=0.6
#w4=0.0 
#w6=0.2
#w7=0.0

w1=0.5
w2=0.0
w3=0.0
w4=0.3 
w6=0.0
w7=0.6


weights = (w1, w2, w3, w4, w6, w7)


L1_results = open(glob.glob(dataDir + "L1/*.csv")[0], 'rb')
L2_results = open(glob.glob(dataDir + "L2/*_Final.csv")[0], 'rb')
L3_results = open(glob.glob(dataDir + "L3/*_Final.csv")[0], 'rb')
L4_results = open(glob.glob(dataDir + "L4/*_Final.csv")[0], 'rb')
L6_results = open(glob.glob(dataDir + "L6/*_Final.csv")[0], 'rb')
L7_results = open(glob.glob(dataDir + "L7/1st/*_Final.csv")[0], 'rb')

ensembled_results = open(dataDir + "Ensembled_results.csv", 'w+b')


line = L1_results.readline()
ensembled_results.write(line)

#throw away 1st line
L2_results.readline()
L3_results.readline()
L4_results.readline()
L6_results.readline()
L7_results.readline()

line = L1_results.readline()

val_list = list()
for x in range(len(weights)):
    val_list.append(0)
    
while line != "":
    img_id, val = line.split(',')
    val_list[0] = int(val[0])

    val_list[1] = int(L2_results.readline().split(',')[1][0])
    
    val_list[2] = int(L3_results.readline().split(',')[1][0])
    
    val_list[3] = int(L4_results.readline().split(',')[1][0])
    
    val_list[4] = int(L6_results.readline().split(',')[1][0])
    
    val_list[5] = int(L7_results.readline().split(',')[1][0])
    
    s = int(img_id.split("_")[0])    
    if s == 88:
        pass
    #result = calc_score(val_list, weights)
    
    
    
    if (val_list[0] == val_list[5]):
        result = val_list[5]
    else:
#         mode = stats.mode([val_list[0], val_list[3], val_list[5]], axis=None)
#         if int(mode[1]) > 1:
#             result = int(mode[0])
#        else:
        category_list = [0,0,0,0,0]
        weightIndex = 0
        for x in val_list:
            category_list[x] = category_list[x] + weights[weightIndex]
            weightIndex = weightIndex + 1
        
        result = np.argmax(category_list)    
    #end else
        
    ensembled_results.write(img_id + "," + str(result) + "\r\n")
    
    line = L1_results.readline()
#end while

L1_results.close()
L2_results.close()
L3_results.close()
L4_results.close()
L6_results.close()
L7_results.close()

ensembled_results.close()
    
    
    
    
