import operator
import math
import pandas as pd
import numpy as np
import operator
from scipy import spatial
import os
def split_data(filename):
    test_file = open('/home/saliency/data_mining/ml-1m/test_file','a+')
    train_file = open('/home/saliency/data_mining/ml-1m/train_file','a+')
    with open(filename) as f:
        lines= f.readlines()
    content = []
    for i in range(0,len(lines)):
        content.append(lines[i].split("::"))
    usr = []
    for i in range(0,len(lines)):
        usr.append(content[i][0])
    for uindex in range(1,6041):
        ulen = usr.count(str(uindex))
        train_len = int(ulen*0.9)
        total = []
        for i in range(0,len(lines)):
            if content[i][0] == str(uindex):
                total.append(content[i])
        total = sorted(total, key=operator.itemgetter(3), reverse=False)
        for k in range(0,len(total)):
            if k>train_len-1:
                test_file.write(total[k][0]+'::'+total[k][1]+'::'+total[k][2]+'::'+total[k][3])
            else:
                train_file.write(total[k][0]+'::'+total[k][1]+'::'+total[k][2]+'::'+total[k][3])
    test_file.close()
    train_file.close()


def data_dict(filename):
    with open(filename) as f:
        lines = f.readlines()
    ##obtain dataset{{'1':{'mID':rate,...}...}}
    userindex_dataset = {}
    lines = [x.strip('\n').split("::") for x in lines]
    usr = set([x[0] for x in lines])
    usr = list(usr)
    for i in range(0,len(usr)):
        print usr[i]
        value = {}
        for k in range(0,len(lines)):
            if(lines[k][0]==usr[i]):
                value[lines[k][1]]=lines[k][2]
        userindex_dataset[usr[i]]= value
    return userindex_dataset

def build_dataset():
    df = pd.read_csv("/home/saliency/data_mining/ml-1m/train_file",sep = '::',header=None)
    df.columns = ['user','mID','rating','timestamp']
    dataset = df.pivot(index='user', columns='mID', values='rating')
    ##dataset.ix[1][x] represents the x movie rated by user 1
    ##dataset[1][u] represents the first movie rated by user u
    return dataset

def find_similar(dataset,ins,kind,user_cluster):
    #kind = 0: find the most similar user to ins
    #kind = 1: find the most similar item to ins
    mrating= {}
    if kind == 0:
        for i in range(0,len(user_cluster)):
            k = user_cluster[i]
            target = [x for x in dataset.ix[k]]
            target = [0 if np.isnan(x) else x for x in target]
            mrating[k] = target
        #mrating[1] is all the rating value of the first user
    else:
        for i in range(1,len(dataset.ix[1])+1):
            if i in dataset:
                target = [x for x in dataset[i]]
                target = [0 if np.isnan(x) else x for x in target]
                mrating.append(target)
            else:
                mrating.append([0]*len(dataset[1]))
    value = {}
    for item in mrating:
        mrating[item] = [x - float(sum(mrating[item]))/float((len(mrating[item])-mrating[item].count(0)))   if x !=0 else 0 for x in mrating[item]]
    for item in mrating:
        if item == ins:
            continue
        else:
            value[item] = 1 - spatial.distance.cosine(mrating[ins],mrating[item])
    ordered=dict(sorted(value.items(), key=lambda x: x[1],reverse=True)[:20]) 
    return ordered 



def cal_similar_user(data,uID,mID):
    user_cluster = []
    os.chdir("/home/saliency/data_mining/user_cluster_20")
    name = []
    for filename in os.listdir("."):
        name.append(filename)
    for fn in name:
        with open(fn) as f:
            lines = f.readlines()
            lines = [x.strip('\n') for x in lines]
            lines = [int(x) for x in lines]
        user_cluster.append(lines)
    for i in range(0,len(user_cluster)):
        if uID in user_cluster[i]:
            similar_user = user_cluster[i]
    order = find_similar(data,uID,0,similar_user)
    rate = {}
    for item in order:
        value = data.ix[item][mID]
        if np.isnan(value):
            value = 0
        rate[item] = value
    fenzi = 0
    fenmu = 0
    for item in order:
        fenzi = fenzi + order[item]*rate[item]
        fenmu = fenmu + order[item]
    rate = fenzi/fenmu
    return rate


def user_result():
    target = open("/home/saliency/data_mining/ml-1m/user_user_result",'a+')
    with open("/home/saliency/data_mining/ml-1m/train_file") as f:
        lines = f.readlines()
    lines = [x.strip('\n').split("::") for x in lines]
    data = build_dataset()
    cal_rate = []
    real_rate = []
    for i in range(0,len(lines)):
        lines[i] = [int(x) for x in lines[i]]
    for i in range(0,len(lines)):
        uID= lines[i][0]
        mID= lines[i][1]
        res = cal_similar_user(data,uID,mID)
        cal_rate.append(res)
        target.write(str(lines[i][0])+'::'+str(lines[i][1])+"::"+str(res)+"::"str(lines[i][3])+'\n')
        real_rate.append(lines[i][2])
    final = 0
    target.close()
    for i in range(cal_rate):
        final = final + pow((cal_rate[i]-real_rate[i]),2)
    final = final/len(lines)
    final = math.sqrt(final)
    return final



def split(a, n):
    k, m = len(a) / n, len(a) % n
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))

def cal_bin():
    with open("/home/saliency/data_mining/ml-1m/train_file") as f:
        lines = f.readlines()
    lines = [x.strip('\n').split("::") for x in lines]
    movieID =  set([x[1] for x in lines])
    movie = list(movieID)
    movie_time = {}
    for k in range(0,len(movie)):
        time = []
        value ={}
        inner = {}
        for i in range(0,len(lines)):
            if lines[i][1] == movie[k]:
                time.append(int(lines[i][3]))
                value[int(lines[i][3])] = int(lines[i][2])
        #movie_time[int(movie[k])]= time
        time.sort()
        time = list(split(time,15))
        for r in range(0,len(time)):
            z = time[r]
            total = 0
            for j in range(0,len(z)):
                total = total + value[z[j]]
            inner[r] = [z,total]
        movie_time[int(movie[k])] = inner
    return movie_time

def dev_u(uID,t):
    with open("/home/saliency/data_mining/ml-1m/train_file") as f:
        lines = f.readlines()
    lines = [x.strip('\n').split("::") for x in lines]
    for i in range(0,len(lines)):
        lines[i] = [int(x) for x in lines[i]]
    date = []
    rate = []
    unum = 0
    for i in range(0,len(lines)):
        lines[i] = [int(x) for x in lines[i]]
    for i in range(0,len(lines)):
        if lines[i][0] == uID:
            date.append(lines[i][3])
            rate.append(lines[i][2])
            unum = unum +1
    mid = np.median(date)
    mean = np.mean(date)
    ###calculate alpha_u
    if mean>mid:
        alpha_u = mid/mean
    else:
        alpha_u = mean/mid
    
    ###calculate sign
    if t<mean:
        sign = -1
    elif t==mean:
        sign =0
    elif t>mean:
        sign = 1
    
    ###calculate bu_sum:
    bu_sum = sum(rate)
    z = (t-mid)/60/60/24
    dev = sign* pow(abs(z),0.4)
    return bu_sum,alpha_u,unum,dev

def cal_baseline(usr,mID,date,mbin):
    with open("/home/saliency/data_mining/ml-1m/train_file") as f:
        lines = f.readlines()
    lines = [x.strip('\n').split("::") for x in lines]
    for i in range(0,len(lines)):
        lines[i] = [int(x) for x in lines[i]]
    sum = 0
    for i in range(0,len(lines)):
        sum = sum + lines[i][2]
    
    ##cal miu
    miu = sum / len(lines)
    bu_sum,alpha_u,unum,dev= dev_u(usr,date)
    bu = bu_sum/unum - miu
    
    print "dev OK"

    ##movie_bin
#   mbin = cal_bin()
    print "bin OK"
    msum = 0
    mnum = 0
    bshift = -1
    for i in range(0,15):
        msum = msum+mbin[mID][i][1]
        mnum = mnum + len(mbin[mID][i][0])
        if date in (mbin[mID][i][0]) == 1 :
            bshift = mbin[mID][i][1] / len(mbin[mID][i][0])
    if bshift == -1:
        bshift = mbin[mID][14][1]/ len(mbin[mID][14][0])
    bshift = bshift-miu
    bi = (msum / mnum) -miu
    ##bshift is bi,Bin(t)
    print miu, bu, alpha_u,dev,bi,bshift
    baseline = miu + bu + alpha_u*dev + bi *bshift
    return baseline

def cal_rate(usr,ins,date,mbin):
    data = build_dataset()
    order = find_similar(data,ins,1)
    fenzi = []
    fenmu = []
    for item in order:
        baseline = cal_baseline(usr,item,date,mbin)
        similar = order[item]
        rate = data.ix[usr][item]
        if np.isnan(rate):
            rate = 0
        fenzi.append((rate-baseline)*similar)
        fenmu.append(similar)
    base_target = cal_baseline(usr,ins,date,mbin)
    res = base_target+(sum(fenzi)/sum(fenmu))
    return res

def item_result():
    with open("/home/saliency/data_mining/ml-1m/train_file") as f:
        lines = f.readlines()
    for i in range(0,len(lines)):
        lines[i] = [int(x) for x in lines[i]]
    mbin = cal_bin()
    cal_rate = []
    real_rate = []
    for i in range(0,len(lines)):
        uID = lines[i][0]
        mID = linse[i][1]
        date = lines[i][3]
        res = cal_rate(uID,mID,date,mbin)
        cal_rate.append(res)
        real_rate.append(lines[i][2])
        target.write(str(lines[i][0])+'::'+str(lines[i][1])+"::"+str(res)+str(lines[i][3])+'\n')
    final = 0
    for i in range(0,len(cal_rate)):
        final = final + pow((cal_rate[i]-real_rate[i]),2)
    final = final/len(lines)
    final = math.sqrt(final)
    target.write("RMSE: "+str(final))
    target.close()
    return final




