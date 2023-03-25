#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import uproot
import math
import csv


# In[2]:


partialTracks=np.load('partialTracks.npy')
injectedTracks=np.load('injectedTracks.npy')

#partialTracks=np.load('partialTrackTests.npy')
#injectedTracks=np.load('injectedTrackTests.npy')


# In[3]:


#Actual track
#targettree = uproot.open('Background/singMum_x2y2z300_370K.root:QA_ana')
#targetdata = targettree.arrays(library="np")


# In[4]:


geomData = np.genfromtxt("ericsBasicGeo.txt.txt", dtype=float)
np.reshape(geomData, (30,27))
print(geomData)


# In[5]:


plt.scatter(partialTracks[:,0],partialTracks[:,1],marker='_')
plt.scatter(injectedTracks[:,0],injectedTracks[:,1],marker='|',color='red')
plt.xlim(0,6)
plt.ylim(0,200)

#note the injected track is from event 96817!



# In[6]:


#remove zeros
partialTracks = partialTracks[~np.all(partialTracks == 0, axis=1)]


# In[7]:


#after zero removed
plt.scatter(partialTracks[:,0],partialTracks[:,1],marker='_')
plt.scatter(injectedTracks[:,0],injectedTracks[:,1],marker='+',color='red')
plt.xlim(0,6)
plt.ylim(0,200)


# In[8]:


#number of hits
nSt1=0
for i in range(7):
    print("detectorId =", i)
    print("hits on detector", np.count_nonzero(partialTracks==i,axis=0))
    nSt1+=np.count_nonzero(partialTracks==i,axis=0)
    print("total hits in station",nSt1)
    
nSt2=0
for i in range(13,19):
    print("detectorId =", i)
    print("hits on detector", np.count_nonzero(partialTracks==i, axis=0))
    nSt2+=np.count_nonzero(partialTracks==i,axis=0)
    print("total hits in station",nSt2)
    
nSt3=0
for i in range(19,31):
    print("detectorId =", i)
    print("hits on detector", np.count_nonzero(partialTracks==i, axis=0))
    nSt3+=np.count_nonzero(partialTracks==i,axis=0)
    print("total hits in station",nSt3)
nHits=nSt1+nSt2+nSt3  
print("Hits in total:",nSt1+nSt2+nSt3)
    


# In[9]:


hitPairX=np.zeros((2000,2))
#Station 3 tracking X planes:
det=1
index=0
indexD=0
for k in range(3):
    detLeft=np.where(partialTracks[:,0]==det)
    detRight=np.where(partialTracks[:,0]==det+1)

    for i in range(len(detLeft[0])):
        Xp=partialTracks[detLeft[0][i]][1]
        for j in range(len(detRight[0])):
            X=partialTracks[detRight[0][j]][1]
            elemDist=abs(Xp-X)
            if(elemDist<=1):
                print("Hit Pair Found", elemDist, Xp, X)
                hitPairX[index,0]=partialTracks[detLeft[0][i],0]
                hitPairX[index,1]=partialTracks[detLeft[0][i],1]
                hitPairX[index+1,0]=partialTracks[detRight[0][j],0]
                hitPairX[index+1,1]=partialTracks[detRight[0][j],1]
                print("Det Id:",partialTracks[detLeft[0][i],0],partialTracks[detRight[0][j],0])
                print("Elem Id:",partialTracks[detLeft[0][i],1],partialTracks[detRight[0][j],1])
                print("Elem Diff", elemDist)
                print("index",index)
                index+=2   

            else:
                print("Not a Hit pair:",elemDist,Xp)

    det+=2
    print(det)
#_+_+_+_+_+_+_+_+_+_+_+



# In[10]:


#plt.scatter(partialTracks[:,0],partialTracks[:,1],marker='_')
plt.scatter(hitPairX[:,0],hitPairX[:,1],marker='_')
plt.scatter(injectedTracks[:,0],injectedTracks[:,1],color='red',marker='+')
plt.xlim(0,6)
plt.ylim(0,200)


# In[11]:


#Uwindow

cosine=math.cos(.244)
sine=math.sin(.244)
tx=0.15
ty=0.1
delta=0.05
L=1.2192
URadius=(L*sine/2)+tx*abs(geomData[5][1]-geomData[3][1])*cosine+ty*abs(geomData[5][1]+geomData[3][1])*sine+2*geomData[5][4]+delta
VRadius=tx*abs(geomData[5][1]+geomData[1][1]-2*geomData[3][1])*cosine+ty*abs(geomData[5][1]-geomData[1][1])*sine+2*geomData[1][4]
print(VRadius)


# In[12]:


hitPairX=np.zeros((2000,2))
elemID=np.zeros((3,200))
#Station 3 tracking X planes:
det=1
index=0
indexD=0
for k in range(3):
    detLeft=np.where(partialTracks[:,0]==det)
    detRight=np.where(partialTracks[:,0]==det+1)
    indexD=0

    for i in range(len(detLeft[0])):
        Xp=partialTracks[detLeft[0][i]][1]
        for j in range(len(detRight[0])):
            X=partialTracks[detRight[0][j]][1]
            
            elemDist=abs(Xp-X)
            if(elemDist<=1):
                print("Hit Pair Found", elemDist, Xp, X)
                hitPairX[index,0]=partialTracks[detLeft[0][i],0]
                hitPairX[index,1]=partialTracks[detLeft[0][i],1]
                hitPairX[index+1,0]=partialTracks[detRight[0][j],0]
                hitPairX[index+1,1]=partialTracks[detRight[0][j],1]
                print("Det Id:",partialTracks[detLeft[0][i],0],partialTracks[detRight[0][j],0])
                print("Elem Id:",partialTracks[detLeft[0][i],1],partialTracks[detRight[0][j],1])
                print("Elem Diff", elemDist)
                print("index",index)
                elemID[k][indexD]=X
                index+=2
                indexD+=1

            else:
                print("Not a Hit pair:",elemDist,Xp)

    det+=2
    print(det)
print("-=-=--=-=-=-=-=-=-=-==-=-=-=-=-")  





# In[13]:


print("-=-=-=--=-=-=U plane=-=-=-=-=-")

i=0
j=0
NumberX=np.count_nonzero(elemID[1])
NumberU=np.count_nonzero(elemID[2])
NumberV=np.count_nonzero(elemID[0])

if(NumberX!=1):
    XPosition=np.zeros(len(NumberX))
else:
    XPosition=np.zeros(1)
ucenter=np.zeros(len(XPosition))
index=np.where(hitPairX[:,0]==4)
indexD=np.where(hitPairX[:,0]==6)

for i in range(len(index[0])):
    for j in range(NumberX):
        XPosition[j]=hitPairX[index[i],1]-((geomData[3][2]+1)/2)*(geomData[3][4])+geomData[3][5]+geomData[3][7]*geomData[3][10]+geomData[3][12]*geomData[3][4]+geomData[3][26]
        ucenter[j]=XPosition[j]*cosine
        print(ucenter[j])
    


# In[14]:


print("-=-=-=--=-=-=V plane=-=-=-=-=-")

i=0
j=0
k=0


if(NumberU!=1):
    UPosition=np.zeros(NumberU)
else:
    UPosition=np.zeros(1)
    
if(NumberV!=1):
    VPosition=np.zeros(NumberV)
else:
    VPosition=np.zeros(1)
    
indexU=np.where(hitPairX[:,0]==6)
indexV=np.where(hitPairX[:,0]==2)
indexX=np.where(hitPairX[:,0]==4)

vcenter=np.zeros(len(indexV[0]))

for i in range(NumberU):
    UPosition[i]=hitPairX[indexX[0][i],1]-((geomData[5][2]+1)/2)*(geomData[5][4])+geomData[5][5]+geomData[5][7]*geomData[5][10]+geomData[5][12]*geomData[5][4]+geomData[5][26]
    print(UPosition[i])
    for j in range(NumberV):
        VPosition[j]=hitPairX[indexV[0][j],1]-((geomData[1][2]+1)/2)*(geomData[1][4])+geomData[1][5]+geomData[1][7]*geomData[1][10]+geomData[1][12]*geomData[1][4]+geomData[1][26]
        print(VPosition)
        for k in range(NumberX):
            for l in range(NumberU):
                vcenter[j]=2*XPosition[k]*cosine-UPosition[l]
                print(vcenter[j])

            


# In[15]:


i=0


for i in range(NumberU):
       # if(hitPairX[indexD[0][i],1]<=ucenter[i]+URadius and hitPairX[indexD[0][i],1]>=ucenter[i]-URadius):
        if(UPosition[i]>=ucenter[i]-URadius and UPosition[i]<=ucenter[i]+URadius):

            print("hit confirmed in U")

        else:
            
            print("hit removed", hitPairX[indexU[0][i]])
            print("not between", ucenter[i]+URadius, "and", ucenter[i]-URadius)
            
            hitPairX[indexU[0][i]]=[0,0]
            hitPairX[indexU[0][i]-1]=[0,0]

i=0            
for i in range(NumberV):
    if(VPosition[i]>=vcenter[i]-VRadius and VPosition[i]<=vcenter[i]+VRadius):
        print("hit confirmed in V")
        print (vcenter[i])

    else:
        print("hit removed")
        print("Hits", hitPairX[indexV[0][i]], vcenter[i]+VRadius, vcenter[i]-VRadius)
        hitPairX[indexV[0][i]]=[0,0]
        hitPairX[indexV[0][i]-1]=[0,0]


# In[16]:


plt.scatter(hitPairX[:,0],hitPairX[:,1],marker='_')
plt.scatter(injectedTracks[:,0],injectedTracks[:,1],color='red',marker='+')
plt.xlim(0,6)
plt.ylim(0,200)
i=0
for i in range(10):
    print(hitPairX[i])


# In[ ]:




