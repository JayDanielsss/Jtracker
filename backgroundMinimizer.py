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


#Testing the minimizer. Adding tracks to partial Tracks

partialTracks[1000]=[13,38]
partialTracks[1001]=[14,39]


# In[4]:


geomData = np.genfromtxt("ericsBasicGeo.txt.txt", dtype=float)
np.reshape(geomData, (30,27))
print(geomData)


# In[5]:


plt.scatter(partialTracks[:,0],partialTracks[:,1],marker='_')
plt.scatter(injectedTracks[:,0],injectedTracks[:,1],marker='|',color='red')
plt.xlim(13,18)
plt.ylim(0,200)

#note the injected track is from event 96817!



# In[6]:


#remove zeros
partialTracks = partialTracks[~np.all(partialTracks == 0, axis=1)]


# In[7]:


#after zero removed
plt.scatter(partialTracks[:,0],partialTracks[:,1],marker='_')
plt.scatter(injectedTracks[:,0],injectedTracks[:,1],marker='+',color='red')
plt.xlim(13,18)
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
det=13
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
plt.xlim(0,30)
plt.ylim(0,200)


# In[11]:


#Uwindow

Station = 2
if(Station == 1):
    UID=5
    XID=3
    VID=1
elif(Station == 2):
    UID=17
    XID=15
    VID=13

cosine=math.cos(.244)
sine=math.sin(.244)
tx=0.15
ty=0.1
delta=0.05
L=1.2192
URadius=abs((geomData[UID][11]*sine/2))+tx*abs(geomData[UID][1]-geomData[XID][1])*cosine+ty*abs(geomData[UID][1]-geomData[XID][1])*sine+2*geomData[UID][4]+delta
VRadius=tx*abs(geomData[UID][1]+geomData[VID][1]-2*geomData[XID][1])*cosine+ty*abs(geomData[UID][1]-geomData[VID][1])*sine+2*geomData[VID][4]
print(URadius)


# In[12]:


hitPairX=np.zeros((2000,2))
elemID=np.zeros((3,200))
#Station 3 tracking X planes:
det=13
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


#plt.scatter(partialTracks[:,0],partialTracks[:,1],marker='_')
plt.scatter(hitPairX[:,0],hitPairX[:,1],marker='_')
plt.scatter(injectedTracks[:,0],injectedTracks[:,1],color='red',marker='+')
plt.xlim(13,18)
plt.ylim(0,200)


# In[14]:


print("-=-=-=--=-=-=U plane=-=-=-=-=-")

i=0
j=0
k=0
NumberX=np.count_nonzero(elemID[1])
NumberU=np.count_nonzero(elemID[2])
NumberV=np.count_nonzero(elemID[0])

if(NumberX!=1):
    XPosition=np.zeros(NumberX)
else:
    XPosition=np.zeros(1)
ucenter=np.zeros(NumberU)
index=np.where(hitPairX[:,0]==16)
indexD=np.where(hitPairX[:,0]==18)

for i in range(NumberX):
    XPosition[i]=(hitPairX[index[0],1][i]-(geomData[XID][2]+1)/2)*geomData[XID][4]+geomData[XID][5]+geomData[XID][7]*geomData[XID][10]+geomData[XID][12]*geomData[XID][4]+geomData[XID][26]
    print(hitPairX[index[0],1][i])
    print(XPosition[i])
for j in range(NumberU):
    for k in range(len(XPosition)):
        ucenter[j]=XPosition[k]*cosine
        print(ucenter[j])
    


# In[15]:


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
    
indexU=np.where(hitPairX[:,0]==18)
indexV=np.where(hitPairX[:,0]==14)
indexX=np.where(hitPairX[:,0]==16)

vcenter=np.zeros(len(indexV[0]))


for i in range(NumberX):
    for j in range(NumberU):
        UPosition[j]=(hitPairX[indexU[0][i],1]-(geomData[UID][2]+1)/2)*(geomData[UID][4])+geomData[UID][5]+geomData[UID][7]*geomData[UID][10]+geomData[UID][12]*geomData[UID][4]+geomData[UID][26]
        print(UPosition[j])
        for k in range(NumberV):
            VPosition[k]=(hitPairX[indexV[0][k],1]-(geomData[VID][2]+1)/2)*(geomData[VID][4])+geomData[VID][5]+geomData[VID][7]*geomData[VID][10]+geomData[VID][12]*geomData[VID][4]+geomData[VID][26]
            print(VPosition)
            vcenter[k]=2*XPosition[i]*cosine-UPosition[j]
            print(vcenter[k])

            


# In[16]:


i=0


for i in range(NumberU):
       # if(hitPairX[indexD[0][i],1]<=ucenter[i]+URadius and hitPairX[indexD[0][i],1]>=ucenter[i]-URadius):
        if(UPosition[i]>ucenter[i]-URadius and UPosition[i]<ucenter[i]+URadius):

            print("hit confirmed in U", UPosition[i], "vs", ucenter[i]-URadius, "and", ucenter[i]+URadius)

        else:
            
            print("hit removed", hitPairX[indexU[0][i]])
            print("not a hit" , UPosition[i], "vs", ucenter[i]-URadius, "and", ucenter[i]+URadius)
            
            hitPairX[indexU[0][i]]=[0,0]
            hitPairX[indexU[0][i]-1]=[0,0]

i=0            
for i in range(NumberV):
    if(VPosition[i]>vcenter[i]-VRadius and VPosition[i]<vcenter[i]+VRadius):
        print("hit confirmed in V",VPosition[i], "vs", vcenter[i]-VRadius, "and", vcenter[i]+VRadius)

    else:
        print("hit removed in V",VPosition[i], "vs", vcenter[i]-VRadius, "and", vcenter[i]+VRadius)
        hitPairX[indexV[0][i]]=[0,0]
        hitPairX[indexV[0][i]-1]=[0,0]


# In[18]:


plt.scatter(hitPairX[:,0],hitPairX[:,1],marker='_')
plt.scatter(injectedTracks[:,0],injectedTracks[:,1],color='red',marker='+')
plt.xlim(13,18)
plt.ylim(0,200)
i=0
for i in range(10):
    print(hitPairX[i])


# In[23]:


hitPairX[index[0],1][1]


# In[34]:


indexV=np.where(hitPairX[:,0]==2)
print(indexV)


# In[7]:


partialTracks[1000]


# In[37]:


geomData[0][11]


# In[ ]:




