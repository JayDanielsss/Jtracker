#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import uproot
import math
import csv
import random
import itertools
from scipy import stats
import dearpygui.dearpygui as dpg



# In[2]:


c=+1#random.choice((+1,-1))


if(c==0):
    #comparison

    partialTracks=np.load("injectedTrackscompare2.npy")
    injectedTracks=np.load("injectedTrackscompare2.npy")

if(c==+1):
    partialTracks=np.load('partialTrackTests6.npy')
    #injectedTracks=np.load('injectedTrackTests3.npy')
    injectedTracks=np.load('injectedTracksTests6.npy')
    #injectedTrack2=np.load('injectedTrack2.npy')


    #partialTracks=np.load('partialTracks.npy')
    #injectedTracks=np.load('injectedTracks.npy')

if(c==-1):
    partialTracks=np.load('partialTrackTestsPositive.npy')
    injectedTracks=np.load('injectedTrackTestsPositive.npy')


# In[3]:


print("Here are the hits")
partialTracks


# In[4]:


#Testing the minimizer. Adding tracks to partial Tracks
testing = True

if(testing):
    #Gives a random number of partial tracks
    m=30#random.randrange(3,50)
    print(m)
    d=0
    for i in range(m):
        a=random.randrange(200)
        b=random.randrange(3)
        c=random.choice((1,3,5,13,15,17,19,21,23,25,27,29))
        partialTracks[200+d]=[c,a]
        print(partialTracks[200+d])
        partialTracks[200+d+1]=[c+1,a+b]
        print(partialTracks[200+d+1])
        d+=2
        print(d)


# In[5]:


#Parameters based on geometery of the detector. 
geomData = np.genfromtxt("ericsBasicGeo.txt.txt", dtype=float)
np.reshape(geomData, (30,27))
print(geomData)


# In[6]:


#remove zeros
partialTracks = partialTracks[~np.all(partialTracks == 0, axis=1)]


# In[7]:


#number of hits
nSt1=0
for i in range(7):
    print("detectorId =", i)
    print("hits on detector", np.count_nonzero(partialTracks[:,0]==i,axis=0))
    nSt1+=np.count_nonzero(partialTracks[:,0]==i,axis=0)
    print("total hits in station",nSt1)
    
nSt2=0
for i in range(13,19):
    print("detectorId =", i)
    print("hits on detector", np.count_nonzero(partialTracks[:,0]==i, axis=0))
    nSt2+=np.count_nonzero(partialTracks[:,0]==i,axis=0)
    print("total hits in station",nSt2)
    
nSt3m=0
for i in range(19,25):
    print("detectorId =", i)
    print("hits on detector", np.count_nonzero(partialTracks[:,0]==i, axis=0))
    nSt3m+=np.count_nonzero(partialTracks[:,0]==i,axis=0)
    print("total hits in station",nSt3m)

nSt3p=0
for i in range(25,31):
    print("detectorId =", i)
    print("hits on detector", np.count_nonzero(partialTracks[:,0]==i, axis=0))
    nSt3p+=np.count_nonzero(partialTracks[:,0]==i,axis=0)
    print("total hits in station",nSt3p)
nHits=nSt1+nSt2+nSt3m+nSt3p  
print("Hits in total:",nSt1+nSt2+nSt3m+nSt3p)
    


# In[8]:


#plt.scatter(partialTracks[:,0],partialTracks[:,1],marker='_')
print("This is the detector before Cleaning.")
fig,ax=plt.subplots()
ax.scatter(partialTracks[:,0],partialTracks[:,1],marker='_')
ax.scatter(injectedTracks[:,0],injectedTracks[:,1],color='red',marker='|')
plt.xlim(12,31)
plt.ylim(0,200)
plt.show()


# In[9]:


#Determines hit pairs through the whole detector

hitPairX=np.zeros((len(partialTracks)*6,2))
det=1
index=0
indexD=0
for k in range(15):
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
plt.scatter(injectedTracks[:,0],injectedTracks[:,1],color='red',marker='|')
plt.xlim(12,30)
plt.ylim(0,200)


# In[11]:


#math and values from paper


tx=.1
ty=.1

#cosine=math.cos(.244)
#sine=math.sin(.244)

#Station values

def Radius(Station):


    if(Station == 1):
        UID=4
        XID=2
        VID=0
        delta=5
        cosine=geomData[UID][10]
        sine=geomData[UID][15]

    elif(Station == 2):
        UID=16
        XID=15
        VID=12
        #V=14
        delta=5
        cosine=geomData[UID][10]
        sine=geomData[UID][15]

    elif(Station == 3):
        UID=23
        XID=21
        VID=19
        delta=15
        cosine=geomData[UID][10]
        sine=geomData[UID][15]

    elif(Station == 4):
        UID=29
        XID=27
        VID=25
        delta=15
        cosine=geomData[UID][10]
        sine=geomData[UID][15]


    URadius=abs((geomData[XID][11]*sine/2))+tx*abs(geomData[UID][1]-geomData[XID][1])*cosine+ty*abs(geomData[UID][1]-geomData[XID][1])*sine+2*geomData[UID][4]+delta
    
    
    
    VR1=geomData[UID][4]*2*cosine
    VR2=abs((geomData[UID][1]+geomData[VID][1]-2*geomData[XID][1])*cosine*tx)
    VR3=abs((geomData[VID][1]-geomData[UID][1])*sine*ty)
    VRadius=VR1+VR2+VR3+2*geomData[UID][4]
    
    
    #VRadius=tx*abs(geomData[UID][1]+geomData[VID][1]-2*geomData[XID][1])*cosine+ty*abs(geomData[UID][1]-geomData[VID][1])*sine+2*geomData[VID][4]+2*geomData[VID][4]*cosine
    print(URadius)
    print(VRadius)
    
    return(URadius,VRadius,VID,XID,UID, cosine, sine)


# In[12]:


def getPosition(ID,Number, index):
    
    Position=[]
    Position=np.append(Position,[((hitPairX[index,1]-(geomData[ID][2]+1)/2)*geomData[ID][4]+geomData[ID][5]+geomData[ID][7]*geomData[ID][10]+geomData[ID][12]*geomData[ID][15]+geomData[ID][26],index)])
   # print("elemid",hitPairX[index[0,1][i])

    #if(Number!=1):
       # Position=np.zeros((Number,2))
    #else:
     #   Position=np.zeros((1,2))
        
    #for i in range(Number):
        #Position[i]=(hitPairX[index[0],1][i]-(geomData[ID][2]+1)/2)*geomData[ID][4]+geomData[ID][5]+geomData[ID][7]*geomData[ID][10]+geomData[ID][12]*geomData[ID][4]+geomData[ID][26],index[0][i]
        #Position=np.append(Position,[((hitPairX[index[0],1][i]-(geomData[ID][2]+1)/2)*geomData[ID][4]+geomData[ID][5]+geomData[ID][7]*geomData[ID][10]+geomData[ID][12]*geomData[ID][15]+geomData[ID][26],index[0][i])])
        #print("elemid",hitPairX[index[0],1][i])
    Position=np.reshape(Position, (-1,2))
    return(Position)


# In[13]:


def getElemIDFromPos(ID,position):
    E=[]
    for i in range(len(position)):
        E=np.append(E,[((position[i]-(+geomData[ID][5]+geomData[ID][7]*geomData[ID][10]+geomData[ID][12]*geomData[ID][15]+geomData[ID][26]))/geomData[ID][4])+((geomData[ID][2]+1)/2)])
    return(E)


# In[14]:


def compareArray(a,b):
   # n = min(len(a), len(b))
    #out_idx = np.flatnonzero(a[:n] == b[:n])
    #out_val = a[out_idx]
    
    out_idx=[]
    for i in range(len(a)):
        for j in range(len(b)):
            if(a[i]==b[j]):
                out_idx=np.append(out_idx,[i])
                
    out_idx=np.unique(out_idx)
    
    
    return(out_idx)


# In[15]:


def buildTracklet(Tracklet,s):

    Tracklets=np.zeros((len(Tracklet),6,2))

    if(s==2):

        for l in range(len(Tracklet)):    
            Tracklets[l][0]=hitPairX[Tracklet[l,0].astype(int)]
            Tracklets[l][1]=hitPairX[Tracklet[l,0].astype(int)+1]

            Tracklets[l][2]=hitPairX[Tracklet[l,1].astype(int)-1]
            Tracklets[l][3]=hitPairX[Tracklet[l,1].astype(int)]

            Tracklets[l][4]=hitPairX[Tracklet[l,2].astype(int)]
            Tracklets[l][5]=hitPairX[Tracklet[l,2].astype(int)+1]

    if(s==3 or s==4):
        for l in range(len(Tracklet)):    

            Tracklets[l][0]=hitPairX[Tracklet[l,0].astype(int)-1]
            Tracklets[l][1]=hitPairX[Tracklet[l,0].astype(int)]

            Tracklets[l][2]=hitPairX[Tracklet[l,1].astype(int)-1]
            Tracklets[l][3]=hitPairX[Tracklet[l,1].astype(int)]

            Tracklets[l][4]=hitPairX[Tracklet[l,2].astype(int)-1]
            Tracklets[l][5]=hitPairX[Tracklet[l,2].astype(int)]









    

    return(Tracklets)


# In[16]:


#Setting what the planes are based off the station
print("This is the start of the loop")
#s=4
#file1 = open("comparelog.txt","w")

TrackletSt2=[]
TrackletSt3pm=[]
for s in range(2,5):
    URadius,VRadius,VID,XID,UID, cosine, sine=Radius(s)
    #Shows what station:
    print("we are in Station:", s)
    if(s==2):
        V=geomData[VID][0]
        X=V+3
        U=V+4
        
    
    else:
        
        V=geomData[VID][0]
        X=V+2
        U=V+4
    print(URadius,VRadius,VID,XID,UID,V,X,U)


   
    
    print("This is V:", V)
    print("This is X",X)
    print("This is U", U)

    i=0
    j=0
    k=0

    indexU=np.where(hitPairX[:,0]==U)
    indexV=np.where(hitPairX[:,0]==V)
    indexX=np.where(hitPairX[:,0]==X)
    print("IndexX",indexX)

    NumberX=np.count_nonzero(indexX)
    print("NumberX",NumberX)
    NumberU=np.count_nonzero(indexU)
    print("NumberU",NumberU)

    NumberV=np.count_nonzero(indexV)
    print("NumberV",NumberV)
    
    if(NumberU == 0 or NumberV == 0 or NumberX == 0):
        print("empty station")
        if(NumberU>0):
            for i in range(NumberU):
                hitPairX[indexU[0][i]]=[0,0]
                hitPairX[indexU[0][i]-1]=[0,0]
        if(NumberV>0):
            for i in range(NumberV):
                hitPairX[indexV[0][i]]=[0,0]
                hitPairX[indexV[0][i]-1]=[0,0]
        if(NumberX>0):
            for i in range(NumberX):
                hitPairX[indexX[0][i]]=[0,0]
                hitPairX[indexX[0][i]-1]=[0,0]
                
    TrackletXU=[]
    TrackletVXU=[]
    vcplt=[]
    ucplt=[]
                
    for i in range(NumberX):
        print("collecting the XPosition")
        XPosition=getPosition(XID,NumberX,indexX[0][i])
        print("This is XPosition: ", XPosition)

        print("Collecting UCenter")
        ucenter=[XPosition[0,0]*cosine,XPosition[0,1]]
        
        print("This is the ucenter: ", ucenter)    
        
        for j in range(NumberU):
            print("collecting the UPosition")
            UPosition=getPosition(UID,NumberU,indexU[0][j])
            print("This is UPosition: ", UPosition)

            print("making UX Track")
            if(UPosition[0,0]>ucenter[0]-URadius and UPosition[0,0]<ucenter[0]+URadius):
                print("hit!")
                print(UPosition[0,0],ucenter[0]-URadius,ucenter[0]+URadius,)   
                TrackletXU=np.append(TrackletXU,[ucenter[1],UPosition[0,1]])
                
            print("collecting vcenter")
            vcenter=[2*ucenter[0]-UPosition[0,0],ucenter[1],UPosition[0,1]]
            print("This is vcenter: ", vcenter)
            
            for k in range(NumberV):
                print("collecting the VPosition")
                VPosition=getPosition(VID,NumberV,indexV[0][k])
                print("This is VPosition: ", VPosition)
                
                if(VPosition[0,0]>vcenter[0]-VRadius and VPosition[0,0]<vcenter[0]+VRadius):
                    print("hit!")
                    print(VPosition[0,0],vcenter[0]-VRadius,vcenter[0]+VRadius,)
                    TrackletVXU=np.append(TrackletVXU,[VPosition[0,1],vcenter[1],vcenter[2]])
                    ucplt=np.append(ucplt,(ucenter))
                    vcplt=np.append(vcplt,(vcenter))

                    
    TrackletXU=np.reshape(TrackletXU,(-1,2))
    TrackletVXU=np.reshape(TrackletVXU,(-1,3))
    ucplt=np.reshape(ucplt,(-1,2))
    vcplt=np.reshape(vcplt,(-1,3))

    print("Lets combine the tracklets")

    Tracklets=buildTracklet(TrackletVXU,s)
    Tracklets

    if(s==2):
        print("saving Tracklets 2")

        TrackletSt2=np.append(TrackletSt2,Tracklets)

    if(s==3 or s==4):
        print("saving Tracklets 3pm")
        TrackletSt3pm=np.append(TrackletSt3pm,Tracklets)



#reshaping to organize properly
TrackletSt2=np.reshape(TrackletSt2,(-1,6,2))
TrackletSt3pm=np.reshape(TrackletSt3pm,(-1,6,2))


# In[17]:


print("plotting Station 2")

cmap = plt.cm.get_cmap("plasma")
marker = itertools.cycle((',', '+', '.', 'o', '*','_')) 

for i in range(len(TrackletSt2)):

    plt.scatter(TrackletSt2[i][:,0],TrackletSt2[i][:,1],marker=next(marker))


plt.scatter(injectedTracks[:,0],injectedTracks[:,1],color='k',marker='|')

plt.xlim(12,18.5)
plt.ylim(0,200)





# In[18]:


print("plotting Station 3mp")

cmap = plt.cm.get_cmap("plasma")
marker = itertools.cycle((',', '+', '.', 'o', '*','_')) 

for i in range(len(TrackletSt3pm)):

    plt.scatter(TrackletSt3pm[i][:,0],TrackletSt3pm[i][:,1],marker=next(marker))


plt.scatter(injectedTracks[:,0],injectedTracks[:,1],color='k',marker='|')

plt.xlim(18.5,30.5)
plt.ylim(0,200)





# In[19]:


# string to search in file
with open(r'comparison/log.txt', 'r') as fp, open('Kinfo.txt','w') as fk:
    # read all lines using readline()
    lines = fp.readlines()
    for row in lines:
        # check if string present on a current line
        word = 'jindex'
        #print(row.find(word))
        # find() method returns -1 if the value is not found,
        # if found it returns index of the first occurrence of the substring
        if row.find(word) != -1:
            print('string exists in file')
            print('line Number:', lines.index(row))
            fk.write(row)


# In[20]:


ktrackerData = np.genfromtxt('Kinfo.txt', delimiter=',')
ktrackerData = ktrackerData[~np.isnan(ktrackerData)]
ktrackerData = np.reshape(ktrackerData,(-1,9))
print(ktrackerData)


# In[21]:


ktdata=[]
for i in range(len(ktrackerData)):
    if(ktrackerData[i,8]==1):
        ktdata=np.append(ktdata,ktrackerData[i])


# In[22]:


ktrackerData = np.reshape(ktdata,(-1,9))


# In[23]:


kvcenter=[]
kucenter=[]
kumax=[]
kumin=[]
kvmax=[]
kvmin=[]
kupos=[]
kxpos=[]
kvpos=[]

for i in range(len(ktrackerData)):
    if(ktrackerData[i,0]==0.0):
        kxpos=np.append(kxpos,[ktrackerData[i,1],ktrackerData[i,7]])
    
    if(ktrackerData[i,0]==1.0):
        kupos=np.append(kupos,[ktrackerData[i,1],ktrackerData[i,7]])
        kumin=np.append(kumin,[ktrackerData[i,2],ktrackerData[i,7]])
        kumax=np.append(kumax,[ktrackerData[i,3],ktrackerData[i,7]])
        kucenter=np.append(kucenter,[ktrackerData[i,4],ktrackerData[i,7]])
        
    if(ktrackerData[i,0]==2.0):
        kvpos=np.append(kvpos,[ktrackerData[i,1],ktrackerData[i,7]])
        kvmin=np.append(kvmin,[ktrackerData[i,2],ktrackerData[i,7]])
        kvmax=np.append(kvmax,[ktrackerData[i,3],ktrackerData[i,7]])
        kvcenter=np.append(kvcenter,[ktrackerData[i,4],ktrackerData[i,7]])
        
kumax=np.reshape(kumax,(-1,2))
kumin=np.reshape(kumin,(-1,2))
kvmax=np.reshape(kvmax,(-1,2))
kvmin=np.reshape(kvmin,(-1,2))
kucenter=np.reshape(kucenter,(-1,2))
kvcenter=np.reshape(kvcenter,(-1,2))
kxpos=np.reshape(kxpos,(-1,2))
kupos=np.reshape(kupos,(-1,2))
kvpos=np.reshape(kvpos,(-1,2))





# In[24]:


kumax=np.delete(kumax,np.where(kumax[:,1]!=s),0)
kumin=np.delete(kumin,np.where(kumin[:,1]!=s),0)
kvmax=np.delete(kvmax,np.where(kvmax[:,1]!=s),0)
kvmin=np.delete(kvmin,np.where(kvmin[:,1]!=s),0)
kucenter=np.delete(kucenter,np.where(kucenter[:,1]!=s),0)
kvcenter=np.delete(kvcenter,np.where(kvcenter[:,1]!=s),0)

kxpos=np.delete(kxpos,np.where(kxpos[:,1]!=s),0)
kupos=np.delete(kupos,np.where(kupos[:,1]!=s),0)
kvpos=np.delete(kvpos,np.where(kvpos[:,1]!=s),0)




# In[25]:


def getRatio(ktracker,python):
    ratio=ktracker/python
    
    plt.hlines(y=ratio[0],xmin=0,xmax=1,color='black',linestyle='-')
    plt.hlines(y=ratio[1],xmin=0,xmax=1,color='black',linestyle='-')
    plt.hlines(y=1,xmin=0,xmax=1,color='red',linestyle='-')

    return(ratio)


# In[26]:


#vMin=vcenter-VRadius
#vMax=vcenter+VRadius

#getRatio(kvcenter[:,0],vcenter[:,0])


# In[27]:


print("plotting Station 3pm")

cmap = plt.cm.get_cmap("plasma")
marker = itertools.cycle((',', '+', '.', 'o', '*','_')) 

#for i in range(len(TrackletSt3pm)):

    #plt.scatter(TrackletSt3pm[i][:,0],TrackletSt3pm[i][:,1],marker=next(marker))

#Python

print("lets get plottable centers")
vc=getElemIDFromPos(VID,vcplt[:,0])
uc=getElemIDFromPos(UID,ucplt[:,0])

for i in range(len(vc)):    
    plt.hlines(y=vc[i],xmin=(VID-1),xmax=(VID+1),color='black',linestyle='-')
for i in range(len(uc)):    
    plt.hlines(y=uc[i],xmin=(UID-1),xmax=(UID+1),color='black',linestyle='-')
    
vmin=getElemIDFromPos(VID,vcplt[:,0]-VRadius)
vmax=getElemIDFromPos(VID,vcplt[:,0]+VRadius)

umin=getElemIDFromPos(UID,ucplt[:,0]-URadius)
umax=getElemIDFromPos(UID,ucplt[:,0]+URadius)
for i in range(len(vmin)):
    plt.vlines(x = VID, ymin = vmin[i], ymax = vmax[i], colors = 'black', label ='V window')
    
for i in range(len(umin)):
    plt.vlines(x = UID, ymin = umin[i], ymax = umax[i], colors = 'black', label ='U window')

#ktracker


kvc=getElemIDFromPos(VID,kvcenter[:,0])
kuc=getElemIDFromPos(UID,kucenter[:,0])

kvMin=getElemIDFromPos(VID,kvmin[:,0])
kvMax=getElemIDFromPos(VID,kvmax[:,0])

kuMin=getElemIDFromPos(UID,kumin[:,0])
kuMax=getElemIDFromPos(UID,kumax[:,0])

for i in range(len(kvc)):
    plt.hlines(y=kvc[i],xmin=VID-1,xmax=VID+1,color='red',linestyle='-')
for i in range(len(kuc)):
    plt.hlines(y=kuc[i],xmin=UID-1,xmax=UID+1,color='red',linestyle='-')
for i in range(len(kvMin)):
    plt.vlines(x = VID, ymin = kvMin[i], ymax = kvMax[i], colors = 'red', label ='U window')

for i in range(len(kuMin)):
    plt.vlines(x = UID, ymin = kuMin[i], ymax = kuMax[i], colors = 'red', label ='U window')

plt.scatter(injectedTracks[:,0],injectedTracks[:,1],color='red',marker='+')

plt.xlim(12,30)
plt.ylim(0,200)


# In[32]:


cmap = plt.cm.get_cmap("plasma")
marker = itertools.cycle((',', '+', '.', 'o', '*','_')) 
plt.scatter(injectedTracks[:,0],injectedTracks[:,1],color='k',marker='d')

for i in range(len(TrackletSt2)):

    plt.scatter(TrackletSt2[i][:,0],TrackletSt2[i][:,1],marker=next(marker))
    
for i in range(len(TrackletSt3pm)):

    plt.scatter(TrackletSt3pm[i][:,0],TrackletSt3pm[i][:,1],marker=next(marker))
    

plt.xlim(12,30.5)
plt.ylim(0,200)

