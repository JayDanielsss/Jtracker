#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt
import uproot
import random
import numba


# In[2]:


targettree = uproot.open('Background/singMum_x2y2z300_370K.root:QA_ana')
targetdata = targettree.arrays(library="np")


# In[3]:


detectorid=targettree["detectorID"].arrays(library="np")["detectorID"]
elementid=targettree["elementID"].arrays(library="np")["elementID"]


# In[4]:


@numba.jit(nopython=True)
def clean(events):
    for j in range(len(events)):
        for i in range(500):
            if(events[j][i]>1000):
                events[j][i]=0
    return events

#Clean input data
elementid=clean(elementid)
detectorid=clean(detectorid)


# In[5]:


plt.title("HitMatrix")
#plt.xlim(0,30)
#plt.ylim(0,200)
plt.scatter(detectorid[0],elementid[0],marker='_')


# In[24]:


ptracksElem=np.zeros(500)
ptracksDet=np.zeros(500)
m=random.randrange(0,50) #Random number of partial tracks to input
for i in range(m):
        for j in range(200):
            n=random.randrange(len(elementid)) #Select random event number
            st=random.randrange(1,79) #Selects station
            r=random.randrange(0,6)
            if(st<=34):
                for l in range(0,6):
                    ptracksElem[j]=elementid[n][l]
                    ptracksDet[j]=detectorid[n][l]
                for k in range(random.randrange(0,4)):    
                    ptracksElem[j]=elementid[n][l+r]
                    ptracksDet[j]=detectorid[n][l+r]
                    ptracksElem[j]=elementid[n][l-r]
                    ptracksDet[j]=detectorid[n][l-r]
                    #print(ptracks[j])
            if(st>34 and st<=62):
                for l in range(12,18):
                    ptracksElem[j]=elementid[n][l]
                    ptracksDet[j]=detectorid[n][l]
                for k in range(random.randrange(0,4)):    
                    ptracksElem[j]=elementid[n][l+r]
                    ptracksDet[j]=detectorid[n][l+r]
                    ptracksElem[j]=elementid[n][l-r]
                    ptracksDet[j]=detectorid[n][l-r]
            if(st>62):
                for l in range(18,30):
                    ptracksElem[j]=elementid[n][l]
                    ptracksDet[j]=detectorid[n][l]
                for k in range(random.randrange(0,4)):    
                    ptracksElem[j]=elementid[n][l+r]
                    ptracksDet[j]=detectorid[n][l+r]
                    ptracksElem[j]=elementid[n][l-r]
                    ptracksDet[j]=detectorid[n][l-r]


# In[25]:


plt.title("HitMatrix")
#plt.xlim(0,30)
#plt.ylim(0,25)
plt.scatter(ptracksDet[:],ptracksElem[:],marker='_')


# In[ ]:




