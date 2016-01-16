from __future__ import division
import matplotlib.pylab as plt
import numpy as np
import random
import pandas as pd
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn import preprocessing

class ts_cluster(object):
    def __init__(self,num_clust=100):
        '''
        num_clust is the number of clusters for the k-means algorithm
        assignments holds the assignments of data points (indices) to clusters
        centroids holds the centroids of the clusters
        '''
        self.num_clust=num_clust
        self.assignments={}
        self.centroids=[]
    
    def compa_clust(self,s1,centroid,w):        
        centroid_part = centroid[:,:5]
        self.assign = pd.Series([[] for i in range(len(centroid))],index=np.arange(len(centroid)))

        for ind,i in enumerate(s1):
            min_dist=float('inf')
            #closest_clust=None
            for c_ind,j in enumerate(centroid_part):
                    
                if self.LB_Keogh(i,j,5)<min_dist:
                    cur_dist=self.SpatioTemporalDis(i, j, w)
                        
                    if cur_dist<min_dist:
                        min_dist=cur_dist
                        closest_clust=c_ind
            
            
            self.assign[closest_clust].append(ind)

        print self.assign
        #print s1[1]
        self.s2 = pd.Series([[] for i in range(len(s1))],index=np.arange(len(s1)))
        #print self.s2
        for key in self.assign.index:
            for k in self.assign[key]:
                self.s2[k] = centroid[key,6:].tolist()
                #print self.s2[k]
        self.s2 = np.array(self.s2)
        return self.s2
        
    
    
    def k_means_clust(self,data,num_iter,w,progress=True):
        
        '''
        k-means clustering algorithm for time series data.  dynamic time warping Euclidean distance
         used as default similarity measure. 
        '''
        self.centroids=random.sample(data,self.num_clust)
        print len(self.centroids)
        for n in range(num_iter):
            if progress:
                print 'iteration '+str(n+1)
            
            self.assignments={}
            for ind,i in enumerate(data):
                min_dist=float('inf')
                #closest_clust=None
                for c_ind,j in enumerate(self.centroids):
                    if self.LB_Keogh(i,j,5)<min_dist:
                        cur_dist=self.SpatioTemporalDis(i, j, w)
                        
                        if cur_dist<min_dist:
                            min_dist=cur_dist
                            closest_clust=c_ind
                
                if closest_clust in self.assignments:
                    self.assignments[closest_clust].append(ind)
                else:
                    self.assignments[closest_clust]=[]
                    
            print len(self.assignments)
            #recalculate centroids of clusters
            for key in self.assignments:
                clust_sum=0
                for k in self.assignments[key]:
                    clust_sum=clust_sum+data[k]
                self.centroids[key]=[m/len(self.assignments[key]) for m in clust_sum]
            
    def get_centroids(self):
        return self.centroids
        
    def get_assignments(self):
        return self.assignments
        
    def plot_centroids(self):
        for i in self.centroids:
            plt.plot(i)
        plt.show()
        
    def SpatioTemporalDis(self,s1,s2,w):
        
        f1,f2 = s1[0:2],s2[0:2]
        v1,v2 = s1[2:],s2[2:]
        disInvari = self.InvarDistance(f1, f2)
        disVari   = self.DTWDistance(v1, v2, w)
        
        return np.sqrt(disInvari+disVari)
    
    
    def InvarDistance(self,f1,f2):
        '''calculates the invariant features distance using Euclidean distance'''
         
        dis = spatial.distance.euclidean(f1, f2)
        
        return dis
        
    def DTWDistance(self,s1,s2,w=None):
        '''
        Calculates dynamic time warping Euclidean distance between two
        sequences. Option to enforce locality constraint for window w.
        '''
        DTW={}
        
        if w:
            w = max(w, abs(len(s1)-len(s2)))
    
            for i in range(-1,len(s1)):
                for j in range(-1,len(s2)):
                    DTW[(i, j)] = float('inf')
            
        else:
            for i in range(len(s1)):
                DTW[(i, -1)] = float('inf')
            for i in range(len(s2)):
                DTW[(-1, i)] = float('inf')
        
        DTW[(-1, -1)] = 0
        
        for i in range(len(s1)):
            if w:
                for j in range(max(0, i-w), min(len(s2), i+w)):
                    dist= (s1[i]-s2[j])**2
                    DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
            else:
                for j in range(len(s2)):
                    dist= (s1[i]-s2[j])**2
                    DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
            
        return (DTW[len(s1)-1, len(s2)-1])      
        
    def LB_Keogh(self,s1,s2,r):
        '''
        Calculates LB_Keough lower bound to dynamic time warping. Linear
        complexity compared to quadratic complexity of dtw.
        '''
        LB_sum=0
        for ind,i in enumerate(s1):
            
            lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
            upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
            
            if i>upper_bound:
                LB_sum=LB_sum+(i-upper_bound)**2
            elif i<lower_bound:
                LB_sum=LB_sum+(i-lower_bound)**2
            
        return np.sqrt(LB_sum)
    
if __name__ == "__main__":
    
    ''''x=np.linspace(0,50,100)
    ts1=np.array(pd.Series(3.1*np.sin(x/1.5)+3.5))
    ts2=np.array(pd.Series(2.2*np.sin(x/3.5+2.4)+3.2))
    ts3=np.array(pd.Series(0.04*x+3.0))
    data = np.vstack((ts1,ts2,ts3))
    
    train = np.genfromtxt('train111.csv', delimiter='\t')
    data1 = np.vstack((train[:,:-1]))
    print (data1).shape
    '''
    '''
    data = pd.read_csv('train.csv')
    data2 = pd.read_csv('test.csv')
    
    #print data2.iloc[40000:].shape
    #data = data.dropna(axis=1, thresh=10000) #if Nan is over 10000, then drop this column
    for col in data.columns:
        data[col]  = data[col].fillna(data[col].median()) 
    for col in data2.columns:
        data2[col] = data2[col].fillna(data2[col].median())
    #print data2.loc[:,'Ret_MinusTwo']
    #print data2.loc[40000:,'Ret_MinusTwo']
    features ,features2 = data.loc[:,'Feature_2':'Feature_25'],data2.loc[:,'Feature_2':'Feature_25']
    features  = preprocessing.StandardScaler().fit(features).transform(features)
    features2 = preprocessing.StandardScaler().fit(features2).transform(features2)
    features ,features2 = np.array(features), np.array(features2)
      
    pca = PCA(n_components=2) #use PCA to decompose the features
    features  = pd.DataFrame(pca.fit_transform(features)) 
    features2 = pd.DataFrame(pca.fit_transform(features2))
    
    train = pd.concat((features, data.loc[:,'Ret_MinusTwo':'Ret_2'],data.loc[:,'Ret_180':'Ret_PlusTwo']),axis =1)
    #train = np.array(train)
    test  = pd.concat((features2, data2.loc[:,'Ret_MinusTwo':'Ret_2']),axis = 1)
    
    #test  = np.array(test) #np.array((data2.loc[:,'Feature_24':'Ret_2']))
    train.to_csv('train_cluster.csv')
    test.to_csv('test_cluster.csv')
    '''
    #train = pd.read_csv('train_cluster.csv')
    test  = pd.read_csv('test_cluster.csv')
    #train = train.iloc[:,1:]
    test  = test.iloc[:,1:] 
    
    #train = np.array(train)
    test  = np.array(test)

    #clu = ts_cluster()
    #clu.k_means_clust(train, 10, 5)
    #centroid = np.array(clu.get_centroids())
    #centroidd = pd.DataFrame(centroid)
    #centroidd.to_csv('centroid.csv')
    
    clu = ts_cluster()
    centroid  = (pd.read_csv('centroid.csv'))
    centroid  = centroid.iloc[:,1:]
    centroid  = np.array(centroid)
    #print len(centroid)
    #clu.plot_centroids()
    result = clu.compa_clust(test, centroid, 5) 
    #result.to_csv('final_result11.csv') 
    #print result.shape
    f = open('part1.txt', 'a')
    
    for line in result:
        f.write("%f %f\n" %(line[0], line[1]))
    f.close()
        