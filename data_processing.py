import pandas as pd
from pyproj import Proj, transform
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt


# load all nodes' information including the Latitude and Longtitude
df=pd.read_csv("d08_text_meta_2019_10_24.csv")


#load the nodes which are invovled in the graph
id=[]
for line in open("nodes_map.txt"):
    line=line[:-1]
    id.append(int(line))
id[-1]=827738

# filter the table
for num in range(len(df)):
    if df["ID"][num] not in id:
        df.drop([num],inplace=True)


#convert the geography into node position based on California location
def inverse_conv(x, y, target_proj="epsg:3310"):
    """

    :param x: Longitude
    :param y: Latitude
    :return: X and Y coordinates for the given projection system
    """
    inProj = Proj(init='epsg:4326')

    outProj = Proj(init=target_proj)

    x1, y1 = x, y
    x2, y2 = transform(inProj, outProj, x1, y1)

    return x2, y2

# convert latitude and longtitude into the calculable distance
longtitude = list(df["Longitude"])
latitude = list(df["Latitude"])
longtitude_list = []
latitude_list = []

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.warn("deprecated", DeprecationWarning)
    for i in range(len(longtitude)):
        longtitude_update, latitude_update = inverse_conv(longtitude[i], latitude[i])
        longtitude_list.append(longtitude_update)
        latitude_list.append(latitude_update)
        if (i % 100 == 0):
            print(i)

#generate adjacent matrix
A = np.zeros([len(longtitude_list), len(longtitude_list)])
for i in range(len(longtitude_list)):
    for j in range(0,len(longtitude_list)):
        distance=pow(pow(longtitude_list[i]-longtitude_list[j],2)+pow(latitude_list[i]-latitude_list[j],2),0.5)
        if distance==0:
            A[i,j]=0
        else:
            A[i,j]=1./distance
np.save("adjacent_new.npy", A)

#calculate weight and edge index
adjacent=np.load("./adjacent_new.npy")
first_line=[]
second_line=[]
weight=[]
for i in range(adjacent.shape[0]):
    for j in range(adjacent.shape[1]):
        if adjacent[i][j] !=0:
            first_line.append(i)
            second_line.append(j)
            weight.append(adjacent[i][j])

np_first_line=np.array(first_line).reshape(len(weight),1)
np_second_line=np.array(second_line).reshape(len(weight),1)
edge_index_traffic=np.concatenate([np_first_line,np_second_line],axis=1).transpose((1,0))
np.save("weight_new.npy", weight)
np.save("edge_index_traffic_new.npy", edge_index_traffic)