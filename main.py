#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 18:49:16 2018

@author: chaitu65c
"""

from matplotlib import pyplot as plt
from antcolony import ant_colony
import pandas as pd
from random_and_greedy import random_walks, greedy
from GeneticAlgorithm import geneticAlgorithm as ga
from GeneticAlgorithm import City
import graph
from collections import defaultdict
import time

def get_distance(stuff):
    dist = 0
    #print(stuff)
    for i in range(1,len(stuff)):
        dist += graph.euclidian_distance(stuff[i-1],stuff[i])
    return dist


if __name__ == '__main__':
    #test_nodes = {0: (0, 7), 1: (3, 9), 2: (12, 4), 3: (14, 11), 4: (8, 11),	5: (15, 6), 6: (6, 15), 7: (15, 9), 8: (12, 10), 9: (10, 7)}
    ACOdata = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    gadata = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    randomdata = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    greddydata = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    counter= 1
    print("generating graps")
    graphs = graph.generate_graphs(5,50,graph.great_circle_distance)
    graphs = [graphs[1]]
    print("Start")
    for g in graphs:
        print(g.all_vertex_coordinates())
        length = len(g.all_vertex_coordinates())/10
        graphnodes = g.all_vertex_coordinates()
        key = 0
        if length == 1:
            key = 0
        elif length == 2:
            key = 1
        elif length == 3:
            key = 2
        elif length  ==4:
            key = 3
        else:
            key = 4
        # ant colony
        anttime = time.time()
        cityList = []
        q = defaultdict()
        count = 0
        for i in g.all_vertex_coordinates():
            q[count] = i
            count += 1
            cityList.append(City(x=i[0],y=i[1]))        
        print("Start ant colony")
        answer = ant_colony(dict(q), graph.great_circle_distance)
        e = answer.mainloop()
        a = ACOdata[key]
        #print(e)
        a[0] += time.time()-anttime
        a[1] += get_distance([q[y] for y in e])
        a[2] += 1
        ACOdata[key] = a
        print("Done ant colony")
        
        """#Genetic Algorithm
        gat = time.time()
        print('Start GA')
        answer = ga(population=cityList, popSize=100, eliteSize=20, mutationRate=0.015, generations=20)
        a = gadata[key]
        a[0] += time.time()-gat
        a[1] += get_distance(answer)
        a[2] += 1
        gadata[key] = a
        print('End GA')"""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        allpath = [] #put ordered tuple coordinates here
        allpath = [graphnodes[i] for i in e]
        
        Xf = [int(x[0]) for x in allpath]
        Yf = [int(y[1]) for y in allpath]
        print('Entire Path: {}\n'.format(allpath))
        print('X coord: {}\n'.format(Xf))
        print('Y coord: {}\n'.format(Yf))
        
        plt.plot(Xf,Yf) #most points
        plt.plot([Xf[-1],Xf[0]], [Yf[-1], Yf[0]]) #last stretch
        plt.plot(Xf[-1], Yf[-1], 'ro') #end point before finish
        plt.plot(Xf[0], Yf[0], 'go') #start point
    
        for xy in zip(Xf, Yf):                                       # <--
            ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') # <--
        plt.title('Ant Colony TSM')
        plt.grid()
        plt.show()
            
        #Random
        ra = time.time()
        print('Start Random')
        answerr = random_walks(g.all_vertex_coordinates())
        a = randomdata[key]
        a[0] += time.time()-ra
        a[1] += get_distance(answerr)
        a[2] += 1
        randomdata[key] = a
        print('End Random')
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        allpath = [] #put ordered tuple coordinates here
        
        allpath = []
        allpath = answerr
        
        Xf = [int(x[0]) for x in allpath]
        Yf = [int(y[1]) for y in allpath]        
        print('Entire Path: {}\n'.format(allpath))
        print('X coord: {}\n'.format(Xf))
        print('Y coord: {}\n'.format(Yf))
        
        plt.plot(Xf,Yf) #most points
        plt.plot([Xf[-1],Xf[0]], [Yf[-1], Yf[0]]) #last stretch
        plt.plot(Xf[-1], Yf[-1], 'ro') #end point before finish
        plt.plot(Xf[0], Yf[0], 'go') #start point
    
        for xy in zip(Xf, Yf):                                       # <--
            ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') # <--
        plt.title('Random Walks TSM')
        plt.grid()
        plt.show()
        
        #Greedy
        print('Greedy Start')
        gr = time.time()
        answerg = greedy(g, graph.great_circle_distance)
        a = greddydata[key]
        a[0] += time.time()-gr
        a[1] += get_distance(answerg)
        a[2] += 1
        greddydata[key] = a
        print('Iter {} done'.format(counter))
        print('Greedy end')
        counter += 1
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        allpath = [] #put ordered tuple coordinates here
        
        allpath = answerg
        
        Xf = [int(x[0]) for x in allpath]
        Yf = [int(y[1]) for y in allpath]
        print('Entire Path: {}\n'.format(allpath))
        print('X coord: {}\n'.format(Xf))
        print('Y coord: {}\n'.format(Yf))
        
        plt.plot(Xf,Yf) #most points
        plt.plot([Xf[-1],Xf[0]], [Yf[-1], Yf[0]]) #last stretch
        plt.plot(Xf[-1], Yf[-1], 'ro') #end point before finish
        plt.plot(Xf[0], Yf[0], 'go') #start point
    
        for xy in zip(Xf, Yf):                                       # <--
            ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') # <--
        plt.title('Greedy Algorithm TSM')
        plt.grid()
        plt.show()
    print('Done')
    total_time1 = [f[0]/f[2] for f in ACOdata]
    #total_time2 = [f[0]/f[2] for f in gadata]
    total_time3 = [f[0]/f[2] for f in randomdata]
    total_time4 = [f[0]/f[2] for f in greddydata]
    
    #...we can make a colony of ants...
    #colony = ant_colony(test_nodes, distance)
    #...that will find the optimal solution with ACO
    #answer = colony.mainloop()
    #print(answer)
    print(ACOdata)
    print(gadata)
    print(randomdata)
    print(greddydata)
    
    
    plt.plot(total_time1, [50,100,200,500,1000], 'go-', label='Ant Colony')
    #plt.plot(total_time2, [50,100,200,500,1000], 'ro-',  label='Genetic Algorithm')
    plt.plot(total_time3, [50,100,200,500,1000], 'bo-', label='Greedy')
    plt.plot(total_time4, [50,100,200,500,1000], 'mo-',  label='Random Walks')
    plt.axis([0.0, 10.0, 0.0, 30.0])
    plt.xlabel('Size')
    plt.ylabel('Time Taken')
    plt.title('Size of Algorithm vs. Time Taken')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    allpath = [] #put ordered tuple coordinates here
    
    for t in range(len(answerr)):
        allpath.append(answerr[t])
    
    Xf = [x[0] for x in allpath]
    Yf = [y[1] for y in allpath]
    print('Entire Path: {}\n'.format(allpath))
    print('X coord: {}\n'.format(Xf))
    print('Y coord: {}\n'.format(Yf))
    
    plt.plot(Xf,Yf) #most points
    plt.plot([Xf[-1],Xf[0]], [Yf[-1], Yf[0]]) #last stretch
    plt.plot(Xf[-1], Yf[-1], 'ro') #end point before finish
    plt.plot(Xf[0], Yf[0], 'go') #start point
    
    for xy in zip(Xf, Yf):                                       # <--
        ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') # <--
    plt.title('Genetic Algorithm TSM')
    plt.grid()
    plt.show()