# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 18:07:12 2018

@author: Sahil
"""
"""
Random path, start at a city and randomly select the next city from the remaining not visited cities until all cities are visited.
"""

#graph is a list of tuples of lat, long
import graph

def random_walks(g):
    from random import randint
    graph = g
    start_index = randint(0, len(graph) - 1)
    ret = []

    first_point = graph[start_index]
    ret.append(first_point)
    del graph[start_index]
    
    while len(graph) > 0:
        index = randint(0, len(graph) - 1)
        point = graph[index]
        ret.append(point)
        del graph[index]
    
    ret.append(first_point)
    return ret

"""
Greedy, start a city select as next city the unvisited city that is closest to the current city
"""

def greedy(g, distance):
    from random import randint
    import operator
    graph = g
    new_g = graph.all_vertex_coordinates()
    start_index = randint(0, len(new_g) - 1)
    visited = []
    first = []
    first_point = new_g[start_index]
    first.append(first_point)
    visited.append(first_point)
    
    while len(visited) < len(new_g):
        dist_dict = graph.neighbor_weights(first_point)
        dict_list = sorted(dist_dict.items(), key=operator.itemgetter(1))
        
        for i in dict_list:
            if i[0] not in visited:
                visited.append(i[0])
                first_point = i[0]
                break
            else:
                continue
    
    visited.append(first[0])
    return visited
    
    
    
    
    