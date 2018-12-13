from random import uniform
from math import atan, sin, cos, pi

class CompleteGraph:
    def __init__(self, weight_function):
        self.vertices = {}
        self.weight_function = weight_function
        self.neighbors = {}
        
    def add_vertex(self, name, coordinates):
        if name not in self.vertices.keys():
            self.vertices[name] = coordinates
            
    def neighbor_weights(self, vertex):
        if vertex not in self.neighbors.keys():  
            neighbors = {}
            for x in self.vertices.keys():
                if x != vertex:
                    neighbors[x]=self.weight_function(vertex,self.vertices[x])
            self.neighbors[vertex] = neighbors
        return self.neighbors[vertex]
    
    def all_vertex_coordinates(self):
        return list(self.vertices.values())
    
    def get_weight_function(self):
        return self.weight_function

def euclidian_distance(v1, v2):
    total = 0
    for x1, x2 in zip(v1,v2):
        total += (x2-x1)**2
    return total**0.5

def to_radians(coord):
    return (x*pi/180 for x in coord)

def great_circle_distance(v1, v2):
    earth_radius = 3959
    lat1, long1 = to_radians(v1)
    lat2, long2 = to_radians(v2)
    delta = abs(long2 - long1)
    angle_dif = atan(((cos(lat2)*sin(delta))**2 + (cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(delta))**2)**0.5 / (sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(delta)))
    return earth_radius * angle_dif

def make_graph(iterable, limit, weight_function):
    graph = CompleteGraph(weight_function)
    for x in iterable:
        graph.add_vertex(x, (uniform(-limit,limit), uniform(-limit,limit)))
    return graph

def generate_graphs(n_cases, limit, weight_function):
    graph_lst = []
    sizes = [10, 20, 30, 40, 50]
    for _ in range(int(n_cases/len(sizes))):
        for size in sizes:
            graph_lst.append(make_graph(range(size), limit, weight_function))
    return graph_lst

# cases = generate_graphs(20, 180)
# for x in cases:
#     print(x)