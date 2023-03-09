import copy
import numpy as np
import matplotlib.pyplot as plt
import torch

# Here, a graph G is a tensor of order 3 representing the faces of a piece and we stock it as (P, G) where P is the set of points of interest (linked by faces).

# This script converts a graph into a cloud of dots (points set) on the surface represented by the graph.

### Types and devices
dtypef = torch.float32
dtypei = torch.int32

### Tools
class graph():
    def __init__(self, Vertices, Edges, Faces):
        self.vertices = torch.Tensor(Vertices)
        self.edges = Edges
        self.faces = Faces

    def showGraph(self, withpoints, pointsPerFace):
        # Parameters :
        # _ withpoints : If True, shows random points on the surface
        # _ pointsPerFace : number of random points par face

        # Shows the 3D object modelized by the graph
        ax = plt.axes(projection='3d')
        sline = np.linspace(0, 1, 1000)
        x_dots = []
        y_dots = []
        z_dots = []
        for face in self.faces:
            zline = torch.rand(pointsPerFace, len(face))
            zline = torch.transpose(torch.div(torch.transpose(zline, 0, 1), torch.sum(zline, dim=1)), 0, 1)
            points = torch.matmul(zline, self.vertices[face])
            for i in range(len(face)):
                x_dots.append(points[i][0])
                y_dots.append(points[i][1])
                z_dots.append(points[i][2])

        for edge in self.edges:
            x_line = sline*(self.vertices[edge[0]][0]).item() + (1-sline)*(self.vertices[edge[1]][0]).item()
            y_line = sline*(self.vertices[edge[0]][1].item()) + (1-sline)*(self.vertices[edge[1]][1].item())
            z_line = sline*(self.vertices[edge[0]][2].item()) + (1-sline)*(self.vertices[edge[1]][2].item())
            ax.plot3D(x_line, y_line, z_line, 'gray')

        if(withpoints):
            ax.scatter3D(np.array(x_dots), np.array(y_dots), np.array(z_dots), color='green')
        plt.show()
    
    def fuse(other):
        pass

def update(bin_list):
    # Parameters :
    # _ bin_list : the list that represents the binary number to update

    # Returns :
    #The binary number updated
    updated_list = copy.deepcopy(bin_list)
    n = len(bin_list)
    for i in range(n):
        if(bin_list[i] == 0):
            updated_list[i] = 1
            break
        else:
            updated_list[i] = 0

    return updated_list

def hyper_rect(origin, dir, ndim = 3):
    # Parameters :
    # _ origin : the point of a vertex of the hyper rectangle
    # _ dir : the vector representing the size of the hyper rectangle

    # Returns :
    # Graph : the graph of a hyper-rectangle generated with parameters
    Edges = []
    Faces = []
    if(ndim == 2):
        #Edges = [[0, 0], [0, 1]]
        Faces = [[[0, 0], [0, 1]], [[0, 0], [1, 0]], [[1, 0], [1, 1]], [[0, 1], [1, 1]]]
    elif(ndim == 3):
        Edges = [[[0, 0, 0], [0, 0, 1]], [[0, 0, 1], [0, 1, 1]], [[0, 1, 1], [0, 1, 0]], [[0, 1, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 0]], [[1, 1, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 1]], [[1, 0, 1], [1, 1, 1]], [[0, 0, 0], [1, 0, 0]], [[0, 0, 1], [1, 0, 1]], [[0, 1, 0], [1, 1, 0]], [[0, 1, 1], [1, 1, 1]]]
        Faces = [[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]], [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]], [[1, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1]], [[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 1]], [[1, 1, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0]]]

    n1 = len(Faces)
    n2 = len(Faces[0])
    Powers = np.array([[[2**k for k in range(ndim)] for j in range(n2)] for i in range(n1)])
    true_Faces = np.sum(np.array(Faces)*Powers, axis=2)
    n1 = len(Edges)
    n2 = len(Edges[0])
    Powers = np.array([[[2**k for k in range(ndim)] for j in range(n2)] for i in range(n1)])
    true_Edges = np.sum(np.array(Edges)*Powers, axis=2)
    points = []
    bin_incr = [0 for k in range(ndim)]
    for i in range(2**ndim):
        points.append([origin[k] + bin_incr[k]*dir[k] for k in range(ndim)])
        bin_incr = update(bin_incr)

    Graph = graph(points, true_Edges, true_Faces)
    return Graph


### Tests
Graph_rect = hyper_rect([1, 5, 6], [1, 2, 1], 3)
Graph_rect.showGraph(True, 4)