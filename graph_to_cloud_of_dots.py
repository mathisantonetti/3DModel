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
def norm2(x):
    return np.sqrt(np.dot(x, x))

class Convexhull():
    def __init__(self, point):
        d = len(point)
        if(d > 2):
            print("Warning : convex hull for dimensions > 2 are not yet implemented")
        self.points = [point]

    def isInterior(self, point):
        # Parameters :
        # _ point : the point to check

        # Returns True if the point is in the convex hull interior, False otherwise
        barycenter = np.mean(self.points, axis=0)
        N = len(self.points)
        if(N == 1):
            return False
        
        for i in range(-1, N-1):
            normal_i = np.array([self.points[i+1][1] - self.points[i][1], -(self.points[i+1][0] - self.points[i][0])])
            normal_i = normal_i*(2*(np.dot(normal_i, barycenter - self.points[i]) >= 0) - 1)
            #print(normal_i)
            if(np.dot(normal_i, point - self.points[i]) < -1e-4):
                return False
        return True

    def update(self, point):
        # Parameters :
        # _ point : the point to cover by the convex hull

        # Computes the new convex hull updated with the new point
        if(self.isInterior(point) == False):
            barycenter = np.mean(self.points, axis=0)
            N = len(self.points)
            Imax, Imin, angle_max, angle_min = 0, 0, -1.0, 1.0
            for i in range(N):
                angle = np.arccos(np.dot(self.points[i] - point, barycenter - point)/(norm2(point - barycenter)*norm2(self.points[i] - point)))
                signvec = np.array([barycenter[1] - point[1], -(barycenter[0] - point[0])])
                angle *= np.sign(np.dot(signvec, self.points[i] - point))
                #print("angle loop : i = ", i, " , angle = ", angle)
                if(angle > angle_max):
                    Imax, angle_max = i, angle
                if(angle < angle_min):
                    Imin, angle_min = i, angle
            
            if(Imax < Imin):
                Itemp = Imin
                Imin = Imax
                Imax = Itemp

            #print("Imin = ", Imin, " ; Imax = ", Imax)
            if(Imax == 0):
                if(N >= 2):
                    if(norm2(self.points[0] - point) > norm2(self.points[1] - point)):
                        self.points = [self.points[0], point]
                    else:
                        self.points = [point, self.points[-1]]
                else:
                    self.points.append(point)
            else:
                normal_minmax = np.array([self.points[Imax][1] - self.points[Imin][1], -(self.points[Imax][0] - self.points[Imin][0])])
                normal_minmax = normal_minmax*np.sign(np.dot(normal_minmax, point - self.points[Imin]))
                if(np.dot(self.points[Imin-1] - self.points[Imin], normal_minmax) >= -1e-5):
                    for j in range(Imax+1, N):
                        self.points.pop()
                    for j in range(Imin):
                        self.points.pop(0)
                    self.points.append(point)
                    #print(self.points)
                else:
                    for j in range(Imin+1, Imax-1):
                        self.points.pop(Imin+1)
                    self.points.insert(Imax, point)

    def show(self):
        # Shows the convex hull
        N = len(self.points)
        #for i in range(N):
            #print(self.points[i])
        plt.plot([self.points[i][0] for i in range(-1, N)], [self.points[i][1] for i in range(-1, N)])
        plt.show()





def rotation_matrix(axe, angle):
    # Parameters :
    # _ axe : the rotation axe
    # _ angle : the angle of the rotation

    # Returns :
    # _ mat : the associated rotation matrix 
    mat = torch.zeros((3, 3))
    for i in range(3):
        u = torch.Tensor([(k == i) for k in range(3)])
        p_u = (torch.dot(axe, u)/torch.dot(axe, axe))
        a_u = torch.cross(axe, u)
        for j in range(3):
            mat[i,j] = p_u*axe[j] + (u[j] - p_u*axe[j])*np.cos(angle*np.pi/180) + np.sin(angle*np.pi/180)*a_u[j]

    #print(axe, mat)
    return mat


class graph():
    def __init__(self, Vertices, Edges, Faces):
        self.vertices = torch.Tensor(Vertices)
        self.edges = Edges
        self.faces = Faces

    def __removePoint(self, index):
        pass

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
    
    def rigid_transfo(self, translation, rotation, resize):
        # Parameters :
        # _ translation : vector defining coordinate-wise translation
        # _ rotation : vector made up of Euler angles for the rotation
        # _ resize : vector indicating the resizing factor of the homotethy

        # Returns :
        # Nothing

        N, d = self.vertices.shape
        minPoint, maxPoint = copy.deepcopy(self.vertices[0]), copy.deepcopy(self.vertices[0])
        for i in range(1, N):
            for j in range(d):
                minPoint[j] = min(minPoint[j], self.vertices[i, j])
                maxPoint[j] = max(maxPoint[j], self.vertices[i, j])
        
        boxcenteredPoint = (maxPoint + minPoint)/2

        # resizing
        for i in range(N):
            self.vertices[i] = boxcenteredPoint + (self.vertices[i] - boxcenteredPoint)*resize
        
        # translating
        self.vertices = self.vertices + translation

        # rotating
        for i in range(3):
            rotate_mat = rotation_matrix(torch.Tensor([(k == i) for k in range(3)]), rotation[i])
            self.vertices = boxcenteredPoint[None, :] + torch.matmul(self.vertices - boxcenteredPoint[None, :], torch.transpose(rotate_mat, 0, 1))

    def extrude(self, dir):
        # Parameters :
        # _ dir : direction of extraction

        # Computes the new graph with extrudage
        dirTensor = torch.Tensor(dir)
        N, d = self.vertices.shape
        sortedValues, indices = torch.sort(torch.matmul(self.vertices, dirTensor))
        points = []
        CH = Convexhull(self.vertices[indices[N-1]])
        for i in range(1, N):
            point = self.vertices[indices[N-1-i]]
            if(CH.isInterior(point) == False):
                points.append(indices[N-1-i])
            CH.update(point)

        # Identify faces, edges made with only points in points

        # Copy faces, edges with a new numerotation

        # Remove the identified original faces, edges

        # Create translated points copied from points numeroted accordingly to copied edges & faces

        # append faces, edges, points to the graph

    def fuse(self, other):
        # Parameters :
        # _ other : the other graph to fuse

        # Fuses the graph with other
        pass

def update(bin_list):
    # Parameters :
    # _ bin_list : the list that represents the binary number to update

    # Returns :
    # _ updated_list : The binary number updated
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
# Convex hull
def testCh1():
    Ch_test = Convexhull(np.array([0.0, 0.0]))
    Ch_test.update(np.array([1.0, 1.0]))
    Ch_test.update(np.array([2.0, 0.0]))
    Ch_test.show()
    Ch_test.update(np.array([3.0, 2.0]))
    Ch_test.show()
    Ch_test.update(np.array([3.0, -1.0]))
    Ch_test.show()
    Ch_test.update(np.array([4.0, 0.0]))
    Ch_test.show()
    print(Ch_test.isInterior(np.array([3.0, 5.0])))
    print(Ch_test.isInterior(np.array([3.0, 1.0])))

testCh1()

# Graph
def testGraph1():
    Graph_rect = hyper_rect([1, 5, 6], [1, 2, 1], 3)
    Graph_rect.rigid_transfo(torch.Tensor([0, 0, 1]), torch.Tensor([0, 60, 0]), torch.Tensor([1, 1, 1]))
    Graph_rect.showGraph(True, 4)

#testGraph1()