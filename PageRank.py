"""The Page Rank Algorithm.
Will Melville
1-15-19
"""

from scipy.sparse import dok_matrix
import numpy as np
import scipy.linalg as la
from collections import Counter
import networkx as nx

# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        self.n (int): The number of nodes in the graph
        self.labels (list(str)): labels for the nodes in the graph.
        self.Ahat ((n,n) ndarray): the modified and normalized adjacency matrix
        that is column stochastic and has no sinks.
    """

    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        self.n = len(A)
        #first store the labels as an attribute. If none are provided, use the default value
        if labels == None:
            self.labels = [i for i in range(self.n)]
        elif len(labels) != self.n:
            #there aren't enough labels
            raise ValueError('Number of labels must match number of nodes')
        else:
            self.labels = labels
        #check A for sinks
        for i in range(self.n):
            if np.all(A[:,i]==0):
                A[:,i] = np.ones(self.n)
        #now make A column stochastic and store it as an attribute
        self.Ahat = A / A.sum(axis=0)


    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #use scipy.linalg.solve to solve for p
        p = la.solve(np.eye(self.n) - epsilon*self.Ahat, ((1-epsilon)/self.n) * np.ones(self.n))
        #normalize p
        p = p / sum(p)
        #return the dictionary mapping labels to p
        d = {}
        for i in range(self.n):
            d.update({self.labels[i]: p[i]})
        return d


    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #define the matrix B,
        B = epsilon * self.Ahat + ((1-epsilon) / self.n) * np.ones((self.n, self.n))
        #p is the eigenvector with corresponding eigenvalue of 1
        vals, vecs = np.linalg.eig(B)
        #find the index of vals such that vals[i] = 1
        for i in range(self.n):
            if np.allclose(vals[i], 1):
                break
        #The ith column of vecs is p, normalize p
        p = vecs[:,i]
        p = p / sum(p)
        #return the dictionary
        return{self.labels[i]: p[i] for i in range(self.n)}


    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #choose the initial vector p as [1/n, 1/n, ... , 1/n]
        p0 = np.ones(self.n) / self.n
        #iterate at most maxiter times
        for _ in range(maxiter):
            #assign p1
            p1 = epsilon*(self.Ahat @ p0) + ((1-epsilon) / self.n) * np.ones(self.n)
            #check for convergence using the one norm
            if la.norm(p1 - p0, ord=1) < tol:
                #normalize p then return
                p1 = p1/sum(p1)
                return {self.labels[i]: p1[i] for i in range(self.n)}
            #it hasn't converged yet, so reassign p0
            p0 = p1
        #it never converged, normalize p then return
        p0=p0/sum(p0)
        return {self.labels[i]: p0[i] for i in range(self.n)}

def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    sorted = []
    keys = list(d.keys())
    vals = list(d.values())
    n = len(keys)
    #loop until sorted is the same length as n, so every node has been accounted for
    while len(sorted) < n:
        #sort from highest to lowest, so find the max of values
        i = vals.index(max(vals))
        sorted.append(keys[i])
        #remove keys[i] and vals[i] from the list and loop again
        keys.remove(keys[i])
        vals.remove(vals[i])
    return sorted


def rank_ncaa_teams(filename='ncaa2019.csv', epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    #read the file
    file = open(filename, 'r')
    lines = file.readlines()
    #close the file
    file.close()
    #Use Counter to get the team names, start with line 1 to ignore the header
    teams = Counter(lines[1].strip().split(','))
    for line in lines[2:]:
        #now loop through the lines and update the Counter
        teams.update(Counter(line.strip().split(',')))
    team_names = set(teams.keys())    #using set ensures no repeated team names
    team_names = list(team_names)
    team_names.sort()
    n = len(team_names)
    #make a dictionary mapping a team_name to its index
    team_dict = {team_names[i]: i for i in range(n)}
    #now loop through lines again to make the adjacency matrix
    A = np.zeros((n,n))
    for line in lines[1:]:
        #create a link from the loser to the winner. Loser is the second column
        line_list = line.strip().split(',')
        A[team_dict[line_list[0]], team_dict[line_list[1]]] += 1
    #now we can use A and team_names in a DiGraph class
    ranking = DiGraph(A, team_names)
    #return the ranking
    return get_ranks(ranking.itersolve(epsilon = epsilon))
