import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
class Node:

    def __init__(self, value, number, connections=None):
        self.index = number
        self.connections = connections
        self.value = value
        self.parent =None

class Queue:
    def __init__(self):
        self.queue = []
    
    def push(self,value):
        self.queue.append(value)
    
    def pop(self):
        if len(self.queue)<1:
            return None
        return self.queue.pop(0)
    
    def is_empty(self):
        return len(self.queue)==0
    
class Network:
    
    def __init__(self, nodes=None):
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes 

    def get_mean_degree(self):
        #Your code  for task 3 goes here
        total =0
        for node in self.nodes:
            total += node.connections.count(1)
        num_nodes = self.nodes[-1].index + 1
        
        return total/num_nodes
    
    def get_mean_clustering(self):
        #Your code for task 3 goes here
        clustering_co = []
        for (index_l, node) in enumerate(self.nodes):
            neighbours = node.connections.count(1)
            possible_triangles = neighbours * (neighbours-1)/2
            if possible_triangles == 0:
                continue
            temp_connections = []
            for (index_temp,connection) in enumerate(node.connections):
                if connection ==1:
                    temp_connections.append(index_temp)
            connections= [i for i in temp_connections]      # Don't change this line, only solution that i found works 
            total = 0
            for counting in connections:
                index_m= temp_connections.pop(0)
                for index_h in temp_connections:
                    m_connection = self.nodes[index_m].connections
                    if m_connection[index_l]==1 and m_connection[index_h] ==1:
                        total+=1
            clustering_co.append(total/possible_triangles)
                        
        num_nodes = self.nodes[-1].index + 1
        return float(sum(clustering_co)/num_nodes)

    def get_mean_path_length(self):
        #Your code for task 3 goes here
        mean_path_lenght_pn=[]
        for node in self.nodes:
            start_index = node.index
            neigh = node.connections
            nodes_num = len(neigh)
            
            path_length_per_node =[]
            
            for end_index in range(0,nodes_num):
                if start_index == end_index:
                    continue
                queue = Queue()
                queue.push(start_index)
                visited = []
                while not queue.is_empty():
                    node_check = queue.pop()
                    if node_check == end_index:
                        break
                    neighbour_index = [i for i, n in enumerate(self.nodes[node_check].connections) if n ==1]
                    if neighbour_index == []:
                        break
                    else:
                        for neighbour in neighbour_index:
                            if neighbour not in visited:
                                queue.push(neighbour)
                                visited.append(node_check)
                                self.nodes[neighbour].parent=node_check
                            if neighbour == end_index:
                                self.nodes[neighbour].parent=node_check
                                queue=Queue()
                                break
                        
                check = self.nodes[end_index]
                self.nodes[start_index].parent=None
                rought = []
                while check.parent != None:
                    rought.append(check.index)
                    temp = check
                    check=self.nodes[check.parent]
                    temp.parent=None
                if rought == []:
                    continue
                path_length_per_node.append(len(rought))
            if len(path_length_per_node)==0:
                mean_path_lenght_pn.append(0.0)
            else:
                mean_path_lenght_pn.append(sum(path_length_per_node)/len(path_length_per_node))
        mean_path_length=sum(mean_path_lenght_pn)/nodes_num
            
        return round(mean_path_length,15)
                        
    def make_random_network(self, N, connection_probability):
        '''
        This function makes a *random* network of size N.
        Each node is connected to each other node with probability p
        '''

        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index+1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1
        for node in self.nodes:
            print(node.index,node.connections)
            
            
    def make_ring_network(self, N, neighbour_range=1):
        print("mean")
        #Your code  for task 4 goes here

    def make_small_world_network(self, N, re_wire_prob=0.2):
        print("mean")
        #Your code for task 4 goes here

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
        ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i+1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

def test_networks():

    #Ring network
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number-1)%num_nodes] = 1
        connections[(node_number+1)%num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing ring network")
    assert(network.get_mean_degree()==2), network.get_mean_degree()
    assert(network.get_mean_clustering()==0), network.get_mean_clustering()
    assert(network.get_mean_path_length()==2.777777777777778), network.get_mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number+1)%num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing one-sided network")
    assert(network.get_mean_degree()==1), network.get_mean_degree()
    assert(network.get_mean_clustering()==0),  network.get_mean_clustering()
    assert(network.get_mean_path_length()==5), network.get_mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
    assert(network.get_mean_clustering()==1),  network.get_mean_clustering()
    assert(network.get_mean_path_length()==1), network.get_mean_path_length()

    print("All tests passed")

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''

'''setting 2D array, each value has value 1 or -1 which depends on the random probability
'''

agree = 1
disagree = -1
population_size = 100
probability_option = 0.5
population = np.random.choice([agree, disagree], size = [population_size,population_size],
                              p=(probability_option, 1-probability_option))

# Di_math function, row and col controls the opinion of original guy
def calculate_agreement(population, row, col, external=0.0):
    size = len(population)
    # deal with the boundary value problem by using % operate
    agreement = (population[row, col] * population[row, (col + 1) % size] +
    population[row, col] * population[row, (col - 1) % size] +
    population[row, col] * population[(row - 1) % size, col] +
    population[row, col] * population[(row + 1) % size, col]+
    (external * population[row,col]))

    return agreement
# renew the ising model. Shows how External and alpha affect the Di/agreement value.
def ising_step(population, alpha, external): # three input arguments control the p and agreement math function
    '''
    This function will perform a single update of the Ising model
    '''
    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)

    agreement = calculate_agreement(population, row, col, external)
# agreement < 0 the original guy will flip his opinion
    if agreement < 0:
        population[row, col] *= -1

    elif agreement > 0:
        p = math.e ** (-agreement / alpha)
        # the original opinion will flip with the probability p when agreement >0
        population[row, col] = np.random.choice([population[row, col], population[row, col] * -1],
                                                 p=(1-p, p))
        # flip the opinion if agreement equals zero
    elif agreement == 0:
        population[row,col] = population[row, col] * -1

    return population

def plot_ising(im, population):
    '''
    This function will display a plot of the Ising model
    '''
    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.1)

# if the user does not set the alpha and external value, these will be assigned the input arguments
def ising_main(population, alpha=None, external=0.0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, alpha, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)

def test_ising(): # This function will test the calculate_agreement function in the Ising model

    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1)==4), "Test 1"

    population[1, 1] = 1.
    assert(calculate_agreement(population,1,1)==-4), "Test 2"

    population[0, 1] = 1.
    assert(calculate_agreement(population,1,1)==-2), "Test 3"

    population[1, 0] = 1.
    assert(calculate_agreement(population,1,1)==0), "Test 4"

    population[2, 1] = 1.
    assert(calculate_agreement(population,1,1)==2), "Test 5"

    population[1, 2] = 1.
    assert(calculate_agreement(population,1,1)==4), "Test 6"

    "Testing external pull"
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1,1)==3), "Test 7"
    assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
    assert(calculate_agreement(population,1,1,10)==-6), "Test 9"
    assert(calculate_agreement(population,1,1,-10)==14), "Test 10"

    print("Tests passed")



def ising_main(population, alpha=None, external=0.0):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population,alpha, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''

def defuant_main():
    print("mean")
	#Your code for task 2 goes here

def test_defuant():
    print("mean")
	#Your code for task 2 goes here


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main():
	#You should write some code for handling flags here
    parser = argparse.ArgumentParser(description='process the users_s input ')
    parser.add_argument("-network",action="store",type=int,default=False)
    parser.add_argument("-test_network",action="store_true",default=False)

    #adding four command-line parameter
    parser.add_argument('-ising_model', action='store_true')
    parser.add_argument('-test_ising', action='store_true')
    parser.add_argument('-external', type=float, default=0.0)
    parser.add_argument('-alpha', type=float, default=1.0)

    args=parser.parse_args()
    
    if args.network:
        net=Network()
        print(args.network)
        net.make_random_network(args.network,0.3)
        print("Mean degree:",net.get_mean_degree())
        print("Mean path length:",net.get_mean_path_length())
        print("Mean clustering co-efficient:",net.get_mean_clustering())
        net.plot()
        plt.show()
    if args.test_network:
        test_networks()

        # if the user enter the ising_model or test_ising, running these two funtions
    if args.ising_model:
        ising_main(population, alpha=args.alpha, external=args.external)

    if args.test_ising:
        test_ising()
        
if __name__=="__main__":
	main()
