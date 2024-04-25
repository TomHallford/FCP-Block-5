import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
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
        '''
        Function for calulating the mean degree of a network
        - sums the number of neighbours each node has
        - find the total number of nodes in the network
        - retirns the value produced by the deviding the two previous values 
        '''
        total =0
        for node in self.nodes:
            total += node.connections.count(1)
        num_nodes = self.nodes[-1].index + 1
        
        return total/num_nodes

    def get_mean_clustering(self):
        #Your code for task 3 goes here
        '''
        Functions for finding the mean clustering in a network
        -checks each node for the number of neighbours, if it has less then 2 it will move onto the next node
        -perfomrs calulations for the maxium possible triablges formed by the node and it's connections
        -checks if two connected nodes have a common neghibour, if they do increase the triangle count by 1
        - for a given node devide the total numebr of triangles by the number of possible triangles
        -return the sum the mean clustering for each node devied by the toal number of nodes 
        '''
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
        '''
        function for finding the mean path length for the entire network
        -for each node finds the shortest rought to each other node in the network
        -sums the result and devides the total by the number of nodes that contriute to the toal
        -repeasts for each node and adds the result of each to a total
        -devides the total by the number of nodes in the network
        -returns the result of this
        '''
        mean_path_lenght_pn=[]
        for node in self.nodes:
            start_index = node.index
            neigh = node.connections
            nodes_num = len(neigh)
            
            path_length_per_node =[]
            
            for end_index in range(0,nodes_num):
                if start_index == end_index:
                    continue
                # search to find the shortest path length
                
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
                    temp.parent=None								# prevents overflow between nodes
                if rought == []:
                    continue
                path_length_per_node.append(len(rought))
                
                
            if len(path_length_per_node)==0:
                mean_path_lenght_pn.append(0.0)     # prevents nodes with 0 connecitons haveing a path length of 1
            else:
                mean_path_lenght_pn.append(sum(path_length_per_node)/len(path_length_per_node))
        mean_path_length=sum(mean_path_lenght_pn)/nodes_num
            
        return round(mean_path_length,15)				#needed to past test as the number producded has 16 dp
                        
    def make_random_network(self, N, connection_probability):
        '''
        This function makes a *random* network of size N.
        Each node is connected to each other node with probability p
        '''

        self.nodes = []
        for node_number in range(N):
            value = np.random.choice([1,-1],p=(0.5,0.5))
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index+1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1
            
            
    def make_ring_network(self, N, neighbour_range=1):
        print("mean")
        #Your code  for task 4 goes here

    def make_small_world_network(self, N, re_wire_prob=0.2):
        print("mean")
        #Your code for task 4 goes here

    def ising_update(self,alpha):
        '''
        function for updating a single node in the ising_network
        '''
        random_index=random.randint(0,len(self.nodes)-1)
        node = self.nodes[random_index]
        
        neighbour_index_list=[i for (i,n) in enumerate(node.connections) if n == 1]
        agreement = 0
        for neighbour_index in neighbour_index_list:
            agreement += node.value * self.nodes[neighbour_index].value
        
        if agreement <= 0:
            node.value *= -1
        elif agreement >0:
            p = math.e ** (-agreement / alpha)
            node.value = np.random.choice([node.value,node.value*-1],p=(1-p,p))
        
    def plot(self,ax=False):
        if not ax:
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

            circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.Set1(node.value))
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


def defuant_model_calc(x, T, beta):

    """
    calculates the 2 new opinions based of each other

    :param x: a length 2 list of neighbours opinions, where x[0] is xi and x[1] is xj
    :param T: is the threshold constant for which the opinions need to be close enough to, to change
    :param beta: how much an opinion updates if it is close enough to another
    :return: 2 variables of which are the new opinions
    """

    if abs(x[0] - x[1]) < T:
        xi = x[0]
        xj = x[1]
        x[0] = xi + (beta*(xj - xi))
        x[1] = xj + (beta*(xi - xj))

    return x[0], x[1]


def new_iteration(neighbour_list, T, beta):

    """
    iterates through the list of neighbour opinions, creating the next set of opinions

    :param neighbour_list: list containing the opinions of the neighbours
    :param T: (passed into defuant calc)
    :param beta: (passed into defuant calc)
    :return:
    """

    # iterates through the list
    for i in range(0, len(neighbour_list)):

        # needs to handle the scenario of the neighbour being at the ends of the list
        if i == 0:
            r_n = 1
        elif i == (len(neighbour_list) - 1):
            r_n = -1
        else:
            r_n = random_neighbour()

        # updates the values of the list
        x = [neighbour_list[i], neighbour_list[i + r_n]]
        x = defuant_model_calc(x, T, beta)
        neighbour_list[i] = x[0]
        neighbour_list[i + r_n] = x[1]

    return neighbour_list


def opinion_mapping(time_step, neighbour_list, T, beta):

    """
    iterates through the time steps creating new opinions then plotting them, at the end it shows the scatter
    graph and a histogram of the opinions

    :param time_step: how many times the neighbour opinions are going to be iterated through
    :param neighbour_list: list of neighbour opinions
    :param T: (passed for defuant calc)
    :param beta: (passed for defuant calc)
    :return: doesn't explicitly return anything but plots the graphs needed
    """

    fig, (lax, rax) = plt.subplots(1, 2)

    # plotting the scatter graph and then creating the histogram
    for i in range(0, time_step + 1):
        rax.scatter([i] * len(neighbour_list), neighbour_list, color="red")
        new_iteration(neighbour_list, T, beta)
    lax.hist(neighbour_list)

    # detailing the graphs
    fig.suptitle("coupling: " + str(beta) + "  threshold: " + str(T))
    lax.set_xlabel("opinion")
    rax.set_xlabel("time step")
    rax.set_ylabel("opinion")

    plt.show()


def neighbour_create(n):

    """
    using list comprehension it creates a list of random opinions of length n

    :param n: specifies the length of neighbour opinions
    :return: a list of random values ranging between 0 and 1
    """
    neighbour_list = [random.random() for i in range(n)]
    return neighbour_list


def random_neighbour():
    """generates a random number to choose a random neighbour"""
    return random.choice([-1, 1])


def defuant_main(T, beta, N, time_step):
    """creates the neighbour list using the passed values"""
    neighbour_list = neighbour_create(N)
    opinion_mapping(time_step, neighbour_list, T, beta)


def test_defuant():
    # opinion case where their difference is below the threshold, x = [0.4, 0.5]
    assert 0.42001 > defuant_model_calc([0.4, 0.5], 0.2, 0.2)[0] > 0.41999, "xi defuant model calc fail"
    assert 0.48001 > defuant_model_calc([0.4, 0.5], 0.2, 0.2)[1] > 0.47999, "xj defuant model calc fail"
    print("passed calc")

    # opinion case where their difference is above the threshold, x = [0.2, 0.8]
    assert defuant_model_calc([0.2, 0.8], 0.2, 0.2)[0] == 0.2, "xi threshold comparison fail"
    assert defuant_model_calc([0.2, 0.8], 0.2, 0.2)[1] == 0.8, "xj threshold comparison fail"
    print("passed threshold comparison")

    # correctly passes one iteration with both cases of the opinions being in an out of threshold range
    assert 0.43201 > new_iteration([0.4, 0.5], 0.2, 0.2)[0] > 0.43199, "xi iteration fail"
    assert 0.46800 > new_iteration([0.4, 0.5], 0.2, 0.2)[1] > 0.46799, "xj iteration fail"
    # to note, the values of the xi/xj iteration check is different due to the calculations occur twice
    assert new_iteration([0.2, 0.8], 0.2, 0.2)[0] == 0.2, "xi threshold iteration fail"
    assert new_iteration([0.2, 0.8], 0.2, 0.2)[1] == 0.8, "xj threshold iteration fail"
    print("passed iteration")

    # checks neighbour random generation
    for i in range(10):
        rand_neigh = random_neighbour()
        assert rand_neigh == 1 or rand_neigh == -1, "random neighbour fail"
    print("passed random neighbour")

    # checks that opinions range between 0 and 1
    list_of_rand_opinions = neighbour_create(100)
    for i in range(100):
        assert 1 >= list_of_rand_opinions[i] >= 0, "opinion out of range"
    print("passed opinion check")


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''


def main():
    # You should write some code for handling flags here
    parser = argparse.ArgumentParser(description='process the users_s input ')
    
    # flags for task_3 Network
    parser.add_argument("-network",action="store",type=int,default=False)
    parser.add_argument("-test_network",action="store_true",default=False)

    # adding four command-line parameter
    parser.add_argument('-ising_model', action='store_true')
    parser.add_argument('-test_ising', action='store_true')
    parser.add_argument('-external', type=float, default=0.0)
    parser.add_argument('-alpha', type=float, default=1.0)

    # flags for the defuant model
    parser.add_argument("-defuant", action="store_true", default=False)
    parser.add_argument("-threshold", type=float, default=0.2)
    parser.add_argument("-beta", type=float, default=0.2)
    parser.add_argument("-n_count", type=int, default=100)
    parser.add_argument("-time_step", type=int, default=100)
    parser.add_argument("-test_defuant", action="store_true", default=False)

    # flag for task_5
    parser.add_argument("-use_network", action="store",type=int, default=False)
    
    args=parser.parse_args()
    
    if args.network:
        net=Network()
        print(args.network)
        net.make_random_network(args.network,0.5)
        print("Mean degree:",net.get_mean_degree())
        print("Mean path length:",net.get_mean_path_length())
        print("Mean clustering co-efficient:",net.get_mean_clustering())
        net.plot()
        plt.show()
    if args.test_network:
        test_networks()

    if args.defuant:
        defuant_main(args.threshold, args.beta, args.n_count, args.time_step)
    if args.test_defuant:
        test_defuant()

    # if the user enter the ising_model or test_ising, running these two funtions
    if args.ising_model:
        #checks if ths ising_model is using a network or grid for representation
        if args.use_network:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_axis_off()
            plt.ion()
            ising_network = Network()
            ising_network.make_random_network(args.use_network, 0.5)
            ising_network.plot(ax)
            plt.pause(0.1)
            for i in range(100):
                ising_network.ising_update(args.alpha)
                ising_network.plot(ax)
                plt.pause(0.1)
            
        else:
            ising_main(population, alpha=args.alpha, external=args.external)
    if args.test_ising:
        test_ising()

        
if __name__=="__main__":
    main()
