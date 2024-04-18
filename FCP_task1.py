import math
import numpy as np
import matplotlib.pyplot as plt
import argparse



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



# plotting code
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

# The terminal 
def main():
    parser = argparse.ArgumentParser(description='process the users_s input ')
    #adding four command-line parameter
    parser.add_argument('-ising_model', action='store_true')
    parser.add_argument('-test_ising', action='store_true')
    parser.add_argument('-external', type=float, default=0.0)
    parser.add_argument('-alpha', type=float, default=1.0)

   # store the data on the args.
    args= parser.parse_args()

#if the user enter the ising_model or test_ising, running these two funtions
    if args.ising_model:
        ising_main(population, alpha=args.alpha, external=args.external)

    if args.test_ising:
        test_ising()

if __name__ =='__main__':
    main()