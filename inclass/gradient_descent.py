import random

def function(x):
    return x**4 - 3*x**3 + 2

def derivative(x):
    return 4*x**3 - 9*x**2

def gd(x, alpha, terminate):
    gradient = derivative(x)
    step = -alpha*gradient
    while abs(step) > terminate:
        x += step
        gradient = derivative(x)
        step = -alpha*gradient
    return x

def monte_gd(alpha, terminate, n):
    # Run gradient descent n times
    samples = []
    for i in range(n):
        # Start from a random point
        x = 10*random.random() - 5
        samples.append(gd(x, alpha, terminate))

    # Get the resulting value for each sample
    results = []
    for sample in samples:
        results.append(function(sample))
    
    # Zip the samples and results together
    zipped = list(zip(samples, results))

    # Sort the zipped list by the results
    zipped.sort(key=lambda x: x[1])

    # Return the best sample
    return zipped[0][0]

if __name__ == "__main__":
    # Start from a random point
    x = 10*random.random() - 5

    # Set learning rate
    alpha = 0.01
    
    # Set termination condition
    terminate = 0.0001

    # Run gradient descent
    x = gd(x, alpha, terminate)
    
    # Print the resulting x value
    print("GD (once):", x)

    trials = 1000
    # Run monte carlo gradient descent
    monte_x = monte_gd(alpha, terminate, trials)

    # Print the resulting x value
    print("GD (" + str(trials) + " times):", monte_x)
