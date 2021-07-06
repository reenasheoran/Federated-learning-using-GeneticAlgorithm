import pygad
from pygad import gann


GANN_instance = pygad.gann.GANN(num_solutions=6,
                                num_neurons_input=2,
                                num_neurons_hidden_layers=[2],
                                num_neurons_output=2)

GANN_instance.population_networks
print(GANN_instance.population_networks)

population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)
print(population_vectors)