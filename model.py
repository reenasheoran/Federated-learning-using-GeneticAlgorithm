import pygad
from pygad import gann

data = pd.read_csv('data/data1.csv')

X= data.drop('y',axis=1)
Y= data.y

model = None

# Preparing the NumPy array of the inputs.
data_inputs = numpy.array(X)

# Preparing the NumPy array of the outputs.
data_outputs = numpy.array(Y)

num_classes = 2
num_inputs = 64

num_solutions = 6 # number of clients (or network size)
GANN_instance = pygad.gann.GANN(num_solutions=num_solutions,
                                num_neurons_input=num_inputs,
                                num_neurons_hidden_layers=[2],
                                num_neurons_output=num_classes,
                                hidden_activations=["relu"],
                                output_activation="softmax")

GANN_instance.population_networks
print(GANN_instance.population_networks)

population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)
print(population_vectors)
