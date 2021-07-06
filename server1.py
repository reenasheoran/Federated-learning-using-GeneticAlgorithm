import pickle
import socket
import time
import threading
import pygad
import pygad.nn
import pygad.gann
import numpy
import pandas as pd

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

# Creating support for multiple clients
class SocketThread(threading.Thread):
    
    # function capturing info of each client
    def __init__(self, connection, client_info, buffer_size=1024, recv_timeout=10000):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    # function for getting the data multiple times from the client
    def recv(self):
        received_data = b''
        while True:
            try:
                data = connection.recv(self.buffer_size)
                received_data += data
                

                if data == b'': # Received no message from the client.
                    received_data = b''
                    # If still nothing received for a number of seconds specified by the recv_timeout attribute, return with status 0 to close the connection.
                    if (time.time() - self.recv_start_time) > self.recv_timeout:
                        return None, 0 # None means no more messages and 0 means the connection is no longer active and it should be closed.
                elif str(data)[-2] == '.':
                    print("All data ({data_len} bytes) Received from {client_info}.".format(client_info=self.client_info, data_len=len(received_data)))
                    

                    if len(received_data) > 0:
                        try:
                            # In our case, an object representing the ML model will be sent over the socket.
                            # Thus there must be a mechanism to encode/decode objects. 
                            # The pickle library exists for this purpose. 
                            # Because everything in Python is an object, pickle can work with any data.
                            
                            # Decoding the data from bytes to string.
                            received_data = pickle.loads(received_data)
                            
                            # Returning the decoded data.
                            return received_data, 1

                        except BaseException as e:
                            print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                            return None, 0
                else:
                    # In case data are received from the client, update the recv_start_time to the current time to reset the timeout counter.
                    self.recv_start_time = time.time()

            except BaseException as e:
                print("Error Receiving Data from the Client: {msg}.\n".format(msg=e))
                return None, 0
    
    def model_averaging(self, model, other_model):
        model_weights = pygad.nn.layers_weights(last_layer=model, initial=False)
        other_model_weights = pygad.nn.layers_weights(last_layer=other_model, initial=False)
        
        new_weights = numpy.array(model_weights + other_model_weights)/2

        pygad.nn.update_layers_trained_weights(last_layer=model, final_weights=new_weights)

    def reply(self, received_data):
            global GANN_instance, data_inputs, data_outputs, model
        #if (type(received_data) is dict):
            if (("data" in received_data.keys()) and ("subject" in received_data.keys())):
                subject = received_data["subject"]
                print("Client's Message Subject is {subject}.".format(subject=subject))

                print("Replying to the Client.")
                if subject == "echo":
                    try:
                        data = {"subject": "model", "data": GANN_instance}
                        response = pickle.dumps(data)
                    except BaseException as e:
                        print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                elif subject == "model":
                    try:
                        GANN_instance = received_data["data"]
                        best_model_idx = received_data["best_solution_idx"]

                        best_model = GANN_instance.population_networks[best_model_idx]
                        if model is None:
                            model = best_model
                        else:
                            predictions = pygad.nn.predict(last_layer=model, data_inputs=data_inputs)
    
                            error = numpy.sum(numpy.abs(predictions - data_outputs))
    
                            # In case a client sent a model to the server despite that the model error is 0.0. In this case, no need to make changes in the model.
                            if error == 0:
                                data = {"subject": "done", "data": None}
                                response = pickle.dumps(data)
                                return

                            self.model_averaging(model, best_model)

                        # print(best_model.trained_weights)
                        # print(model.trained_weights)

                        predictions = pygad.nn.predict(last_layer=model, data_inputs=data_inputs)
                        print("Model Predictions: {predictions}".format(predictions=predictions))

                        error = numpy.sum(numpy.abs(predictions - data_outputs))
                        print("Error = {error}".format(error=error))

                        if error != 0:
                            data = {"subject": "model", "data": GANN_instance}
                            response = pickle.dumps(data)
                        else:
                            data = {"subject": "done", "data": None}
                            response = pickle.dumps(data)

                    except BaseException as e:
                        print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                else:
                    response = pickle.dumps("Response from the Server")
                            
                try:
                    self.connection.sendall(response)
                except BaseException as e:
                    print("Error Sending Data to the Client: {msg}.\n".format(msg=e))

            else:
                print("The received dictionary from the client must have the 'subject' and 'data' keys available. The existing keys are {d_keys}.".format(d_keys=received_data.keys()))
        #else:
            #print("A dictionary is expected to be received from the client but {d_type} received.".format(d_type=type(received_data)))


    def run(self):
        # Transmitting data of any length between the Client and the Server
        while True:
            self.recv_start_time = time.time()
            time_struct = time.localtime()
            date_time = "Waiting to Receive Data Starting from {day}/{month}/{year} {hour}:{minute}:{second} MDT".format(year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
            print(date_time)
            received_data, status = self.recv()
            if status == 0:
                self.connection.close()
                print("Connection Closed with {client_info} either due to inactivity for {recv_timeout} seconds or due to an error.".format(client_info=self.client_info, recv_timeout=self.recv_timeout), end="\n\n")
                break

            self.reply(received_data)

# creating a socket

family=socket.AF_INET # address family using IPV4
type=socket.SOCK_STREAM # transmission of data (sending/receiving) is done via TCP. for UDP, use SOCK_DGRAM
serversocket = socket.socket(family=family, type=type) # socket is created as an instance of the socket.socket class
print("Socket created!!!")

# Binding the Socket to an IP Address and Port Number
serversocket.bind(("localhost", 5000)) # localhost can later be changed to IP address using ipconfig info
print("Socket binded with IP address and port!!!")

# Listening for Incoming Connections

# The listen() method puts the socket into a listening mode for incoming connection from the clients
serversocket.listen(1) 
print("Socket listening for incoming client connection ...")

# Accepting Connections
all_data = b""
while True:
    accept_timeout = 1000
    serversocket.settimeout(accept_timeout)
    try:
        # connection is a new socket object referring to the connection between the client and the server. Through this socket, data can be sent/received from one side to another.
        # address is a tuple holding the socket information of the client (address and port number)
        connection, client_info = serversocket.accept()
        print("New Connection from {client_info}.".format(client_info=client_info))
        socket_thread = SocketThread(connection=connection,
                                     client_info=client_info, 
                                     buffer_size=1024,
                                     recv_timeout=10000)
        socket_thread.start()
    except socket.timeout:
        # Closing the Socket
        serversocket.close()
        print("(Timeout) Socket Closed Because no Connections Received.\n")
        break
