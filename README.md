# Federated learning using Genetic Algorithm
This is a demo project for applying the concepts of federated learning (FL) in Python using socket programming. The ML model is created using PyGAD package which trains ML models using the genetic algorithm (GA). I tried to create a classification model for bank problem using this approach. The classification goal is to predict if the client will subscribe a term deposit or not.<br>
![Concept](https://github.com/reenasheoran/Federated-learning-using-GeneticAlgorithm/blob/main/static/concept.ppm)
## Table of content
[Project Overview](#Project-Overview)<br>
[Motivation](#Motivation)<br>
[Federated Learning](#Federated-Learning)<br>
[Genetic Algorithm](#Genetic-Algorithm)<br>
[Installation](#Installation)<br>
[Project Files](#Project-Files)<br>
[Data Collection](#Data-Collection)<br>
[Data Preparation](#Data-Preparation)<br>
[Working of the Project](#Working-of-the-Project)<br>
[Screen Shots](#Screen-Shots)<br>
## Project Overview
In this project I tried to train the machine learning (ML) models using Federated Learning. The ML model is created using PyGAD, which is trained using the genetic algorithm (GA). The training data is available on client-side only and server has no access to that data. The test data is available on server side, which is used for model evaluation.
I have created a centralized client-server architecture in which a central server is used to orchestrate the different steps of the algorithms and coordinate all the participating nodes during the learning process. The server is responsible for the nodes selection at the beginning of the training process and for the aggregation of the received model updates.
## Motivation
Most of the data in the world are contained not publicly on servers, but on client-side devices and sources (i.e. mobile phones, edge devices). Typically, users don’t end up directly sharing their data anywhere outside their devices. And if they do, there’s no guarantee of privacy. <br>
Federated Learning (FL) is a new paradigm for building machine learning (ML) models that keeps user data private. In this way, ML models are typically trained on data that’s not been collected and stored in a centralized server. In fact, the data is decentralized and scattered across clients. Through this project, I'll apply the concepts of FL in Python to build a demonstration of how a system like this might work.
## Federated Learning
Federated learning (also known as collaborative learning) is a machine learning technique that trains an algorithm across multiple decentralized edge devices or servers holding local data samples, without exchanging them. This approach stands in contrast to traditional centralized machine learning techniques where all the local datasets are uploaded to one server, as well as to more classical decentralized approaches which often assume that local data samples are identically distributed(wikipedia).<br>
Federated learning enables multiple actors to build a common, robust machine learning model without sharing data, thus allowing to address critical issues such as data privacy, data security, data access rights and access to heterogeneous data. <br><br>
Assuming a federated round composed by one iteration of the learning process, the learning procedure can be summarized as follows:<br>
1. **Initialization:** according to the server inputs, a machine learning model (e.g., linear regression, neural network, boosting) is chosen to be trained on local nodes and initialized. Then, nodes are activated and wait for the central server to give the calculation tasks.<br>
2. **Client selection:** a fraction of local nodes is selected to start training on local data. The selected nodes acquire the current statistical model while the others wait for the next federated round.<br>
3. **Configuration:** the central server orders selected nodes to undergo training of the model on their local data in a pre-specified fashion (e.g., for some mini-batch updates of gradient descent).<br>
4. **Reporting:** each selected node sends its local model to the server for aggregation. The central server aggregates the received models and sends back the model updates to the nodes. It also handles failures for disconnected nodes or lost model updates. The next federated round is started returning to the client selection phase.<br>
5. **Termination:** once a pre-defined termination criterion is met (e.g., a maximum number of iterations is reached or the model accuracy is greater than a threshold) the central server aggregates the updates and finalizes the global model.<br>
## Genetic Algorithm
A genetic algorithm is a search-based algorithm used for solving optimization problems in machine learning. This algorithm is important because it solves difficult problems that would take a long time to solve. <br><br>
The following are some of the basic terminologies that can help us to understand genetic algorithms:<br>
1. **Population:** This is a subset of all the probable solutions that can solve the given problem.<br>
2. **Chromosomes:** A chromosome is one of the solutions in the population.<br>
3. **Gene:** This is an element in a chromosome.<br>
4. **Allele:** This is the value given to a gene in a specific chromosome.<br>
5. **Fitness function:** This is a function that uses a specific input to produce an improved output. The solution is used as the input while the output is in the form of solution suitability.<br>
6. **Genetic operators:** In genetic algorithms, the best individuals mate to reproduce an offspring that is better than the parents. Genetic operators are used for changing the genetic composition of this next generation.<br><br>
### Genetic Algorithms working
Genetic algorithms use the evolutionary generational cycle to produce high-quality solutions. They use various operations that increase or replace the population to provide an improved fit solution.Genetic algorithms follow the following phases to solve complex optimization problems:<br>
1. **Initialization**: The genetic algorithm starts by generating an initial population. This initial population consists of all the probable solutions to the given problem. The most popular technique for initialization is the use of random binary strings.<br>
2. **Fitness assignment**: The fitness function helps in establishing the fitness of all individuals in the population. It assigns a fitness score to every individual, which further determines the probability of being chosen for reproduction. The higher the fitness score, the higher the chances of being chosen for reproduction.<br>
3. **Selection**: In this phase, individuals are selected for the reproduction of offspring. The selected individuals are then arranged in pairs of two to enhance reproduction. These individuals pass on their genes to the next generation.The main objective of this phase is to establish the region with high chances of generating the best solution to the problem (better than the previous generation). The genetic algorithm uses the fitness proportionate selection technique to ensure that useful solutions are used for recombination.<br>
4. **Reproduction**:This phase involves the creation of a child population. The algorithm employs variation operators that are applied to the parent population. The two main operators in this phase include crossover and mutation.<br>
5. **Crossover**: This operator swaps the genetic information of two parents to reproduce an offspring. It is performed on parent pairs that are selected randomly to generate a child population of equal size as the parent population.<br>
6. **Mutation**: This operator adds new genetic information to the new child population. This is achieved by flipping some bits in the chromosome. Mutation solves the problem of local minimum and enhances diversification. The following image shows how mutation is done.<br>
7. **Replacement**: Generational replacement takes place in this phase, which is a replacement of the old population with the new child population. The new population consists of higher fitness scores than the old population, which is an indication that an improved solution has been generated.<br>
8. **Termination**: After replacement has been done, a stopping criterion is used to provide the basis for termination. The algorithm will terminate after the threshold fitness solution has been attained. It will identify this solution as the best solution in the population.<br>
## Installation
This project is developed using python 3.8. If you are using any lower version of python then I recommend you to upgrade your python to the latest version by using pip command. Follow the steps below to run this project locally.
```
git clone https://github.com/reenasheoran/Federated-learning-using-GeneticAlgorithm.git
cd Federated-learning-using-GeneticAlgorithm
pip install -r requirements.txt
python server1.py
python client1.py
python client2.py
```
## Project Files
The project has the following files:<br>
1. **server1.py**: It performs the following tasks:-<br>
  1. Firstly, it creates an initial model that is to be trained on the clients' devices using FL and sends the model to the selected client nodes. <br>
  2. Secondly, it gets the trained models from each client node.<br> 
  3. Thirdly, it tests the received models and calculate errors.<br>
  4. Lastly, it aggregates the weights of all models and repeats the process.<br>
3. **client1.py**: It trains the model sent by the server on its own local data to which server has no access.<br>
4. **client2.py**: Another client that trains the server's model on its own local data to which server has no access.<br>
## Data Collection
The data is taken from UCI Machine Learning Repository https://archive.ics.uci.edu/ml/datasets/bank+marketing (bank-additional-full.csv ) containing 41188 instances and 20 features, ordered by date (from May 2008 to November 2010)).<br>
**Features** <br>
1 - age (numeric)<br>
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')<br>
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)<br>
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')<br>
5 - default: has credit in default? (categorical: 'no','yes','unknown')<br>
6 - housing: has housing loan? (categorical: 'no','yes','unknown')<br>
7 - loan: has personal loan? (categorical: 'no','yes','unknown')<br><br>
_related with the last contact of the current campaign:_<br>
8 - contact: contact communication type (categorical: 'cellular','telephone')<br>
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')<br>
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')<br>
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model. <br><br>
_other attributes:_<br>
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)<br>
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)<br>
14 - previous: number of contacts performed before this campaign and for this client (numeric)<br>
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')<br><br>
_social and economic context attributes:_<br>
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)<br>
17 - cons.price.idx: consumer price index - monthly indicator (numeric)<br>
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)<br>
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)<br>
20 - nr.employed: number of employees - quarterly indicator (numeric)<br>

**Output variable (desired target)**<br>
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')<br>
## Data Preparation
To train a model using FL, the training data is distributed across the clients. The server itself does not have any training data, just test data to assess the model received from the clients. <br>
For the execution purpose of this project the bank data was horizontally splitted into three equal parts ( rows each). Out of these three dataset parts, one was kept at server side (as test data) and the remaining two parts were kept at two different client locations to which server has no access at all.
## Working of the Project
The interaction between client and server is implemented by using programming sockets with the socket Python library. The client-server architecture will handle multiple clients simultaneously. Using PyGAD, I also created a feed-forward neural network (FFNN), also using a genetic algorithm (GA).
 Using FL, a model can be created directly out of such private data. 
## Screen Shots
![image1](https://github.com/reenasheoran/Federated-learning-using-GeneticAlgorithm/blob/main/static/7.png)
![image2](https://github.com/reenasheoran/Federated-learning-using-GeneticAlgorithm/blob/main/static/4.png)
![image3](https://github.com/reenasheoran/Federated-learning-using-GeneticAlgorithm/blob/main/static/1.png)
![image4](https://github.com/reenasheoran/Federated-learning-using-GeneticAlgorithm/blob/main/static/5.png)
![image5](https://github.com/reenasheoran/Federated-learning-using-GeneticAlgorithm/blob/main/static/2.png)
![image6](https://github.com/reenasheoran/Federated-learning-using-GeneticAlgorithm/blob/main/static/3.png)
![image7](https://github.com/reenasheoran/Federated-learning-using-GeneticAlgorithm/blob/main/static/6.png)

