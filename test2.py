import mnist_loader
import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.SGD(training_data[:1000], 30, 10, 0.5,
net.SGD(training_data, 30, 10, 0.5,
    lmbda=5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    early_stopping_n=0)