import numpy
# --------------------------------------------------------------------------------------------------------------------
class classifier_FC_numpy(object):
    def __init__(self):
        self.name = "FC_numpy"
        self.verbose = True
        self.epochs = 100000
        self.learning_rate=1e-4
        self.params_values = None
        self.NN_ARCHITECTURE = [
            {"input_dim": 2, "output_dim": 25, "activation": "relu"},
            {"input_dim": 25, "output_dim": 50, "activation": "relu"},
            {"input_dim": 50, "output_dim": 50, "activation": "relu"},
            {"input_dim": 50, "output_dim": 25, "activation": "relu"},
            {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
        ]
# ----------------------------------------------------------------------------------------------------------------
    def maybe_reshape(self, X):
        if numpy.ndim(X) == 2:
            return X
        else:
            return numpy.reshape(X, (X.shape[0], -1))
#----------------------------------------------------------------------------------------------------------------------
    def sigmoid(self,Z):return 1/(1 + numpy.exp(-Z))
# ----------------------------------------------------------------------------------------------------------------------
    def relu(self,Z):return numpy.maximum(0, Z)
# ----------------------------------------------------------------------------------------------------------------------
    def sigmoid_backward(self,dA, Z):
        sig = self.sigmoid(Z)
        return dA * sig * (1 - sig)
# ----------------------------------------------------------------------------------------------------------------------
    def relu_backward(self,dA, Z):
        dZ = numpy.array(dA, copy = True)
        dZ[Z <= 0] = 0
        return dZ
# ----------------------------------------------------------------------------------------------------------------------
    def init_layers(self,nn_architecture, seed=99):
        numpy.random.seed(seed)
        params_values = {}
        for idx, layer in enumerate(nn_architecture):
            layer_idx = idx + 1
            layer_input_size  = layer["input_dim"]
            layer_output_size = layer["output_dim"]
            params_values['W' + str(layer_idx)] = numpy.random.randn(layer_output_size, layer_input_size) * 0.1
            params_values['b' + str(layer_idx)] = numpy.random.randn(layer_output_size, 1) * 0.1

        return params_values
# ----------------------------------------------------------------------------------------------------------------------
    def single_layer_forward_propagation(self,A_prev, W_curr, b_curr, activation="relu"):

        Z_curr = numpy.dot(W_curr, A_prev) + b_curr

        if activation is "relu":
            activation_func = self.relu
        else:
            activation_func = self.sigmoid

        return activation_func(Z_curr), Z_curr
# ----------------------------------------------------------------------------------------------------------------------
    def full_forward_propagation(self, X):

        memory = {}
        A_curr = numpy.transpose(X)

        NN_ARCHITECTURE = self.NN_ARCHITECTURE

        for idx, layer in enumerate(NN_ARCHITECTURE):
            layer_idx = idx + 1
            A_prev = A_curr

            activ_function_curr = layer["activation"]
            W_curr = self.params_values["W" + str(layer_idx)]
            b_curr = self.params_values["b" + str(layer_idx)]
            A_curr, Z_curr = self.single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

            memory["A" + str(idx)] = A_prev
            memory["Z" + str(layer_idx)] = Z_curr

        return A_curr, memory
# ----------------------------------------------------------------------------------------------------------------------
    def get_cost_value(self,Y_hat, Y):

        m = Y_hat.shape[1]
        cost = -1 / m * (numpy.dot(Y, numpy.log(Y_hat).T) + numpy.dot(1 - Y, numpy.log(1 - Y_hat).T))
        return numpy.squeeze(cost)
# ----------------------------------------------------------------------------------------------------------------------
    def convert_prob_into_class(self,probs):
        probs_ = numpy.copy(probs)
        probs_[probs_ > 0.5] = 1
        probs_[probs_ <= 0.5] = 0
        return probs_
# ----------------------------------------------------------------------------------------------------------------------
    def get_accuracy_value(self,Y_hat, Y):
        Y_hat_ = self.convert_prob_into_class(Y_hat)
        return (Y_hat_ == Y).all(axis=0).mean()
# ----------------------------------------------------------------------------------------------------------------------
    def single_layer_backward_propagation(self,dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):

        m = A_prev.shape[1]

        if activation is "relu":
            backward_activation_func = self.relu_backward
        else:
            backward_activation_func = self.sigmoid_backward

        dZ_curr = backward_activation_func(dA_curr, Z_curr)

        dW_curr = numpy.dot(dZ_curr, A_prev.T) / m
        db_curr = numpy.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = numpy.dot(W_curr.T, dZ_curr)
        return dA_prev, dW_curr, db_curr
# ----------------------------------------------------------------------------------------------------------------------
    def full_backward_propagation(self,Y_hat, Y, memory, params_values, nn_architecture):
        grads_values = {}

        # number of examples
        m = Y.shape[1]
        # a hack ensuring the same shape of the prediction vector and labels vector
        Y = Y.reshape(Y_hat.shape)

        # initiation of gradient descent algorithm
        dA_prev = - (numpy.divide(Y, Y_hat) - numpy.divide(1 - Y, 1 - Y_hat))

        for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
            # we number network layers from 1
            layer_idx_curr = layer_idx_prev + 1
            # extraction of the activation function for the current layer
            activ_function_curr = layer["activation"]

            dA_curr = dA_prev

            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]

            W_curr = params_values["W" + str(layer_idx_curr)]
            b_curr = params_values["b" + str(layer_idx_curr)]

            dA_prev, dW_curr, db_curr = self.single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr

        return grads_values
# ----------------------------------------------------------------------------------------------------------------------
    def update(self,params_values, grads_values, nn_architecture, learning_rate):

        # iteration over network layers
        for layer_idx, layer in enumerate(nn_architecture, 1):
            params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
            params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

        return params_values
# ----------------------------------------------------------------------------------------------------------------------
    def train(self, X, Y):

        NN_ARCHITECTURE = self.NN_ARCHITECTURE
        # initiation of neural net parameters

        Y[Y<0]=0
        Y = numpy.transpose(Y.reshape((Y.shape[0], 1)))

        self.params_values = self.init_layers(NN_ARCHITECTURE, 2)
        # initiation of lists storing the history
        # of metrics calculated during the learning process
        cost_history = []
        accuracy_history = []

        # performing calculations for subsequent iterations
        for i in range(self.epochs):
            # step forward
            Y_hat, cashe = self.full_forward_propagation(X)

            # calculating metrics and saving them in history
            cost = self.get_cost_value(Y_hat, Y)
            cost_history.append(cost)
            accuracy = self.get_accuracy_value(Y_hat, Y)
            accuracy_history.append(accuracy)

            # step backward - calculating gradient
            grads_values = self.full_backward_propagation(Y_hat, Y, cashe, self.params_values, NN_ARCHITECTURE)
            # updating model state
            self.params_values = self.update(self.params_values, grads_values, NN_ARCHITECTURE, self.learning_rate)

            if (i % 1000 == 0) and self.verbose:
                print("Iteration: {:} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))



        return
# ----------------------------------------------------------------------------------------------------------------------
    def predict(self,X):
        pred, memory = self.full_forward_propagation(X)
        pred = pred[0]
        res = numpy.vstack((pred,1-pred))
        return res.T
# ----------------------------------------------------------------------------------------------------------------------