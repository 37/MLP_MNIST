from __future__ import print_function
__docformat__ = 'restructuredtext en'

import timeit
import numpy
import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data

class HiddenLayer(object):
    def __init__ (self, rnd, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rnd.uniform(
                    low = -numpy.sqrt(6. / (n_in, n_out)),
                    high = numpy.sqrt(6. / (n_in, n_out)),
                    size = (n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) = self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        # Parameters of the model
        self.params = [self.W, self.b]

class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):

        # create hidden layer with tanh activation function
        self.hiddenLayer = HiddenLayer(
            rnd = rng,
            input = input,
            n_in = n_in,
            n_out = n_hidden,
            activation = T.tanh
        )

        # The logistic regression layer gets as inputs the output of the hiddne layer
        self.logRegressionLayer = LogisticRegression(
            input = self.hiddenLayer.output,
            n_in = n_hidden,
            n_out = n_out
        )

        # Regularisation option L1 norm
        self.L1 = (
            abs(self.hiddenLayer.W).sum(),
            + abs(self.logRegressionLayer.W).sum()
        )

        # Regularisation option L2 sqrt
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        self.input = input


def test_mlp(learning_rate=0.01, L1_reg=0.00, l2_reg=0.0001, n_epochs=1000, dataset='csv', batch_size=20, n_hidden=500):
    print('loading data:')
    datasets = load_data(dataset)
    print('...done!')
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ########################
    ## Building the model ##
    ########################
    print('building the model:')

    # Allocate symbolic variables for the data
    index = T.lscalar() # Index to a minibatch
    x = T.matrix('x')   # Data presented as a rasterized image
    y = T.ivector('y')  # Labels presented as a 1D vector

    rnd = numpy.random.RandomState(1234)

    # Construct MLP class
    classifier = MLP(
        rng = rng,
        input = x,
        n_in = (28 * 28),
        n_hidden = n_hidden,
        n_out = 10
    )

    # Cost minimized as the -ve log likelihood of model + regularisation terms(L1 + L2)
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # Function to test mistakes of model on minibatch using Theano
    test_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(y),
        givens = {
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(y),
        givens = {
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # Compute the gradient of the cost with respect to theta, store list as gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)
    ]

    # Compile a theano function 'train_model' that returns the cost and updates the parameters
    # of the model based on the rules defined in 'updates'
    train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        givens = {
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print('...done!')
    ########################
    ## Training the model ##
    ########################
    print('Training the model:')

    patience = 10000 # Look at at-least this many examples
    patience_increase = 2 # wait this much longer when a new best is found
    improvement_threshold = 0.995 # a relative improvement if this much is considered significant
    validation_frequency = min( #check at least this num of minibatches before validation
        n_train_batches,
        patience // 2
    )
    # set defaults
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()
    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch ++
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


if __name__ == '__main__':
    test_mlp()
