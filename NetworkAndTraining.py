import scipy.io
import lasagne
import numpy as np
import theano.tensor as T
import theano
import time
import pickle
import h5py

class Unpool2DLayer(lasagne.layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, ds, **kwargs):

        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if (isinstance(ds, int)):
            raise ValueError('ds must have len == 2')
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must have len == 2')
            if ds[0] != ds[1]:
                raise ValueError('ds should be symmetric (I am lazy)')
            self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        ds = self.ds
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(ds[0], axis=2).repeat(ds[1], axis=3)




class BatchNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, axes=None, epsilon=0.01, alpha=0.5,
                 nonlinearity=None, **kwargs):
        """
        Instantiates a layer performing batch normalization of its inputs,
        following Ioffe et al. (http://arxiv.org/abs/1502.03167).

        @param incoming: `Layer` instance or expected input shape
        @param axes: int or tuple of int denoting the axes to normalize over;
            defaults to all axes except for the second if omitted (this will
            do the correct thing for dense layers and convolutional layers)
        @param epsilon: small constant added to the standard deviation before
            dividing by it, to avoid numeric problems
        @param alpha: coefficient for the exponential moving average of
            batch-wise means and standard deviations computed during training;
            the larger, the more it will depend on the last batches seen
        @param nonlinearity: nonlinearity to apply to the output (optional)
        """
        super(BatchNormLayer, self).__init__(incoming, **kwargs)
        if axes is None:
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        self.epsilon = epsilon
        self.alpha = alpha
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity
        shape = list(self.input_shape)
        broadcast = [False] * len(shape)
        for axis in self.axes:
            shape[axis] = 1
            broadcast[axis] = True
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all dimensions/axes not normalized over.")
        dtype = theano.config.floatX
        self.mean = self.add_param(lasagne.init.Constant(0), shape, 'mean',
                                   trainable=False, regularizable=False)
        self.std = self.add_param(lasagne.init.Constant(1), shape, 'std',
                                  trainable=False, regularizable=False)
        self.beta = self.add_param(lasagne.init.Constant(0), shape, 'beta',
                                   trainable=True, regularizable=True)
        self.gamma = self.add_param(lasagne.init.Constant(1), shape, 'gamma',
                                    trainable=True, regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            # use stored mean and std
            mean = self.mean
            std = self.std
        else:
            # use this batch's mean and std
            mean = input.mean(self.axes, keepdims=True)
            std = input.std(self.axes, keepdims=True)
            # and update the stored mean and std:
            # we create (memory-aliased) clones of the stored mean and std
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_std = theano.clone(self.std, share_inputs=False)
            # set a default update for them
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * mean)
            running_std.default_update = ((1 - self.alpha) * running_std +
                                          self.alpha * std)
            # and include them in the graph so their default updates will be
            # applied (although the expressions will be optimized away later)
            mean += 0 * running_mean
            std += 0 * running_std
        std += self.epsilon
        mean = T.addbroadcast(mean, *self.axes)
        std = T.addbroadcast(std, *self.axes)
        beta = T.addbroadcast(self.beta, *self.axes)
        gamma = T.addbroadcast(self.gamma, *self.axes)
        normalized = (input - mean) * (gamma / std) + beta
        return self.nonlinearity(normalized)


def batch_norm(layer,**kwargs):
    """
    Convenience function to apply batch normalization to a given layer's output.
    Will steal the layer's nonlinearity if there is one (effectively introducing
    the normalization right before the nonlinearity), and will remove the
    layer's bias if there is one (because it would be redundant).
    @param layer: The `Layer` instance to apply the normalization to; note that
        it will be irreversibly modified as specified above
    @return: A `BatchNormLayer` instance stacked on the given `layer`
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return BatchNormLayer(layer, nonlinearity=nonlinearity,**kwargs)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def load_dataset_Train_Validation():
    TrainValidationData = h5py.File('Path_to_database_train_validation.mat')


    TrainX = TrainValidationData['TrainDataX']
    TrainY = TrainValidationData['TrainDataY']
    ValidationX = TrainValidationData['ValidationDataX']
    ValidationY = TrainValidationData['ValidationDataY']

    print('Reorienting the Data...')

    TrainX = np.float32(np.transpose(TrainX , axes=(2,1,0)))
    TrainX = TrainX.reshape(-1,1,96,128)
    TrainX = TrainX / np.float32(256)
    TrainY = np.float32(np.transpose(TrainY , axes=(2,1,0)))
    TrainY = TrainY.reshape(-1,1,96,128)
    ValidationX = np.float32(np.transpose(ValidationX , axes=(2,1,0)))
    ValidationX = ValidationX.reshape(-1,1,96,128)
    ValidationX = ValidationX / np.float32(256)
    ValidationY = np.float32(np.transpose(ValidationY , axes=(2,1,0)))
    ValidationY = ValidationY.reshape(-1,1,96,128)
    return TrainX , TrainY , ValidationX , ValidationY


TrainX , TrainY , ValidationX , ValidationY = load_dataset_Train_Validation()


input_var = T.tensor4('inputs')
target_var = T.tensor4('targets')
batch_size = 100
weight_decay = 0.00014

print('Building the model...')

networki = lasagne.layers.InputLayer(shape=(None , 1 , None , None),name='networki')
netL1 = lasagne.layers.Conv2DLayer(networki , num_filters=10 , filter_size=(3,3),pad='same',name='netL1')
netL1BN = batch_norm(netL1 , name = 'netL1BN')
netL2 = lasagne.layers.Conv2DLayer(netL1BN , num_filters=10,filter_size=(5,5),pad='same',name='netL2')
netL2BN = batch_norm(netL2,name='netL2BN')
netL3 = lasagne.layers.Conv2DLayer(netL2BN,num_filters=20,filter_size=(7,7) , pad='same',name='netL3')
netL3BN = batch_norm(netL3,name='netL3BN')
netL4 = lasagne.layers.Conv2DLayer(netL3BN,num_filters=20,filter_size=(9,9),pad='same',name='netL4')
netL4BN = batch_norm(netL4,name='netL4BN')
netL5 = lasagne.layers.Conv2DLayer(netL4BN,num_filters=30,filter_size=(11,11),pad='same',name='netL5')
netL5BN = batch_norm(netL5,name='netL5BN')
netL6 = lasagne.layers.Conv2DLayer(netL5BN,num_filters=30,filter_size=(13,13),pad='same',name='netL6')
netL6BN = batch_norm(netL6,name='netL6BN')
netL7 = lasagne.layers.Conv2DLayer(netL6BN,num_filters=40,filter_size=(15,15),pad='same',name='netL7')
netL7BN = batch_norm(netL7,name='netL7BN')
netL8 = lasagne.layers.Conv2DLayer(netL7BN,num_filters=30,filter_size=(13,13),pad='same',name='netL8')
netL8BN = batch_norm(netL8,name='netL8BN')
netMerge1 = lasagne.layers.ConcatLayer((netL6BN,netL8BN),name='netMerge1')
netL9 = lasagne.layers.Conv2DLayer(netMerge1,num_filters=30,filter_size=(11,11),pad='same',name='netL9')
netL9BN = batch_norm(netL9,name='netL9BN')
netMerge2 = lasagne.layers.ConcatLayer((netL5BN,netL9BN),name='netMerge2')
netL10 = lasagne.layers.Conv2DLayer(netMerge2,num_filters=20,filter_size=(9,9),pad='same',name='netL10')
netL10BN = batch_norm(netL10,name='netL10BN')
netMerge3 = lasagne.layers.ConcatLayer((netL4BN,netL10BN),name='netMerge3')
netL11 = lasagne.layers.Conv2DLayer(netMerge3,num_filters=20,filter_size=(7,7),pad='same',name='netL11')
netL11BN = batch_norm(netL11,name='netL11BN')
netL12 = lasagne.layers.Conv2DLayer(netL11BN,num_filters=10,filter_size=(5,5),pad='same',name='netL12')
netL12BN = batch_norm(netL12,name='netL12BN')
networkOut = lasagne.layers.Conv2DLayer(netL12BN,num_filters=1,filter_size=(3,3),pad='same'
                                        ,nonlinearity=lasagne.nonlinearities.sigmoid,name='networkOut')


output = T.add(lasagne.layers.get_output(networkOut , inputs=input_var),np.finfo(np.float32).eps)

loss = lasagne.objectives.binary_crossentropy(output , target_var)

loss = loss.mean()

params = lasagne.layers.get_all_params(networkOut , trainable = True)
updates = lasagne.updates.nesterov_momentum(loss , params , learning_rate = 0.01 , momentum=0.9)
#updates = lasagne.updates.sgd(loss , params , learning_rate = 0.01)
test_output = T.add(lasagne.layers.get_output(networkOut ,inputs=input_var, deterministic = True),np.finfo(np.float32).eps)
test_loss = lasagne.objectives.binary_crossentropy(test_output , target_var)
test_loss = test_loss.mean()

train_func = theano.function([input_var , target_var] , loss , updates=updates)
valid_func = theano.function([input_var , target_var] , test_loss)

print('Training...')
TRAINLOSS = np.array([])
VALIDATIONLOSS = np.array([])
num_epochs = 1000
validationErrorBest = 100000;

for epoch in range(num_epochs):

    train_err = 0
    train_batches = 0

    start_time = time.time()
    for batch in iterate_minibatches(TrainX , TrainY , batch_size , shuffle=True):
        inputs , targets = batch

        train_err += train_func(inputs , targets)
        train_batches += 1
        err = valid_func(inputs , targets)

    val_err = 0
    val_batches = 0

    for batch in iterate_minibatches(ValidationX , ValidationY , batch_size , shuffle=False):
        inputs , targets = batch
        err = valid_func(inputs , targets)
        val_err += err
        val_batches +=1

    validationError = val_err / val_batches
    if validationError < validationErrorBest:
            validationErrorBest = validationError
            with open('netBestAttempt9SPDNN.pickle' , 'wb') as handle:
                print('saving the model....')
                pickle.dump(networkOut , handle)


    print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    TRAINLOSS = np.append(TRAINLOSS , train_err / train_batches)
    VALIDATIONLOSS = np.append(VALIDATIONLOSS , val_err / val_batches)
    scipy.io.savemat('LossesAttempt9SPDNN.mat',mdict={'TrainLoss':TRAINLOSS , 'ValidationLoss':VALIDATIONLOSS})


print 'Training Finished'
