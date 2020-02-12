from keras import backend as K
from keras.layers import Activation
from keras.optimizers import Optimizer
from keras.utils.generic_utils import get_custom_objects

class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x, beta = 1):
    return (K.sigmoid(beta * x) * x)

get_custom_objects().update({'swish': Swish(swish)})

class iRprop_(Optimizer):
    def __init__(self, init_alpha=0.01, scale_up=1.2, scale_down=0.5, min_alpha=0.00001, max_alpha=50., **kwargs):
        super(iRprop_, self).__init__(**kwargs)
        self.init_alpha = K.variable(init_alpha, name='init_alpha')
        self.scale_up = K.variable(scale_up, name='scale_up')
        self.scale_down = K.variable(scale_down, name='scale_down')
        self.min_alpha = K.variable(min_alpha, name='min_alpha')
        self.max_alpha = K.variable(max_alpha, name='max_alpha')

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        alphas = [K.variable(K.ones(shape) * self.init_alpha) for shape in shapes]
        old_grads = [K.zeros(shape) for shape in shapes]
        self.weights = alphas + old_grads
        self.updates = []

        for p, grad, old_grad, alpha in zip(params, grads, old_grads, alphas):
            grad = K.sign(grad)
            new_alpha = K.switch(
                K.greater(grad * old_grad, 0),
                K.minimum(alpha * self.scale_up, self.max_alpha),
                K.switch(K.less(grad * old_grad, 0),K.maximum(alpha * self.scale_down, self.min_alpha),alpha)    
            )

            grad = K.switch(K.less(grad * old_grad, 0),K.zeros_like(grad),grad)
            new_p = p - grad * new_alpha 

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))
            self.updates.append(K.update(alpha, new_alpha))
            self.updates.append(K.update(old_grad, grad))

        return self.updates

    def get_config(self):
        config = {
            'init_alpha': float(K.get_value(self.init_alpha)),
            'scale_up': float(K.get_value(self.scale_up)),
            'scale_down': float(K.get_value(self.scale_down)),
            'min_alpha': float(K.get_value(self.min_alpha)),
            'max_alpha': float(K.get_value(self.max_alpha)),
        }
        base_config = super(iRprop_, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

