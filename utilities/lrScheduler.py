#Library Imports
import math
import weakref
import warnings
from .optimizer import Optimizer
from functools import wraps
#Using Adam optimizer with 
#Beta_1=0.9, Beta_2=0.98, and Epsilon=10^-9

#Learning rate varies over course of training
#lrate = sqrt(d_model)*min((1/sqrt(step_num)), step_num*(1/warmup_steps*sqrt(warmup_steps)))
#warmup_steps is 4,000
class lrScheduler:
    #Initialization of Class
    def __init__(self, optimizer, model_dim, warmup_steps=4000, last_epoh=-1):
        #Store Values
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.model_dim = model_dim
        #Begin Calculations
        self.invsqrt_dim = (1/math.sqrt(model_dim))
        self.invsqrt_warmup = (1/(warmup_steps*math.sqrt(warmup_steps)))
        self.initial_lr = self.invsqrt_dim*1*self.invsqrt_warmup
        #Initialize Learning Rate In Optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group['lr'] = self.initial_lr
            last_epoch = 0
        self.last_epoch = last_epoch

        #Ensure optimizer.step() is called prior to lr_scheduler.step()
        def with_OptimizerCounter(method):
            #Return if the step method of the optimizer has already been replaced
            if getattr(method, '_with_counter', False):
                return method
            #Keep a reference to the optimizer and get unbounded methods for class
            instance_ref = weakref.ref(method.__self__)
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)
            
            wrapper._with_counter = True
            return wrapper

            self.optimizer.step = with_OptimizerCounter(self.optimizer.step)
            self.optimizer._step_count = 0
            self._step_count = 0
            self.step(last_epoch)
    #Ensure optimizer.step() is called prior to this method
    def step(self, epoch=None):
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("optimizer.step() has not been overriden after the learning rate scheduler initialization")
            elif self.optimizer._step_count < 1:
                warnings.warn("optimizer.step() was not called prior to lrScheduler.step()")
        self._step_count += 1
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if self._step_count <= self.warmup_steps:
            for group in self.optimizer.param_groups:
                self.lrAdj = self.invsqrt_dim*self._step_count*self.invsqrt_warmup
                group['lr'] = self.lrAdj
        elif self._step_count > self.warmup_steps:
            self.invsqrt_step = (1/math.sqrt(self._step_count))
            for group in self.optimizer.param_groups:
                self.lrAdj = self.invsqrt_dim*self.invsqrt_step
                group['lr'] = self.lrAdj