# Abstract base class for data managers

from abc import ABC, abstractmethod
import numpy as np
from multiprocessing import Process, Queue, Pipe


class Data_manager(ABC):
     
    def __init__(self,read_dataset=True):
        # data preparations go here
        pass

    @abstractmethod
    def make_batch(self,rng_key,subset,n_examples):
        # Prepare a batch of the indicated specification.
        # Should return a dict with the elements specified below.
        """
        batch = {'states': [...],
                 'actions': [...],
                 'auxes': [...], # optional
                 'meta_data': [...], # optional
                 }
        return batch
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_example_count(self,subset):
        # return the number of example sequences in the indicated set.
        # return n_data
        raise NotImplementedError
    
    @abstractmethod
    def get_step_count(self,subset=None,index=None):
        # return the number of steps in the indicated sequence.
        # return n_steps
        raise NotImplementedError
    
    def get_planning_mask(self,n_steps):
        # Should return None or a binary array indicating which values of the 
        # action sequence should be updated each step of the planning process.
        # Return None to let all values be updated.
        # Otherwise, the mask should be shaped like an action sequence of length
        # n_steps, with 0 values indicating that a value should be kept static 
        # and 1 values indicating that a value should be updated.
        return None
    
    def modify_planning_batch(self,batch,n_strains):
        # Override this to modify how the parallel search strains for planning are initialised.
        # E.g. if there are multiple equivalent representations of the task, it can be helpful
        # to initialise different strains with different representations.
        return batch
    
    def generate_plan_equivalents(plan,task):
        return plan
    
    @abstractmethod
    def calculate_scores(self,predicted_state,ground_truth_state,predicted_aux,ground_truth_aux):
        # Calculate score(s) for the given prediction.
        # Should return a list of score values.
        # Format of the score is free, so you can e.g. return a string including a score name and value.
        # The score list will be included in relevant calls to draw_panel.
        raise NotImplementedError
    
    def planning_step_task_preprocessing(self,task):
        # Runs before each iteration of the plan search process.
        # Can be used to e.g. update task content on basis of the actions being evaluated in this planning iteration.
        # Note that this method's time cost adds to each iteration of the planning process.
        return task
    
    def shutdown(self):
        print('data manager shuts down')
