''' Import Libraries'''
import pandas as pd
import numpy as np
from discriminants import GaussianDiscriminant, MultivariateGaussian


class Classifier:
    ''' This is a class prototype for any classifier. It contains two empty methods: predict, fit'''
    def __init__(self):
        self.model_params = {}
        pass
    
    def predict(self, x):
        '''This method takes in x (numpy array) and returns a prediction y'''
        raise NotImplementedError
    
    def fit(self, x, y):
        '''This method is used for fitting a model to data: x, y'''
        raise NotImplementedError

class Prior(Classifier):
    def __init__(self):
        super().__init__()
        self.class_priors = {}
    
    def fit(self, x, y):
        '''Calculate prior probabilities from training labels'''
        unique_classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        self.class_priors = {cls: count / total_samples for cls, count in zip(unique_classes, counts)}
    
    def predict(self, x):
        '''Predict based only on the prior (always predicting the most frequent class)'''
        return max(self.class_priors, key=self.class_priors.get)

class DiscriminantClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.classes = {}
    
    def set_classes(self, *discs):
        '''Store discriminant objects in self.classes'''
        self.classes = {disc.name: disc for disc in discs}
    
    def fit(self, dataframe, label_key='Labels', default_disc = MultivariateGaussian):
        '''Train the model by creating a discriminant for each class.'''
        classes = dataframe[label_key].unique()
        for target_class in classes:
            class_data = dataframe[dataframe[label_key] == target_class]
            feature_data = class_data.drop(columns=[label_key]).to_numpy()
            self.classes[target_class] = default_disc(feature_data, prior= len(feature_data)/len(dataframe), name=target_class)
    
    def predict(self, x):
        '''Return the class corresponding to the highest discriminant score.'''
        scores = {}
        for target_class, disc in self.classes.items():
            scores[target_class] = disc.calc_discriminant(x)
        max_score_label = max(scores, key=scores.get)
        return max_score_label

    def pool_variances(self):
        '''Calculate the shared covariance matrix and update class parameters'''
        n_samples = sum(len(disc.params['mu']) for disc in self.classes.values()) 
        # If we consider degrees of freedom, we divide by n-1 for pooled variance
        pool_variance = sum((len(disc.params['mu']) - 1) * disc.params['sigma'] for disc in self.classes.values()) / n_samples  
        for disc in self.classes.values():
            disc.params['sigma'] = pool_variance 


       
        
        
        
