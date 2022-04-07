import json
import os
import pandas as pd
import numpy as np
from codecarbon import EmissionsTracker as ET
from opacus import PrivacyEngine 

class responsibleModel:
    
    __modelname__ = ""
    __modeltype__ = ""
    __emissions__ = 0.0
    __classbalance__ = 0.0
    __explained__ = False
    __epsilon__ = 0.0
    __tracker__ = None
    __privacy_engine__ = None
    
    def __init__(self,):
        self.__modelname__ = ""
        self.__modeltype__ = ""
        self.__emissions__ = 0.0
        self.__classbalance__ = 0.0
        self.__explained__ = False
        self.__epsilon__ = 0.0
        
        self.__tracker__ = ET(project_name = "",
            measure_power_secs = 15,
            save_to_file = False)
        
        self.__privacy_engine__ = PrivacyEngine()

    def __init__(self, 
                 modelname: str,
                 modeltype:str,
                 explained:bool = False,
                 emissions:float = 0.0,
                 classbalance:float= 0.0,
                 epsilon:float = 0.0):
        
        self.__modelname__ = modelname
        self.__modeltype__ = modeltype
        self.__emissions__ = emissions
        self.__classbalance__ = classbalance
        self.__explained__ = explained
        self.__epsilon__ = epsilon
    
        self.__tracker__ = ET(project_name = modelname,
            measure_power_secs = 15,
            save_to_file = False)
        
        self.__privacy_engine__ = PrivacyEngine()
    
    def explained(self, isexplained: bool):
        self.__explained__ = isexplained

    def emissions(self, carbon_emissions: float):
        self.__emissions__ = carbon_emissions

    def classbalance(self, minclass: float):
        self.__classbalance__ = minclass

    def epsilon(self, privacy_epsilon: bool):
        self.__epsilon__ = privacy_epsilon

    def model(self, modeltype: str):
        self.__modeltype__ = modeltype
        
    def __calculate_emissions_index(self):

        if self.__emissions__ <= 500:
            emissionIndex = 3
        elif self.__emissions__ > 500 and self.__emissions__ <= 10000:
            emissionIndex = 2
        else:
            emissionIndex = 1

        return emissionIndex

    def __calculate_privacy_index(self):
        if self.__epsilon__ <= 1:
            privacyIndex = 3
        elif self.__epsilon__ > 1 and self.__epsilon__ <= 10:
            privacyIndex = 2
        else:
            privacyIndex = 1

        return privacyIndex

    def __calculate_explainability_index(self):

        expIndex = 1

        if self.__modeltype__ == "deeplearning":
            return expIndex

        if self.__explained__ == True:
            expIndex = 3
        else:
            expIndex = 2

        return expIndex

    def __calculate_bias_index(self):
        if self.__classbalance__ >= 0.4:
            bindex = 3
        elif self.__classbalance__ > 0.2 and self.__classbalance__ < 0.4:
            bindex = 2
        else:
            bindex = 1

        return bindex
    
    def describe(self):
        return json.dumps(self.__dict__)
    
    def model_rai_components(self):
        
        emission_index = self.__calculate_emissions_index()
        privacy_index = self.__calculate_privacy_index()
        bias_index = self.__calculate_bias_index()
        explain_index = self.__calculate_explainability_index()
        RAI_index = self.rai_index()
        
        value = json.dumps({"model name": self.__modelname__,
                            "model type": self.__modeltype__,
                            "rai index": RAI_index,
                            "emissions": emission_index,
                            "privacy": privacy_index,
                            "bias": bias_index,
                            "explainability": explain_index})

        return value
        
    def rai_index(self):
    
        index = 0.0
        weights = 0.2

        emission_index = self.__calculate_emissions_index()
        privacy_index = self.__calculate_privacy_index()
        bias_index = self.__calculate_bias_index()
        explain_index = self.__calculate_explainability_index()

        index = weights * (emission_index + privacy_index + bias_index + explain_index)

        return index

    def track_emissions(self):
        # Calculate Emissions
        self.__tracker__.start()
        
    def stop_tracking(self):
        self.__emissions__ =  self.__tracker__.stop()
        
    def calculate_bias(self, df_label: str):
        
        # Get the number of classes & samples
        label_classes = df_label.value_counts(ascending=True)
        totalvalues = label_classes.sum()
        min_class_count = label_classes.values[1]
        
        #calcualte the bias
        self.__classbalance__ = min_class_count / totalvalues
        
    def privatize(self, model, optimizer, dataloader):
        model, optimizer, dataloader = self.__privacy_engine__.make_private(module=model,
                                                                            optimizer=optimizer,
                                                                            data_loader=dataloader,
                                                                            noise_multiplier=1.0,
                                                                            max_grad_norm=1.0,
                                                                            )
        
        self.__epsilon__ = self.__privacy_engine__.get_privacy_spent()
        
        return model, optimizer, dataloader
                
class models:
    model_list = []
    
    def __init__(self):
        self.model_list = []
    
    def add_model(self, modelname, modeltype, explained, emissions, bias, epsilon):
        model = responsibleModel(modelname, modeltype, explained, emissions, bias, epsilon)
        self.model_list.append(model)
        
    def add_model(self, model):
        self.model_list.append(model)
        
    def remove_model(self, modelname):
        self.model_list.remove(modelname)
        
    def list_models(self):
        model_json = ""
        for model in self.model_list:
            model_json += model.describe() 
            if model != self.model_list[-1]:
                model_json += ","
                                
            model_json += "\n"
            
        model_json = "[" + model_json + "]"
        
        return model_json
    
    def get_model(self, modelname):
        for model in self.model_list:
            if model.__modelname__ == modelname:
                return model
        return None
    
    def rank_models(self, rank_type = None):
        sorted_json = ""
        sorted_models = sorted(self.model_list, key=lambda x: x.rai_index(), reverse=True)
        for model in sorted_models:
            sorted_json += model.model_rai_components()
            if(model != sorted_models[-1]):
                sorted_json += ","
            sorted_json += "\n"
            
        sorted_json = "[" + sorted_json + "]"
        return sorted_json