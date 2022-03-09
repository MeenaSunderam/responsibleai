import math

explainable = False
emissions = 0.0
bias = 0.0
diff_privacy = False

def set_explainability(isexplainaable: bool):
    explainable = isexplainaable
    
def set_emissions(emissions: float):
    emissions = emissions

def set_bias(bias: float):
    bias = bias

def set_diff_privacy(isdiffprivacy: bool):
    diff_privacy = isdiffprivacy

def rai_index():
    index = 0.0
    
    #calcuate explainability index
    exp_index = 0.0
    if explainable == True:
        exp_index = 3.0
    else:
        exp_index = 1.0
        
    #calcuate emissions index
    if emissions > 0.50 and emissions < 1.0:
        emissions_index = 2.0
    elif emissions >= 1.0:
        emissions_index = 1.0
    else:
        emissions_index = 3.0
        
    #calcuate Privacy index
    priv_index = 0.0
    if diff_privacy == True:
        priv_index = 3.0
    else:
        priv_index = 1.0
    
    index = (exp_index + emissions_index + priv_index) / 3.0
    return index
