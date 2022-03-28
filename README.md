# SPECTRE - With great powers comes great Responsibility

## What is SPECTRE

SPECTRE is a AI Governance framework.  With increased use of AI / ML in critical decision making process, SPECTRE provides a objective index for measuring the health of the model across the following dimensions

- Security:  Who can access the Model in production
- Privacy:  Differential Privacy of data used in model training exercise
- Explainability:  Ability of the model to explain the prediction
- Consistency:  Model consistency across multiple test data sets
- Transparency:  Interpretability of the Model
- Responsble: Responsibility on Bias and Fairness of the Model
- Environment friendly:  Carbon Emissions of the Model training / Inference

## How does it work

1.  Build the Model using framework of your choice.  
2.  Use any package to decipher the SPECTRE components.  For eg. Code Carbon for Carbon Emissions, FairTorch for Bias, SageMaker Clarify for Bias, MSFT Interpret ML for Transparency of the Model etc...
4.  Import the SPECTRE package (Python package)
5.  Add the details of the SPECTRE components of your model to SPECTRE Framework
6.  SPECTRE calculates an Index (float) that denotes the health of the model.  Higher the number better the model


## Support

At the time of this writing, the MVP of the SPECTRE framework supports 
- Differential Privacy
- Explainability
- Responsibility 
- Environment Friendly (Carbon emissions)

You can add a bunch of Models for a specific use case to the SPECTRE framework and SPECTRE ranks them on the basis of its over health or health per dimension


## Installation

Currently available in test.pypi

To install from test.pypy use, 

pip install -i https://test.pypi.org/simple/responsibleML

## Usage

from aigovernance import responsibleML

responsibleModel = responsibleML.rML(modeltype = "", 
                                     explained = False, 
                                     emissions = 700, 
                                     bias = 0.2, 
                                     epsilon = 0.8)

raiIndex = responsibleModel.rai_index()
