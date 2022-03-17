# Responsible ML

## Support

Responsible ML supports creating a Responsible AI / ML Index covering
1. Carbon Emissions
2. Imbalance class - Bias on labels
3. Differential Privacy 
4. Explainability of the model

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
