# SPECTRE - with great powers comes great responsibility

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

![image](https://user-images.githubusercontent.com/7538839/160517464-a716c6d9-bbf3-4255-8710-d090e11abf1a.png)

You can add a bunch of Models for a specific use case to the SPECTRE framework and SPECTRE ranks them on the basis of its over health or health per dimension

## Pre-requisistes

1. Emission tracking is calcualted using [Code Carbon](https://codecarbon.io/).  Install Code Carbon using

>     !pip install codecarbon

2.Differential privacy is calculated usig [Opacus](https://opacus.ai/).  Install Opacus

>     !pip install opacus

3.Responsible index is calcualted using ResponsibleML.  Install ResponsibleML

>     !pip install -i https://test.pypi.org/simple/ spectre

## Usage

1. Import the package in your code

>     from aigovernance import spectre

2.Initiate a Model and a Model List

>     r_model = spectre.responsibleModel(modelname, modeltype)
>     model_list = spectre.models()

3. Bias Information - During data engineering phase of your ML life cycle, send the label data frame (Y) to responsibleModel

>     r_model.calcualte_bias(label_df)

4. Model Explainability - given there is no straight forward way to identify if the model explainability is performed, you will have to add it independently

>     r_model.explained(isExplained)

5. Carbon Emissions - For calculating carbon emissions, before you start your model training, start the tracker.  Once the model training is complete, stop the tracker

>     r_model.track_emissions()
> 
>     << your model training>>
> 
>     r_model.stop_tracking()

6. Differential Privacy - [Work in Progress]

7. Responsible Index:  You can now retrieve the responsible index of the model using

>     responsible_index = r_model.rai_index()

8. Given the nature of ML, you will be building multiple models.  Use steps 2 to 7 initiate the model with responsible dimension.  You can now add the models you created on to a model list

>     model_list.add_model(r_model)

9. You can list all the models and their responsible score using 

>     model_list.list_models()

10. finally, to identify the most responsible model that you can use for your production use case, you can get the models ranked by their responsibility index
> 
>     model_list.rank_models()
