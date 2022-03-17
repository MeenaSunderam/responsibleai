class rML:
    
    __modeltype__ = ""
    __emissions__ = 0.0
    __bias__ = 0.0
    __explained__ = False
    __epsilon__ = 0.0

    def __init__(self, modeltype:str,
                 explained:bool,
                 emissions:float,
                 bias:float,
                 epsilon:float):

        self.__modeltype__ = modeltype
        self.__emissions__ = emissions
        self.__bias__ = bias
        self.__explained__ = explained
        self.__epsilon__ = epsilon
    
    def explained(self, isexplained: bool):
        self.__explained__ = isexplained

    def emissions(self, carbon_emissions: float):
        self.__emissions__ = carbon_emissions

    def bias(self, label_bias: float):
        self.__bias__ = label_bias

    def epsilon(self, privacy_epsilon: bool):
        self.__epsilon__ = privacy_epsilon

    def model(self, modeltype: str):
        self.__modeltype__ = modeltype
        
    def __calculate_emissions_index(self):

        print(self.__emissions__)

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
        if self.__bias__ <= 0.25 and self.__bias__ >= -0.25:
            bindex = 3
        elif self.__bias__ > 0.5 or self.__bias__ < -0.5:
            bindex = 1
        else:
            bindex = 2

        return bindex
    
    def rai_index(self):
    
        index = 0.0
        weights = 0.2

        emission_index = self.__calculate_emissions_index()
        privacy_index = self.__calculate_privacy_index()
        bias_index = self.__calculate_bias_index()
        explain_index = self.__calculate_explainability_index()

        print(emission_index)
        print(privacy_index)
        print(bias_index)
        print(explain_index)

        index = weights * (emission_index + privacy_index + bias_index + explain_index)

        return index