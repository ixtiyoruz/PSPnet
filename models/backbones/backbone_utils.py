import warnings 

class BackboneUtils:
    
    def __init__(self, name):
        """
        layer name differs if you use model with different named layers,
        """
        self.name = name
        self.feature_layer_names = {"resnet50":("activation5c_branch2c"), "resnet101":("conv5_block3_out"), "resnet152":("")}
        self.auxillary_layer_names = {"resnet50":("activation4f_branch2c"), "resnet101":("conv4_block23_out"), "resnet152":("")}
        self.names = list(self.feature_layer_names.keys())
        if(not self.name in self.names):
            warnings.warn('The name ' + self.name + " does not exist in our backbone utils")

    def get_auxillary_layer(self, ):
        return self.feature_layer_names[self.name]

    def get_feature_layer(self, ):
        return self.feature_layer_names[self.name]