class DimensionException(Exception):
    """ Dimension Exception error, 
    Raised if the dimension of 2 values are not the same 
    before performing specific operation"

    Attributes:
        dimA : dimensionA
        dimB : dimensionB
        message -- explanation of the error
    """

    def __init__(self, dimA,dimB, message="Dimensions are not equals"):
        self.dimA = dimA
        self.dimB = dimB
        self.message = message
        super().__init__(self.message)
        