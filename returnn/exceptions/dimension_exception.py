class DimensionException(Exception):
    """ Dimension Exception error. Raised if the dimension are not the same"

    Attributes:
        dimA : dimensionA
        dimB : dimensionB
        message -- explanation of the error
    """

    def __init__(self, dimA, dimB, message="Dimensions are not equals"):
        self.dimA = dimA
        self.dimB = dimB
        self.message = message
        super().__init__(self.message)
