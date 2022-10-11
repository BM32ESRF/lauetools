# Simple minded model objects for Laue spots
class Spot(object):
    """
    Simple minded object that represents a fitted laue spot
    """

    def __init__(self, data, listoffields):
        for val, field in zip(data, listoffields):
            self.__setattr__(field, float(val))


#            print "%s field is :" % field, self.__getattribute__(field)

#    def GetSizeInMb(self):
#        return self.sizeInBytes / (1024.0 * 1024.0)

def GetAllspots(listofdata, listoffields):
    """
    Return a list of spotscollection of tracks
    """
    #    print Spot(listofdata[0], listoffields)
    return [Spot(data, listoffields) for data in listofdata]
