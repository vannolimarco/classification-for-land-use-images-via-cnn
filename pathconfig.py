class paths(object):
    """
    This is a class with goals to call all data paths from it. It simplifies and streamlines the code from long paths.
    It is used following this rules:
    - in the file needs to include the file : import pathconfig,
    - create object from class : paths = pathconfig.paths()
    - call path from property of class: for example path_flow_train = paths.FLOW_TRAIN
    Change all path in order to set own path and used them in the code.
    I remember that for path mappings the path are the same. So use this class to call them.
    Many files were deleted so
    """

    def __init__(self):
        self.DATA_DIR = '.\\data\\'    #the folder resources


        # Resources path base (Dataset UCMerced)
        self.BASE_FLOW = _BASE_FLOW = '.\\data\\flow'
        self.UCMERCED_LANDUSE_DATASET = _UCMERCED_LANDUSE_DATASET = self.DATA_DIR + 'UCMerced_LandUse\\Images\\'
        self.UCMERCED_LANDUSE_DATASET_ZIP = _UCMERCED_LANDUSE_DATASET_ZIP = self.DATA_DIR + 'UCMerced_LandUse.zip'

        #bottleneck features
        self.BN_TEST_X = _BN_TEST_X = self.DATA_DIR + 'bn_test_X.npy'
        self.BN_TEST_Y = _BN_TEST_Y = self.DATA_DIR + 'bn_test_y.npy'
        self.BN_TRAIN_X = _BN_TRAIN_X = self.DATA_DIR + 'bn_train_X.npy'
        self.BN_TRAIN_Y = _BN_TRAIN_X = self.DATA_DIR + 'bn_train_y.npy'
        self.BN_VALIDATE_X = _BN_VALIDATE_X = self.DATA_DIR + 'bn_validate_X.npy'
        self.BN_VALIDATE_y = _BN_VALIDATE_Y = self.DATA_DIR + 'bn_validate_Y.npy'

        # models of CNN and VGG
        self.MODEL_CNN = _MODEL_CNN = self.DATA_DIR + 'models\\model_cnn\\model.json'
        self.WEIGHTS_CNN = _WEIGHTS_CNN = self.DATA_DIR + 'models\\model_cnn\\model.h5'

        self.MODEL_VGG = _MODEL_VGG = self.DATA_DIR + 'models\\model_vgg\\model.json'
        self.WEIGHTS_VGG = _WEIGHTS_VGG = self.DATA_DIR + 'models\\model_vgg\\model.h5'

        # train,test,validation path
        self.FLOW_TRAIN= _UCMERCED_LANDUSE_DATASET = self.DATA_DIR + 'flow\\train\\'
        self.FLOW_TEST = _UCMERCED_LANDUSE_DATASET = self.DATA_DIR + 'flow\\test\\'
        self.FLOW_VALIDATE = _UCMERCED_LANDUSE_DATASET = self.DATA_DIR + 'flow\\validate\\'


