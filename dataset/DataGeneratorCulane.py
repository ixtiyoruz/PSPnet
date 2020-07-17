import numpy as np
import cv2
from tensorflow.keras.utils import Sequence

def rgb_to_onehot(self, rgb_arr, color_dict = None):
        if(color_dict == None):
            color_dict = {
                      0: (0,0,0),
                      1: (0,255,0),
                      2: (0,255,255),
                      3: (255,255,0),
                      4: (255,0,0)
                      }
        num_classes = len(color_dict)
        shape = rgb_arr.shape[:2]+(num_classes,)
        arr = np.zeros( shape, dtype=np.float32 )
        for i, cls in enumerate(color_dict):
            arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
        return arr
def gray_to_onehot(gray, color_dict = None):
        
        if(color_dict is None):
            color_dict = {
                      0: (0,0,0),
                      1: (0,255,0),
                      2: (0,255,255),
                      3: (255,255,0),
                      4: (255,0,0)
                      }
            
        num_classes = len(color_dict)
        shape = gray.shape+(num_classes,)
        arr = np.zeros( shape, dtype=np.float32 )
        for i, cls in enumerate(color_dict):
            wx, wy = np.where(gray == cls)
            arr[wx,wy,i] = 1.0
        # ignoring label 255
        wx, wy = np.where(gray == 255)
        arr[wx,wy,0] = 1.0
        return arr

def gray_to_onehot_all(grays, color_dict = None):
    results = []
    for i in range(len(grays)):
        results.append(gray_to_onehot(grays[i], color_dict))
    return np.array(results)

class DataGenerators(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, img_paths, label_paths, to_fit=True, batch_size=32, dim=(256, 256),
                 n_channels=3, n_classes=10, shuffle=True, transform=None, output_of_models=None):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of label channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.transform = transform
        self.on_epoch_end()
        self.curr_index = 0
        self.colors = np.array([[255,255,255], [0, 0, 0]])
        self.output_of_models = output_of_models
        self.color_dict = {
                      0: (0,0,0),
                      1: (0,255,0)
                      }
    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.img_paths) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # if(self.curr_index > self.__len__()):
        #     self.on_epoch_end()
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            if X.shape[1] != y.shape[1] or X.shape[2] != y.shape[2]:
                raise (RuntimeError("Image & label shape mismatch between " + indexes[0] + ":" + indexes[-1] + "\n"))
            
            if(not self.transform == None):
                for i in range(len(self.transform)):
                    X, y = self.transform[i]((X,y))
            
            y = gray_to_onehot_all(y, self.color_dict)
            
            if(self.output_of_models is None):
                return (X, y)
            else:
                return (X, {self.output_of_models[0]:y, self.output_of_models[1]:y})
        else:
            return X
    # def next(self):
    #     X, y, y = self.__getitem__(self.curr_index)
    #     self.curr_index +=1
    #     return X, y, y

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.curr_index = 0

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], 3))

        # Generate data
        for i, index in enumerate(list_IDs_temp):
            limg = self._load_image(self.img_paths[index])

            # Store sample
            X[i] = limg

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        if(self.n_channels == 1):
            y = np.empty((self.batch_size, self.dim[0], self.dim[1]), dtype=int)
        else:
            y = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels), dtype=int)

        # Generate data
        for i, index in enumerate(list_IDs_temp):
            # Store sample
            y[i] =  self._load_image(self.label_paths[index])        
        y[y>0]=1#-0----------------------------------------------------------
        return y
#    def one_hot_encode_all(self, masks, mode='broadcasting',n_classes=4, ignore_label =[255,255,255]):
#        """
#            one hot encoding an mask image,
#            input mask is one image can be [n x w x h] or [n x w x h x 3] shape
#        """
#        return np.array([self.one_hot_encode(mask,mode,n_classes,ignore_label) for mask in masks])
#    
#    def one_hot_encode(self, mask, mode='broadcasting',n_classes=4, ignore_label =[255,255,255]):
#        """
#            one hot encoding an mask image,
#            input mask is one image can be w x h or w x h x 3 shape
#        """
#        shape = np.shape(mask)
#        ch = 3
#        if(len(shape) ==3):
#            ch = shape[-1]
#        else:
#            ch = 1
#        if(ch == 3 and mode == 'broadcasting'):
#            """
#            mask can get no more than 8 categories
#            """
#            ignore =None
#            if(not ignore_label == None):
#                ignore = (np.array(ignore_label)==255).dot([4,2,1]) # ignore value will be labelled as 0 current ignore label is 7, 
#            b = (mask==255).dot([4,2,1])
#            v_uns = np.unique((self.colors==255).dot([4,2,1])) # unique values
#            if(not ignore == None):
#                v_uns = np.delete(v_uns, np.where((v_uns == ignore))[0][0])
#                b[b == ignore] = 0
#            assert len(v_uns) == n_classes, ' the number of unique colors must be equal to number of classes'
#            return np.flip((b[..., None] == v_uns).astype(int), -1) # reversing the labels so they will be in increasing order
#        if(ch ==1):
#            if(not ignore_label == None):
#                if(type(ignore_label) == type([]) or type(ignore_label) == type(np.array([]))):
#                    ignore_label = 255
#            b = np.zeros((shape[0],shape[1], n_classes))
#            v_uns = np.unique(mask) # unique values
#            for i in v_uns:
#                cls_vec = np.zeros(n_classes)
#                if(i == ignore_label): # i will never be None so no need to check
#                    cls_vec[-1] = 1
#                else:
#                    cls_vec[int(n_classes-i-1)] = 1
#                b[np.where(mask== i)] = cls_vec
#            return b
#        return None
    
    def _load_image(self, image_path):
        """Load  image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path, -1)
        img = cv2.resize(img, (self.dim[1], self.dim[0]), interpolation=cv2.INTER_NEAREST)
        # img = np.zeros((self.dim[0], self.dim[1], 3))
        return img




class DataIterators(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, img_paths, label_paths, to_fit=True, batch_size=32, dim=(256, 256),
                n_channels=3, n_classes=10, shuffle=True, transform=None, output_of_models=None, color_dict = {
                      0: (0,0,0),
                      1: (0,255,0)
                      }
                      ):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of label channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.transform = transform
        self.on_epoch_end()
        self.curr_index = 0
        self.colors = np.array([[255,255,255], [0, 0, 0]])
        self.output_of_models = output_of_models
        self.color_dict = color_dict
    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.img_paths) / self.batch_size))
    
    def get_data_iterator(self):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        while(True):
            if(self.curr_index >= self.__len__()):
                self.on_epoch_end()
#             Generate indexes of the batch
            indexes = self.indexes[self.curr_index * self.batch_size:(self.curr_index + 1) * self.batch_size]
            self.curr_index = self.curr_index + 1
            # Find list of IDs
            list_IDs_temp = [k for k in indexes]
    
            # Generate data
            X = self._generate_X(list_IDs_temp)
    
            if self.to_fit:
                y = self._generate_y(list_IDs_temp)
                x_n, y_n = [],[]
                for i in range(len(X)):
                    Xt, yt = X[i], y[i]
                    # Xt = np.reshape(Xt, [*(Xt.shape), 1])
                    for augmenter in self.transform:                        
                        Xt, yt = augmenter((Xt, yt))     
                    Xt = cv2.resize(Xt, ( self.dim[1],self.dim[0] ),interpolation = cv2.INTER_NEAREST)
                    yt = cv2.resize(yt, ( self.dim[1],self.dim[0] ),interpolation = cv2.INTER_NEAREST)
                    # Xt = np.reshape(Xt, [*(Xt.shape), 1])
                    x_n.append(Xt)
                    y_n.append(yt)
                
                X = np.array(x_n)
                y = np.array(y_n)
                
                # cv2.imshow("image", np.uint8(X[0]))
                # cv2.waitKey(0)
                y = gray_to_onehot_all(y, self.color_dict)
    
                yield (X, y)
            else:
                yield X
    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.curr_index = 0

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, 3))
        X = []
        # Generate data
        for i, index in enumerate(list_IDs_temp):
            # Store sample
            X.append(self._load_image(self.img_paths[index], False))
        X = np.array(X)

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        # if(self.n_channels == 1):
            # y = np.empty((self.batch_size, *self.dim), dtype=int)
        # else:
            # y = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=int)
        y = []
        # Generate data
        for i, index in enumerate(list_IDs_temp):
            # Store sample
            y.append(self._load_image(self.label_paths[index]))
        y = np.array(y)
        y[y>0] = 1
        return y
    

    def _load_image(self, image_path, gray=False):
        """Load  image
        :param image_path: path to image to load
        :return: loaded image
        """
        if(gray):
            img = cv2.imread(image_path, 0)
            
        else:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # img = cv2.resize(img, (self.dim[1], self.dim[0]), interpolation=cv2.INTER_NEAREST)
        return img
#
#if __name__ == "__main__":
#    a = np.zeros((3,3))
#    a[0][0] = 1
#    a[0][1] = 1
#    a[1][0] = 2
#    a[1][1] = 2
#    a[2][0] = 3
#    a[2][1] = 255
#    # a[0,0] = [255,0,0]
#    # a[0,1] = [255,255,0]
#    # a[1,0] = [255,0,255]
#    # a[1,1] = [255,255,255]
#    b = one_hot_encode(a, n_classes=4)
#    # b = np.flip(b, -1)