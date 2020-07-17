import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from utils._utils import make_image

class DataGenerators(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, train_names, labels, image_path, mask_path,
                 to_fit=True, batch_size=32, dim=(256, 256),
                 n_channels=3, n_classes=10, shuffle=True, transform=None, output_of_models=None):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.train_names = train_names
        self.labels = labels
        self.image_path = image_path
        self.mask_path = mask_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.transform = transform
        self.on_epoch_end()
        self.curr_index = 0
        self.curr_epoch = 0
        self.colors = np.array([[255,255,255], [0, 0, 0]])
        self.output_of_models = output_of_models
    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.train_names) / self.batch_size))
    
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
            y = self.one_hot_encode_all(y, n_classes=self.n_classes, ignore_label=None)

            return (X, {self.output_of_models[0]:y, self.output_of_models[1]:y, self.output_of_models[2]:y})
        else:
            return X
    # def next(self):
    #     X, y, y = self.__getitem__(self.curr_index)
    #     self.curr_index +=1
    #     return X, y, y

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
#        image = make_image()
#        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
#        writer = tf.summary.FileWriter(self.log_folder+'/train/logs')
#        writer.add_summary(summary, self.curr_epoch)
#        writer.close()
        
        self.indexes = np.arange(len(self.train_names))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.curr_index = 0
        self.curr_epoch =self.curr_epoch+ 1

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, index in enumerate(list_IDs_temp):
            # Store sample
            X[i] = self._load_image(self.image_path + self.train_names[index])

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=int)

        # Generate data
        for i, index in enumerate(list_IDs_temp):
            # Store sample
            y[i] = self._load_image(self.mask_path + self.labels[index])

        return y
    def one_hot_encode_all(self, masks, mode='broadcasting',n_classes=4, ignore_label =[255,255,255]):
        """
            one hot encoding an mask image,
            input mask is one image can be [n x w x h] or [n x w x h x 3] shape
        """
        return np.array([self.one_hot_encode(mask,mode,n_classes,ignore_label) for mask in masks])
    
    def one_hot_encode(self, mask, mode='broadcasting',n_classes=4, ignore_label =[255,255,255]):
        """
            one hot encoding an mask image,
            input mask is one image can be w x h or w x h x 3 shape
        """
        shape = np.shape(mask)
        ch = 3
        if(len(shape) ==3):
            ch = shape[-1]
        else:
            ch = 1
        if(ch == 3 and mode == 'broadcasting'):
            """
            mask can get no more than 8 categories
            """
            ignore =None
            if(not ignore_label == None):
                ignore = (np.array(ignore_label)==255).dot([4,2,1]) # ignore value will be labelled as 0 current ignore label is 7, 
            b = (mask==255).dot([4,2,1])
            v_uns = np.unique((self.colors==255).dot([4,2,1])) # unique values
            if(not ignore == None):
                v_uns = np.delete(v_uns, np.where((v_uns == ignore))[0][0])
                b[b == ignore] = 0
            # print(len(v_uns), n_classes, v_uns)
            assert len(v_uns) == n_classes, ' the number of unique colors must be equal to number of classes'
            return np.flip((b[..., None] == v_uns).astype(int), -1) # reversing the labels so they will be in increasing order
        if(ch ==1):
            if(not ignore_label == None):
                if(type(ignore_label) == type([]) or type(ignore_label) == type(np.array([]))):
                    ignore_label = 255
            b = np.zeros((shape[0],shape[1], n_classes))
            v_uns = np.unique(mask) # unique values
            for i in v_uns:
                cls_vec = np.zeros(n_classes)
                if(i == ignore_label): # i will never be None so no need to check
                    cls_vec[-1] = 1
                else:
                    cls_vec[int(n_classes-i-1)] = 1
                b[np.where(mask== i)] = cls_vec
            return b
        return None

    def _load_image(self, image_path):
        """Load  image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path, -1)
        img = cv2.resize(img, (self.dim[0], self.dim[1]), interpolation=cv2.INTER_NEAREST)
        return img

class DataIterators():
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, train_names, labels, image_path, mask_path,
                 to_fit=True, batch_size=32, dim=(256, 256),
                 n_channels=3, n_classes=10, shuffle=True, transform=None, output_of_models=None):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.train_names = train_names
        self.labels = labels
        self.image_path = image_path
        self.mask_path = mask_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.transform = transform
        self.on_epoch_end()
        self.curr_index = 0
        self.curr_epoch = 0
        self.colors = np.array([[255,255,255], [0, 0, 0]])
        self.output_of_models = output_of_models
    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.train_names) / self.batch_size))
    
    def get_data_iterator(self):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        while(True):
            if(self.curr_index > self.__len__()):
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
                if X.shape[1] != y.shape[1] or X.shape[2] != y.shape[2]:
                    raise (RuntimeError("Image & label shape mismatch between " + indexes[0] + ":" + indexes[-1] + "\n"))
                
                if(not self.transform == None):
                    for i in range(len(self.transform)):
                        X, y = self.transform[i]((X,y))
                y = self.one_hot_encode_all(y, n_classes=self.n_classes, ignore_label=None)
    
                yield (X, y,y)
            else:
                yield X
    # def next(self):
    #     X, y, y = self.__getitem__(self.curr_index)
    #     self.curr_index +=1
    #     return X, y, y

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
#        image = make_image()
#        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
#        writer = tf.summary.FileWriter(self.log_folder+'/train/logs')
#        writer.add_summary(summary, self.curr_epoch)
#        writer.close()
        
        self.indexes = np.arange(len(self.train_names))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.curr_index = 0
        self.curr_epoch =self.curr_epoch+ 1

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, index in enumerate(list_IDs_temp):
            # Store sample
            X[i] = self._load_image(self.image_path + self.train_names[index])

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=int)

        # Generate data
        for i, index in enumerate(list_IDs_temp):
            # Store sample
            y[i] = self._load_image(self.mask_path + self.labels[index])

        return y
    def one_hot_encode_all(self, masks, mode='broadcasting',n_classes=4, ignore_label =[255,255,255]):
        """
            one hot encoding an mask image,
            input mask is one image can be [n x w x h] or [n x w x h x 3] shape
        """
        return np.array([self.one_hot_encode(mask,mode,n_classes,ignore_label) for mask in masks])
    
    def one_hot_encode(self, mask, mode='broadcasting',n_classes=4, ignore_label =[255,255,255]):
        """
            one hot encoding an mask image,
            input mask is one image can be w x h or w x h x 3 shape
        """
        shape = np.shape(mask)
        ch = 3
        if(len(shape) ==3):
            ch = shape[-1]
        else:
            ch = 1
        if(ch == 3 and mode == 'broadcasting'):
            """
            mask can get no more than 8 categories
            """
            ignore =None
            if(not ignore_label == None):
                ignore = (np.array(ignore_label)==255).dot([4,2,1]) # ignore value will be labelled as 0 current ignore label is 7, 
            b = (mask==255).dot([4,2,1])
            v_uns = np.unique((self.colors==255).dot([4,2,1])) # unique values
            if(not ignore == None):
                v_uns = np.delete(v_uns, np.where((v_uns == ignore))[0][0])
                b[b == ignore] = 0
            # print(len(v_uns), n_classes, v_uns)
            assert len(v_uns) == n_classes, ' the number of unique colors must be equal to number of classes'
            return np.flip((b[..., None] == v_uns).astype(int), -1) # reversing the labels so they will be in increasing order
        if(ch ==1):
            if(not ignore_label == None):
                if(type(ignore_label) == type([]) or type(ignore_label) == type(np.array([]))):
                    ignore_label = 255
            b = np.zeros((shape[0],shape[1], n_classes))
            v_uns = np.unique(mask) # unique values
            for i in v_uns:
                cls_vec = np.zeros(n_classes)
                if(i == ignore_label): # i will never be None so no need to check
                    cls_vec[-1] = 1
                else:
                    cls_vec[int(n_classes-i-1)] = 1
                b[np.where(mask== i)] = cls_vec
            return b
        return None

    def _load_image(self, image_path):
        """Load  image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path, -1)
        img = cv2.resize(img, (self.dim[0], self.dim[1]), interpolation=cv2.INTER_NEAREST)
        return img

if __name__ == "__main__":
    a = np.zeros((3,3))
    a[0][0] = 1
    a[0][1] = 1
    a[1][0] = 2
    a[1][1] = 2
    a[2][0] = 3
    a[2][1] = 255
    # a[0,0] = [255,0,0]
    # a[0,1] = [255,255,0]
    # a[1,0] = [255,0,255]
    # a[1,1] = [255,255,255]
    b = one_hot_encode(a, n_classes=4)
    # b = np.flip(b, -1)
    print(a)
    print(b)