from datasets.base import DataLoader
from datasets.modules import conv_module_vgg
from datasets.iterator import MyScheme
from datasets.__local__ import data_path, port_train, port_val
from datasets.preprocessing import filename_to_array_imagenet
from utils.misc import flush_last_line
from config import Configuration as Cfg

import numpy as np
import cPickle as pickle
import time
import os
import fnmatch
import scipy.io as io

from picklable_itertools import imap

from fuel.datasets import Dataset
from fuel.streams import DataStream
from fuel.transformers import Transformer
from fuel.streams import ServerDataStream
from fuel.server import start_server


class ImageNet_Base(DataLoader):

    def __init__(self):

        DataLoader.__init__(self)

        self.dataset_name = "imagenet"

        self.n_train = 1281167
        self.n_val = 50000
        self.n_test = -1

        self.n_classes = 1000

        Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

        self.data_path = data_path
        self.port_train = port_train
        self.port_val = port_val

        self.on_memory = False
        Cfg.store_on_gpu = False


class ImageNet_DataLoader(ImageNet_Base):

    def __init__(self):

        ImageNet_Base.__init__(self)

        data_stream_train = ServerDataStream(('filenames',), False,
                                             port=self.port_train)
        self.get_epoch_train = data_stream_train.get_epoch_iterator

        data_stream_val = ServerDataStream(('filenames',), False,
                                           port=self.port_val)
        self.get_epoch_val = data_stream_val.get_epoch_iterator

    def build_architecture(self, nnet):

        shape = (None, 3, 224, 224)
        nnet.addInputLayer(shape=shape)

        conv_module_vgg(nnet, nfilters=64, nrepeat=2)
        conv_module_vgg(nnet, nfilters=128, nrepeat=2)
        conv_module_vgg(nnet, nfilters=256, nrepeat=3)
        conv_module_vgg(nnet, nfilters=512, nrepeat=3)
        conv_module_vgg(nnet, nfilters=512, nrepeat=3)

        nnet.addDenseLayer(num_units=4096)
        nnet.addReLU()
        nnet.addDropoutLayer(p=0.5)
        nnet.addDenseLayer(num_units=4096)
        nnet.addReLU()
        nnet.addDropoutLayer(p=0.5)
        nnet.addSVMLayer()

    def check_specific(self):

        # store primal variables on RAM
        assert not Cfg.store_on_gpu

    def load_weights(self, nnet):

        print("Loading weights...")
        pickled_file = open('../data/imagenet/vgg16.pkl', "r")
        pickled_dict = pickle.load(pickled_file)

        # loading weights
        pickled_weights = pickled_dict['param values']
        for layer in nnet.trainable_layers:
            layer.W.set_value(pickled_weights.pop(0))
            layer.b.set_value(pickled_weights.pop(0))

        pickled_file.close()
        flush_last_line()
        print("Weights loaded.")

    def get_epoch_test(self, *args, **kwargs):

        raise NotImplementedError('Test data iterator not implemented')


def find_jpeg_files(root_dir):

    matches = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*.JPEG'):
            matches.append(os.path.join(root, filename))

    return matches


class ImageNet_Server(Dataset, ImageNet_Base):

    provides_sources = ('filenames',)

    def __init__(self, which_set):

        Dataset.__init__(self)
        ImageNet_Base.__init__(self)

        assert which_set in ('train', 'val')

        # store which data set to build
        self.which_set = which_set

        self.port = self.port_train if which_set == 'train' else self.port_val

        self.batch_size = Cfg.batch_size

        if self.which_set == 'val':
            assert self.batch_size == 1, \
                "Use batch size of 1 at validation time"

        self.find_files_and_labels()

        self.iteration_scheme = MyScheme(examples=self.n_samples,
                                         batch_size=self.batch_size)

        self.image_reader = \
            ImageReader(filename_to_label=self.filename_to_label,
                        which_set=which_set,
                        data_stream=
                        DataStream.default_stream(dataset=self,
                                                  iteration_scheme=
                                                  self.iteration_scheme))

    def start(self):

        start_server(self.image_reader, self.port)

    def find_files_and_labels(self):

        if self.which_set == "train":
            my_data_path = "%strain/" % data_path
        elif self.which_set == "val":
            my_data_path = "%sval/" % data_path

        # find data on file system
        self.filenames = find_jpeg_files(my_data_path)

        # avoid issues with batch size for now
        n_filenames = len(self.filenames) -\
            (len(self.filenames) % self.batch_size)
        self.filenames = self.filenames[:n_filenames]

        # get number of data samples found
        self.n_samples = len(self.filenames)
        print("%i samples for %s" % (self.n_samples, self.which_set))

        # load mapping from synsets to labels
        synsets_file = open("../data/imagenet/synsets.txt", "r")
        synset_list = synsets_file.read().split("\n")

        # store dictionary mapping a wnid to an index
        # (according to VGG / Caffe mapping)
        self.synset_to_label = dict()
        for idx in range(1000):
            synset = synset_list[idx]
            self.synset_to_label[synset] = idx

        # dummy function for now
        self.filter_sources = lambda x: x

        if self.which_set == "train":

            # shuffle data (always in the same way for reproducibility)
            np.random.seed(self.seed)
            np.random.shuffle(self.filenames)

            # store function mapping filename ot label
            self.filename_to_label = self.filename_to_label_train

        else:

            # load mapping from wnid to index (according to ImageNet challenge)
            cls_loc_file = data_path.replace("images/",
                                             "ILSVRC2014_devkit/data/" +
                                             "meta_clsloc.mat")
            contents = io.loadmat(cls_loc_file)
            synsets = contents['synsets']

            wnid_to_label = dict()
            for tuple_ in synsets[0][:1000]:
                # label in 0-999 instead of 1-1000
                label = int(tuple_[0]) - 1
                wnid = tuple_[1][0]
                wnid_to_label[wnid] = label

            # map index of ImageNet to
            self.imgnet_to_vgg = dict()
            for wnid in wnid_to_label.keys():
                imgnet_idx = wnid_to_label[wnid]
                vgg_idx = self.synset_to_label[wnid]
                self.imgnet_to_vgg[imgnet_idx] = vgg_idx

            # open filename with ground truth of validation set
            label_filename = data_path.replace("images/",
                                               "ILSVRC2014_devkit/data/" +
                                               "ILSVRC2014_clsloc_" +
                                               "validation_ground_truth.txt")
            label_file = open(label_filename, "r")

            # labels are given in the form of 1-1000, change to 0-999
            self.labels = np.array(label_file.read().split("\n")[:-1])\
                .astype(np.int32) - 1

            # store function mapping filename ot label
            self.filename_to_label = self.filename_to_label_val

    def filename_to_label_train(self, filename):

        # parse string of filename to find its wnid
        wnid = filename.split("/")[-2]

        # retrieve label from wnid dictionary
        label = self.synset_to_label[wnid]

        return label

    def filename_to_label_val(self, filename):

        # parse string of filename to find its index
        str_idx = filename.split("/")[-1].replace(".JPEG", "").split("_")[-1]

        # files are indexed in the form of 1-50,000, change to 0-49,999
        idx = int(str_idx) - 1

        # retrieve ImageNet label from file index
        label = self.labels[idx]

        # retrieve VGG label from ImageNet to VGG mapping
        label = self.imgnet_to_vgg[label]

        return label

    def get_data(self, state=None, request=None):

        if isinstance(request, list):
            batch_index = request[0] / self.batch_size
            data = imap(self.filter_sources,
                        [self.filenames[idx] for idx in request])
        else:
            raise ValueError("request should be a list instance")
        return (data, batch_index)


class ImageReader(Transformer):
    """

    """
    def __init__(self, filename_to_label, which_set, data_stream):

        super(ImageReader, self).__init__(data_stream)

        self.filename_to_label = filename_to_label

        assert which_set in ("train", "val")

        # use crops at training time only
        self.crop = (which_set == "train")

        # store pixel mean (BGR channels) and make array broadcastable
        self.mean = np.array([103.939, 116.779, 123.68]).reshape((3, 1, 1)) \
            .astype(np.float32)

    def get_data(self, request=None):

        clock = -time.time()

        # get iterator of filenames from data set and batch index to pass
        iterator, batch_index = next(self.child_epoch_iterator)

        # initialize containers
        X, y = [], []
        for filename in iterator:

            # store array
            X.append(filename_to_array_imagenet(filename, self.mean,
                                                self.crop))

            # store label
            y.append(self.filename_to_label(filename))

        # transform lists of arrays to stacked arrays
        # NB: this will require to use a batch size of 1 at validation time
        # (images may have different sizes)
        X = np.stack(X).astype(np.float32)
        y = np.stack(y).astype(np.int32)

        clock += time.time()
        print("Batch loading took %g s" % clock)

        return (X, y, batch_index)
