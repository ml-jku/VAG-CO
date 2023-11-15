
import numpy as np
from functools import partial

class DataBuffer():
    ### TODO actually Hb and Nb are already random, all that has to be changed is Sb
    def __init__(self, Hb, Nb, Sb, mini_Hb, mini_Nb, mini_Sb):
        self.Hb = Hb
        self.Nb = Nb
        self.Sb = Sb

        self.Hb_list = np.arange(0, Hb)
        self.Nb_list = np.arange(0, Nb)
        self.Sb_list = np.arange(0, Sb)
        Hb_Sb_arr = np.repeat(np.arange(0, Sb)[np.newaxis, :], mini_Nb, axis=0)
        self.Nb_Hb_Sb_arr = np.repeat(Hb_Sb_arr[np.newaxis, :], mini_Hb, axis=0)

        self.H_splits = int(self.Hb / mini_Hb)
        self.N_splits = int(self.Nb / mini_Nb)
        self.S_splits = int(self.Sb / mini_Sb)

        #
        self.HH, self.NN = np.meshgrid(np.arange(0, self.H_splits),np.arange(0, self.N_splits))
        self.HH = np.ravel(self.HH)
        self.NN = np.ravel(self.NN)


    def shuffle(self):
        np.random.shuffle(self.Hb_list)
        np.random.shuffle( self.Nb_list)
        #self.Sb_list = jax.random.shuffle(keys[3], self.Sb_list)
        ### TODO sb_lsit should have shape Hb Nb Sb

    def shuffle_Sb(self):
        self.Nb_Hb_Sb_arr  = shuffle_along_axis(self.Nb_Hb_Sb_arr, -1)


    def split_Sb(self):
        mini_Sb_lists = np.array_split(self.Nb_Hb_Sb_arr, self.S_splits, axis = -1)

        return mini_Sb_lists


    def split_in_minibatches(self):
        self.shuffle()
        mini_Hb_lists = np.array_split(self.Hb_list, self.H_splits)
        mini_Nb_lists = np.array_split(self.Nb_list, self.N_splits)

        order = self.shuffle_order()
        return (mini_Hb_lists, mini_Nb_lists), order


    def shuffle_order(self):
        self.HH = shuffle_along_axis(self.HH, 0)
        self.NN = shuffle_along_axis(self.NN, 0)
        return (self.HH, self.NN)

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)
