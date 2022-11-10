
import os
import pickle5 as pickle
from pathlib import Path
import random
import string

def path_of(location):
    me_dir, me_file= os.path.split(os.path.abspath(__file__))
    return os.path.join(me_dir, location)



def load_pkl(filename):
    filename= path_of(filename)
    data= None
    with open(filename, "rb") as handle:
        data= pickle.load(handle)
        handle.close()
    return data

def store_pkl(object, filename):
    filename= path_of(filename)
    with open(filename, "wb") as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

def is_valid_file(filename):
    filename= path_of(filename)
    file= Path(filename)
    if file.is_file():
        return True
    return False


def to_same_shape(arr_of_items, required_shape):
    if len(arr_of_items) == 0:
        print("Error: tried to make shape: ", required_shape, ", but item is empty...")
        exit()
    if len(arr_of_items)>required_shape:
        return arr_of_items[:required_shape]
    res= []
    ind=0
    while len(res)<required_shape:
        res.append(arr_of_items[ind])
        ind= (ind+1)%len(arr_of_items)
    return res

def get_rand_str(size):
    return "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(size))

def save_model(keras_model, save_folder="./models", save_filename= None):
# def save_model(keras_model, save_folder="/home/sriram_1801cs37/prabir/GCN/models", save_filename= None):
    if save_filename is None:
        me_dir, me_file= os.path.split(os.path.abspath(__file__))
        save_filename= me_file.split(".")[0]+".h5"
    op_file= path_of(save_folder +"/"+save_filename)
    keras_model.save(op_file)
    print("\n\n    Saved_model:", save_filename, "\n\n")

    