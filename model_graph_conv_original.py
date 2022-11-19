# this function generates the GCN based model

import keras
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import fractional_matrix_power

# generate the graph based movie embedding 
def gcn_calc_movie(x):
    np_x= x.numpy()    
    sample_count= len(np_x)
    A= cosine_similarity(np_x) # A is the Adjacency Matrix
    cutoff_edge_matrix= 0.76
    A[A > cutoff_edge_matrix] = 1
    A[A < cutoff_edge_matrix] = 0

    degree= dict()
    for ind_i, val_i in enumerate(A):
        degree.update({
            ind_i: len(np.where(val_i==1)[0]),
        })
    degree_matrix= [[0]*sample_count for _ in range(sample_count)]
    for k in degree.keys():
        degree_matrix[k][k]= degree.get(k)

    half_norm_degree= fractional_matrix_power(degree_matrix, -0.5)
    res= half_norm_degree.dot(A).dot(half_norm_degree).dot(np_x)
    return tf.convert_to_tensor(res)


class Brain:
    def __init__(self, ip_m_size, ip_u_size):
        self.make_model(ip_m_size, ip_u_size)
        
    def make_model(self, ip_m_size, ip_u_size):
        #model sequence
        ip_m= keras.layers.Input(shape=(ip_m_size,), name="input_movies")        
        gcn_transform_m= keras.layers.Lambda(gcn_calc_movie, dynamic=True, output_shape=(ip_m_size), name="gcn_movies_1")(ip_m)     
        gcn_op_m= keras.layers.Dense(units= int(ip_m_size*2/3), activation= "tanh")(gcn_transform_m)
        ip_u= keras.layers.Input(shape=(ip_u_size,), name="input_users")
        d_m= keras.layers.Dense(units= int(ip_m_size*2/3), activation="tanh")(gcn_op_m)
        d_u= keras.layers.Dense(units= int(ip_u_size*2/3), activation="tanh")(ip_u)
        concated= keras.layers.concatenate([d_m, d_u])
        d_c1= keras.layers.Dense(units= int((ip_m_size+ip_u_size)/2), activation="tanh")(concated)
        d_c2= keras.layers.Dense(units=1024, activation="sigmoid")(d_c1)
        d_c3= keras.layers.Dense(units=128, activation="sigmoid")(d_c2)
        d_c4= keras.layers.Dense(units=1, activation="sigmoid")(d_c3)

        model.compile(loss="mse", optimizer="adam")
        self.model= model


    def fit(self, xs, ys, epochs=1, batch_size=64, verbose=2):
        self.model.fit(xs, ys, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)

    def predict(self, xs, batch_size=None):
        return self.model.predict(xs, batch_size=batch_size)

    def save(self, op_file):
        self.model.save(op_file) 




# if __name__ == "__main__":
#     m_size, u_size= 1234, 1345
#     model= Brain(m_size, u_size)    
#     sample_count= 1000

#     xs_m= np.random.random(sample_count*m_size).reshape(-1, m_size)
#     xs_u= np.random.random(sample_count*u_size).reshape(-1, u_size)
#     ys= np.random.random(sample_count*1).reshape(-1)

#     model.fit([xs_m, xs_u], ys, epochs=10)

#     model.save("graph_model_1.h5")


