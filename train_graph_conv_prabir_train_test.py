from email.contentmanager import raw_data_manager
import helpers
import random
import numpy as np
import sys
import pandas as pd
import math



import os

# train_graph_conv_prabir.py

from sklearn.metrics import mean_squared_error as mse

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import model_graph_conv_original as neural_network
# import model_graph_conv as neural_network
# import model_no_graph_conv as neural_network
import ndcg_k as ndcg



'''
'/home/sriram_1801cs37/prabir/frames5/'

movie_enc= helpers.load_pkl('/home/sriram_1801cs37/prabir/GCN/enc_movie.obj')
user_enc= helpers.load_pkl('/home/sriram_1801cs37/prabir/GCN/enc_user.obj')

'''







movie_enc= helpers.load_pkl('/home/prabir_prj22/sriram_1801cs37/prabir/fresh_GCN/embeddings/ML/enc_movie.obj')
# genre_enc = helpers.load_pkl('/home/prabir_prj22/sriram_1801cs37/prabir/fresh_GCN/training/genre_gcn/enc_genere_1.pkl')   ### change#1
user_enc= helpers.load_pkl('/home/prabir_prj22/sriram_1801cs37/prabir/fresh_GCN/embeddings/ML/enc_user.obj')
# genre_enc = helpers.load_pkl('/home/sriram_1801cs37/prabir/GCN/enc_genere.pkl')   ### change#1





# movie_enc= helpers.load_pkl("../liv_data/objs/enc_movie.obj")
# user_enc= helpers.load_pkl("../liv_data/objs/enc_user.obj")

modality = "ML_text"

movie_shape= len(movie_enc.get(list(movie_enc.keys())[0]))
# genre_shape= len(genre_enc.get(list(genre_enc.keys())[0]))
user_shape= len(user_enc.get(list(user_enc.keys())[0]))

batch_size = 64




print("Mov")


def shuffle_single_epoch(ratings):
    data_copied= ratings.copy()
    random.shuffle(data_copied)
    return data_copied


def normalize(rate):
    return rate/5
def de_normalize(rate):
    return rate*5



def copy_to_fix_shape(movie, user, rate, req_len):
    assert len(movie)==len(user)==len(rate)
    new_movie, new_user, new_rate= movie[:], user[:], rate[:]
    while len(new_movie)<req_len:
        r_ind= random.randint(0, len(movie)-1)
        new_movie.append(movie[r_ind])
        new_user.append(user[r_ind])
        # new_genre.append(genre[r_ind])
        new_rate.append(rate[r_ind])
    return new_movie, new_user, new_rate


def get_nth_batch(ratings, n, batch_size= batch_size, take_entire_data= False):
    users= []
    movies= []
    rates= []
    # genre = [] ### change#4
    if take_entire_data:
        slice_start= 0
        slice_end= len(ratings)
    else:
        if (n+1)*batch_size>len(ratings):
            print("OUT OF RANGE BATCH ID")
        slice_start= n*batch_size
        slice_end= (n+1)*batch_size
    for user_id, movie_id, rate in ratings[slice_start: slice_end]:
        if user_enc.get(user_id) is None or movie_enc.get(movie_id) is None:
            continue
        users.append(user_enc.get(user_id))
        # print(user_enc.get(user_id)) 
        # movies.append(movie_enc.get(movie_id))  ## multiplied with rate        
        
        mv_enc = [item * rate for item in movie_enc.get(movie_id)]
        movies.append(mv_enc)  ## multiplied with rate         
              
        # genre.append(genre_enc.get(movie_id))  ### change#5
        rates.append(normalize(rate))   ##############
        
        

    if not take_entire_data:
        movies, users, rates= copy_to_fix_shape(movies, users, rates, batch_size)   ###############
    users= np.array(users)
    movies= np.array(movies)
    # genre = np.array(genre)
    rates= np.array(rates)

    users = scaler.fit_transform(users) 
    movies = scaler.fit_transform(movies) 
    
#     

    return movies, users, rates


# log = open("log.txt", 'a')

# def oprint(message):
#     print(message)
#     global log
#     log.write(message)
#     return()








def train(model, data, test_data= None, no_of_epoch= 50):
    total_batches_train= int(len(data)/batch_size)
    for epoch in range(no_of_epoch):
        print("\n\n---- EPOCH: ", epoch, "------\n\n")
#         oprint("\n\n---- EPOCH: ", epoch, "------\n\n")
        data= shuffle_single_epoch(data)    #########################
        for batch_id in range(total_batches_train):
            print("Epoch: ", epoch+1, " Batch: ", batch_id)
#             oprint("Epoch: ", epoch+1, " Batch: ", batch_id)
#             movies, users, rates= get_nth_batch(data, batch_id)   ###################
            movies, users, rates= get_nth_batch(data, batch_id)   ### Change#3
            
            #             print(f"shape of genre = {len(genre)}")
            # model.fit([movies, users], rates, genre, batch_size=batch_size, epochs=1, verbose=2)
            model.fit([movies, users], rates, batch_size=batch_size, epochs=1, verbose=2) # 1.d: 21.10.22
        if test_data is not None:
            test(model, test_data, take_entire_data=False, save=True, epoch=epoch, batch=batch_id)  ###############



lest_rmse= float("inf")
def test(model, data, save= True, take_entire_data= True, epoch = 100000, batch = 100000):
    eval = []
    if take_entire_data:


        ndcg_k1 = 4
        ndcg_k2 = 10
        ndcg_1 = ndcg_2 = 0

        ################################# generating the evaluation score on test data  #######################
        # print("test entered")
       #         movies, users, res_true, _, _= get_nth_batch(data, 0, take_entire_data=take_entire_data)
 
        xyz = pd.DataFrame(data, columns = ["userId", "movieId", "rating"])
        arr = xyz[["userId"]]
        arr = arr["userId"].unique()
        
        y_true, y_pred= np.array([]), np.array([])
        # arr=help.fun()  ## arr holds all the unique user id only
        final_ans=0.0
        final_recall=0.0
        final_recall_k=0.0
        n1=len(arr)  ## total number of user
        # print(f"total number of user = {n1}")
        # list_of_movie_pred=[]
        # list_of_movie_true=[]
        cnt=0
#         arr=arr.sort()  
        # n1=3000   ## consider only the 3000 users
#         global epoch
        for i in arr:
#             print("user: ",cnt)
            if cnt==3000:
                break
            # print(f"count = {cnt}")
            users=[]
            movies=[]
            # genres = []
            rates=[]
            df=xyz[xyz['userId']==i]   ## xyz holds all the user-movie-rating values
#             print(df)
            for index,movieid in df.iterrows():
                user_id=int(movieid['userId'])
                movie_id=int(movieid['movieId'])
                rate=movieid['rating']
                if user_enc.get(user_id) is None or movie_enc.get(movie_id) is None or len(user_enc.get(user_id))!=1243:
                    continue
                users.append(user_enc.get(user_id))
                movies.append(movie_enc.get(movie_id))
                # genres.append(genre_enc.get(movie_id))
                rates.append(normalize(rate))
            if len(users)==0:
                continue
            users= np.array(users)  ## sigle user indicated by i
            movies= np.array(movies)  ## all movies rated by user i
            # genres = np.array(genres)
            # print(f"user_id = {user_id}, number of movies = {len(movies)}")
            # print(f"Movies length = {len(movies)}")

            res_pred = model.predict([movies, users])
            res_true= np.array(rates)
            res_pred= np.array(res_pred).reshape(-1)
#             print(len(res_true))
#             print(len(res_pred))
#             k=3.5
            
            y_true= np.concatenate([y_true, res_true])
            y_pred= np.concatenate([y_pred, res_pred])
            res_true= de_normalize(res_true)
            res_pred= de_normalize(res_pred)
#             user.append([res_true,res_pred])

            cnt+=1

             #--------------------------ndcg bolck: --------------------
            ndcg_1 += ndcg.my_ndcg(res_true, res_pred, ndcg_k1)
            ndcg_2 += ndcg.my_ndcg(res_true, res_pred, ndcg_k2)
            #-----------------------end of ndcg block------------------

                   
            dict_true=[]
            dict_pred=[]
            for j in range(0,len(res_true)):
                dict_true.append([res_true[j],j])
                dict_pred.append([res_pred[j],j])
            sorter = lambda x: (-x[0], x[1])
#             print(dict_true)
#             print(dict_pred)
            dict_true = sorted(dict_true, key=sorter)
            dict_pred = sorted(dict_pred, key=sorter)
            rel=0
            for j in range(0,len(dict_pred)):
                x=dict_pred[j][1]
                y=dict_true[j][1]
                if x==y:
                    rel+=1
            dict_pred=dict_pred[:4]    ##### K = 10
            dict_true=dict_true[:4]    ##### K = 10
#             print(dict_true)
#             print(dict_pred)
            res=0.0
            gtp=0
            recall=0.0
            for j in range(0,len(dict_pred)):
#                 movie_pred.append(dict_pred[j][1])
#                 movie_true.append(dict_true[j][1])
                x=dict_pred[j][1]
                y=dict_true[j][1]
                if x==y:
                    gtp+=1
                    res+=float(gtp)/(j+1)
                    if gtp==1:
                        recall=1.0/(j+1)
            
            if gtp==0:
                continue
            # MRR
            final_recall+=recall
            
            # MAP
            res=res/gtp
            final_ans+=res
            
            #RECALL@K
            rel=float(gtp)/rel
            final_recall_k+=rel

            # cnt1+=1

        avg_ndcg_4 = ndcg_1/cnt
        avg_ndcg_10 = ndcg_2/cnt

        n1 = cnt
            
        final_recall_k=final_recall_k/n1    
        final_ans=final_ans/n1
        final_recall=final_recall/n1
#         save_file_per_epoch=str(epoch)
#         np.save(save_file_per_epoch,user)
        # print("MAP: ",final_ans," MRR: ",final_recall,"R: ",final_recall_k)
        eval.append(final_ans)
        eval.append(final_recall)
        eval.append(final_recall_k)
        # eval = [final_ans, final_recall, final_recall_k] ## map@K, mar@k, r@k


        ####################################  end of evaluation on test data  #################################

#         movies, users, res_true= get_nth_batch(data, 0, take_entire_data=take_entire_data)
# #         movies, users, genre_sim, res_true= get_nth_batch(data, 0, take_entire_data=take_entire_data)  ### change#3
#         res_pred= model.predict([movies, users], batch_size=batch_size)
#         res_true= np.array(res_true)
#         res_pred= np.array(res_pred).reshape(-1)
#         assert len(res_true)==len(res_pred)
    else:
        total_batches_test=int(len(data)/batch_size)
        y_true, y_pred= np.array([]), np.array([])
        for batch_id in range(total_batches_test+1):
            movies, users, rates= get_nth_batch(data, batch_id)   ####
            pred= model.predict([movies, users], batch_size=batch_size)
            pred= pred.reshape(-1)
            assert len(rates)==len(pred)
            y_true= np.concatenate([y_true, rates])
            y_pred= np.concatenate([y_pred, pred])


    y_true= de_normalize(y_true)   
    y_pred= de_normalize(y_pred)

    y_true = np.nan_to_num(y_true)
    y_pred = np.nan_to_num(y_pred)
    # print(f"y_ture = {len(y_true)}, y_pred = {len(y_pred)}")
    # print(f"y_ture shape = {y_true.shape}, y_pred shape = {y_pred.shape}")
    # with open("Ys.txt", "a") as h:
    #     print("-------------------------y_true-----------------------------------", file=h)
    #     print(y_true, file=h)
    #     print("----------------------y_pred--------------------------------", file=h)
    #     print(y_pred, file=h)
    
    
    
    
    # res_pred= np.array([round(x) for x in res_pred])
    # for x in range(200):
    #     print(y_true[x], " : ", y_pred[x])



    rmse= calc_rms(y_true, y_pred)
    y_pred= np.array([round(x) for x in y_pred])
    rmse_n= calc_rms(y_true, y_pred)





    # print(f"training rmse = {rmse} and training normalized rmse = {rmse_n}")
    # print("rmse: ", rmse, " rmse_norm: ", rmse_n)
#     oprint("rmse: ", rmse, " rmse_norm: ", rmse_n)
    
    global lest_rmse
    if save and lest_rmse>rmse:
        lest_rmse= rmse

        with open("/home/prabir_prj22/sriram_1801cs37/prabir/fresh_GCN/training/genre_gcn/record/rmse_chain_weightedEnc_t.txt", "a") as h:   #prabir
            print(f"Epoch = {epoch}, batch = {batch}, RMSE = {lest_rmse}", file=h)
        


        # helpers.save_model(model)       
        helpers.save_model(model,save_folder = "/home/prabir_prj22/sriram_1801cs37/prabir/fresh_GCN/training/genre_gcn/models/ML",save_filename = "model_ml_weightedEnc_t.h5")   # by prabir
        
    if save is False:     #  by prabir
        eval.append(rmse)
        eval.append(rmse_n)
        eval.append(avg_ndcg_4)
        eval.append(avg_ndcg_10)
        return(eval)

def calc_rms(t, p):
    return math.sqrt(mse(t, p))

def train_test_ext(train_obj, test_obj):
# def train_test_ext(train_obj, test_obj, cut_off):
    # model= neural_network.make_model(movie_shape, user_shape, window_size=batch_size)

    model= neural_network.Brain(movie_shape, user_shape)
    # model= neural_network.Brain(movie_shape, genre_shape, user_shape)
    # model= neural_network.Brain(movie_shape, genre_shape, user_shape, cut_off)
    # model= neural_network.Brain(movie_shape, genre_shape, user_shape, cut_off)
    train(model, data=train_obj, test_data=test_obj)

    #---------------------------------------------------------------
    # #     test(model, data= test_obj, save= False, take_entire_data=False)
    # (rmse, rmse_n) =  test(model, data= test_obj, save= False, take_entire_data=False)
    # with open("rmse.txt", "a") as h:   #prabir
    #     print("\n")  
    #     print("RMSE =  ",rmse, file=h) 
    #     print("Normalised RMSE =  ",rmse_n, file=h) 
    #---------------------------------------------------------------------


def test_saved(saved_model, test_file_path):
    import keras
    model= keras.models.load_model(saved_model)
    # ratings_test_path= helpers.path_of(test_file_path)

    ratings_test_path = test_file_path

    test_obj= helpers.load_pkl(ratings_test_path)
#     test(model, data= test_obj, take_entire_data=True)
    eval = test(model, data= test_obj, save = False, take_entire_data=True)
    with open("/home/prabir_prj22/sriram_1801cs37/prabir/fresh_GCN/training/genre_gcn/record/ml_weightedEnc_t.txt", "a") as h:   #prabir
        print(f"modality = {modality}, MAP = {eval[0]}, MRR = {eval[1]}, R = {eval[2]}, RMSE = {eval[3]}, Normalized RMSE = {eval[4]}, NDCG@4 = {eval[5]}, NDCG@10 = {eval[6]}", file=h)
        # print("\n")  
        # print("RMSE =  ",rmse, file=h) 
        # print("Normalised RMSE =  ",rmse_n, file=h) 


if __name__=="__main__":



    print("Movie_shape: ", movie_shape)
    # print("Genre_shape: ", genre_shape)
    print("User_shape: ", user_shape)

    print("batch_size:", batch_size)

    print("\n\n")

    train_obj= helpers.load_pkl("/home/prabir_prj22/sriram_1801cs37/prabir/fresh_GCN/embeddings/ML/u1.train.obj")
    test_obj= helpers.load_pkl("/home/prabir_prj22/sriram_1801cs37/prabir/fresh_GCN/embeddings/ML/u1.test.obj")

    # for cut_off in range(9,1,-1):

    #     with open("/home/prabir_prj22/sriram_1801cs37/prabir/fresh_GCN/training/genre_gcn/record/rmse_chain.txt", "a") as h:   #prabir
    #         print(f"................................... Threshold = {cut_off/10}......................", file=h)

    #     train_test_ext(train_obj, test_obj, cut_off)
        
    #     saved_model = "/home/prabir_prj22/sriram_1801cs37/prabir/fresh_GCN/training/genre_gcn/models/ML/model_ml_all.h5"
    #     test_file_path = ("/home/prabir_prj22/sriram_1801cs37/prabir/fresh_GCN/embeddings/ML/u1.test.obj")
        
    #     test_saved(saved_model, test_file_path)
    # print(f"Modality completed = {modality}")

    train_test_ext(train_obj, test_obj)
    saved_model = "/home/prabir_prj22/sriram_1801cs37/prabir/fresh_GCN/training/genre_gcn/models/ML/model_ml_weightedEnc_t.h5"
    test_file_path = ("/home/prabir_prj22/sriram_1801cs37/prabir/fresh_GCN/embeddings/ML/u1.test.obj")
    
    test_saved(saved_model, test_file_path)
    print(f"Modality completed = {modality}")



