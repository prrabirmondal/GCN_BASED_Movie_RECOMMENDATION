# GCN_BASED_Movie_RECOMMENDATION
Graph Convolution Network Based Movie Recommendation System
Get the [paper](https://dl.acm.org/doi/pdf/10.1145/3555776.3577853)

The Recommendation System (RS) development and recommending customers’ preferred products to the customer are highly desirable motives in today’s digital market. Most of the RSs are mainly
based on textual information of the engaged entities in the platform
and the ratings provided by the users to the products. This paper
develops a movie recommendation system where the cold-start
problem relating to rating information dependency has been dealt
with and the multi-modality approach is introduced. The proposed
method differs from existing approaches in three main aspects: (a)
implementation of knowledge graph for text embedding, (b) besides textual information, other modalities of movies like video,
and audio are employed rather than rating information for generating movie/user representation and this approach deals with the
cold-start problem effectively, (c) utilization of graph convolutional
network (GCN) for generating some further hidden features and
also for developing regression system.

## FILES DETAILS:

**train_graph_conv_prabir_train_test.py:** FIND THE DRIVER FUNCTION HERE. ALL THE TRAINING AND TESTING IS DONE HERE

**model_graph_conv_original.py:** FIND THE MODEL DECLARATION AND DEFINATION PART HERE

**helpers.py:** HELPER FUCTION USED FOR DATASET LOADING, SAVING, PREPROCESSING ETC.

**ndcg_k.py:** IT CALCULATES THE NDCG VALUE FOR THE GIVEN VALUE OF K
