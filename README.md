This Repo contains three folders and each of those have the followings,

    • In Audio Folder:
      
      opensmile_feature.ipynb: This notebbok contains code for extracting features from audio of trailers.
      
      Audio_Model_Movie_Lens.ipynb: This notebook contains code for creating audio embedding using End to end neural network.
      
    • In text Folder:

	movielens_textual_embedding.ipynb : This file contains code for creating textual 	embedding.
      
      
    • In Video Folder:

	Frames_Extraction.ipynb: Contains 3 code:
            ▪ get_frames: To extract all frames from the video trailer
            ▪ get_K_Frames: Extract 5 frames from the centroid of the cluster using K-means
            ▪ get_entropy: Extract top 3 frames having max information 

	K-Means_CNN Model (MovieLens).ipynb: Contain code for CNN to generate video 	embeddings for the entropy frames

