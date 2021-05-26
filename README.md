# Crisis-Computing-Early-Fusion-Workflow

<h2> Steps of the Workflow </h2>

<h3>Pre-Workflow Computations </h3>

<b>Images:</b>
  1. Get Informative and Non-Informative Images
          
          bin/utils.get_images()
          
  2. Assign labels to the images --> '_0': Non-Informative, '_1': Informative
   
         bin/utils.add_labels()
  
  3. Get unique image ID's and their corresponding names
      
         bin/utils.get_ids()
  
  4. Prefix split labels to the images based on tweets data split --> 'train_', 'test_' and 'val_'

          bin/utils.get_image_splits()

| Input Image Name | Output Image Name |
|------------------|-------------------|
| 917791044158185473_0.jpg| train_917791044158185473_0_0.jpg|
| 917873017065160704_0.jpg| val_917873017065160704_0_1.jpg|
| 917830092620836864_2.jpg| test_917830092620836864_2_0.jpg|

<b>Text:</b>
  1. Combine all tweets into one CSV file

          bin/utils.get_tweets()    
   
| Input Tweets | Output |
|------------------|-------------------|
| california_wildfires_final_data.tsv| combined.csv|
  mexico_earthquake_final_data.tsv
  
  2. Split the tweets into train, val and test randomly.
      
         bin/utils.split_tweets()
         
         
<h3>Preprocess: Resize all the images (N parallel Jobs) </h3>

Resizes all the input images to 600 x 600 dimension

        
          python bin/preprocess_images.py
      
Outputs: resized_train_917791044158185473_0_0.jpg, ... ,  resized_val_917873017065160704_0_1.jpg

<h3>Preprocess: Formats all the tweets (N parallel Jobs) </h3>
 
Cleans all the tweets by removing unnecessay words, characters and URLs 

    
          python bin/preprocess_tweets.py
      
Outputs: processed_train_tweets.csv, processed_test_tweets.csv, processed_val_tweets.csv

 
<h3>Train Model </h3>

Trains models on the processed dataset


For Images:
      
          python bin/main_supcon.py 
     
For Text:
      
          python bin/train_bert.py
     
Outputs: 
  Image: supcon_final_model.pth <br>
  Text: bert_final_model
  
<h3>Generate Embeddings </h3>

Produces 128-dim embeddings for Images and 64-dim embeddings for Text

For Images:
      
          python bin/generate_supcon_embeddings.py 
     
For Text:
      
          python bin/generate_bert_embeddings.py
     
Outputs: 
  Image: supcon_train_embeddings.csv, supcon_test_embeddings.csv <br>
  Text: bert_train_embeddings.csv, bert_test_embeddings.csv

<h3>Early Fusion </h3>

Uses the generated embeddings, concats them and is given as an input to a Multi-layered Perceptron. Produces Confusion Matrix and Classification Performance report.
      
          python bin/early_fusion.py 
     
Outputs: early_fusion_MLP.png, early_fusion_MLP.csv

<h3>Workflow Diagram </h3>

![workflow](/Early_Fusion.png)
