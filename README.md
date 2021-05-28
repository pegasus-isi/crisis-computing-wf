# Crisis-Computing-Late-Fusion-Workflow

<h2> Dataset </h2>
This repository contains two types of dataset: <br>

<b>partial_data:</b> To run the workflow on a smaller dataset to debug issues.<br>
<b>full_data:</b> To run the workflow on the entire dataset.<br>
  
In order to change between the two please follow these steps:
  1. Go to bin/utils.py
  2. Locate the 'DATA_FOLDER' variable on the 12th line.
  3. If using partial data:
             
         DATA_FOLDER = '/partial_data'
  4. If using full data:
 
         DATA_FOLDER = '/full_data'

Additionally, follow these steps to download pre-trained GloVe embeddings:
  1. Go to http://nlp.stanford.edu/data/glove.twitter.27B.zip
  2. Unzip the file
  3. Copy the 'glove.twitter.27B.200d.txt' and place it inside <b>full_data</b> and <b>partial_data</b>
  4. Default embedding path in late_fusion_workflow.py has been set to full_data in the parser arguments.

<b>**Important Note:**</b>

For all the Training, HPO and Inference scripts: <br>
If using <b>partial_data</b> set BATCH_SIZE = 2 <br>
If using <b>full_data</b> set BATCH_SIZE>8  <br>

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

<h3>HPO </h3>

Performs Hyper-parameter tuning using optuna

For Images:
      
          python bin/hpo_train_resnet.py --trails=10
     
For Text:
      
          python bin/hpo_train_bilstm.py --trials=10
     
Outputs: 
  Image: best_resnet_hpo_params.txt, hpo_crisis_resnet.pkl
  Text: best_bilstm_hpo_params.txt, hpo_crisis_resnet.pkl
 
<h3>Train Model </h3>

Trains models with best hyperparameters obtained from HPO


For Images:
      
          python bin/train_resnet.py 
     
For Text:
      
          python bin/train_bilstm.py
     
Outputs: 
  Image: Loss_curve_resnet.png, Accuracy_curve_resnet.png, resnet_final_model.pth <br>
  Text: Loss_curve_bilstm.png, Accuracy_curve_bilstm.png, bilstm_final_model.h5
  
<h3>Model Inference </h3>

Produces confusion matrix and model output for train and test data

For Images:
      
          python bin/resnet_inference.py 
     
For Text:
      
          python bin/bilstm_inference.py
     
Outputs: 
  Image: resnet_train_output.csv, resnet_test_output.csv, resnet_confusion_matrix.png<br>
  Text: bilstm_train_output.csv, bilstm_test_output.csv, bilstm_confusion_matrix.png

<h3>Late Fusion </h3>

Uses Mean Probability Concatenation, Logistic Regression and MLP Decision Policy to fuse image and text model outputs. Generates Performance statistics report.
      
          python bin/late_fusion.py 
     
Outputs: confusion_matrix_MPC.png, confusion_matrix_LR.png, confusion_matrix_MLP.png, report_MPC.png, report_LR.png, report_MLP.png

<h3>Workflow Diagram </h3>

![workflow](/Late_Fusion.png)
