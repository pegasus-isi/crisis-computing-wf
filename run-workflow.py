import glob 
import os
import cv2
import numpy as np
from Pegasus.api import *
from pathlib import Path
import logging
import argparse
import time

from bin.utils import get_images, add_labels, get_data_splits, get_tweets

logging.basicConfig(level=logging.DEBUG)
props = Properties()
props["pegasus.mode"] = "development"
props.write()

# File Names
GLOVE_EMBEDDING_FILE = 'glove.twitter.27B.200d.txt'
GRAPH_FILENAME = "crisis-wf.dot"

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def run_pre_workflow():
    """
    Executes pre-workflow tasks:
        1. Get Informative and Non-Informative images from the dataset directory.
        2. Add label _0 for Non-Informative and _1 for Informative class respectively.
        3. Split Images into train, test and val set.
        4. Combine all tweets into one CSV file.

    returns paths to each image in the dataset, path to the tweets csv file, and its filename.
    """
    # retrieves images with their paths
    informative, non_informative = get_images()

    # appends labels _0 and _1 as per the class
    add_labels(informative, 'informative')
    add_labels(non_informative, 'non_informative')

    # splits data into train, validation and test
    image_dataset = get_data_splits()

    # returns path to the csv containing all tweets and its name
    all_tweets_path, tweets_file_name = get_tweets()

    return image_dataset, all_tweets_path, tweets_file_name


def replica_catalogue(dataset, all_tweets_path, tweets_file_name, EMBEDDING_BASE_PATH):

    """
    Function to add paths of dataset to the replica catalogue. 
    Returns File object of entire image dataset and combined tweets csv.
    :params: dataset - list of image paths
             all_tweets_path - path to the combined tweets csv
             tweets_file_name - filename of the combined tweets csv
    """
    rc = ReplicaCatalog()

    # list of entire dataset
    dataset_images = []
    dataset_images.extend(dataset['train'])
    dataset_images.extend(dataset['val'])
    dataset_images.extend(dataset['test'])

    # list of input file objects
    input_images = []

    for image_path in dataset_images:
        name = image_path.split("/")[-1]
        image_file = File(name)
        input_images.append(image_file)
        rc.add_replica("local", image_file,  image_path)

    tweets_csv_name = File(tweets_file_name)
    glove_embeddings = File('glove.twitter.27B.200d.txt')
    
    resnet_checkpoint = open("checkpoint_resnet.pth", 'w')
    resnet_checkpoint_object = File("checkpoint_resnet.pth")
    rc.add_replica("local", resnet_checkpoint_object, os.path.join(os.getcwd(), "checkpoint_resnet.pth"))    

    hpo_checkpoint = open("checkpoint_hpo.pkl", 'w')
    hpo_checkpoint_object = File("checkpoint_hpo.pkl")
    rc.add_replica("local", hpo_checkpoint_object, os.path.join(os.getcwd(), "checkpoint_hpo.pkl"))

    # Supcon file objects
    supcon_checkpoint = open("checkpoint_supcon.pth", 'w')
    supcon_checkpoint_object = File("checkpoint_supcon.pth")
    rc.add_replica("local", supcon_checkpoint_object, os.path.join(os.getcwd(), "checkpoint_supcon.pth"))

    supcon_util_obj = File('supcon_util.py')
    rc.add_replica("local", supcon_util_obj, os.path.join(os.getcwd(), "bin/supcon_util.py"))

    resnet_big_obj = File('resnet_big.py')
    rc.add_replica("local", resnet_big_obj, os.path.join(os.getcwd(), "bin/resnet_big.py"))

    losses_obj = File('losses.py')
    rc.add_replica("local", losses_obj, os.path.join(os.getcwd(), "bin/losses.py"))


    rc.add_replica("local", tweets_csv_name, all_tweets_path)
    rc.add_replica("local", glove_embeddings, os.path.join(os.getcwd(), os.path.join(EMBEDDING_BASE_PATH, GLOVE_EMBEDDING_FILE)))            
    rc.write()

    return input_images, tweets_csv_name, glove_embeddings, resnet_checkpoint_object, hpo_checkpoint_object, supcon_checkpoint_object, supcon_util_obj, resnet_big_obj, losses_obj



def transformation_catalogue():
    """
    Function to add executable files in the bin folder to the transformation catalogue.
    """
    tc = TransformationCatalog()

    # Add docker container
    crisis_container = Container(
                'crisis_container',
                Container.DOCKER,
                image = "docker://slnagark/crisis_wf:latest",
                arguments="--runtime=nvidia --shm-size=1gb"
                        ).add_env(TORCH_HOME="/tmp")


    # preprocessing scripts
    preprocess_images = Transformation(
                            "preprocess_images",
                            site = "local",
                            pfn = os.path.join(os.getcwd(), "bin/preprocess_images.py"), 
                            is_stageable = True,
                            container=crisis_container
                        )

    preprocess_tweets = Transformation(
                            "preprocess_tweets",
                            site = 'local',
                            pfn = os.path.join(os.getcwd(), "bin/preprocess_tweets.py"), 
                            is_stageable = True,
                            container=crisis_container
                        )

    split_tweets = Transformation(
                        "split_tweets",
                        site = 'local',
                        pfn = os.path.join(os.getcwd(), "bin/split_tweets.py"),
                        is_stageable = True,
                        container=crisis_container
                    )

    # HPO, training and inference scripts for ResNet-50
    hpo_train_resnet = Transformation(
                        "hpo_train_resnet",
                        site = 'local',
                        pfn = os.path.join(os.getcwd(), "bin/hpo_train_resnet.py"),
                        is_stageable = True,
                        container=crisis_container
                    )

    train_resnet = Transformation(
                        "train_resnet",
                        site = 'local',
                        pfn = os.path.join(os.getcwd(), "bin/train_resnet.py"),
                        is_stageable = True,
                        container=crisis_container
                    )

    resnet_inference = Transformation(
                        "resnet_inference",
                        site = 'local',
                        pfn = os.path.join(os.getcwd(), "bin/resnet_inference.py"),
                        is_stageable = True,
                        container=crisis_container
                    )

    # HPO, training and inference scripts for Bi-LSTM

    hpo_train_bilstm = Transformation(
                        "hpo_train_bilstm",
                        site = 'local',
                        pfn = os.path.join(os.getcwd(), "bin/hpo_train_bilstm.py"),
                        is_stageable = True,
                        container=crisis_container
                    )

    train_bilstm = Transformation(
                        "train_bilstm",
                        site = 'local',
                        pfn = os.path.join(os.getcwd(), "bin/train_bilstm.py"),
                        is_stageable = True,
                        container=crisis_container
                    )
    
    bilstm_inference = Transformation(
                        "bilstm_inference",
                        site = 'local',
                        pfn = os.path.join(os.getcwd(), "bin/bilstm_inference.py"),
                        is_stageable = True,
                        container=crisis_container
                    )

    # late fusion script
    late_fusion = Transformation(
                        "late_fusion",
                        site = 'local',
                        pfn = os.path.join(os.getcwd(), "bin/late_fusion.py"),
                        is_stageable = True,
                        container=crisis_container
                    )

    # train SupCon
    main_supcon = Transformation(
                        "main_supcon",
                        site = 'local',
                        pfn = os.path.join(os.getcwd(), "bin/main_supcon.py"),
                        is_stageable = True,
                        container=crisis_container
                    )

    tc.add_containers(crisis_container)
    tc.add_transformations(preprocess_tweets, preprocess_images, split_tweets, train_resnet, hpo_train_resnet, train_bilstm, hpo_train_bilstm, resnet_inference, bilstm_inference, late_fusion, main_supcon)
    tc.write()

    return preprocess_tweets, preprocess_images, split_tweets, train_resnet, hpo_train_resnet, train_bilstm, hpo_train_bilstm, resnet_inference, bilstm_inference, late_fusion, main_supcon


def split_preprocess_jobs(preprocess_images_job, input_images, prefix):
    """
    Function to split the preprocessing task into parallel N jobs.
    Returns File objects of the preprocessed images
    :params: preprocess_images_job - list of N preprocessing jobs
             input_images - File objects of input image dataset
             prefix - keyword to add to the new file names ('resized_')

    """
    resized_images = []
    
    for i in range(len(input_images)):
        curr = i % len(preprocess_images_job)
        preprocess_images_job[curr].add_inputs(input_images[i])
        out_file = File(prefix + str(input_images[i]))
        preprocess_images_job[curr].add_outputs(out_file)
        resized_images.append(out_file)
        
    return resized_images

def plot_workflow_graph(wf):
    """
    Plots the graph of the workflow.
    """
    wf.graph(include_files=True, no_simplify=True, label="xform-id", output = GRAPH_FILENAME)
    
    return


def run_workflow(EMBEDDING_BASE_PATH):

    image_dataset, all_tweets_path, tweets_file_name = run_pre_workflow()

    input_images, tweets_csv_name, glove_embeddings, resnet_checkpoint_object, hpo_checkpoint_object, supcon_checkpoint_object, supcon_util_obj, resnet_big_obj, losses_obj = replica_catalogue(image_dataset, all_tweets_path, tweets_file_name, EMBEDDING_BASE_PATH)

    preprocess_tweets, preprocess_images, split_tweets, train_resnet, hpo_train_resnet, train_bilstm, hpo_train_bilstm, resnet_inference, bilstm_inference, late_fusion, main_supcon = transformation_catalogue()
    
    
    wf = Workflow('Crisis_Computing_Workflow')

    # ---------------------------------------------------    TEXT PIPELINE     ------------------------------------------------------ 

    # Job 1: Preprocess tweets
    preprocessed_tweets = File('preprocessed_tweets.csv')

    job_preprocess_tweets = Job(preprocess_tweets)\
                            .add_inputs(tweets_csv_name)\
                            .add_outputs(preprocessed_tweets)

    # Job 2: Split tweets
    train_tweets = File('train_tweets.csv')
    val_tweets = File('val_tweets.csv')
    test_tweets = File('test_tweets.csv')

    job_split_tweets = Job(split_tweets)\
                        .add_inputs(preprocessed_tweets, *input_images)\
                        .add_outputs(train_tweets, val_tweets, test_tweets)

    # Job 3: HPO Bi-LSTM
    bilstm_hpo_study = File('hpo_crisis_bilstm.pkl')
    bilstm_best_params = File('best_bilstm_hpo_params.txt')

    job_hpo_train_bilstm = Job(hpo_train_bilstm)\
                            .add_inputs(glove_embeddings, train_tweets, val_tweets, test_tweets)\
                            .add_outputs(bilstm_best_params, bilstm_hpo_study)\
                            .add_args('--trials', BILSTM_NUM_TRIALS)

    # Job 4: Train Bi-LSTM using best parameters from HPO study and output loss and accuracy curves
    trained_bilstm_model = File('bilstm_final_model.h5')    
    bilstm_loss_curve = File('Loss_curve_bilstm.png')
    bilstm_accuracy_curve = File('Accuracy_curve_bilstm.png')


    job_train_bilstm = Job(train_bilstm)\
                        .add_inputs(glove_embeddings, train_tweets, val_tweets, test_tweets, bilstm_best_params)\
                        .add_outputs(bilstm_loss_curve, bilstm_accuracy_curve, trained_bilstm_model)

    # Job 5: Run inference on best Bi-LSTM  model to produce output on test dataset along with confusion matrix
    bilstm_train_output_prob = File('bilstm_train_output.csv')
    bilstm_test_output_prob = File('bilstm_test_output.csv')
    bilstm_confusion_matrix = File('bilstm_confusion_matrix.png')

    job_bilstm_inference = Job(bilstm_inference)\
                            .add_inputs(train_tweets, val_tweets, test_tweets, trained_bilstm_model)\
                            .add_outputs(bilstm_train_output_prob, bilstm_test_output_prob, bilstm_confusion_matrix)


    # ---------------------------------------------------    IMAGE PIPELINE     ------------------------------------------------------ 

    # Job 1: Preprocess images
    prefix = "resized_"
    job_preprocess_images = [Job(preprocess_images) for i in range(NUM_WORKERS)]
    resized_images = split_preprocess_jobs(job_preprocess_images, input_images, prefix)

    # Job 2: HPO ResNet-50
    resnet_hpo_study = File('hpo_crisis_resnet.pkl')
    resnet_best_params = File('best_resnet_hpo_params.txt')

    job_hpo_train_resnet = Job(hpo_train_resnet)\
                            .add_inputs(*resized_images)\
                            .add_args('--trials', RESNET_NUM_TRIALS)\
                            .add_checkpoint(hpo_checkpoint_object, stage_out=True)\
                            .add_outputs(resnet_best_params)\
                            .add_profiles(Namespace.PEGASUS, key="maxwalltime", value=MAXTIMEWALL)


    # Job 3: Train ResNet-50 using best parameters from HPO study and output loss and accuracy curves
    trained_resnet_model = File('resnet_final_model.pth')
    resnet_loss_curve = File('Loss_curve_resnet.png')
    resnet_accuracy_curve = File('Accuracy_curve_resnet.png')

    job_train_resnet = Job(train_resnet)\
                        .add_inputs(*resized_images, resnet_best_params)\
                        .add_checkpoint(resnet_checkpoint_object, stage_out=True)\
                        .add_outputs(resnet_loss_curve, resnet_accuracy_curve, trained_resnet_model)\
                        .add_profiles(Namespace.PEGASUS, key="maxwalltime", value=MAXTIMEWALL)


    # Job 4: Run inference on best ResNet-50  model to produce output on test dataset along with confusion matrix
    resnet_train_output_prob = File('resnet_train_output.csv')
    resnet_confusion_matrix = File('resnet_confusion_matrix.png')
    resnet_test_output_prob = File('resnet_test_output.csv')   

    job_resnet_inference = Job(resnet_inference)\
                            .add_inputs(*resized_images, trained_resnet_model)\
                            .add_outputs(resnet_train_output_prob, resnet_test_output_prob, resnet_confusion_matrix)

    
    
    # ---------------------------------------------------    LATE FUSION    ------------------------------------------------------ 

    # Job 1: Late Fusion
    confusion_matrix_MPC = File('late_fusion_MPC.png')
    confusion_matrix_LR = File('late_fusion_LR.png')
    confusion_matrix_MLP = File('late_fusion_MLP.png')
    report_MLP = File('late_fusion_MLP.csv')
    report_MPC = File('late_fusion_MPC.csv')
    report_LR = File('late_fusion_LR.csv')

    job_late_fusion = Job(late_fusion)\
                        .add_inputs(resnet_train_output_prob, resnet_test_output_prob, bilstm_train_output_prob, bilstm_test_output_prob)\
                        .add_outputs(confusion_matrix_MPC, confusion_matrix_LR, confusion_matrix_MLP, report_MLP, report_MPC, report_LR)

    # ---------------------------------------------------    EARLY FUSION    ------------------------------------------------------ 
    
    #Job 1: Train SupCon Model

    supcon_final_model = File('supcon_final_model.pth')

    job_train_supcon = Job(main_supcon)\
                        .add_inputs(*resized_images, supcon_util_obj, resnet_big_obj, losses_obj)\
                        .add_checkpoint(supcon_checkpoint_object, stage_out=True)\
                        .add_outputs(supcon_final_model)\
                        .add_args('--batch_size', SUPCON_BATCH_SIZE, '--epochs', SUPCON_EPOCHS, '--size', SUPCON_RESIZE_IMAGE)\
                        .add_profiles(Namespace.PEGASUS, key="maxwalltime", value=MAXTIMEWALL)


    wf.add_jobs(job_preprocess_tweets, job_split_tweets, *job_preprocess_images, job_hpo_train_resnet, job_train_resnet, job_hpo_train_bilstm, job_train_bilstm, job_resnet_inference, job_bilstm_inference, job_late_fusion, job_train_supcon)

    try:
        wf.plan(submit=True)
        wf.wait()
        wf.statistics()
    except PegasusClientError as e:
        print(e.output)
    
    plot_workflow_graph(wf)
    
    return

def main():

    start = time.time()

    global ARGS
    global NUM_WORKERS
    global RESNET_NUM_TRIALS
    global BILSTM_NUM_TRIALS
    global MAXTIMEWALL
    global EMBEDDING_BASE_PATH
    global SUPCON_BATCH_SIZE
    global SUPCON_EPOCHS
    global SUPCON_RESIZE_IMAGE


    parser = argparse.ArgumentParser(description="Crisis Computing Workflow")   

    parser.add_argument('--embedding_path', type=str, default='dataset_temp/',help='path to glove embedding')
    parser.add_argument('--bilstm_trials', type=int, default=1, help = "number of Bi-LSTM trials")   
    parser.add_argument('--resnet_trials', type=int, default=1, help = "number of ResNet-50 trials") 
    parser.add_argument('--num_workers', type=int, default= 5, help = "number of workers")
    parser.add_argument('--maxwalltime', type=int, default= 30, help = "maxwalltime")
    parser.add_argument('--supcon_bs', type=int, default= 2, help = "Batch size for SupCon model") # change this default to 16/64/256 depending on gpu
    parser.add_argument('--supcon_epochs', type=int, default= 1, help = "Epochs for SupCon model") # change to 1000
    parser.add_argument('--supcon_img_size', type=int, default= 64, help = "Image Size to be fed to SupCon model") # if one gpu then 128 should be the max size

    ARGS                = parser.parse_args()
    EMBEDDING_BASE_PATH = ARGS.embedding_path
    BILSTM_NUM_TRIALS   = ARGS.bilstm_trials
    RESNET_NUM_TRIALS   = ARGS.resnet_trials
    NUM_WORKERS         = ARGS.num_workers
    MAXTIMEWALL         = ARGS.maxwalltime
    SUPCON_BATCH_SIZE   = ARGS.supcon_bs
    SUPCON_EPOCHS       = ARGS.supcon_epochs
    SUPCON_RESIZE_IMAGE = ARGS.supcon_img_size

    # execute pre-workflow tasks
    run_pre_workflow()

    # run the workflow
    run_workflow(EMBEDDING_BASE_PATH)

    exec_time = time.time() - start
    print('Execution time in seconds: ' + str(exec_time))

    return

if __name__ == "__main__":
    main()