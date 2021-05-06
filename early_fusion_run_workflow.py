import glob 
import os
import cv2
import numpy as np
from Pegasus.api import *
from pathlib import Path
import logging
import argparse
import time

from bin.utils import get_images, add_labels, get_ids, get_image_splits, get_tweets, split_tweets

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
    labelled_informative = add_labels(informative, 'informative')
    labelled_non_informative = add_labels(non_informative, 'non_informative')
    
    total_images = labelled_informative + labelled_non_informative
    
    # get unique image ids and their corresponding names
    unique_ids, id_to_image_dict = get_ids(total_images)
    
    # returns path to the csv containing all tweets and its name
    all_tweets_path = get_tweets(unique_ids)
    
    #split tweets into train val and test
    train_tweets_path, val_tweets_path, test_tweets_path  = split_tweets(all_tweets_path)
    
    # add prefix train, val , test to image file name
    image_dataset = get_image_splits(train_tweets_path, val_tweets_path, test_tweets_path, id_to_image_dict)
    
    return train_tweets_path, val_tweets_path, test_tweets_path, image_dataset


def replica_catalogue(train_tweets_path, val_tweets_path, test_tweets_path, dataset_images, EMBEDDING_BASE_PATH):

    """
    Function to add paths of dataset to the replica catalogue. 
    Returns File object of entire image dataset and combined tweets csv.
    :params: dataset - list of image paths
             all_tweets_path - path to the combined tweets csv
             tweets_file_name - filename of the combined tweets csv
    """
    rc = ReplicaCatalog()

    # list of input file objects
    input_images = []

    for image_path in dataset_images:
        name = image_path.split("/")[-1]
        image_file = File(name)
        input_images.append(image_file)
        rc.add_replica("local", image_file,  image_path)

    glove_embeddings = File('glove.twitter.27B.200d.txt')
    

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

    train_tweets_name = File(train_tweets_path.split('/')[-1])
    val_tweets_name = File(val_tweets_path.split('/')[-1])
    test_tweets_name = File(test_tweets_path.split('/')[-1])
    
    rc.add_replica("local", train_tweets_name, train_tweets_path)
    rc.add_replica("local", val_tweets_name, val_tweets_path)
    rc.add_replica("local", test_tweets_name, test_tweets_path)

    rc.add_replica("local", glove_embeddings, os.path.join(os.getcwd(), os.path.join(EMBEDDING_BASE_PATH, GLOVE_EMBEDDING_FILE)))            
    rc.write()

    return input_images, train_tweets_name, val_tweets_name, test_tweets_name, glove_embeddings, supcon_checkpoint_object, supcon_util_obj, resnet_big_obj, losses_obj



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

    # train SupCon
    main_supcon = Transformation(
                        "main_supcon",
                        site = 'local',
                        pfn = os.path.join(os.getcwd(), "bin/main_supcon.py"),
                        is_stageable = True,
                        container=crisis_container
                    )

    # generate SupCon Embeddings
    gen_supcon_embed = Transformation(
                        "generate_supcon_embeddings",
                        site = 'local',
                        pfn = os.path.join(os.getcwd(), "bin/generate_supcon_embeddings.py"),
                        is_stageable = True,
                        container=crisis_container
                    )

    tc.add_containers(crisis_container)
    tc.add_transformations(gen_supcon_embed, preprocess_tweets, preprocess_images, split_tweets, main_supcon)
    tc.write()

    return gen_supcon_embed, preprocess_tweets, preprocess_images, split_tweets, main_supcon


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

    train_tweets_path, val_tweets_path, test_tweets_path, image_dataset = run_pre_workflow()

    input_images, train_tweets, val_tweets, test_tweets, glove_embeddings, supcon_checkpoint_object, supcon_util_obj, resnet_big_obj, losses_obj = replica_catalogue(train_tweets_path, val_tweets_path, test_tweets_path, image_dataset, EMBEDDING_BASE_PATH)

    gen_supcon_embed, preprocess_tweets, preprocess_images, split_tweets,main_supcon = transformation_catalogue()
    
    wf = Workflow('Crisis_Computing_Workflow')

    # ---------------------------------------------------    TEXT PIPELINE     ------------------------------------------------------ 

    # Job 1: Preprocess tweets
    preprocessed_train_tweets = File('preprocessed_train_tweets.csv')
    preprocessed_val_tweets = File('preprocessed_val_tweets.csv')
    preprocessed_test_tweets = File('preprocessed_test_tweets.csv')
    
    job_preprocess_tweets = [Job(preprocess_tweets) for i in range(3)]
    job_preprocess_tweets[0].add_inputs(train_tweets)
    job_preprocess_tweets[0].add_outputs(preprocessed_train_tweets)
    job_preprocess_tweets[0].add_args('--filename', 'train_tweets.csv')
        
    job_preprocess_tweets[1].add_inputs(val_tweets)
    job_preprocess_tweets[1].add_outputs(preprocessed_val_tweets)
    job_preprocess_tweets[1].add_args('--filename', 'val_tweets.csv')
    
    job_preprocess_tweets[2].add_inputs(test_tweets)
    job_preprocess_tweets[2].add_outputs(preprocessed_test_tweets)
    job_preprocess_tweets[2].add_args('--filename', 'test_tweets.csv')


    # ---------------------------------------------------    IMAGE PIPELINE     ------------------------------------------------------ 

    # Job 1: Preprocess images
    prefix = "resized_"
    job_preprocess_images = [Job(preprocess_images) for i in range(NUM_WORKERS)]
    resized_images = split_preprocess_jobs(job_preprocess_images, input_images, prefix)


    # ---------------------------------------------------    EARLY FUSION    ------------------------------------------------------ 
    
    #Job 1: Train SupCon Model
    supcon_final_model = File('supcon_final_model.pth')

    job_train_supcon = Job(main_supcon)\
                        .add_inputs(*resized_images, supcon_util_obj, resnet_big_obj, losses_obj)\
                        .add_checkpoint(supcon_checkpoint_object, stage_out=True)\
                        .add_outputs(supcon_final_model)\
                        .add_args('--batch_size', SUPCON_BATCH_SIZE, '--epochs', SUPCON_EPOCHS, '--size', SUPCON_RESIZE_IMAGE)\
                        .add_profiles(Namespace.PEGASUS, key="maxwalltime", value=MAXTIMEWALL)

    #Job 2: Generate SupCon Embeddings
    supcon_train_embeddings = File('train_supcon_embeddings.csv')
    supcon_test_embeddings = File('test_supcon_embeddings.csv')

    job_gen_supcon_embed = Job(gen_supcon_embed)\
                        .add_inputs(*resized_images, supcon_final_model, resnet_big_obj)\
                        .add_outputs(supcon_train_embeddings, supcon_test_embeddings)


    wf.add_jobs(job_gen_supcon_embed, *job_preprocess_tweets, *job_preprocess_images, job_train_supcon)

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
    global MAXTIMEWALL
    global EMBEDDING_BASE_PATH
    global SUPCON_BATCH_SIZE
    global SUPCON_EPOCHS
    global SUPCON_RESIZE_IMAGE


    parser = argparse.ArgumentParser(description="Crisis Computing Workflow")   

    parser.add_argument('--embedding_path', type=str, default='dataset_temp/',help='path to glove embedding')
    parser.add_argument('--num_workers', type=int, default= 5, help = "number of workers")
    parser.add_argument('--maxwalltime', type=int, default= 30, help = "maxwalltime")
    parser.add_argument('--supcon_bs', type=int, default= 2, help = "Batch size for SupCon model") # change this default to 16/64/256 depending on gpu
    parser.add_argument('--supcon_epochs', type=int, default= 1, help = "Epochs for SupCon model") # change to 1000
    parser.add_argument('--supcon_img_size', type=int, default= 64, help = "Image Size to be fed to SupCon model") # if one gpu then 128 should be the max size

    ARGS                = parser.parse_args()
    EMBEDDING_BASE_PATH = ARGS.embedding_path
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