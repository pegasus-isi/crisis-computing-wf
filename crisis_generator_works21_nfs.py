#!/usr/bin/env python3

# Important Libraries
import os
from Pegasus.api import *
from pathlib import Path
import logging
import argparse
import time
from bin.utils import get_images, add_labels, get_ids, get_image_splits, get_tweets, split_tweets


logging.basicConfig(level=logging.DEBUG)

# File Names
GLOVE_EMBEDDING_FILE = 'glove.twitter.27B.200d.txt'
GRAPH_FILENAME = "crisis-wf.dot"

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def run_pre_workflow():
    """
    Executes pre-workflow tasks:
        1. Get Informative and Non-Informative images from the dataset directory.
        2. Add label _0 for Non-Informative and _1 for Informative class respectively.
        3. Combine all tweets into one CSV file.
        4. Split Tweets into train, test and val set.
        5. Split Images into train, test and val set.
        

    returns paths to each image in the dataset, path to the train, test and val tweets csv file
    """
    
    
    # retrieves images with their paths
    informative, non_informative = get_images()

    # appends labels _0 and _1 as per the class
    # returns a list of tuples with old name - new name
    labelled_informative = add_labels(informative, 'informative')
    labelled_non_informative = add_labels(non_informative, 'non_informative')
    
    total_images = labelled_informative + labelled_non_informative
    
    # get unique image ids and their corresponding names
    # use new name from the tuples
    unique_ids, id_to_image_dict = get_ids([image_name[0] for image_name in total_images])
 
    # returns path to the csv containing all tweets and its name
    all_tweets_path = get_tweets(unique_ids)
    
    #split tweets into train val and test randomly
    train_tweets_path, val_tweets_path, test_tweets_path  = split_tweets(all_tweets_path)
    
    # add prefix train, val , test to image file name based on the split from tweets.
    image_dataset = get_image_splits(train_tweets_path, val_tweets_path, test_tweets_path, id_to_image_dict)
    
    return train_tweets_path, val_tweets_path, test_tweets_path, image_dataset


def replica_catalog(train_tweets_path, val_tweets_path, test_tweets_path, dataset_images, EMBEDDING_BASE_PATH):

    """
    Function to add paths of dataset to the replica catalogue. 
    Returns File object of entire image dataset and combined tweets csv.
    :params: dataset_images - list of image paths
             {train, val, test}_tweets_path - path to the all tweets csv
             EMBEDDING_BASE_PATH - path to the pre-trained GLOVE embedding
             
    """
    rc = ReplicaCatalog()
    data_location = "file:///nfs/shared/panorama/data/CrisisComputing"

    # list of input file objects
    input_images = []

    # Adding Images to the replica catalogue
    for image_path in dataset_images:
        name = image_path[1].split("/")[-1]
        image_file = File(name)
        input_images.append(image_file)
        
        path_split = image_path[0].split("/")
        rc.add_replica("condorpool", image_file, os.path.join(data_location, path_split[-3], path_split[-2], path_split[-1]))

   
    glove_embeddings = File('glove.twitter.27B.200d.txt')
    
    # File objects for train, val and test tweets csv
    train_tweets_name = File(train_tweets_path.split('/')[-1])
    val_tweets_name = File(val_tweets_path.split('/')[-1])
    test_tweets_name = File(test_tweets_path.split('/')[-1])
    
    path_split = train_tweets_path.split('/')
    rc.add_replica("condorpool", train_tweets_name, os.path.join(data_location, path_split[-2], path_split[-1]))
    path_split = val_tweets_path.split('/')
    rc.add_replica("condorpool", val_tweets_name, os.path.join(data_location, path_split[-2], path_split[-1]))
    path_split = test_tweets_path.split('/')
    rc.add_replica("condorpool", test_tweets_name, os.path.join(data_location, path_split[-2], path_split[-1]))
    
    rc.add_replica("condorpool", glove_embeddings, os.path.join(data_location, "glove_twitter", GLOVE_EMBEDDING_FILE))          
    rc.write()

    return input_images, train_tweets_name, val_tweets_name, test_tweets_name, glove_embeddings


def pegasus_properties():
    props = Properties()
    props["pegasus.mode"] = "development"
    props["pegasus.transfer.links"] = "true"
    props["pegasus.transfer.bypass.input.staging"] = "true"
    props["pegasus.transfer.threads"] = "8"
    props["pegasus.monitord.encoding"] = "json"
    props["pegasus.catalog.workflow.amqp.events"] = "stampede.*"
    props["pegasus.catalog.workflow.amqp.url"] = "amqps://panorama:panorama@hostname:5671/panorama/monitoring"
    props.write()


def sites_catalog():
    sc = SiteCatalog()

    local = Site("local")\
                .add_directories(
                    Directory(Directory.SHARED_SCRATCH, "/nfs/shared/panorama/scratch")
                        .add_file_servers(FileServer("file:///nfs/shared/panorama/scratch", Operation.ALL)),
                    Directory(Directory.SHARED_STORAGE, "/nfs/shared/panorama/storage")
                        .add_file_servers(FileServer("file:///nfs/shared/panorama/storage", Operation.ALL))
                )

    condorpool = Site("condorpool")\
                .add_directories(
                    Directory(Directory.SHARED_SCRATCH, "/nfs/shared/panorama/scratch")
                        .add_file_servers(FileServer("file:///nfs/shared/panorama/scratch", Operation.ALL)),
                    Directory(Directory.SHARED_STORAGE, "/nfs/shared/panorama/storage")
                        .add_file_servers(FileServer("file:///nfs/shared/panorama/storage", Operation.ALL))
                )\
                .add_condor_profile(universe="vanilla")\
                .add_pegasus_profile(
                    style="condor",
                    data_configuration="nonsharedfs",
                    auxillary_local="true",
                    cores=8,
                    memory=2048,
                    runtime=4800,
                    grid_start_arguments="-m 10"
                )\
                .add_env(key="PEGASUS_HOME", value="/nfs/shared/panorama/pegasus-5.1.0panorama")\
                .add_env(key="KICKSTART_MON_URL", value="rabbitmqs://panorama:panorama@hostname:15671/api/exchanges/panorama/monitoring/publish")\
                .add_env(key="PEGASUS_TRANSFER_PUBLISH", value="1")\
                .add_env(key="PEGASUS_AMQP_URL", value="amqps://panorama:panorama@hostname:5671/panorama/monitoring")
    
    sc.add_sites(local, condorpool)
    sc.write()


def transformation_catalog():
    """
    Function to add executable files in the bin folder to the transformation catalogue.
    """
    tc = TransformationCatalog()

    crisis_container = Container(
                'crisis_computing_container',
                Container.SINGULARITY,
                image = "file:///nfs/shared/panorama/containers/crisis-computing_latest.sif",
                image_site = "condorpool",
                mounts = ["/nfs/shared:/nfs/shared"]
    ).add_env(TORCH_HOME="/tmp")\
    .add_env(key="PEGASUS_HOME", value="/nfs/shared/panorama/pegasus-5.1.0panorama")


    # preprocessing scripts
    preprocess_images = Transformation(
                            "preprocess_images",
                            site = "condorpool",
                            pfn = "file:///nfs/shared/panorama/bin/CrisisComputing/preprocess_images.py", 
                            is_stageable = False,
                            container=crisis_container
                        )

    preprocess_tweets = Transformation(
                            "preprocess_tweets",
                            site = "condorpool",
                            pfn = "file:///nfs/shared/panorama/bin/CrisisComputing/preprocess_tweets.py", 
                            is_stageable = False,
                            container=crisis_container
                        )

    
    # HPO, training and inference scripts for ResNet-50
    hpo_train_resnet = Transformation(
                        "hpo_train_resnet",
                        site = "condorpool",
                        pfn = "file:///nfs/shared/panorama/bin/CrisisComputing/hpo_train_resnet.py",
                        is_stageable = False,
                        container=crisis_container
                  )\
                  .add_pegasus_profile(cores=24, gpus=1, memory=131072, runtime=43200, grid_start_arguments="-G -m 10")\
                  .add_env(key="KICKSTART_MON_GRAPHICS_PCIE", value="TRUE")

    train_resnet = Transformation(
                        "train_resnet",
                        site = "condorpool",
                        pfn = "file:///nfs/shared/panorama/bin/CrisisComputing/train_resnet.py",
                        is_stageable = False,
                        container=crisis_container
                  )\
                  .add_pegasus_profile(cores=24, gpus=1, memory=131072, runtime=43200, grid_start_arguments="-G -m 10")\
                  .add_env(key="KICKSTART_MON_GRAPHICS_PCIE", value="TRUE")

    resnet_inference = Transformation(
                        "resnet_inference",
                        site = "condorpool",
                        pfn = "file:///nfs/shared/panorama/bin/CrisisComputing/resnet_inference.py",
                        is_stageable = False,
                        container=crisis_container
                  )\
                  .add_pegasus_profile(cores=24, gpus=1, memory=131072, runtime=43200, grid_start_arguments="-G -m 10")\
                  .add_env(key="KICKSTART_MON_GRAPHICS_PCIE", value="TRUE")

    # HPO, training and inference scripts for Bi-LSTM

    hpo_train_bilstm = Transformation(
                        "hpo_train_bilstm",
                        site = "condorpool",
                        pfn = "file:///nfs/shared/panorama/bin/CrisisComputing/hpo_train_bilstm.py",
                        is_stageable = False,
                        container=crisis_container
                  )\
                  .add_pegasus_profile(cores=24, gpus=1, memory=131072, runtime=43200, grid_start_arguments="-G -m 10")\
                  .add_env(key="KICKSTART_MON_GRAPHICS_PCIE", value="TRUE")

    train_bilstm = Transformation(
                        "train_bilstm",
                        site = "condorpool",
                        pfn = "file:///nfs/shared/panorama/bin/CrisisComputing/train_bilstm.py",
                        is_stageable = False,
                        container=crisis_container
                  )\
                  .add_pegasus_profile(cores=24, gpus=1, memory=131072, runtime=43200, grid_start_arguments="-G -m 10")\
                  .add_env(key="KICKSTART_MON_GRAPHICS_PCIE", value="TRUE")
    
    bilstm_inference = Transformation(
                        "bilstm_inference",
                        site = "condorpool",
                        pfn = "file:///nfs/shared/panorama/bin/CrisisComputing/bilstm_inference.py",
                        is_stageable = False,
                        container=crisis_container
                  )\
                  .add_pegasus_profile(cores=24, gpus=1, memory=131072, runtime=43200, grid_start_arguments="-G -m 10")\
                  .add_env(key="KICKSTART_MON_GRAPHICS_PCIE", value="TRUE")

    # late fusion script
    late_fusion = Transformation(
                        "late_fusion",
                        site = "condorpool",
                        pfn = "file:///nfs/shared/panorama/bin/CrisisComputing/late_fusion.py",
                        is_stageable = False,
                        container=crisis_container
                  )\
                  .add_pegasus_profile(cores=24, gpus=1, memory=131072, runtime=43200, grid_start_arguments="-G -m 10")\
                  .add_env(key="KICKSTART_MON_GRAPHICS_PCIE", value="TRUE")


    tc.add_containers(crisis_container)
    tc.add_transformations(preprocess_tweets, preprocess_images, train_resnet, hpo_train_resnet, train_bilstm, hpo_train_bilstm, resnet_inference, bilstm_inference, late_fusion)
    tc.write()

    return  preprocess_tweets, preprocess_images, train_resnet, hpo_train_resnet, train_bilstm, hpo_train_bilstm, resnet_inference, bilstm_inference, late_fusion


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
    """
    Funtion to run the entire Pegasus Workflow.
        1. Call Pre-workflow Function
        2. Call Replica Catalogue function
        3. Call Transformation Catalogue function
        4. Text Pipeline Jobs
        5. Image Pipeline Jobs
        6. Late Fusion Job
    """
    train_tweets_path, val_tweets_path, test_tweets_path, image_dataset = run_pre_workflow()

    input_images, train_tweets, val_tweets, test_tweets, glove_embeddings = replica_catalog(train_tweets_path, val_tweets_path, test_tweets_path, image_dataset, EMBEDDING_BASE_PATH)

    preprocess_tweets, preprocess_images, train_resnet, hpo_train_resnet, train_bilstm, hpo_train_bilstm, resnet_inference, bilstm_inference, late_fusion = transformation_catalog()
    
    sites_catalog()

    pegasus_properties()
    
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


    # Job 2: HPO Bi-LSTM
    bilstm_best_params = File('best_bilstm_hpo_params.txt')

    job_hpo_train_bilstm = Job(hpo_train_bilstm)\
                            .add_inputs(glove_embeddings, preprocessed_train_tweets, preprocessed_val_tweets, preprocessed_test_tweets)\
                            .add_outputs(bilstm_best_params)\
                            .add_args('--trials', BILSTM_NUM_TRIALS)


    # Job 3: Train Bi-LSTM using best parameters from HPO study and output loss and accuracy curves
    trained_bilstm_model = File('bilstm_final_model.h5')    
    bilstm_loss_curve = File('Loss_curve_bilstm.png')
    bilstm_accuracy_curve = File('Accuracy_curve_bilstm.png')


    job_train_bilstm = Job(train_bilstm)\
                        .add_inputs(glove_embeddings, preprocessed_train_tweets, preprocessed_val_tweets, preprocessed_test_tweets, bilstm_best_params)\
                        .add_outputs(bilstm_loss_curve, bilstm_accuracy_curve, trained_bilstm_model)\


    # Job 4: Run inference on best Bi-LSTM  model to produce output on test dataset along with confusion matrix
    bilstm_train_output_prob = File('bilstm_train_output.csv')
    bilstm_test_output_prob = File('bilstm_test_output.csv')
    bilstm_confusion_matrix = File('bilstm_confusion_matrix.png')

    job_bilstm_inference = Job(bilstm_inference)\
                            .add_inputs(preprocessed_train_tweets, preprocessed_val_tweets, preprocessed_test_tweets, trained_bilstm_model)\
                            .add_outputs(bilstm_train_output_prob, bilstm_test_output_prob, bilstm_confusion_matrix)


    # ---------------------------------------------------    IMAGE PIPELINE     ------------------------------------------------------ 

    
    # Job 1: Preprocess images
    prefix = "resized_"
    job_preprocess_images = [Job(preprocess_images) for i in range(NUM_WORKERS)]
    resized_images = split_preprocess_jobs(job_preprocess_images, input_images, prefix)

    # Job 2: HPO ResNet-50
    resnet_best_params = File('best_resnet_hpo_params.txt')

    job_hpo_train_resnet = Job(hpo_train_resnet)\
                            .add_inputs(*resized_images)\
                            .add_args('--trials', RESNET_NUM_TRIALS)\
                            .add_outputs(resnet_best_params)


    # Job 3: Train ResNet-50 using best parameters from HPO study and output loss and accuracy curves
    trained_resnet_model = File('resnet_final_model.pth')
    resnet_loss_curve = File('Loss_curve_resnet.png')
    resnet_accuracy_curve = File('Accuracy_curve_resnet.png')

    job_train_resnet = Job(train_resnet)\
                        .add_inputs(*resized_images, resnet_best_params)\
                        .add_outputs(resnet_loss_curve, resnet_accuracy_curve, trained_resnet_model)


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

    wf.add_jobs(*job_preprocess_tweets, *job_preprocess_images, job_bilstm_inference, job_hpo_train_bilstm, job_train_bilstm, job_hpo_train_resnet, job_train_resnet,  job_resnet_inference, job_late_fusion)

    wf.write()
    #try:
        #wf.plan(submit=False, sites=["condorpool"], output_sites=["condorpool"], dir="submit")
        #wf.wait()
        #wf.statistics()
    #except PegasusClientError as e:
    #    print(e.output)
    
    #plot_workflow_graph(wf)
    
    return


def main():

    start = time.time()

    global ARGS
    global NUM_WORKERS
    global RESNET_NUM_TRIALS
    global BILSTM_NUM_TRIALS
    global EMBEDDING_BASE_PATH


    parser = argparse.ArgumentParser(description="Crisis Computing Workflow")   

    parser.add_argument('--embedding_path', type=str, default='full_data/glove_twitter',help='path to glove embedding')
    parser.add_argument('--bilstm_trials', type=int, default=1, help = "number of Bi-LSTM trials")   
    parser.add_argument('--resnet_trials', type=int, default=1, help = "number of ResNet-50 trials") 
    parser.add_argument('--num_workers', type=int, default=5, help = "number of workers")
   
    ARGS                = parser.parse_args()
    EMBEDDING_BASE_PATH = ARGS.embedding_path
    BILSTM_NUM_TRIALS   = ARGS.bilstm_trials
    RESNET_NUM_TRIALS   = ARGS.resnet_trials
    NUM_WORKERS         = ARGS.num_workers

    # run the workflow
    run_workflow(EMBEDDING_BASE_PATH)

    exec_time = time.time() - start
    print('Execution time in seconds: ' + str(exec_time))

    return

if __name__ == "__main__":
    main()
