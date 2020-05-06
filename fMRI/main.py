import os
import yaml
import argparse
import numpy as np

from utils import check_folder, read_yaml, save_yaml, write, get_subject_name, get_output_name, aggregate_cv, create_maps, fetch_masker, fetch_data, get_nscans
from utils import get_splitter_information, get_compression_information, get_data_transformation_information, get_encoding_model_information
from task import Task
from logger import Logger
from regression_pipeline import Pipeline
from encoding_models import EncodingModel
from splitter import Splitter
from data_transformation import Transformer
from data_compression import Compressor



if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="""Main script that compute the R2 maps for a given subject and model.""")
    parser.add_argument("--yaml_file", type=str, default="/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/code/fMRI/template.yml", 
                            help="Path to the yaml containing the parameters of the script execution.")

    args = parser.parse_args()
    parameters = read_yaml(args.yaml_file)
    input_path = parameters['input']
    output_path_ = parameters['output']
    subject = get_subject_name(parameters['subject'])
    output_path = get_output_name(output_path_, parameters['language'], subject, parameters['model_name'])
    logs = Logger(get_output_name(output_path_, parameters['language'], subject, parameters['model_name'], 'logs.txt'))
    save_yaml(parameters, output_path + 'config.yml')

    logs.info("Fetching maskers...", end='\n')
    masker = fetch_masker(parameters['masker_path'], parameters['language'], parameters['path_to_fmridata'], input_path, logger=logs)
    logs.validate()
    smoothed_masker = fetch_masker(parameters['smoothed_masker_path'], parameters['language'], parameters['path_to_fmridata'], input_path, smoothing_fwhm=5, logger=logs)
    logs.validate()

    logs.info("Retrieve arguments for each model...")
    kwargs_splitter = get_splitter_information(parameters)
    kwargs_compression = get_compression_information(parameters)
    kwargs_transformation = get_data_transformation_information(parameters)
    kwargs_encoding_model = get_encoding_model_information(parameters)
    logs.validate()

    logs.info("Instanciations of the classes...")
    splitter = Splitter(**kwargs_splitter)
    compressor = Compressor(**kwargs_compression)
    transformer = Transformer(**kwargs_transformation)
    encoding_model = EncodingModel(**kwargs_encoding_model)
    logs.validate()

    logs.info("Defining Pipeline flow...")
    splitter_cv_external = Task([splitter.split], 
                                name='splitter_cv_external')
    ## Internal Pipeline
    splitter_cv_internal = Task([splitter.split], 
                                input_dependencies=[splitter_cv_external],
                                name='splitter_cv_internal', 
                                flatten_inputs=[True]) # define the splitting strategy
    compressor_internal = Task([compressor.compress], 
                                input_dependencies=[splitter_cv_internal],
                                name='compressor_internal', 
                                flatten_inputs=[True], 
                                unflatten_output='automatic') # define the data compression method
    transform_data_internal = Task([transformer.make_regressor, transformer.standardize], 
                                input_dependencies=[splitter_cv_internal, compressor_internal],
                                name='transform_data_internal', 
                                flatten_inputs=[True, True], 
                                unflatten_output='automatic') # functions in Task.functions are read left to right
    encoding_model_internal = Task([encoding_model.grid_search], 
                                input_dependencies=[splitter_cv_internal, transform_data_internal],
                                name='encoding_model_internal', 
                                flatten_inputs=[True, True], 
                                unflatten_output='automatic',
                                special_output_transform=aggregate_cv)
    ## External Pipeline
    compressor_external = Task([compressor.compress], 
                                input_dependencies=[splitter_cv_external, encoding_model_internal], 
                                name='compressor_external', 
                                flatten_inputs=[True, False])
    transform_data_external = Task([transformer.make_regressor, transformer.standardize], 
                                input_dependencies=[splitter_cv_external, compressor_external], 
                                name='transform_data_external', 
                                flatten_inputs=[True, False])
    encoding_model_external = Task([encoding_model.evaluate], 
                                input_dependencies=[splitter_cv_external, transform_data_external, encoding_model_internal], 
                                name='encoding_model_external', 
                                flatten_inputs=[True, False, False])
    
    # Creating tree structure (for output/input flow)
    splitter_cv_external.set_children_tasks([splitter_cv_internal, compressor_external])
    splitter_cv_internal.set_children_tasks([compressor_internal])
    compressor_internal.set_children_tasks([transform_data_internal])
    transform_data_internal.set_children_tasks([encoding_model_internal])
    encoding_model_internal.set_children_tasks([compressor_external])
    compressor_external.set_children_tasks([transform_data_external])
    transform_data_external.set_children_tasks([encoding_model_external])
    logs.validate()
    
    logs.info("Fetching and preprocessing input data...")
    stimuli_representations_paths, fMRI_paths = fetch_data(parameters['path_to_fmridata'], input_path, 
                                                            subject, parameters['language'], parameters['models'])
    stimuli_representations = transformer.process_representations(stimuli_representations_paths, parameters['models'])
    fMRI_data = transformer.process_fmri_data(fMRI_paths, masker)
    logs.validate()
    
    logs.info("Executing pipeline...", end='\n')
    pipeline = Pipeline()
    pipeline.fit(splitter_cv_external, logs) # retrieve the flow from children and input_dependencies
    maps = pipeline.compute(stimuli_representations, fMRI_data, output_path, logger=logs)
    
    logs.info("Aggregating over cross-validation results...")
    maps = {key: np.mean(np.stack(np.array([dic[key] for dic in maps]), axis=0), axis=0) for key in maps[0]}
    logs.validate()
    
    logs.info("Plotting...", end='\n')
    ## R2
    output_path = get_output_name(output_path_, parameters['language'], subject, parameters['model_name'], 'R2')
    create_maps(masker, maps['R2'], output_path, vmax=None, logger=logs)
    create_maps(smoothed_masker, maps['R2'], output_path, vmax=None, logger=logs)
    ## Pearson
    output_path = get_output_name(output_path_, parameters['language'], subject, parameters['model_name'], 'Pearson_coeff')
    create_maps(masker, maps['Pearson_coeff'], output_path, vmax=None, logger=logs)
    create_maps(smoothed_masker, maps['Pearson_coeff'], output_path, vmax=None, logger=logs)
    ## Alpha (not exactly what should be done: averaging alphas)
    output_path = get_output_name(output_path_, parameters['language'], subject, parameters['model_name'], 'alpha')
    create_maps(masker, maps['alpha'], output_path, vmax=None, logger=logs)
    create_maps(smoothed_masker, maps['alpha'], output_path, vmax=None, logger=logs)
    logs.validate()
    
    print("Model: {} for subject: {} --> Done".format(parameters['model_name'], subject))
