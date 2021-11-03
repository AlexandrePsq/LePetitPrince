# -*- coding: utf-8 -*-
import os
import yaml
import argparse
import numpy as np

from utils import check_folder, read_yaml, save_yaml, write, get_subject_name, get_output_name, aggregate_cv, create_maps, fetch_masker, fetch_data, get_nscans
from utils import get_splitter_information, get_compression_information, get_data_transformation_information, get_estimator_model_information
from task import Task
from logger import Logger
from regression_pipeline import Pipeline
from estimator_models import EstimatorModel
from splitter import Splitter
from data_transformation import Transformer
from data_compression import Compressor



if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="""Fast version of the main script that compute the R2 maps for a given subject and model, without nested cross-validation.""")
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
    kwargs = {
        'detrend': parameters['detrend'],
        'standardize': parameters['standardize'],
        'high_pass': parameters['high_pass'],
        'low_pass': parameters['low_pass'],
        'mask_strategy': parameters['mask_strategy'],
        #'dtype': parameters['dtype'],
        'memory_level': parameters['memory_level'],
        'smoothing_fwhm': parameters['smoothing_fwhm'],
        'verbose': parameters['verbose'],
        't_r': parameters['tr']
        }
    masker = fetch_masker(
        parameters['masker_path'], 
        parameters['language'], 
        parameters['path_to_fmridata'], 
        input_path, 
        logger=logs,
        **kwargs)
    logs.validate()

    logs.info("Retrieve arguments for each model...")
    kwargs_splitter = get_splitter_information(parameters)
    kwargs_compression = get_compression_information(parameters)
    kwargs_transformation = get_data_transformation_information(parameters)
    kwargs_estimator_model = get_estimator_model_information(parameters)
    logs.validate()

    logs.info("Instanciations of the classes...")
    splitter = Splitter(**kwargs_splitter)
    compressor = Compressor(**kwargs_compression)
    transformer = Transformer(**kwargs_transformation)
    estimator_model = EstimatorModel(**kwargs_estimator_model)
    logs.validate()

    logs.info("Defining Pipeline flow...")
    ## Pipeline
    splitter_cv_external = Task([splitter.split], 
                                name='splitter_cv_external'
                               )
    compressor_external = Task([compressor.compress], 
                                input_dependencies=[splitter_cv_external], 
                                name='compressor_external', 
                                flatten_inputs=[True])
    transform_data_external = Task([transformer.make_regressor, transformer.scale], 
                                input_dependencies=[splitter_cv_external, compressor_external], 
                                name='transform_data_external', 
                                flatten_inputs=[True, False],
                                parallel=parameters['parallel'])
    estimator_model_external = Task([estimator_model.evaluate], 
                                input_dependencies=[splitter_cv_external, transform_data_external], 
                                name='estimator_model_external', 
                                flatten_inputs=[True, False],
                                parallel=parameters['parallel'])
    
    # Creating tree structure (for output/input flow)
    splitter_cv_external.set_children_tasks([compressor_external])
    compressor_external.set_children_tasks([transform_data_external])
    transform_data_external.set_children_tasks([estimator_model_external])
    logs.validate()

    try:
        logs.info("Fetching and preprocessing input data...")
        stimuli_representations_paths, fMRI_paths = fetch_data(parameters['path_to_fmridata'], input_path, 
                                                                subject, parameters['language'], parameters['models'])
        stimuli_representations = transformer.process_representations(stimuli_representations_paths, parameters['models'])
        fMRI_data = transformer.process_fmri_data(fMRI_paths, masker, parameters['add_noise_to_constant'])
        logs.validate()
        
        logs.info("Executing pipeline...", end='\n')
        pipeline = Pipeline()
        pipeline.fit(splitter_cv_external, logs) # retrieve the flow from children and input_dependencies
        maps = pipeline.compute(stimuli_representations, fMRI_data, output_path, logger=logs)

        logs.info("Aggregating over cross-validation results...")
        maps_aggregated = {key: np.mean(np.stack(np.array([dic[key] for dic in maps]), axis=0), axis=0) for key in maps[0]}
        logs.validate()

        logs.info("Saving model weights...")
        for key in maps_aggregated.keys():
            output_path = get_output_name(output_path_, parameters['language'], subject, parameters['model_name'], key)
            np.save(output_path + '.npy', maps_aggregated[key])
        logs.validate()
        
        if parameters['save_all_weights']:
            logs.info("Saving non-aggregated weights...")
            for key in maps[0].keys():
                output_path = get_output_name(output_path_, parameters['language'], subject, parameters['model_name'], key)
                for run in range(1, len(maps) + 1):
                    np.save(output_path + '_run{}.npy'.format(run), maps[run-1][key])
            logs.validate()
                  
        
        logs.info("Plotting...", end='\n')
        kwargs = {
            'detrend': False,
            'standardize': False,
            'high_pass': None,
            'low_pass': None,
            'mask_strategy': parameters['mask_strategy'],
            #'dtype': parameters['dtype'],
            'memory_level': parameters['memory_level'],
            'smoothing_fwhm': None,
            'verbose': parameters['verbose'],
            't_r': None
        }
        masker.set_params(**kwargs)
        ## R2
        output_path = get_output_name(output_path_, parameters['language'], subject, parameters['model_name'], 'R2')
        create_maps(masker, maps_aggregated['R2'], output_path, vmax=None, logger=logs, distribution_min=-10, distribution_max=1)
        ## Pearson
        output_path = get_output_name(output_path_, parameters['language'], subject, parameters['model_name'], 'Pearson_coeff')
        create_maps(masker, maps_aggregated['Pearson_coeff'], output_path, vmax=None, logger=logs, distribution_min=-10, distribution_max=1)
        ## Alpha (not exactly what should be done: averaging alphas)
        output_path = get_output_name(output_path_, parameters['language'], subject, parameters['model_name'], 'alpha')
        create_maps(masker, maps_aggregated['alpha'], output_path, vmax=None, logger=logs)
        logs.validate()
    except Exception as err:
        logs.error(str(err))
    
    print("Model: {} for subject: {} --> Done".format(parameters['model_name'], subject))
