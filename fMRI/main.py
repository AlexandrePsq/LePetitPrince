import os
import yaml
import argparse
import numpy as np

from sklearn.linear_model import Ridge

from utils import check_folder, read_yaml, write, get_subject_name, output_name, structuring_inputs, aggregate_cv, create_maps, fetch_masker, fetch_data
from task import Task
from logger import Logger
from regression_pipeline import Pipeline
from encoding_models import EncodingModel
from splitter import Splitter
from data_transformation import Transformer
from data_compression import Compressor



if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="""Main script that compute the R2 maps for a given subject and model.""")
    parser.add_argument("--yaml_file", type=str, default="/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/code/fMRI/template.yml", help="Path to the yaml containing the parameters of the script execution.")
    parser.add_argument("--input", type=str, help="Path to the folder containing the representations.")
    parser.add_argument("--output", type=str, help="Path to the output folder.")
    parser.add_argument("--logs", type=str, default=None, help="Path to the log file.")

    args = parser.parse_args()
    parameters = read_yaml(args.yaml_file)
    input_path = args.input
    output_path = args.output
    subject = get_subject_name(parameters['subject'])
    output_path = output_name(output_path, subject, parameters['model_name'])
    logs = Logger(os.path.join(args.logs, '{}_{}.txt'.format(subject, parameters['model_name'])))
    logs.info("Fetching maskers...")
    masker = fetch_masker(parameters['masker_path'], parameters['language'], parameters['path_to_fmridata'], input_path)
    smoothed_masker = fetch_masker(parameters['smoothed_masker_path'], parameters['language'], parameters['path_to_fmridata'], input_path, smoothing_fwhm=5)
    logs.validate()

    logs.info("Structuring inputs for later computation...")
    indexes, new_indexes, offset_type_list, duration_type_list, compression_types, n_components_list = structuring_inputs(parameters['models'], parameters['nb_runs'])
    logs.validate()

    logs.info("Instanciations of the classes...")
    transformer = Transformer(parameters['tr'], parameters['nscans'], new_indexes, offset_type_list, duration_type_list, parameters['offset_path'], parameters['duration_path'], parameters['language'], parameters['hrf'])
    encoding_model = EncodingModel(model=Ridge(), alpha=None, alpha_min_log_scale=2, alpha_max_log_scale=4, nb_alphas=25)
    splitter = Splitter(1)
    compressor = Compressor(n_components_list, indexes, compression_types)
    logs.validate()

    logs.info("Defining Pipeline flow...")
    splitter_cv_ext = Task([splitter.split], name='splitter_cv_ext')
    ## Pipeline int
    splitter_cv_int = Task([splitter.split], [splitter_cv_ext],
                                name='splitter_cv_int')
    compressor_int = Task([compressor.compress], [splitter_cv_int],
                                name='compressor_int', flatten=True) # define here the compression method wanted
    transform_data_int = Task([transformer.standardize, transformer.make_regressor], [splitter_cv_int, compressor_int],
                                name='transform_data_int', flatten=True) # functions in Task objects are read right to left
    encoding_model_int = Task([encoding_model.grid_search], [splitter_cv_int, transform_data_int],
                                name='encoding_model_int', flatten=True,
                                special_output_transform=aggregate_cv)
    ## Pipeline ext
    compressor_ext = Task([compressor.compress], [splitter_cv_ext, encoding_model_int], name='compressor_ext')
    transform_data_ext = Task([transformer.standardize, transformer.make_regressor], [splitter_cv_ext, compressor_ext], name='transform_data_ext')
    encoding_model_ext = Task([encoding_model.evaluate], [splitter_cv_ext, transform_data_ext, encoding_model_int], name='encoding_model_ext')
    
    # Creating tree structure
    splitter_cv_ext.set_children([splitter_cv_int, compressor_ext, encoding_model_ext])
    splitter_cv_int.set_children([compressor_int, transform_data_int, encoding_model_int])
    compressor_int.set_children([transform_data_int])
    transform_data_int.set_children([encoding_model_int])
    encoding_model_int.set_children([compressor_ext, encoding_model_ext])
    compressor_ext.set_children([transform_data_ext])
    transform_data_ext.set_children([encoding_model_ext])
    logs.validate()
    
    logs.info("Formatting input...")
    deep_representations_paths, fMRI_paths = fetch_data(parameters['path_to_fmri_data'], input_path, subject, parameters['language'])
    deep_representations = transformer.process_representations(deep_representations_paths, parameters['models'])
    fMRI_data = transformer.process_fmri_data(fMRI_paths, masker)
    logs.validate()
    
    logs.info("Executing pipeline...")
    pipeline = Pipeline()
    pipeline.fit([splitter_cv_ext], logs) # retrieve the flow from children and parents dependencies
    maps = pipeline.compute(deep_representations, fMRI_data, output_path, logger=logs)
    logs.validate()
    
    logs.info("Aggregating over cross-validation results...")
    maps = {key: np.mean(np.stack(np.array([dic[key] for dic in maps]), axis=0), axis=0) for key in maps[0]}
    logs.validate()
    
    logs.info("Plotting...")
    ## R2
    output_path = os.path.join(output_path, 'R2')
    create_maps(masker, maps['R2'], output_path, vmax=None, logger=logs)
    create_maps(smoothed_masker, maps['R2'], output_path, vmax=None, logger=logs)
    ## Pearson
    output_path = os.path.join(output_path, 'Pearson_coeff')
    create_maps(masker, maps['Pearson_coeff'], output_path, vmax=None, logger=logs)
    create_maps(smoothed_masker, maps['Pearson_coeff'], output_path, vmax=None, logger=logs)
    ## Alpha (not exactly what should be done: averaging alphas)
    output_path = os.path.join(output_path, 'alpha')
    create_maps(masker, maps['alpha'], output_path, vmax=None, logger=logs)
    create_maps(smoothed_masker, maps['alpha'], output_path, vmax=None, logger=logs)
    logs.validate()
    
    print("Model: {} for subject: {} --> Done".format(parameters['model_name'], subject))
