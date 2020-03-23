import os
import yaml
import argparse


from utilities import check_folder, read_yaml, write, get_subject_name
from task import Task
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
    log_file = args.logs
    input_path = args.input
    output_path = args.output
    subject = get_subject_name(parameters['subject'])
    
    check_folder(os.path.join(output_path, subject))
    transformer = Transformer()
    encoding_model = EncodingModel()
    
    # Pipeline flow
    splitter_cv_ext = Task([Splitter().split], name='splitter_cv_ext')
    ## Pipeline int
    splitter_cv_int = Task([Splitter().split], [splitter_cv_ext], name='splitter_cv_int')
    compressor_int = Task([Compressor().pca], [splitter_cv_int], name='compressor_int') # define here the compression method wanted
    transform_data_int = Task([transformer.standardize, transformer.make_regressor], [compressor_int], name='transform_data_int') # functions in Task objects are read right to left
    encoding_model_int = Task([encoding_model.predict, encoding_model.fit], [transform_data_int], name='encoding_model_int')
    ## Pipeline ext
    compressor_ext = Task([Compressor().pca], [splitter_cv_ext, encoding_model_int], name='compressor_ext')
    transform_data_ext = Task([transformer.standardize, transformer.make_regressor], [compressor_ext], name='transform_data_ext')
    encoding_model_ext = Task([encoding_model.predict, encoding_model.fit], [transform_data_ext], name='encoding_model_ext')
    
    # Creating tree structure
    splitter_cv_ext = splitter_cv_ext.set_children([splitter_cv_int, compressor_ext])
    splitter_cv_int = splitter_cv_int.set_children([compressor_int])
    compressor_int = compressor_int.set_children([transform_data_int])
    transform_data_int = transform_data_int.set_children([encoding_model_int])
    encoding_model_int = encoding_model_int.set_children([compressor_ext])
    compressor_ext = compressor_ext.set_children([transform_data_ext])
    transform_data_ext = transform_data_ext.set_children([encoding_model_ext])    
    
    deep_representations_paths, fMRI_paths = fetch_data(parameters, input)
    deep_representations = transformer.process_representations(deep_representations_paths, parameters['models'])
    fMRI_data = transformer.process_fmri_data(fMRI_paths)
    
    pipeline = Pipeline()
    pipeline.fit([splitter_cv_ext]) # retrieve the flow from children and parents dependencies
    pipeline.compute(deep_representations, fMRI_data, output_path, logs=logs)
    
    print("Model: {} for subject: {} --> Done".format(parameters['model_name'], subject))
