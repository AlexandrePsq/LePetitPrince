import os
import yaml
import argparse


from utilities import check_folder, read_yaml, write, get_subject_name
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
    
    encoding_model_test = EncodingModel()
    encoding_model_valid = EncodingModel()
    transform_data = Transformer()
    compressor = Compressor()
    splitter_cv_int = Splitter()
    splitter_cv_ext = Splitter()
    
    deep_representations_paths, fMRI_paths = fetch_data(parameters, input)
    deep_representations = transform_data.process_representations(deep_representations_paths, parameters['models'])
    fMRI_data = transform_data.process_fmri_data(fMRI_paths)
    
    pipeline = Pipeline()
    pipeline.fit([("splitter_cv_ext", splitter_cv_ext),
                    ("splitter_cv_int", splitter_cv_int),
                    ("compressor", compressor),
                    ("transform_data", transform_data),
                    ("encoding_model_valid", encoding_model_valid),
                    ("encoding_model_test", encoding_model_test)
                ])
    pipeline.compute(deep_representations, fMRI_data, output_path, logs=logs)
    
    print("Model: {} for subject: {} --> Done".format(parameters['model_name'], subject))
