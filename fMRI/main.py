import os
import yaml
import argparse


from utilities import check_folder, read_yaml, write, get_subject_name
from regression_pipeline import Pipeline

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
    
    deep_representations, fMRI_data = fetch_data(parameters, input)
    
    pipeline = Pipeline()
    pipeline.fit(parameters)
    pipeline.compute(deep_representations, fMRI_data, output_path, logs)
    
    print("Model: {} for subject: {} --> Done".format(parameters['model_name'], subject))
