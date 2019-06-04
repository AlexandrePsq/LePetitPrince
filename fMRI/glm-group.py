############################################
# Second level analysis 
# 
#
############################################











if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Objective:\nGenerate design matrices from features in a given language.\n\nInput:\nLanguage and models.""")
    parser.add_argument("--test", type=bool, default=False, action='store_true', help="Precise if we are running a test.")
    parser.add_argument("--language", type=str, default='english', help="Language of the text and the model.")
    parser.add_argument("--models", nargs='+', action='append', default=[], help="Name of the models to use to generate the raw features.")
    parser.add_argument("--overwrite", type=bool, default=False, action='store_true', help="Precise if we overwrite existing files")
    parser.add_argument("--parallel", type=bool, default=True, help="Precise if we run the code in parallel")

    args = parser.parse_args()
    source = 'fMRI'
    input_data_type = 'design-matrices'
    output_data_type = 'glm-indiv'

    for model_ in args.models[0]:
        subjects = subjects_list.get_all(args.language, args.test)
        dm = get_data(args.language, input_data_type, model=model_, source='fMRI', test=args.test)
        fmri_runs = {subject: get_data(args.language, data_type=source, test=args.test, subject=subject) for subject in subjects}

        output_parent_folder = get_output_parent_folder(source, output_data_type, args.language, model_)
        check_folder(output_parent_folder) # check if the output_parent_folder exists and create it if not

        matrices = [transform_design_matrices(run) for run in dm] # list of design matrices (dataframes) where we added a constant column equal to 1
        masker = compute_global_masker(list(fmri_runs.values()))  # return a MultiNiftiMasker object ... computation is sloow

        if args.parallel:
                Parallel(n_jobs=-1)(delayed(do_single_subject)(sub, fmri_runs[sub], matrices, masker, output_parent_folder) for sub in subjects)
            else:
                for sub in subjects:
                    print(f'Processing subject {}...'.format(sub))
                    do_single_subject(sub, fmri_runs[sub], matrices, masker, output_parent_folder)
