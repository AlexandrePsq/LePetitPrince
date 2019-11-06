import os
import numpy as np





job_final = Job(command=["python", "create_maps.py", 
                                "--input", derivatives_path, 
                                "--parameters", parameters_path,
                                "--subject", args.subject,
                                "--fmri_data", fmri_path], 
                        name="Creating the maps.",
                        working_directory=scripts_path)


# significativity retrieval 
    files_list = sorted(glob.glob(os.path.join(yaml_files_path, 'run_*_alpha_*.yml')))
    job2launch = {}
    for yaml_file in files_list:
        info = os.path.basename(yaml_file).split('_')
        run = int(info[1])
        alpha = float(info[3][:-4])
        with open(yaml_file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except :
                print(-1)
                quit()
        if data['voxels']!=[]:
            if str(run) in job2launch.keys():
                job2launch[str(run)].append(alpha)
            else:
                job2launch[str(run)] = [alpha]
    job2launch_path = os.path.join(yaml_files_path, 'job2launch.yml')
    with open(job2launch_path, 'w') as outfile:
        yaml.dump(job2launch, outfile, default_flow_style=False)

    group_significativity = []
    group_score = []
    group_merge =[]
    base_group = []
    count = 1

    for model in parameters['models']:
        model_name = model['name']
        # temporary buffers:
        jobs_score = []
        jobs_perm = []

        # Merging the results and compute significant r2
        job_merge = Job(command=["python", "optimized_merge_results.py", 
                                    "--input_folder", derivatives_path, 
                                    "--yaml_files", yaml_files_path,
                                    "--nb_runs", nb_runs, 
                                    "--nb_voxels", nb_voxels,
                                    "--n_permutations", nb_permutations, 
                                    "--model_name", model_name, 
                                    "--alpha_percentile", alpha_percentile], 
                        name="Merging results for model: {}.".format(model_name),
                        working_directory=scripts_path)

        for run in job2launch.keys():
            native_specification = "-q Nspin_bigM  -l walltime=24:00:00" # 
            features_indexes = ','.join([str(index) for index in model['indexes']])
            job = Job(command=["python", "optimized_significance_clusterized.py", 
                                "--job2launch", job2launch_path, 
                                "--yaml_files_path", yaml_files_path,
                                "--run", run,
                                "--output", derivatives_path, 
                                "--x", design_matrices_path, 
                                "--y", fmri_path, 
                                "--features_indexes", features_indexes,
                                "--model_name", model_name], 
                        name="job {} - alpha {} - model {}".format(run, alpha, model_name), 
                        working_directory=scripts_path,
                        native_specification=native_specification)
            jobs_score.append(job)
            if count == 1:
                base_group.append(job)
            for alpha in job2launch[run]:
                yaml_file = 'run_{}_alpha_{}.yml'.format(run, alpha)
                job_permutations = Job(command=["python", "optimized_generate_distribution.py", 
                                                "--yaml_file", os.path.join(yaml_files_path, yaml_file), 
                                                "--output", derivatives_path, 
                                                "--x", design_matrices_path, 
                                                "--y", fmri_path, 
                                                "--shuffling", shuffling_path, 
                                                "--n_sample", nb_permutations, 
                                                "--model_name", model_name, 
                                                "--features_indexes", features_indexes], 
                                        name="distribution {} - alpha {} - model {} ".format(run, alpha, model_name), 
                                        working_directory=scripts_path,
                                        native_specification=native_specification)
                jobs_perm.append(job_permutations)
                dependencies.append((job, job_permutations))
                dependencies.append((job_permutations, job_merge))
