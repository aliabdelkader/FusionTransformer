import pandas as pd 
import wandb

api = wandb.Api()
entity, project = "<entity>", "FusionTransformer"  # set to your entity and project 
runs = api.runs(project) 

summary_list, config_list, name_list, group_list, start_times = [], [], [], [], []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    # summary_list.append(run.summary._json_dict)

    # # .config contains the hyperparameters.
    # #  We remove special values that start with _.
    # config_list.append(
    #     {k: v for k,v in run.config.items()
    #      if not k.startswith('_')})

    # .name is the human-readable name of the run.
    group_list.append(run.group)
    name_list.append(run.name)
    try:
        start_times.append(run.start_time)
    except:
        start_times.append(None)
        continue

runs_df = pd.DataFrame({
    # "summary": summary_list,
    # "config": config_list,
    "name": name_list,
    "group": group_list,
    "start_time": start_times
    })

runs_df.to_csv("project.csv")


# save_runs = {
# "LidarSeg": ["MONTH_11_DAY_22_HOUR_02_MIN_43_SEC_46"],
# "EarlyFusionTransformer":["MONTH_12_DAY_02_HOUR_06_MIN_51_SEC_52"],
# "ImageSeg": ["MONTH_11_DAY_17_HOUR_23_MIN_14_SEC_54"],
# "ImageSegBilinear": ["MONTH_11_DAY_25_HOUR_02_MIN_42_SEC_29"],
# "LateFusionTransformer": ["MONTH_12_DAY_02_HOUR_07_MIN_00_SEC_25"],
# "MiddleFusionTransformer": ["MONTH_12_DAY_02_HOUR_06_MIN_56_SEC_17"],
# }

# api = wandb.Api()
# entity, project = "<entity>", "FusionTransformer"  # set to your entity and project 
# runs = api.runs(project) 

# for run in runs: 
#     if run.name not in save_runs[run.group]:
#         print(run.group, run.name, run.id)
#         run.delete()
