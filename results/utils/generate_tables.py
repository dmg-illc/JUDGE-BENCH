import json
import pandas as pd
from results.utils.name_mappings import dataset_names, candidate_models, all_models, models_short_names, task_cabapility_mappings, tasks_dataset_names, dataset_human_vs_machines, subtask_to_cat

def get_tasks_sd_krippendorff_alpha(selected_df_all_results):
   """
   Get the tasks, standard deviation of the values and krippendorff alpha
   """
   num_sub_tasks_categorical = (
      selected_df_all_results[selected_df_all_results["type"] == "categorical"]
      .groupby(["dataset", "model"])
      .count()["subtask"]
      .to_frame()
      .reset_index()[["dataset", "subtask"]]
      .groupby(["dataset", "subtask"])
      .count()
      .rename(columns={"subtask": "num_subtasks"})
      .reset_index()
   )
   num_sub_tasks_non_categorical = (
      selected_df_all_results[selected_df_all_results["type"] != "categorical"]
      .groupby(["dataset", "model"])
      .count()["subtask"]
      .to_frame()
      .reset_index()[["dataset", "subtask"]]
      .groupby(["dataset", "subtask"])
      .count()
      .rename(columns={"subtask": "num_subtasks"})
      .reset_index()
   )

   avereage_variance = (
      selected_df_all_results.groupby(["dataset", "subtask"])["value"]
      .var()
      .reset_index()
      .rename(columns={"value": "average variance"})
      .groupby("dataset")["average variance"]
      .mean()
      .reset_index()
      .round(2)
   )
   avereage_std = (
      selected_df_all_results.groupby(["dataset", "subtask"])["value"]
      .std()
      .reset_index()
      .rename(columns={"value": "average std"})
      .groupby("dataset")["average std"]
      .mean()
      .reset_index()
      .round(2)
   )
   # get krippendorff alpha
   krippendorff_alpha = (
      selected_df_all_results.groupby(["dataset", "subtask"])["krippendorff_alpha"]
      .mean()
      .reset_index()
      .groupby("dataset")["krippendorff_alpha"]
      .mean()
      .reset_index()
      .round(2)
   )
   krippendorff_alpha = krippendorff_alpha.fillna("-")
   krippendorff_alpha_std = (
      selected_df_all_results.groupby(["dataset", "subtask"])["krippendorff_alpha"]
      .std()
      .reset_index()
      .groupby("dataset")["krippendorff_alpha"]
      .std()
      .reset_index()
      .round(2)
   )
   krippendorff_alpha_std.rename(
      columns={"krippendorff_alpha": "krippendorff_alpha_std"}, inplace=True
   )
   dataset_task = (
      selected_df_all_results.groupby(["dataset", "subtask"])["task"]
      .first()
      .reset_index()
      .groupby("dataset")["task"]
      .first()
      .reset_index()
   )
   # replace Toxicity \ Safety with Toxicity \& Safety
   dataset_task["task"] = dataset_task["task"].replace(
      {"Toxicity Safety": "Toxicity \& Safety"}
   )
   return num_sub_tasks_categorical, num_sub_tasks_non_categorical, avereage_variance, avereage_std, krippendorff_alpha, krippendorff_alpha_std, dataset_task

def get_model_per_dataset_score(selected_df_all_results, type="categorical"):
   """
   Get the model per dataset score
   """
   num_sub_tasks_categorical, num_sub_tasks_non_categorical, avereage_variance, avereage_std, krippendorff_alpha, krippendorff_alpha_std, dataset_task = get_tasks_sd_krippendorff_alpha(selected_df_all_results)
   # round the std to 2 decimal places
   avereage_std = avereage_std.round(2)
   if type == "categorical":
      selected_df_all_results = selected_df_all_results[
         selected_df_all_results["type"] == type
      ]
   else:
      selected_df_all_results = selected_df_all_results[
         selected_df_all_results["type"] != "categorical"
      ]
   models_per_dataset_score = (
      selected_df_all_results.groupby(["dataset", "model"])[["value","p"]]
      .mean()
      .reset_index()
      .rename(columns={"value": "average performance"})
      .round(2)
   )


   # Step 1: Identify Categorical Columns
   categorical_columns = models_per_dataset_score.select_dtypes(
      include=["category"]
   ).columns

   # Step 2: Add `0` as a Category
   for col in categorical_columns:
      if 0 not in models_per_dataset_score[col].cat.categories:
         models_per_dataset_score[col] = models_per_dataset_score[
               col
         ].cat.add_categories([0])

   # Step 3: Merge and Fill NaNs
   std_df = (
      selected_df_all_results.groupby(["dataset", "model"])["value"]
      .std()
      .reset_index()
      .rename(columns={"value": "std"})
      .round(2)
   )
   models_per_dataset_score = models_per_dataset_score.merge(
      std_df, on=["dataset", "model"]
   )
   models_per_dataset_score.fillna(0)

   models_per_dataset_score = models_per_dataset_score.pivot(
      index="dataset", columns="model", values=["average performance", "std","p"]
   ).reset_index()
   
   models_per_dataset_score = models_per_dataset_score.swaplevel(1, 0, axis=1)
   models_per_dataset_score = models_per_dataset_score.sort_index(axis=1, level=0)
   models_per_dataset_score.columns = [
      "_".join(col).strip() for col in models_per_dataset_score.columns.values
   ]
   models_per_dataset_score.rename(columns={"_dataset": "dataset"}, inplace=True)

   if type == "categorical":
      models_per_dataset_score = models_per_dataset_score.merge(
         num_sub_tasks_categorical, on="dataset"
      )
      models_per_dataset_score["Type"] = "Categorical"
   else:
      models_per_dataset_score = models_per_dataset_score.merge(
         num_sub_tasks_non_categorical, on="dataset"
      )
      models_per_dataset_score["Type"] = "Graded"
   models_per_dataset_score = models_per_dataset_score.merge(
      avereage_std, on="dataset"
   )

   #models_per_dataset_score = models_per_dataset_score.round(2)
   models_per_dataset_score = models_per_dataset_score.merge(
      krippendorff_alpha, on="dataset"
   )
   models_per_dataset_score = models_per_dataset_score.fillna("")
   models_per_dataset_score = models_per_dataset_score.merge(
      dataset_task, on="dataset"
   )
   # sort rows by task
   models_per_dataset_score = models_per_dataset_score.sort_values(by="task")

   for model in candidate_models:
      models_per_dataset_score[model] = (
         models_per_dataset_score[f"{model}_average performance"].astype(str)
         + " ("
         + models_per_dataset_score[f"{model}_std"].astype(str)
         + ")"
      )
      models_per_dataset_score.drop(
         columns=[f"{model}_average performance", f"{model}_std"], inplace=True
      )
      models_per_dataset_score[model] = models_per_dataset_score[model].apply(
         lambda x: x.split(" ")[0] if x.split(" ")[1] == "(0.0)" else x
      )
      models_per_dataset_score[model] = models_per_dataset_score[model].apply(
         lambda x: x.replace("(", "±").replace(")", "")
      )

      models_per_dataset_score[model] = models_per_dataset_score.apply(
         lambda x: x[model].replace("±", "") if x["subtask"] == 1 else x[model],
         axis=1,
      )
   models_per_dataset_score = models_per_dataset_score.round(2)
   models_per_dataset_score = models_per_dataset_score.astype(str)
   models_per_dataset_score = models_per_dataset_score.rename(
      columns=models_short_names
   )
   models_per_dataset_score["Human"] = models_per_dataset_score["dataset"].apply(
      lambda x: dataset_human_vs_machines[x]
   )
   models_per_dataset_score["dataset"] = models_per_dataset_score["dataset"].apply(
      lambda x: x.replace("-acceptability", "")
   )

   models_per_dataset_score = models_per_dataset_score.rename(
      columns={
         "subtask": "\#Subtasks",
         "average std": "$\sigma$",
         "dataset": "Dataset",
         "task": "Task",
         "krippendorff_alpha": "$\\alpha$",
      }
   )
   # combine the datasets and #subtasks
   models_per_dataset_score["Dataset (#Subtasks)"] = (
      models_per_dataset_score["Dataset"]
      + " ("
      + models_per_dataset_score["\#Subtasks"]
      + ")"
   )
   # drop the columns
   models_per_dataset_score.drop(columns=["Dataset", "\#Subtasks"], inplace=True)
   # add column to indicate cabapility
   models_per_dataset_score["Type"] = models_per_dataset_score["Type"].replace(
      {"Categorical": "Cat."}
   )
   models_per_dataset_score["Type"] = models_per_dataset_score["Type"].replace(
      {"Graded": "Grad."}
   )
   models_per_dataset_score = models_per_dataset_score[
      [
         "Dataset (#Subtasks)",
         "Human",
         "Type",
         "GPT-4o",
         "Llama-3-70B",
         "Mixtral-8x22B",
         "Gemini-1.5",
         "Mixtral-8x7B",
         "Comm-R+",
         "$\sigma$",
         "$\\alpha$",
      ]
   ]
   # sort models by cabapability and task
   models_per_dataset_score = models_per_dataset_score.sort_values(by=["Human"])
   models_per_dataset_score["Dataset (#Subtasks)"] = models_per_dataset_score.apply(
      lambda x: f'\\cellcolor{{blue!25}}{x["Dataset (#Subtasks)"]}'
      if x["Human"] == "human"
      else f'\\cellcolor{{red!25}}{x["Dataset (#Subtasks)"]}',
      axis=1,
   )
   # remove Cabap. column
   models_per_dataset_score.drop(columns=["Human"], inplace=True)
   # replace Type with empty string
   models_per_dataset_score["Type"] = ""
   # move it as the first column
   models_per_dataset_score = models_per_dataset_score[
      [
         "Type",
         "Dataset (#Subtasks)",
         "GPT-4o",
         "Llama-3-70B",
         "Mixtral-8x22B",
         "Gemini-1.5",
         "Mixtral-8x7B",
         "Comm-R+",
         "$\sigma$",
         "$\\alpha$",
      ]
   ]
   # reset
   models_per_dataset_score.reset_index(drop=True, inplace=True)
   return models_per_dataset_score


def get_all_models_per_dataset_score(df_all_results, type="categorical"):
   models = [model for model in all_models if not "gpt-3.5" in model]
   num_sub_tasks_categorical, num_sub_tasks_non_categorical, avereage_variance, avereage_std, krippendorff_alpha, krippendorff_alpha_std, dataset_task = get_tasks_sd_krippendorff_alpha(df_all_results)

   if type == "categorical":
      df_all_results = df_all_results[
         df_all_results["type"] == type
      ]
   else:
      df_all_results = df_all_results[
         df_all_results["type"] != "categorical"
      ]
   models_per_dataset_score = (
      df_all_results.groupby(["dataset", "model"])["value"]
      .mean()
      .reset_index()
      .rename(columns={"value": "average performance"})
      .round(2)
   )
   # Step 1: Identify Categorical Columns
   categorical_columns = models_per_dataset_score.select_dtypes(
      include=["category"]
   ).columns

   # Step 2: Add `0` as a Category
   for col in categorical_columns:
      if 0 not in models_per_dataset_score[col].cat.categories:
         models_per_dataset_score[col] = models_per_dataset_score[
               col
         ].cat.add_categories([0])

   # Step 3: Merge and Fill NaNs
   std_df = (
      df_all_results.groupby(["dataset", "model"])["value"]
      .std()
      .reset_index()
      .rename(columns={"value": "std"})
   )
   models_per_dataset_score = models_per_dataset_score.merge(
      std_df, on=["dataset", "model"]
   )
   models_per_dataset_score.fillna(0)

   models_per_dataset_score = models_per_dataset_score.pivot(
      index="dataset", columns="model", values=["average performance", "std"]
   ).reset_index()
   models_per_dataset_score = models_per_dataset_score.swaplevel(1, 0, axis=1)
   models_per_dataset_score = models_per_dataset_score.sort_index(axis=1, level=0)
   models_per_dataset_score.columns = [
      "_".join(col).strip() for col in models_per_dataset_score.columns.values
   ]
   models_per_dataset_score.rename(columns={"_dataset": "dataset"}, inplace=True)
   if type == "categorical":
      models_per_dataset_score = models_per_dataset_score.merge(
         num_sub_tasks_categorical, on="dataset"
      )
      models_per_dataset_score["Type"] = "Categorical"
   else:
      models_per_dataset_score = models_per_dataset_score.merge(
         num_sub_tasks_non_categorical, on="dataset"
      )
      models_per_dataset_score["Type"] = "Graded"
   models_per_dataset_score = models_per_dataset_score.merge(
      avereage_std, on="dataset"
   )

   models_per_dataset_score = models_per_dataset_score.round(2)
   models_per_dataset_score = models_per_dataset_score.merge(
      krippendorff_alpha, on="dataset"
   )
   models_per_dataset_score = models_per_dataset_score.fillna("")
   models_per_dataset_score = models_per_dataset_score.merge(
      dataset_task, on="dataset"
   )
   # sort rows by task
   models_per_dataset_score = models_per_dataset_score.sort_values(by="task")
   for model in models:
      models_per_dataset_score[model] = (
         models_per_dataset_score[f"{model}_average performance"].astype(str)
         + " ("
         + models_per_dataset_score[f"{model}_std"].astype(str)
         + ")"
      )
      models_per_dataset_score.drop(
         columns=[f"{model}_average performance", f"{model}_std"], inplace=True
      )
      models_per_dataset_score[model] = models_per_dataset_score[model].apply(
         lambda x: x.split(" ")[0] if x.split(" ")[1] == "(0.0)" else x
      )
      models_per_dataset_score[model] = models_per_dataset_score[model].apply(
         lambda x: x.replace("(", "±").replace(")", "")
      )
      # remove spaces
      # models_per_dataset_score[model] = models_per_dataset_score[model].apply(lambda x: x.replace(' ', ''))
      # if number of subtasks is 1 then remove the ±
      models_per_dataset_score[model] = models_per_dataset_score.apply(
         lambda x: x[model].replace("±", "") if x["subtask"] == 1 else x[model],
         axis=1,
      )
   models_per_dataset_score = models_per_dataset_score.round(2)
   models_per_dataset_score = models_per_dataset_score.astype(str)
   models_per_dataset_score = models_per_dataset_score.rename(
      columns=models_short_names
   )
   models_per_dataset_score["Human"] = models_per_dataset_score["dataset"].apply(
      lambda x: dataset_human_vs_machines[x]
   )
   models_per_dataset_score["dataset"] = models_per_dataset_score["dataset"].apply(
      lambda x: x.replace("-acceptability", "")
   )

   models_per_dataset_score = models_per_dataset_score.rename(
      columns={
         "subtask": "\#Subtasks",
         "average std": "$\sigma$",
         "dataset": "Dataset",
         "task": "Task",
         "krippendorff_alpha": "$\\alpha$",
      }
   )
   # combine the datasets and #subtasks
   models_per_dataset_score["Dataset (#Subtasks)"] = (
      models_per_dataset_score["Dataset"]
      + " ("
      + models_per_dataset_score["\#Subtasks"]
      + ")"
   )
   # drop the columns
   models_per_dataset_score.drop(columns=["Dataset", "\#Subtasks"], inplace=True)
   # add column to indicate cabapility
   models_per_dataset_score["Type"] = models_per_dataset_score["Type"].replace(
      {"Categorical": "Cat."}
   )
   models_per_dataset_score["Type"] = models_per_dataset_score["Type"].replace(
      {"Graded": "Grad."}
   )
   models_per_dataset_score = models_per_dataset_score[
      [
         "Dataset (#Subtasks)",
         "Human",
         "Type",
         "GPT-4o",
         "Llama-3-70B",
         "Mixtral-8x22B",
         "Gemini-1.5",
         "Mixtral-8x7B",
         "Comm-R+",
         "Comm-R4",
         "Llama-3-8B",
         "Mistral-7B",
         "Starling-7B",
         "OLMo-7B",
      ]
   ]
   # sort models by cabapability and task
   models_per_dataset_score = models_per_dataset_score.sort_values(by=["Human"])
   models_per_dataset_score["Dataset (#Subtasks)"] = models_per_dataset_score.apply(
      lambda x: f'\\cellcolor{{blue!25}}{x["Dataset (#Subtasks)"]}'
      if x["Human"] == "human"
      else f'\\cellcolor{{red!25}}{x["Dataset (#Subtasks)"]}',
      axis=1,
   )
   models_per_dataset_score.drop(columns=["Human"], inplace=True)
   models_per_dataset_score["Type"] = ""
   models_per_dataset_score = models_per_dataset_score[
      [
         "Type",
         "Dataset (#Subtasks)",
         "GPT-4o",
         "Llama-3-70B",
         "Mixtral-8x22B",
         "Gemini-1.5",
         "Mixtral-8x7B",
         "Comm-R+",
         "Comm-R4",
         "Llama-3-8B",
         "Mistral-7B",
         "Starling-7B",
         "OLMo-7B",
      ]
   ]
   # reset
   models_per_dataset_score.reset_index(drop=True, inplace=True)
   return models_per_dataset_score