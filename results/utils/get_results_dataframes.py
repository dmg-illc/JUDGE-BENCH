import json
import pandas as pd
from results.utils.name_mappings import (
    dataset_names,
    candidate_models,
    all_models,
    models_short_names,
    task_cabapility_mappings,
    tasks_dataset_names,
    dataset_human_vs_machines,
    subtask_to_cat,
)

import seaborn as sns


def highlight_col_max(col):
    max_v_idx, max_v = None, -1
    for idx, v in enumerate(col):
        v = float(v.split(" ")[0])
        max_v_idx, max_v = (
            (idx, v) if not pd.isna(v) and v > max_v else (max_v_idx, max_v)
        )
    return [
        "background-color: lightgreen" if idx == max_v_idx else ""
        for idx in range(len(col))
    ]


def json_to_df(fpath):
    json_d = json.load(open(fpath, "r"))
    dataset_results = []
    for model in json_d.keys():
        for subtask in json_d[model].keys():
            task = json_d[model][subtask]["task"]
            cat_type = json_d[model][subtask]["type"]
            expert = json_d[model][subtask]["expert"]

            json_d[model][subtask]["Spearman r (p-value)"] = "{} ({})".format(
                round(json_d[model][subtask]["corr_coeff"]["spearman"], 4),
                round(json_d[model][subtask]["p_value"]["spearman"], 4),
            )
            json_d[model][subtask]["Kappa score"] = "{}".format(
                round(json_d[model][subtask]["kappa_score"], 4)
            )

            json_d[model][subtask]["Instances considered (Total)"] = (
                "{} ({})".format(
                    json_d[model][subtask]["valid_responses"],
                    json_d[model][subtask]["total_responses"],
                )
            )

            krippendorff_alpha = (
                round(json_d[model][subtask]["krippendorff_alpha"], 4)
                if type(json_d[model][subtask]["krippendorff_alpha"]) != str
                else json_d[model][subtask]["krippendorff_alpha"]
            )
            json_d[model][subtask]["krippendorff_alpha"] = "{}".format(
                krippendorff_alpha
            )
            if cat_type == "graded" or cat_type == "continuous":
                value = round(
                    json_d[model][subtask]["corr_coeff"]["spearman"], 4
                )
            else:
                value = round(json_d[model][subtask]["kappa_score"], 4)
            response_ratio = (
                json_d[model][subtask]["valid_responses"]
                / json_d[model][subtask]["total_responses"]
            )
            dataset = fpath.split("/")[-1].split(".")[0]
            dataset_subtask = f"{dataset}_{subtask}"
            model_name = model.split(" (")[0]
            human_vs_machine = dataset_human_vs_machines[dataset]
            open_closed = all_models[model_name]
            json_to_add = {
                "p": 1
                * (json_d[model][subtask]["p_value"]["spearman"] < 0.05),
                "value": value,
                "dataset": dataset,
                "subtask": subtask,
                "task": tasks_dataset_names[dataset],
                "type": cat_type,
                "response_ratio": response_ratio,
                "model": model_name,
                "expert": expert,
                "dataset_subtask": dataset_subtask,
                "krippendorff_alpha": krippendorff_alpha,
                "open_closed": open_closed,
                "capability": task_cabapility_mappings[
                    tasks_dataset_names[dataset]
                ],
                "human_vs_machine": human_vs_machine,
            }
            dataset_results.append(json_to_add)
            for k in [
                "corr_coeff",
                "p_value",
                "total_responses",
                "valid_responses",
                "krippendorff_alpha",
            ]:  # , 'valid_pairs', 'total_pairs']:
                del json_d[model][subtask][k]

    df = pd.DataFrame(json_d).T
    df = df.stack().apply(pd.Series).unstack()
    df.columns = df.columns.swaplevel(0, 1)
    df.sort_index(axis=1, level=0, inplace=True)
    columns = [
        (subtask, "Pearson r (p-value)")
        for model in json_d.keys()
        for subtask in json_d[model].keys()
    ]
    df.style.apply(highlight_col_max, subset=list(set(columns)))
    return df, dataset_results


def plot_human_vs_machine(
    selected_df_all_results, annotation_type="categorical", ax=None
):
    if annotation_type == "categorical":
        filtered_data = selected_df_all_results[
            (selected_df_all_results["type"] == annotation_type)
        ]
    else:
        filtered_data = selected_df_all_results[
            (selected_df_all_results["type"] != "categorical")
        ]
    filtered_data = (
        filtered_data.groupby(["dataset", "model", "human_vs_machine"])[
            "value"
        ]
        .mean()
        .reset_index()
    )
    model_scores = (
        filtered_data.groupby("model")["value"]
        .mean()
        .sort_values(ascending=True)
    )
    filtered_data["model"] = pd.Categorical(
        filtered_data["model"], categories=model_scores.index, ordered=True
    )
    filtered_data = filtered_data.sort_values(by="human_vs_machine")
    sns.set_theme(style="white")
    # replace human with Human and model-generated with Machine-Generated
    filtered_data["human_vs_machine"] = filtered_data[
        "human_vs_machine"
    ].replace({"human": "Human", "model-generated": "Machine-Generated"})
    sns.barplot(
        x="model",
        y="value",
        hue="human_vs_machine",
        data=filtered_data,
        ci=None,
        ax=ax,
        palette="Set2",
    )
    ax.set_xlabel("")
    if annotation_type == "categorical":
        y_label = "Kappa Score"
    else:
        y_label = "Spearman Correlation"
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_ylim(-0.1, 0.7)
    ax.set_title(
        f"{annotation_type.capitalize()} Data", fontsize=12, fontweight="bold"
    )
    # replace the xtikcs with short names
    # get labels from the xticks
    labels = [item.get_text() for item in ax.get_xticklabels()]
    # change the legend title
    ax.legend(title="Data Source", title_fontsize="large")
    ax.set_xticklabels(
        rotation=90,
        ha="right",
        rotation_mode="anchor",
        fontweight="bold",
        labels=[models_short_names[label] for label in labels],
    )
    for t in ax.legend().texts:
        t.set_fontweight("bold")

    return ax, filtered_data


def plot_experts_vs_non_experts(
    selected_df_all_results, annotation_type="categorical", ax=None
):
    if annotation_type == "categorical":
        filtered_data = selected_df_all_results[
            (selected_df_all_results["Annotators"] != "Mixed")
            & (selected_df_all_results["type"] == annotation_type)
        ]
    else:
        filtered_data = selected_df_all_results[
            (selected_df_all_results["Annotators"] != "Mixed")
            & (selected_df_all_results["type"] != "categorical")
        ]
    filtered_data = (
        filtered_data.groupby(["dataset", "model", "Annotators"])["value"]
        .mean()
        .reset_index()
    )
    model_scores = (
        filtered_data.groupby("model")["value"]
        .mean()
        .sort_values(ascending=True)
    )
    filtered_data["model"] = pd.Categorical(
        filtered_data["model"], categories=model_scores.index, ordered=True
    )
    filtered_data = filtered_data.sort_values(by="Annotators")
    sns.set_theme(style="whitegrid")
    # replace Crowdsourced with Non-Experts
    filtered_data["Annotators"] = filtered_data["Annotators"].replace(
        {"Crowdsource": "Non-Experts"}
    )
    sns.barplot(
        x="model",
        y="value",
        hue="Annotators",
        data=filtered_data,
        ci=None,
        ax=ax,
    )
    ax.set_xlabel("")
    if annotation_type == "categorical":
        y_label = "Kappa Score"
    else:
        y_label = "Spearman Correlation"
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_ylim(-0.1, 0.7)
    ax.set_title(f"{annotation_type.capitalize()} Annotations")
    # replace the xtikcs with short names
    # get labels from the xticks
    labels = [item.get_text() for item in ax.get_xticklabels()]
    ax.set_xticklabels(
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontweight="bold",
        labels=[models_short_names[label] for label in labels],
    )

    return ax, filtered_data


def get_model_per_category_score(selected_df_all_results):
    selected_df_all_results = selected_df_all_results[
        selected_df_all_results["type"] != "categorical"
    ]
    models_per_category_score = (
        selected_df_all_results.groupby(["category", "model"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "average performance"})
        .round(4)
    )
    # Step 1: Identify Categorical Columns
    categorical_columns = models_per_category_score.select_dtypes(
        include=["category"]
    ).columns

    # Step 2: Add `0` as a Category
    for col in categorical_columns:
        if 0 not in models_per_category_score[col].cat.categories:
            models_per_category_score[col] = models_per_category_score[
                col
            ].cat.add_categories([0])

    # Step 3: Merge and Fill NaNs
    std_df = (
        selected_df_all_results.groupby(["category", "model"])["value"]
        .std()
        .reset_index()
        .rename(columns={"value": "std"})
    )
    models_per_category_score = models_per_category_score.merge(
        std_df, on=["category", "model"]
    )
    models_per_category_score.fillna(0)

    models_per_category_score = models_per_category_score.pivot(
        index="category",
        columns="model",
        values=["average performance", "std"],
    ).reset_index()
    models_per_category_score = models_per_category_score.swaplevel(
        1, 0, axis=1
    )
    models_per_category_score = models_per_category_score.sort_index(
        axis=1, level=0
    )
    models_per_category_score.columns = [
        "_".join(col).strip()
        for col in models_per_category_score.columns.values
    ]
    models_per_category_score.rename(
        columns={"_dataset": "dataset"}, inplace=True
    )
    models_per_category_score["Type"] = "Graded"

    models_per_category_score = models_per_category_score.round(2)
    models_per_category_score = models_per_category_score.fillna("")

    for model in candidate_models:
        models_per_category_score[model] = (
            models_per_category_score[f"{model}_average performance"].astype(
                str
            )
            + " ("
            + models_per_category_score[f"{model}_std"].astype(str)
            + ")"
        )
        models_per_category_score.drop(
            columns=[f"{model}_average performance", f"{model}_std"],
            inplace=True,
        )
        models_per_category_score[model] = models_per_category_score[
            model
        ].apply(lambda x: x.split(" ")[0] if x.split(" ")[1] == "(0.0)" else x)
        models_per_category_score[model] = models_per_category_score[
            model
        ].apply(lambda x: x.replace("(", "±").replace(")", ""))
        # remove spaces
        # models_per_dataset_score[model] = models_per_dataset_score[model].apply(lambda x: x.replace(' ', ''))
        # if number of subtasks is 1 then remove the ±
        models_per_category_score[model] = models_per_category_score.apply(
            lambda x: (
                x[model].replace("±", "") if x[model][-1] == "±" else x[model]
            ),
            axis=1,
        )
    models_per_category_score = models_per_category_score.round(2)
    models_per_category_score = models_per_category_score.astype(str)
    models_per_category_score = models_per_category_score.rename(
        columns=models_short_names
    )
    models_per_category_score.reset_index(drop=True, inplace=True)
    models_per_category_score = models_per_category_score.rename(
        columns={"_category": "Category"}
    )
    models_per_category_score = models_per_category_score[
        [
            "Category",
            "GPT-4o",
            "Llama-3.1-70B",
            "Mixtral-8x22B",
            "Gemini-1.5",
            "Mixtral-8x7B",
            "Comm-R+",
        ]
    ]
    # reset
    return models_per_category_score
