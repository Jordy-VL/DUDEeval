import argparse
import json
import logging
import os

import evaluate as HF_evaluate
import numpy as np
from munkres import Munkres, make_cost_matrix

question_ids_to_exclude = []

answer_types = {
    "abstractive": "Abstractive",
    "extractive": "Extractive",
    "not-answerable": "Not Answerable",
    #"list/abstractive": "Abstractive List",
    #"list/extractive": "Extractive List"
}


def save_json(file_path, data):
    with open(file_path, "w+") as json_file:
        json.dump(data, json_file)


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def parse_answers(pred_answers):
    if len(pred_answers) == 0:
        logging.warning("Mistaken unanswerable prediction")
        pred_answers = ""

    if isinstance(pred_answers, list):
        if len(pred_answers) > 1:
            logging.warning("Mistaken list prediction, assuming first")
        pred_answers = pred_answers[0]
    return pred_answers


def get_NLS(gt_answers, pred_answers, threshold):
    values = []

    pred_answers = parse_answers(pred_answers)

    for answer in gt_answers:
        # preprocess both the answers - gt and prediction
        gt_answer = " ".join(answer.strip().lower().split())
        det_answer = " ".join(pred_answers.strip().lower().split())

        dist = levenshtein_distance(gt_answer, det_answer)
        length = max(len(answer.upper()), len(pred_answers.upper()))
        values.append(0.0 if length == 0 else float(dist) / float(length))

    question_result = 1 - min(values)

    if question_result < threshold:
        question_result = 0

    return question_result


def get_best_matches_hungarian_munkers(anchor_list, matching_list):

    match_dict = {}
    match_matrix = []
    for anchor_item in anchor_list:
        NLS_dict = {}
        NLS_list = []
        for matching_item in matching_list:
            NLS = get_NLS([anchor_item], matching_item, threshold=0.5)
            NLS_dict[str(matching_item) + " "] = NLS
            NLS_list.append(NLS)

        match_dict[anchor_item] = NLS_dict
        match_matrix.append(NLS_list)

    return match_dict, match_matrix


def get_NLSL(gt_list, pred_list):
    if len(gt_list) < len(pred_list):
        anchor_list, matching_list = gt_list, pred_list

    else:
        anchor_list, matching_list = pred_list, gt_list

    match_dict, cost_matrix = get_best_matches_hungarian_munkers(anchor_list, matching_list)
    num_answers = max(len(set(gt_list)), len(pred_list))

    m = Munkres()
    m_cost_matrix = make_cost_matrix(cost_matrix)
    indexes = m.compute(m_cost_matrix)
    values = [cost_matrix[row][column] for row, column in indexes]
    NLSL = np.sum(values) / num_answers

    return NLSL


def validate_data(gtFilePath, submFilePath):
    """
    Method validate_data: validates that all files in the results folder are correct (have the correct name contents).
                            Validates also that there are no missing files in the folder.
                            If some error detected, the method raises the error
    """

    gtJson = json.load(open(gtFilePath, "rb"))
    submJson = json.load(open(submFilePath, "rb"))

    if "data" not in gtJson:
        raise Exception("The GT file is not valid (no data key)")

    if "dataset_name" not in gtJson:
        raise Exception("The GT file is not valid (no dataset_name key)")

    if gtJson["dataset_name"] != "DUDE Dataset":
        raise Exception("The GT file is not valid dataset_name should be DUDE Dataset")

    if isinstance(submJson, list) is False:
        raise Exception("The Det file is not valid (root item must be an array)")

    if len(submJson) != len(gtJson["data"]):
        raise Exception(
            "The Det file is not valid (invalid number of answers. Expected:"
            + str(len(gtJson["data"]))
            + " Found:"
            + str(len(submJson))
            + ")"
        )

    gtQuestions = sorted([str(r["questionId"]) for r in gtJson["data"]])
    res_id_to_index = {str(r["questionId"]): ix for ix, r in enumerate(submJson)}
    detQuestions = sorted([str(r["questionId"]) for r in submJson])

    if (gtQuestions == detQuestions) is False:
        print(len(gtQuestions), len(detQuestions))
        print(len(set(gtQuestions).intersection(detQuestions)))
        print(gtQuestions[0], detQuestions[0])
        raise Exception("The Det file is not valid. Question IDs must match GT")

    for gtObject in gtJson["data"]:

        try:
            q_id = str(gtObject["questionId"])
            res_ix = res_id_to_index[q_id]

        except:
            raise Exception(
                "The Det file is not valid. Question "
                + str(gtObject["questionId"])
                + " not present"
            )

        else:
            detObject = submJson[res_ix]

            if "answers" not in detObject:
                raise Exception(
                    "Question " + str(gtObject["questionId"]) + " not valid (no answer key)"
                )
    return gtJson, submJson


def evaluate_method(gtJson, submJson, evaluationParams):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """

    show_scores_per_question_type = evaluationParams.answer_types

    res_id_to_index = {str(r["questionId"]): ix for ix, r in enumerate(submJson)}

    perSampleMetrics = {}

    totalANLS = 0
    row = 0

    if show_scores_per_question_type:
        answerTypeTotalANLS = {x: 0 for x in answer_types.keys()}
        answerTypeNumQuestions = {x: 0 for x in answer_types.keys()}

    for gtObject in gtJson["data"]:
        q_id = str(gtObject["questionId"])
        res_ix = res_id_to_index[q_id]
        detObject = submJson[res_ix]

        if q_id in question_ids_to_exclude:
            question_result = 0
            info = "Question EXCLUDED from the result"

        else:
            info = ""
            
            if gtObject["answer_type"] == 'not-answerable': #gracefully deal with not-answerable
                if gtObject["answers"] == []:
                    gtObject["answers"] = ['']
                if detObject["answers"] == []:
                    detObject["answers"] = ['']
            
            if "list" in gtObject["answer_type"]:  
                question_result = get_NLSL(gtObject["answers"], detObject["answers"])

            else:
                question_result = get_NLS(
                    gtObject["answers"], detObject["answers"], evaluationParams.anls_threshold
                )

            totalANLS += question_result

            if show_scores_per_question_type:
                answer_type = gtObject["answer_type"]
                answerTypeTotalANLS[answer_type] += question_result
                answerTypeNumQuestions[answer_type] += 1

        perSampleMetrics[str(gtObject["questionId"])] = {
            "anls": question_result,
            "question": gtObject["question"],
            "gt_answer": gtObject["answers"],
            "answer_prediction": detObject["answers"],
            "answer_confidence": detObject.get("answers_confidence", -1),
            "info": info,
        }
        row = row + 1

    methodMetrics = {
        "anls": 0
        if len(gtJson["data"]) == 0
        else totalANLS / (len(gtJson["data"]) - len(question_ids_to_exclude))
    }

    if evaluationParams.score_calibration:
        # from ANLS, determine exact matches based on ANLS threshold
        # ECE is calculated as a population statistic
        # TODO(Jordy): hashing-based implementation for type-based calculation
        y_correct, p_answers = [], []
        for q in perSampleMetrics:
            m = perSampleMetrics[q]
            y_correct.append(int(m["anls"] >= evaluationParams.anls_threshold))
            confidence = m["answer_confidence"]
            if isinstance(confidence, list):
                if len(confidence) > 1:
                    logging.warning("Mistaken list confidences, assuming first")
                confidence = confidence[0]
            if confidence == -1:  # invalid so cannot evaluate ECE
                break
            p_answers.append(confidence)

        if len(y_correct) == len(perSampleMetrics):  # checks all calculations valid
            y_correct = [0 if x == 1 else 1 for x in y_correct] #since ECE expects class size vectors [argmax in 1D]
            y_correct = np.array(y_correct).astype(int)
            p_answers = np.array(p_answers).astype(np.float32)

            metric = HF_evaluate.load("jordyvl/ece")
            kwargs = dict(
                n_bins=min(len(perSampleMetrics)-1, 100),
                scheme="equal-mass" if len(set(p_answers)) != 1 else "equal-range",
                bin_range=[0,1],
                proxy="upper-edge",
                p=1,
                detail=False,
            )

            ece_result = metric.compute(
                references=y_correct, predictions=np.expand_dims(p_answers, -1), **kwargs
            )
            methodMetrics.update(ece_result)

    answer_types_ANLS = {}

    if show_scores_per_question_type:
        for answer_type, answer_type_str in answer_types.items():
            answer_types_ANLS[answer_type_str] = (
                0
                if len(gtJson["data"]) == 0
                else answerTypeTotalANLS[answer_type] / (answerTypeNumQuestions[answer_type])
            )

    resDict = {
        "result": methodMetrics,
        "scores_by_types": {"anls_per_answer_type": answer_types_ANLS},
        "per_sample_result": perSampleMetrics,
    }
    return resDict


def display_results(results, show_answer_page_position):
    print("\nOverall ANLS: {:1.4f}\n".format(results["result"]["anls"]))
    if "ECE" in results["result"]:
        print("\nOverall ECE: {:1.4f}\n".format(results["result"]["ECE"]))

    if show_answer_page_position:
        print("Answer type \t ANLS")
        for answer_type in answer_types.values():
            print(
                "{:10s}\t{:1.4f}".format(
                    answer_type, results["scores_by_types"]["anls_per_answer_type"][answer_type]
                )
            )

    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DUDE evaluation script.")

    parser.add_argument(
        "-g", "--ground_truth", type=str, help="Path of the Ground Truth file.", required=True
    )
    parser.add_argument(
        "-s",
        "--submission_file",
        type=str,
        help="Path of your method's results file.",
        required=True,
    )

    parser.add_argument(
        "-t",
        "--anls_threshold",
        type=float,
        default=0.5,
        help="ANLS threshold to use (See Scene-Text VQA paper for more info.).",
        required=False,
    )
    parser.add_argument(
        "-a",
        "--answer_types",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to a directory where to copy the file 'results.json' that contains per-sample results.",
        required=False,
    )
    parser.add_argument(
        "-c",
        "--score_calibration",
        default=False,
        action="store_true",
        required=False,
    )
    args = parser.parse_args()

    # Validate the format of ground truth and submission files.
    gtJson, submJson = validate_data(args.ground_truth, args.submission_file)

    # Evaluate method
    results = evaluate_method(gtJson, submJson, args)

    display_results(results, args.answer_types)

    if args.output:
        output_dir = args.output

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        resultsOutputname = os.path.join(output_dir, "results.json")
        save_json(resultsOutputname, results)

        print("All results including per-sample result has been correctly saved!")
