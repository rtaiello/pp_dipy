"""
==========================================
Symmetric Diffeomorphic Registration in 3D
==========================================
This example explains how to model 3D volumes using the Symmetric
Normalization (SyN) algorithm proposed by Avants et al. [Avants09]_
(also implemented in the ANTs software [Avants11]_)

We will model two 3D volumes from the same modality using SyN with the Cross
-Correlation (CC) metric.
"""

import os
import time

import numpy as np
import pandas as pd
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
from dipy.io.image import load_nifti, save_nifti


def dice_coef(y_true, y_pred):
    intersect = np.sum(y_true * y_pred)
    fsum = np.sum(y_true)
    ssum = np.sum(y_pred)
    dice = (2 * intersect) / (fsum + ssum)
    dice = np.mean(dice)
    dice = round(dice, 3)  # for easy reading
    return dice


def dice_coef_multilabel(y_true, y_pred, num_labels):
    dice = 0
    y_true_copy = y_true.copy()
    y_pred_copy = y_pred.copy()
    for index in range(0, num_labels):
        y_true = y_true_copy == index
        y_pred = y_pred_copy == index
        dice += dice_coef(y_true, y_pred)
    return dice / (num_labels)


def rmse(displacement_clear, displacement_pp, pixel_size=1):
    """
    Compute the RMSE between two displacement maps.
    :param displacement_clear:
    :param displacement_pp:
    :param pixel_size:
    :return:
    """
    displacement_diff = displacement_clear - displacement_pp
    displacement_diff *= pixel_size
    norm = np.sqrt(
        (displacement_diff[0]) ** 2
        + (displacement_diff[1]) ** 2
        + (displacement_diff[2]) ** 2
    )
    avg = np.mean(norm)
    error = np.sqrt(avg)
    # error *= pixel_size
    return error


def check_and_create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


dataset = "AbdomenMRCT"


def run(id_patient, df):
    # create a row for the dataframe
    row = {}
    row["id_patient"] = id_patient

    patient_static = f"AbdomenMRCT_000{id_patient}_0000"
    patient_moving = f"AbdomenMRCT_000{id_patient}_0001"
    path_static = f"/home/argentera/taiello/pp_dipy/pp_experiments/data/{dataset}/imagesTr/{patient_static}.nii.gz"
    path_moving = f"/home/argentera/taiello/pp_dipy/pp_experiments/data/{dataset}/imagesTr/{patient_moving}.nii.gz"
    path_patient = f"/home/argentera/taiello/pp_dipy/pp_experiments/data/{dataset}/{patient_static}"
    # check if the patient folder exists and create it if not
    check_and_create_folder(path_patient)

    static, static_affine = load_nifti(path_static)
    moving, moving_affine = load_nifti(path_moving)
    level_iters = [10, 10, 5, 1]
    start = time.time()
    metric = CCMetric(3, mpc=False, sigma_diff=2.0)
    end = time.time()
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
    print("Clear computation")
    mapping = sdr.optimize(static, moving, static_affine, moving_affine)
    path_warped = f"{path_patient}/imagesWarped"
    check_and_create_folder(path_warped)
    path_warped_static = f"{path_warped}/{patient_static}.nii.gz"
    path_warped_moving = f"{path_warped}/{patient_moving}.nii.gz"

    warped_moving = mapping.transform(moving)
    save_nifti(path_warped_moving, warped_moving, moving_affine)
    warped_static = mapping.transform_inverse(static)
    save_nifti(path_warped_static, warped_static, static_affine)

    path_label_static = f"/home/argentera/taiello/pp_dipy/pp_experiments/data/{dataset}/labelsTr/{patient_static}.nii.gz"
    path_label_moving = f"/home/argentera/taiello/pp_dipy/pp_experiments/data/{dataset}/labelsTr/{patient_moving}.nii.gz"
    label_static, label_static_affine = load_nifti(path_label_static)
    label_moving, label_moving_affine = load_nifti(path_label_moving)
    num_labels = len(np.unique(label_static)) - 1
    print(f"Dice coefficient: {patient_static} -> {patient_moving}")
    dice = dice_coef_multilabel(label_static, label_moving, num_labels)
    row["dice_static_moving"] = dice
    print(dice)
    print(f"Dice coefficient: {patient_static} -> Warped {patient_moving}")
    label_warped_moving = mapping.transform(label_moving, interpolation="nearest")
    dice = dice_coef_multilabel(label_static, label_warped_moving, num_labels)
    row["dice_static_warped_moving"] = dice
    print(dice)
    print(f"Dice coefficient: {patient_moving} -> {patient_static}")
    dice = dice_coef_multilabel(label_moving, label_static, num_labels)
    row["dice_moving_static"] = dice
    print(dice)
    print(f"Dice coefficient: {patient_moving} -> Warped {patient_static}")
    label_warped_static = mapping.transform_inverse(
        label_static, interpolation="nearest"
    )
    dice = dice_coef_multilabel(label_moving, label_warped_static, num_labels)
    row["dice_moving_warped_static"] = dice
    print(dice)
    path_warped_label = f"{path_patient}/labelsWarped"
    check_and_create_folder(path_warped_label)
    path_warped_label_static = f"{path_warped_label}/{patient_static}.nii.gz"
    path_warped_label_moving = f"{path_warped_label}/{patient_moving}.nii.gz"
    save_nifti(path_warped_label_static, label_warped_static, label_static_affine)
    save_nifti(path_warped_label_moving, label_warped_moving, label_moving_affine)
    row["time_clear"]= end - start
    row["iteration_clear"] = sdr.iter
    metric_mpc = CCMetric(3, mpc=True, sigma_diff=2.0)
    print("MPC computation")
    sdr_mpc = SymmetricDiffeomorphicRegistration(metric_mpc, level_iters)
    start = time.time()
    mapping_mpc = sdr_mpc.optimize(static, moving, static_affine, moving_affine)
    end = time.time()
    row["norm_cc_clear"] = metric.get_energy()

    path_warped_mpc = f"{path_patient}/imagesWarpedMPC"
    check_and_create_folder(path_warped_mpc)
    path_warped_static_mpc = f"{path_warped_mpc}/{patient_static}.nii.gz"
    path_warped_moving_mpc = f"{path_warped_mpc}/{patient_moving}.nii.gz"

    warped_moving_mpc = mapping_mpc.transform(moving)
    save_nifti(path_warped_moving_mpc, warped_moving_mpc, moving_affine)
    warped_static_mpc = mapping_mpc.transform_inverse(static)
    save_nifti(path_warped_static_mpc, warped_static_mpc, static_affine)
    label_warped_moving_mpc = mapping_mpc.transform(
        label_moving, interpolation="nearest"
    )

    print(f"Dice coefficient: {patient_static} -> Warped {patient_moving}")
    dice = dice_coef_multilabel(label_static, label_warped_moving_mpc, num_labels)
    row["dice_static_warped_moving_mpc"] = dice
    print(dice)
    print(f"Time: {end-start}")
    row["time_ppir"] = end - start
    row['iteration_ppir'] = sdr_mpc.iter
    row["norm_cc_mpc"] = metric_mpc.get_energy()
    print(f"Time per iteration: {(end-start)/sum(level_iters)}")
    row["time_per_iteration"] = (end - start) / sum(level_iters)
    print(f"Num bits: {metric_mpc.get_num_bits()}")
    row["num_bits"] = metric_mpc.get_num_bits()
    print(f"GigaBytes: {metric_mpc.get_num_bits()/(8*1024*1024*1024)}")
    row["gigabytes"] = metric_mpc.get_num_bits() / (8 * 1024 * 1024 * 1024)
    print(f"Dice coefficient: {patient_moving} -> Warped {patient_static}")
    label_warped_static_mpc = mapping_mpc.transform_inverse(
        label_static, interpolation="nearest"
    )
    dice = dice_coef_multilabel(label_moving, label_warped_static_mpc, num_labels)
    print(dice)
    row["dice_moving_warped_static_mpc"] = dice
    path_warped_label_mpc = f"{path_patient}/labelsWarpedMPC"
    check_and_create_folder(path_warped_label_mpc)
    path_warped_label_static_mpc = f"{path_warped_label_mpc}/{patient_static}.nii.gz"
    path_warped_label_moving_mpc = f"{path_warped_label_mpc}/{patient_moving}.nii.gz"
    save_nifti(
        path_warped_label_static_mpc, label_warped_static_mpc, label_static_affine
    )
    save_nifti(
        path_warped_label_moving_mpc, label_warped_moving_mpc, label_moving_affine
    )
    print("RMSE")
    print(
        f"RMSE Displacement: Warped {patient_moving} Clear vs Warped {patient_moving} MPC"
    )
    path_displacement = f"{path_patient}/displacement"
    check_and_create_folder(path_displacement)
    rmse_displacement = rmse(
        mapping.get_forward_field(), mapping_mpc.get_forward_field()
    )
    row["rmse_forward"] = rmse_displacement
    np.save(f"{path_displacement}/clear_forward.npy", mapping.get_forward_field())
    np.save(f"{path_displacement}/mpc_forward.npy", mapping_mpc.get_forward_field())
    print(rmse_displacement)
    print(
        f"RMSE Displacement: Warped {patient_static} Clear vs Warped {patient_static} MPC"
    )
    rmse_displacement = rmse(
        mapping.get_backward_field(), mapping_mpc.get_backward_field()
    )
    row["rmse_backward"] = rmse_displacement
    np.save(f"{path_displacement}/clear_backward.npy", mapping.get_backward_field())
    np.save(f"{path_displacement}/mpc_backward.npy", mapping_mpc.get_backward_field())
    print(rmse_displacement)
    df.loc[len(df)] = row
    df.to_csv(f"{dataset}_results.csv", index=False)


df = pd.DataFrame(
    columns=[
        "id_patient",
        "norm_cc_clear",
        "norm_cc_mpc",
        "dice_static_moving",
        "dice_static_warped_moving",
        "dice_moving_static",
        "dice_moving_warped_static",
        "dice_static_warped_moving_mpc",
        "dice_moving_warped_static_mpc",
        "time_clear",
        "time_ppir",
        "iteration_clear",
        "iteration_ppir",
        "num_bits",
        "gigabytes",
        "rmse_forward",
        "rmse_backward",
    ]
)
for i in range(1, 9):
    run(i, df)
