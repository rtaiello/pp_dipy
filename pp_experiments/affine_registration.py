import numpy as np
import pandas as pd
from dipy.align import affine_registration
from dipy.io.image import load_nifti, save_nifti


def get_displacement(transformation, domain):
    tensor = np.mgrid[0 : domain[0], 0 : domain[1], 0 : domain[2]]
    coordinates_homogenous = np.ones((4, tensor[0].size))
    coordinates_homogenous[0] = tensor[0].flatten()
    coordinates_homogenous[1] = tensor[1].flatten()
    coordinates_homogenous[2] = tensor[2].flatten()

    transformed_coordinates = np.dot(
        np.linalg.inv(transformation), coordinates_homogenous
    )
    shape = tensor[0].shape
    displacement = transformed_coordinates - coordinates_homogenous
    displacement = np.array(
        [
            displacement[0].reshape(shape),
            displacement[1].reshape(shape),
            displacement[2].reshape(shape),
        ]
    )

    return displacement

def rmse(displacement_clear, displacement_mpc, pixel_size=1):
    """
    Compute the RMSE between two displacement maps.
    :param displacement_clear:
    :param displacement_pp:
    :param pixel_size:
    :return:
    """
    displacement_diff = displacement_clear - displacement_mpc
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

df = pd.DataFrame(
    columns=[
        "negative MI clear",
        "negative MI MPC",
        "RMSE",
        "Time joint pdf",
        "Num iterations clear",
        "Num iterations MPC",
        "GBytes joint pdf",
        "Time derivative joint pdf",
        "GBytes derivative joint pdf",
    ]
)
row = {}
static_path = (
    "/home/argentera/taiello/pp_dipy/pp_experiments/data/Adni/template_mr.nii.gz"
)
static_data, static_affine = load_nifti(static_path)
static = static_data
moving_path = (
    "/home/argentera/taiello/pp_dipy/pp_experiments/data/Adni/moving_pt_resized.nii.gz"
)
moving_data, moving_affine = load_nifti(moving_path)
moving = moving_data
nbins = 8
sampling_prop = 0.1
level_iters = [100, 100, 0]
sigmas = [3.0, 1.0, 0.0]
factors = [10, 5, 1]


pipeline = ["center_of_mass", "translation", "rigid", "affine"]

metric_kwargs = {"sampling_proportion": sampling_prop, "mpc": False}
transformed, reg_affine, metric, iter = affine_registration(
    moving,
    static,
    moving_affine=moving_affine,
    static_affine=static_affine,
    nbins=8,
    metric="MI",
    pipeline=pipeline,
    level_iters=level_iters,
    sigmas=sigmas,
    factors=factors,
    **metric_kwargs
)

save_nifti(
    "/home/argentera/taiello/pp_dipy/pp_experiments/data/Adni/warped_clear_moving_pt_resized.nii.gz",
    transformed.astype(np.float32),
    moving_affine,
)
displcement_clear = get_displacement(reg_affine, (120, 110, 140))
np.save(
    "/home/argentera/taiello/pp_dipy/pp_experiments/data/Adni/displacement_clear_moving_pt_resized.npy",
    displcement_clear,
)
row["negative MI clear"] = metric.metric_val
row["Num iterations clear"] = iter
metric_kwargs = {"sampling_proportion": sampling_prop, "mpc": True}
transformed, reg_affine, metric, iter = affine_registration(
    moving,
    static,
    moving_affine=moving_affine,
    static_affine=static_affine,
    nbins=8,
    metric="MI",
    pipeline=pipeline,
    level_iters=level_iters,
    sigmas=sigmas,
    factors=factors,
    **metric_kwargs
)
save_nifti(
    "/home/argentera/taiello/pp_dipy/pp_experiments/data/Adni/warped_mpc_moving_pt_resized.nii.gz",
    transformed.astype(np.float32),
    moving_affine,
)
displcement_mpc = get_displacement(reg_affine, (120, 110, 140))
np.save(
    "/home/argentera/taiello/pp_dipy/pp_experiments/data/Adni/displacement_mpc_moving_pt_resized.npy",
    displcement_mpc,
)
row["negative MI MPC"] = metric.metric_val
row["RMSE"] = rmse(displcement_clear, displcement_mpc)
row["Time joint pdf"] = metric.histogram.time_pdf
gbytes_joint_pdf = metric.histogram.num_bits_pdf / (8 * 1024 ** 3)
row["GBytes joint pdf"] = gbytes_joint_pdf
print(gbytes_joint_pdf)
row["Time derivative joint pdf"] = metric.histogram.time_grad
gbytes_derivative_joint_pdf = metric.histogram.num_bits_grad / (8 * 1024 ** 3)
row["GBytes derivative joint pdf"] = gbytes_derivative_joint_pdf
row["Num iterations MPC"] = iter
print(gbytes_derivative_joint_pdf)
df.loc[len(df)] = row
df.to_csv(
    "/home/argentera/taiello/pp_dipy/pp_experiments/data/Adni/affine_registration.csv"
)
row["Num iterations MPC"] = iter