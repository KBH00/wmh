# deepwmh_segmentation.py

import os
import numpy as np
from deepwmh.analysis.image_ops import remove_3mm_sparks
from deepwmh.utilities.external_call import run_shell
from deepwmh.utilities.misc import contain_duplicates
from deepwmh.utilities.nii_preview import nii_as_gif, nii_slice_range
from deepwmh.utilities.parallelization import run_parallel
from deepwmh.utilities.data_io import get_nifti_pixdim, load_nifti, save_nifti, try_load_gif, try_load_nifti
from deepwmh.main.integrity_check import check_dataset, check_system_integrity
from deepwmh.utilities.file_ops import abs_path, cp, dir_exist, file_exist, gd, gn, join_path, ls, mkdir, rm

import nibabel as nib

def calculate_volumes(in_image_path, seg_image_path):
    image_data, image_header = load_nifti(in_image_path)
    seg_data, seg_header = load_nifti(seg_image_path)
    
    voxel_dims = get_nifti_pixdim(in_image_path)
    print("Voxel Dimensions (in mm):", voxel_dims)
    
    voxel_volume = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]  
    voxel_volume_ml = voxel_volume * 0.001  # Convert to mL
    
    brain_volume_mm3 = np.sum(image_data > 0) * voxel_volume
    seg_volume_mm3 = np.sum(seg_data > 0) * voxel_volume
    
    brain_volume_ml = brain_volume_mm3 * 0.001
    seg_volume_ml = seg_volume_mm3 * 0.001
    
    seg_percentage = (seg_volume_mm3 / brain_volume_mm3) * 100 if brain_volume_mm3 > 0 else 0

    print(f"Brain Volume (approx): {brain_volume_mm3:.2f} mm³ / {brain_volume_ml:.2f} mL")
    print(f"Segmented Volume: {seg_volume_mm3:.2f} mm³ / {seg_volume_ml:.2f} mL")
    print(f"Segmented Percentage: {seg_percentage:.2f}%")
    print(f"Single Voxel Volume: {voxel_volume:.2f} mm³ / {voxel_volume_ml:.6f} mL")
    
    return brain_volume_mm3, seg_volume_mm3, seg_percentage


def overlay_segmentation_on_image(in_image_path, seg_image_path, out_overlay_path):
    image_data, image_header = load_nifti(in_image_path)
    seg_data, _ = load_nifti(seg_image_path)

    overlay_data = np.copy(image_data)
    overlay_data[seg_data > 0] = np.max(image_data) 

    save_nifti(overlay_data, image_header, out_overlay_path)


def _parallel_do_bias_field_correction(params):
    raw_image_path, output_image_path = params
    if not try_load_nifti(output_image_path):
        run_shell('N4BiasFieldCorrection -d 3 -i %s -o %s -c [50x50x50,0.0] -s 2' % \
            (raw_image_path, output_image_path), print_command=False, print_output=False)


def _parallel_3mm_spark_removal(params):
    in_seg, out_seg = params
    if try_load_nifti(out_seg):
        return
    label, header = load_nifti(in_seg)
    vox_size = get_nifti_pixdim(in_seg)
    label0 = remove_3mm_sparks(label, vox_size)
    save_nifti(label0, header, out_seg)


def _parallel_generate_final_GIF(params):
    in_image, in_seg, out_gif = params
    if not try_load_gif(out_gif):
        axis = 'axial'
        data, _ = load_nifti(in_image)
        slice_start, slice_end = nii_slice_range(
            in_image, axis=axis, value=np.min(data)+0.001, percentage=0.999)
        nii_as_gif(
            in_image, out_gif, axis=axis, lesion_mask=in_seg,
            side_by_side=True, slice_range=[slice_start, slice_end])
    else:
        print(f'GIF already exists: {out_gif}')


def _parallel_ROBEX_masking(params):
    case, in_seg, in_flair, ROBEX_sh, out_seg = params
    print(case)
    if not try_load_nifti(out_seg):
        brain_out = join_path(gd(out_seg), f"{case}_brain.nii.gz")
        brain_mask = join_path(gd(out_seg), f"{case}_mask.nii.gz")
        run_shell(f"{ROBEX_sh} {in_flair} {brain_out} {brain_mask}", print_output=False)
        dat, hdr = load_nifti(in_seg)
        accept_mask, _ = load_nifti(brain_mask)
        rm(brain_out)
        rm(brain_mask)
        save_nifti(((dat * accept_mask) > 0.5).astype('float32'), hdr, out_seg)


def run_segmentation(input_image_path, output_folder, thread):
    # Set up necessary paths and variables
    import sys

    case_name = os.path.splitext(os.path.basename(input_image_path))[0]
    trained_model = './model/'
    gpu = 0

    # Prepare the dataset
    test_dataset = {'case': [case_name], 'flair': [input_image_path]}

    # Set environment variables
    os.environ['RESULTS_FOLDER'] = os.path.abspath(trained_model)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['nnUNet_raw_data_base'] = output_folder
    os.environ['nnUNet_preprocessed'] = output_folder

    # Check integrity
    ignore_ANTs = False  # Since skip-bfc is False
    if not check_system_integrity(verbose=True, ignore_ANTs=ignore_ANTs, ignore_FreeSurfer=True, ignore_FSL=True):
        thread.log_signal.emit("System integrity check failed.")
        return

    # Check dataset
    if contain_duplicates(test_dataset['case']):
        thread.log_signal.emit("Case names contain duplicates.")
        return

    if not check_dataset(test_dataset):
        thread.log_signal.emit("Dataset check failed.")
        return

    # Check if model is valid
    if not dir_exist(os.environ['RESULTS_FOLDER']):
        thread.log_signal.emit(f'Directory does not exist: "{os.environ["RESULTS_FOLDER"]}".')
        return

    if not dir_exist(join_path(os.environ['RESULTS_FOLDER'], 'nnUNet')):
        thread.log_signal.emit(f'Invalid model directory. Cannot find directory "nnUNet" in folder "{os.environ["RESULTS_FOLDER"]}".')
        return

    thread.log_signal.emit("Model file is valid.")

    # Creating folders
    mkdir(output_folder)
    image_folder = mkdir(join_path(output_folder, '001_Preprocessed_Images'))
    seg_folder = mkdir(join_path(output_folder, '002_Segmentations'))
    raw_seg_folder = mkdir(join_path(output_folder, '002_Segmentations', '001_raw'))
    post_3mm_folder = mkdir(join_path(output_folder, '002_Segmentations', '002_postproc_3mm'))
    post_fov_folder = mkdir(join_path(output_folder, '002_Segmentations', '003_postproc_fov'))

    # Preprocessing
    thread.log_signal.emit('Pre-processing test images for prediction.')

    # Bias field correction
    n4_tasks = []
    for case_name, input_image in zip(test_dataset['case'], test_dataset['flair']):
        n4_image = join_path(image_folder, f"{case_name}_0000.nii.gz")
        n4_tasks.append((input_image, n4_image))
    run_parallel(_parallel_do_bias_field_correction, n4_tasks, 4, 'preprocessing')
    thread.log_signal.emit('Bias field correction completed.')

    # Predict
    DCNN_trainer_name = 'nnUNetTrainerV2'
    DCNN_network_config = '3d_fullres'
    DCNN_planner_name = 'nnUNetPlansv2.1'

    def _auto_detect_task_name():
        task_folders = ls(join_path(os.environ['RESULTS_FOLDER'], 'nnUNet', DCNN_network_config))
        if len(task_folders) == 0:
            raise RuntimeError(f'Cannot find any task folder in "{os.environ["RESULTS_FOLDER"]}".')
        elif len(task_folders) > 1:
            raise RuntimeError(f'Found multiple task folders in "{os.environ["RESULTS_FOLDER"]}".')
        else:
            return task_folders[0]

    DCNN_task_name = _auto_detect_task_name()
    DCNN_fold = 'all'
    model_name = 'model_best'

    predict_command = f'nnUNet_predict -i {image_folder} -o {raw_seg_folder} -tr {DCNN_trainer_name} -m {DCNN_network_config} -p {DCNN_planner_name} -t {DCNN_task_name} -f {DCNN_fold} -chk {model_name} --disable_post_processing --selected_cases {" ".join(test_dataset["case"])}'
    thread.log_signal.emit(f"Running prediction: {predict_command}")
    run_shell(predict_command)
    thread.log_signal.emit('Prediction completed.')

    # Post-processing: 3mm spark removal
    task_list = []
    for item in ls(raw_seg_folder, full_path=True):
        if item.endswith('.nii.gz'):
            out_seg = join_path(post_3mm_folder, gn(item))
            task_list.append((item, out_seg))
    run_parallel(_parallel_3mm_spark_removal, task_list, 8, "3mm spark removal")
    thread.log_signal.emit('3mm spark removal completed.')

    # Remove FP outside brain tissue
    ROBEX_folder = os.environ.get('ROBEX_DIR')
    if not ROBEX_folder:
        thread.log_signal.emit("ROBEX_DIR environment variable not set.")
        return

    ROBEX_sh = join_path(ROBEX_folder, 'runROBEX.sh')
    ROBEX_bin = join_path(ROBEX_folder, 'ROBEX')

    if not (file_exist(ROBEX_sh) and file_exist(ROBEX_bin)):
        thread.log_signal.emit("Cannot find 'runROBEX.sh' and 'ROBEX' binary file. Please check ROBEX installation.")
        return
    else:
        task_list = []
        for case_name in test_dataset['case']:
            flair = join_path(image_folder, f"{case_name}_0000.nii.gz")
            seg_3mm_pp = join_path(post_3mm_folder, f"{case_name}.nii.gz")
            seg_fov = join_path(post_fov_folder, f"{case_name}.nii.gz")
            task_list.append((case_name, seg_3mm_pp, flair, ROBEX_sh, seg_fov))

        run_parallel(_parallel_ROBEX_masking, task_list, 8, 'masking')
        thread.log_signal.emit('ROBEX masking completed.')

    # Overlay segmentation on image and calculate volumes
    for case_name, input_image in zip(test_dataset['case'], test_dataset['flair']):
        original_image_path = input_image
        seg_image_path = join_path(post_fov_folder, f"{case_name}.nii.gz")  # path to segmentation result
        overlay_image_path = join_path(output_folder, f"{case_name}_overlay.nii.gz")
        overlay_segmentation_on_image(original_image_path, seg_image_path, overlay_image_path)
        thread.log_signal.emit(f"Overlay image saved to {overlay_image_path}")

    # Generate preview GIFs
    thread.log_signal.emit('Prediction done. Now generating GIF previews...')
    preview_folder = mkdir(join_path(output_folder, '003_Previews'))
    gif_tasks = []
    for case_name, input_image in zip(test_dataset['case'], test_dataset['flair']):
        in_seg = join_path(post_fov_folder, f"{case_name}.nii.gz")
        output_gif = join_path(preview_folder, f"{case_name}.gif")
        gif_tasks.append((input_image, in_seg, output_gif))

    # Use run_parallel to process tasks
    run_parallel(_parallel_generate_final_GIF, gif_tasks, 4, "GIF generation")
    thread.log_signal.emit('GIF previews generated.')

    # Emit the gif_signal from the main thread
    for case_name in test_dataset['case']:
        gif_path = join_path(preview_folder, f"{case_name}.gif")
        print(f'Emitting gif_signal with gif_path: {gif_path}')
        thread.gif_signal.emit(gif_path)

    # Calculate volumes
    segmentation_output = post_fov_folder
    thread.log_signal.emit('')
    thread.log_signal.emit('>>> Prediction done.')
    thread.log_signal.emit(f'>>> Raw/preprocessed images can be found in folder "{image_folder}".')
    thread.log_signal.emit(f'>>> Segmentation results can be found in folder "{segmentation_output}".')
    thread.log_signal.emit('')

    for case_name, input_image in zip(test_dataset['case'], test_dataset['flair']):
        original_image_path = input_image
        seg_image_path = join_path(post_fov_folder, f"{case_name}.nii.gz")
        overlay_image_path = join_path(output_folder, f"{case_name}_overlay.nii.gz")
        overlay_segmentation_on_image(original_image_path, seg_image_path, overlay_image_path)

        brain_volume_mm3, seg_volume_mm3, seg_percentage = calculate_volumes(original_image_path, seg_image_path)
        voxel_dims = get_nifti_pixdim(original_image_path)
        voxel_volume = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
        voxel_volume_ml = voxel_volume * 0.001

        volume_text = (
            f"Case: {case_name}\n"
            f"Brain Volume (approx): {brain_volume_mm3:.2f} mm³ / {brain_volume_mm3 * 0.001:.2f} mL\n"
            f"Segmented Volume: {seg_volume_mm3:.2f} mm³ / {seg_volume_mm3 * 0.001:.2f} mL\n"
            f"Segmented Percentage: {seg_percentage:.2f}%\n"
            f"Single Voxel Volume: {voxel_volume:.2f} mm³ / {voxel_volume_ml:.6f} mL\n"
            "-------------------------\n"
        )
        thread.volume_signal.emit(volume_text)
