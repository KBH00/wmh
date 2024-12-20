import os
import argparse
import numpy as np
from deepwmh.analysis.image_ops import remove_3mm_sparks
from deepwmh.utilities.external_call import run_shell
from deepwmh.utilities.misc import contain_duplicates
from deepwmh.utilities.nii_preview import nii_as_gif, nii_slice_range
from deepwmh.utilities.parallelization import run_parallel
from deepwmh.utilities.data_io import get_nifti_pixdim, load_nifti, save_nifti, try_load_gif, try_load_nifti
from deepwmh.main.integrity_check import check_dataset, check_system_integrity
from deepwmh.utilities.file_ops import abs_path, cp, dir_exist, file_exist, gd, gn, join_path, ls, mkdir, rm
import sys
sys.setrecursionlimit(50000)

import numpy as np

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
	if try_load_nifti(output_image_path) == False:
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

# def _parallel_generate_final_GIF(params):
#     in_image, in_seg, out_gif = params
#     if try_load_gif(out_gif) == False:
#         axis = 'sagittal'
#         print("AA")
#         data, _ = load_nifti(in_image)
# 		print()
#         slice_start, slice_end = nii_slice_range(in_image, axis=axis, value = np.min(data) + 0.001, percentage = 0.999)
#         nii_as_gif(in_image, out_gif, axis=axis, lesion_mask=in_seg, side_by_side=True, slice_range=[slice_start, slice_end])

def _parallel_generate_final_GIF(params):
	in_image, in_seg, out_gif = params[0][0], params[0][1], params[0][2]

	if try_load_gif(out_gif) == False:
		axis = 'axial'
		data, _ = load_nifti(in_image)
		slice_start, slice_end = nii_slice_range(in_image, axis=axis, value=np.min(data)+0.001, percentage=0.999)
		nii_as_gif(in_image, out_gif, axis=axis, lesion_mask=in_seg, side_by_side=True, slice_range=[slice_start, slice_end])

def _parallel_ROBEX_masking(params):
	case, in_seg, in_flair, ROBEX_sh, out_seg = params
	print(case)
	if try_load_nifti(out_seg) == False:
		brain_out = join_path(gd(out_seg), case + '_brain.nii.gz')
		brain_mask = join_path(gd(out_seg), case + '_mask.nii.gz')
		run_shell('%s %s %s %s' % (ROBEX_sh, in_flair, brain_out, brain_mask), print_output=False)
		dat, hdr = load_nifti(in_seg)
		accept_mask, _ = load_nifti(brain_mask)
		rm(brain_out)
		rm(brain_mask)
		save_nifti(((dat * accept_mask) > 0.5).astype('float32'), hdr, out_seg)

def main():
	class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, 
					  argparse.RawDescriptionHelpFormatter):  pass

	parser = argparse.ArgumentParser(
		description = 'Do lesion segmentation using pre-trained/installed model.',
		formatter_class = MyFormatter)
	parser.add_argument('-i', '--input-images', default="/home/kbh/Downloads/nii/003_S_6256/Sagittal_3D_FLAIR/2018-03-13_15_53_35.0/I973284/I973284_Sagittal_3D_FLAIR_20180313155336_3_cleaned.nii.gz",
		help='Input image paths for prediction. Multiple input image paths are supported.',
		type=str, nargs='+',required=True)
	parser.add_argument('-n', '--case-names', default="subject1",
		help='Case name for each input image.',
		type=str, nargs='+',required=True)	
	parser.add_argument('-m', '--trained-model', default="./model/",
		help='Root folder of the trained model (containing the entire directory structure).',
		type=str, required=True)
	parser.add_argument('-o', '--output-folder', default="home/kbh/Code/deepwmh/output/",
		help='Folder where pre-processed input images and predicted segmentations will be stored.',
		type=str, required=True)
	parser.add_argument('-g', '--gpu',
		help='GPU id.',
		type=int, default=0)

	# advanced configs below. In most cases you won't need to set these.
	parser.add_argument('--skip-bfc', help='[Advanced] Skip bias field correction.', action = 'store_true')
	parser.add_argument('--custom-task-name', 
		help='[Advanced] Find model in specified task to segment image.', type = str, required = False)

	args = parser.parse_args()
	
	if len(args.case_names) != len(args.input_images):
		raise RuntimeError('Number of input images (%d) should be equal to case names (%d).' %\
			(len(args.input_images), len(args.case_names)))
	
	# check integrity
	ignore_ANTs = True if args.skip_bfc else False
	if check_system_integrity(verbose=True,ignore_ANTs=ignore_ANTs,ignore_FreeSurfer=True,ignore_FSL=True) == False:
		exit(1)

	# check dataset
	test_dataset = {'case':[], 'flair': []}
	if contain_duplicates(args.case_names):
		print('case names contain duplicates.')
		exit(1)
	for case_name, input_image in zip(args.case_names, args.input_images):
		test_dataset['case'].append(case_name)
		test_dataset['flair'].append( abs_path(input_image) )
	if check_dataset(test_dataset) == False:
		exit(1)

	# check if model is valid
	os.environ['RESULTS_FOLDER'] = abs_path(args.trained_model)
	if dir_exist( os.environ['RESULTS_FOLDER'] ) == False:
		raise RuntimeError('Directory not exist: "%s".' % os.environ['RESULTS_FOLDER'])
	if dir_exist( join_path(os.environ['RESULTS_FOLDER'], 'nnUNet') ) == False:
		raise RuntimeError('Invalid model directory. Cannot find directory "nnUNet" in folder "%s".' % \
			os.environ['RESULTS_FOLDER'])
	print('model file is valid.')

	# creating folders
	mkdir(args.output_folder)
	image_folder    = mkdir(join_path(args.output_folder, '001_Preprocessed_Images'))
	seg_folder      = mkdir(join_path(args.output_folder, '002_Segmentations'))
	raw_seg_folder  = mkdir(join_path(args.output_folder, '002_Segmentations', '001_raw'))
	post_3mm_folder = mkdir(join_path(args.output_folder, '002_Segmentations', '002_postproc_3mm'))
	post_fov_folder = mkdir(join_path(args.output_folder, '002_Segmentations', '003_postproc_fov'))
	import nibabel as nib
	# preprocessing
	print('Pre-processing test images for prediction.')
	# bias field correction
	if not args.skip_bfc:
		n4_tasks = []
		for case_name, input_image in zip(test_dataset['case'], test_dataset['flair']):
			test_img = nib.load(input_image)
			print(test_img.shape)
			n4_image = join_path(image_folder, '%s_0000.nii.gz' % case_name)
			n4_tasks.append( (input_image, n4_image) )
		run_parallel(_parallel_do_bias_field_correction, n4_tasks, 4, 'preprocessing')
	else:
		print('** Skipped bias field correction. '
			'Segmentation performance can be suboptimal when images have strong intensity bias, please be aware.')
		for case_name, input_image in zip(test_dataset['case'], test_dataset['flair']):
			out_image = join_path(image_folder, '%s_0000.nii.gz' % case_name)
			cp(input_image, out_image) # rename and copy to destination

	# predict
	DCNN_trainer_name = 'nnUNetTrainerV2'
	DCNN_network_config = '3d_fullres'
	DCNN_planner_name = 'nnUNetPlansv2.1'

	def _auto_detect_task_name():
		task_folders = ls( join_path(os.environ['RESULTS_FOLDER'], 'nnUNet', DCNN_network_config) )
		if len(task_folders) == 0:
			raise RuntimeError('Cannot find any task folder in "%s".' % task_folders)
		elif len(task_folders) > 1:
			raise RuntimeError('Found multiple task folders in "%s".' % task_folders)
		else:
			return task_folders[0]

	DCNN_task_name = args.custom_task_name if args.custom_task_name is not None else _auto_detect_task_name()
	DCNN_fold = 'all'
	model_name = 'model_best'
	os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % args.gpu
	os.environ['nnUNet_raw_data_base'] = image_folder
	os.environ['nnUNet_preprocessed'] = image_folder
	predict_command = 'nnUNet_predict -i %s -o %s -tr %s -m %s -p %s -t %s -f %s -chk %s --disable_post_processing --selected_cases %s ' % \
        (image_folder, raw_seg_folder, DCNN_trainer_name, DCNN_network_config,
        DCNN_planner_name, DCNN_task_name, DCNN_fold, model_name, ' '.join(args.case_names))
	run_shell(predict_command)

	task_list = []
	for item in ls(raw_seg_folder, full_path=True):
		if item[-7:]=='.nii.gz':
			out_seg = join_path(post_3mm_folder, gn(item))
			task_list.append((item, out_seg))
	run_parallel(_parallel_3mm_spark_removal,task_list, 8, "3mm postproc")

	# remove FP outside brain tissue
	ROBEX_folder = os.environ['ROBEX_DIR']
	ROBEX_sh = join_path(ROBEX_folder, 'runROBEX.sh')
	ROBEX_bin = join_path(ROBEX_folder, 'ROBEX')
	if file_exist(ROBEX_sh) == False or file_exist(ROBEX_bin) == False:
		raise RuntimeError("Cannot find 'runROBEX.sh' and 'ROBEX' binary file in "
			"folder '%s', be sure to download and install ROBEX in your local "
			"machine and check the path given is correct." % ROBEX_folder)
	else:
		task_list = []
		for case_name in args.case_names:
			flair = join_path(image_folder, case_name + '_0000.nii.gz')
			seg_3mm_pp = join_path(post_3mm_folder, case_name + '.nii.gz')
			seg_fov = join_path( post_fov_folder, case_name + '.nii.gz' )
			task_list.append( (case_name, seg_3mm_pp, flair, ROBEX_sh, seg_fov ) )

		run_parallel(_parallel_ROBEX_masking, task_list, 8, 'masking')

	for case_name, input_image in zip(args.case_names, args.input_images):
		original_image_path = input_image
		seg_image_path = join_path(post_fov_folder, case_name + '.nii.gz')  # path to segmentation result
		overlay_image_path = join_path("./", f"{case_name}_overlay.nii.gz")
		overlay_segmentation_on_image(original_image_path, seg_image_path, overlay_image_path)
		
	# generate preview
	print('Prediction done. Now generating GIF previews...')
	preview_folder = mkdir(join_path(args.output_folder, '003_Previews'))
	gif_tasks = []
	for case_name, input_image in zip(args.case_names, args.input_images):
		folder_name = '003_postproc_fov'
		in_seg = join_path(seg_folder, folder_name, '%s.nii.gz' % case_name)
		output_gif = join_path(preview_folder, '%s.gif' % case_name)
		gif_tasks.append( (input_image, in_seg, output_gif) )
	
	print(gif_tasks)
	_parallel_generate_final_GIF(gif_tasks)

	segmentation_output = post_fov_folder
	print('')
	print('>>> Prediction done.')
	print('>>> Raw/preprocessed images can be found in folder "%s".' % image_folder)
	print('>>> Segmentation results can be found in folder "%s".' % segmentation_output)
	print('')

	for case_name, input_image in zip(args.case_names, args.input_images):
		original_image_path = input_image
		seg_image_path = join_path(post_fov_folder, case_name + '.nii.gz') 
		overlay_image_path = join_path("./", f"{case_name}_overlay.nii.gz")
		overlay_segmentation_on_image(original_image_path, seg_image_path, overlay_image_path)

		#brain_mask_path = join_path(post_fov_folder, case_name + '_brain_mask.nii.gz')  # Assuming brain mask path
		seg_volume, brain_volume, seg_percentage = calculate_volumes(original_image_path, seg_image_path)
		

