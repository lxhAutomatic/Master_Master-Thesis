# diffusion_msc
For different classification models:

	CheXNet: Use the train.py file in the Classification_Model/ChexPert/bin for training, main hyper-parameters are shown in the file.
	DenseNet: Use the densenet.py file in the Classification_Model/DenseNet for training. Remember to set the path for every input and output file.
	Inception-ResNet: Read the readme.md in the Classification_Model/Inception-ResNet-v2 and follow the instruction to train.

For different label imputation methods:

	Change .fillna(0) and .replace(-1,1) in data preprocessing to .fillna(x) and .replace(-1,x).

For different image masking methods:

	Use the mask.py file in the folder to perform masking operations on the dataset. 
	mask_1 function is the joint mask method.
	mask_2 function is the inverted joint mask method.
	mask_3 function is the square mask method.
	mask_4 function is the separate mask method.

For different augmentation methods:

	Traditional augmentation: According to the flag parameter in the create function, use the augmentation.py file in the Gereration_Model to generate image data sets of different traditional augmentation methods. If you want to generate a combined method, you can directly use multiple methods at the same time 
	DDPM model：use the training_DDPM.py file in the Generation_Model.
	DDIM model：use the main.py file in the Generation_Model/ddim-main.
	Stable Diffusion: use the huggingface.py file in the Generation_Model, the prompt can be changed in the file. Or can use the img2img_old.py file in the stable-diffusion-main/scripts to generate.
	Stable Diffusion Finetune: I use the project from https://github.com/AUTOMATIC1111/stable-diffusion-webui

For different augmentation strategies:

	For the unbalanced augmentation strategy, use the create function in the augmentation.py file in xxxx folder.
	For the balanced augmentation strategy, use the create_2 function in the augmentation.py file in xxxx folder.
	For other augmentation strategy, use the PCA.py file to visualise. The augmentation file is not uploaded yet.

For evaluation metrics:

	All classification metrics such as accuracy, precision are in the eval_classification.py file.
	All generation metrics such as SSIM, PSNR are in the eval_generation.py file.
	Code for the mcnemar's test is in the mcnemar.py file.
	Code for the heatmap is in the heatmap.py file.
