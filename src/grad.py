import os
import glob
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

IMAGE_DIR="/home/admin1/Abishek-Intern/new_atr/data/newdata/test/images"
OUTPUT_DIR="/home/admin1/Abishek-Intern/new_atr/data/newdata/test/images/gradcam_m"
MODEL_WEIGHTS_PATH="/home/admin1/Abishek-Intern/new_atr/new/models/yolo_aircraft.pt"

os.makedirs(OUTPUT_DIR,exist_ok=True)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#LOAD MODEL
model=YOLO(MODEL_WEIGHTS_PATH).to(device)
pure_pytorch_model=model.model
pure_pytorch_model.eval()

#ENABLE GRADIENTS
for param in pure_pytorch_model.parameters():
	param.requires_grad=True

torch.set_grad_enabled(True)

target_layers=[pure_pytorch_model.model[-4]]


#CAM WRAPPER
class YoloGradCAMWrapper(torch.nn.Module):

	def __init__(self,yolo_model):
		super().__init__()
		self.model=yolo_model

	def forward(self,x):

		output=self.model(x)

		if not isinstance(output,(tuple,list)):
			raise RuntimeError(
				f"Unexpected model output type: {type(output)}"
			)

		if len(output)<2:
			raise RuntimeError(
				"Expected YOLO output with prediction and head data"
			)

		predictions=output[0]
		head_outputs=output[1]

		print("\n========== MODEL OUTPUT ==========")

		print(
			"FINAL OUTPUT:",
			predictions.shape,
			predictions.requires_grad
		)

		print(
			"HEAD OUTPUT KEYS:",
			list(head_outputs.keys())
		)

		one2many=head_outputs["one2many"]

		print(
			"ONE2MANY KEYS:",
			list(one2many.keys())
		)

		boxes=one2many["boxes"]
		scores=one2many["scores"]
		feats=one2many["feats"]

		print(
			"BOXES TYPE:",
			type(boxes)
		)

		print(
			"SCORES TYPE:",
			type(scores)
		)

		print(
			"FEATS TYPE:",
			type(feats)
		)

		if torch.is_tensor(boxes):

			print(
				"BOXES SHAPE:",
				boxes.shape,
				"REQUIRES_GRAD:",
				boxes.requires_grad
			)

		if torch.is_tensor(scores):

			print(
				"SCORES SHAPE:",
				scores.shape,
				"REQUIRES_GRAD:",
				scores.requires_grad
			)

		if isinstance(feats,(list,tuple)):

			print(
				"NUMBER OF FEATS:",
				len(feats)
			)

			for i,feat in enumerate(feats):

				if torch.is_tensor(feat):

					print(
						"FEAT",
						i,
						"SHAPE:",
						feat.shape,
						"REQUIRES_GRAD:",
						feat.requires_grad,
						"GRAD_FN:",
						feat.grad_fn
					)

				else:

					print(
						"FEAT",
						i,
						"TYPE:",
						type(feat)
					)

		elif torch.is_tensor(feats):

			print(
				"FEATS SHAPE:",
				feats.shape,
				"REQUIRES_GRAD:",
				feats.requires_grad,
				"GRAD_FN:",
				feats.grad_fn
			)

		#USE DIFFERENTIABLE SCORES IF AVAILABLE
		if torch.is_tensor(scores):

			print(
				"USING ONE2MANY SCORES"
			)

			return scores

		#FALLBACK
		return predictions


#TARGET
class YoloDetectionTarget:

	def __call__(self,model_output):

		print("\n========== TARGET ==========")

		print(
			"TARGET TYPE:",
			type(model_output)
		)

		print(
			"TARGET SHAPE:",
			model_output.shape
		)

		print(
			"TARGET REQUIRES GRAD:",
			model_output.requires_grad
		)

		if model_output.ndim==3:

			model_output=model_output[0]

		#SCORES
		if model_output.ndim==2:

			target=model_output.max()

			print(
				"TARGET SCORE:",
				target.item()
			)

			print(
				"TARGET REQUIRES GRAD:",
				target.requires_grad
			)

			return target

		if model_output.ndim==1:

			target=model_output.max()

			print(
				"TARGET SCORE:",
				target.item()
			)

			print(
				"TARGET REQUIRES GRAD:",
				target.requires_grad
			)

			return target

		raise RuntimeError(
			f"Unsupported target shape: {model_output.shape}"
		)


#WRAPPER
wrapped=YoloGradCAMWrapper(
	pure_pytorch_model
)


#CAM
cam=GradCAM(
	model=wrapped,
	target_layers=target_layers
)


#IMAGE LIST
image_paths=glob.glob(
	os.path.join(IMAGE_DIR,"*.jpg")
)

image_paths+=glob.glob(
	os.path.join(IMAGE_DIR,"*.jpeg")
)

image_paths+=glob.glob(
	os.path.join(IMAGE_DIR,"*.png")
)

print("Found",len(image_paths))

#PROCESS FIRST 5
image_paths=image_paths[:5]


for index,image_path in enumerate(image_paths):

	print(
		f"\nProcessing [{index+1}/{len(image_paths)}]: "
		f"{os.path.basename(image_path)}"
	)

	try:

		#READ IMAGE
		image=cv2.imread(image_path)

		if image is None:

			print(
				"Could not read image"
			)

			continue

		#BGR TO RGB
		image_rgb=cv2.cvtColor(
			image,
			cv2.COLOR_BGR2RGB
		)

		#RESIZE
		input_size=640

		image_resized=cv2.resize(
			image_rgb,
			(input_size,input_size)
		)

		#NORMALIZE
		input_image=image_resized.astype(
			np.float32
		)/255.0

		#TENSOR
		input_tensor=torch.from_numpy(
			input_image
		).permute(
			2,
			0,
			1
		).unsqueeze(
			0
		).to(device)

		input_tensor.requires_grad_(True)

		#CLEAR GRADIENTS
		pure_pytorch_model.zero_grad(
			set_to_none=True
		)

		#CAM
		grayscale_cam=cam(
			input_tensor=input_tensor,
			targets=[
				YoloDetectionTarget()
			]
		)

		grayscale_cam=grayscale_cam[0]

		print(
			"\n========== CAM =========="
		)

		print(
			"CAM MIN:",
			grayscale_cam.min()
		)

		print(
			"CAM MAX:",
			grayscale_cam.max()
		)

		print(
			"CAM MEAN:",
			grayscale_cam.mean()
		)

		print(
			"CAM STD:",
			grayscale_cam.std()
		)

		#OVERLAY
		visualization=show_cam_on_image(
			input_image,
			grayscale_cam,
			use_rgb=True
		)

		#OUTPUT PATH
		output_path=os.path.join(
			OUTPUT_DIR,
			os.path.splitext(
				os.path.basename(image_path)
			)[0]+"_gradcam.jpg"
		)

		#RGB TO BGR
		visualization_bgr=cv2.cvtColor(
			visualization,
			cv2.COLOR_RGB2BGR
		)

		#SAVE
		cv2.imwrite(
			output_path,
			visualization_bgr
		)

		print(
			"Saved:",
			output_path
		)

	except Exception as e:

		print(
			f"Error processing {image_path}: "
			f"{type(e).__name__}: {e}"
		)

print("\nDone")
