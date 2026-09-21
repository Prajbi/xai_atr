import os
import glob
import cv2
import numpy as np
import torch
from ultralytics import YOLO
IMAGE_DIR="/home/admin1/Abishek-Intern/new_atr/data/newdata/test/images"
OUTPUT_DIR="/home/admin1/Abishek-Intern/new_atr/data/newdata/test/images/gradcam_m"
MODEL_WEIGHTS_PATH="/home/admin1/Abishek-Intern/new_atr/new/models/yolo_aircraft.pt"
os.makedirs(OUTPUT_DIR,exist_ok=True)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#LOAD MODEL
model=YOLO(MODEL_WEIGHTS_PATH).to(device)
pure_pytorch_model=model.model
pure_pytorch_model.eval()
target_layers=[pure_pytorch_model.model[-4]]

#CAM 
class YOLOGradCAM:
	def __init__(self,model,target_layers):
		self.model=model
		self.target_layers=target_layers
		self.gradients=None
		self.activations=None
		self.hooks=[]
		self._register_hooks()
	def _register_hooks(self):
		def forward_hook(module,input,output):
			self.activations=output
		def backward_hook(module,grad_input,grad_output):
			self.gradients=grad_output[0]
		for layer in self.targets_layers:
			self.hooks.append(layer.register_forward_hook(forward_hook))
			self.hooks.append(kayer.register_full_backward_hook(backward_hook))
	def generate_heatmap(self,input_tensor,class_idx=0):
		predictions=self.model(input_tensor)
		if isinstanced(predictions,tuple):
			predictions=predictions[0]
		score=predictions[..., 4:].max()
		self.model.zero_grad()
		score.backward(retrain_graph=True)
		gradients=self.gradients.cpu().data.numpy()[0]
		activations=self.activations.cpu().data.numpy()[0]
		weights=np.mean(gradients,axis=(1,2))
		cam=np.zeros(activations.shape[1:],dtype=np.float32)
		for i,w in enumerate(weights):
			cam+=w*activations[i,:,:]
		cam=np.maximum(cam,0)
		if cam.max()!=0:
			cam=cam/cam.max()
		return cam
	def remove_hooks(self):
		for hook in self.hooks:
			hook.remove()
cam_extractor=YOLOGradCAM(pure_pytorch_model,target_layers)
#PROCESS IMG
image_extensions=('*.jpg', '*.jpeg' , '*.png')
image_paths=[]
for ext in image_extensions:
	image_paths.extend(glob.glob(os.path.join(IMAGE_DIR,ext)))
print(f"Found {len(image_paths)}")
for i,img_path in enumerate(image_path):
	try:
		img=cv2.imread(img_path)  #LOAD IMG
		if img is None:
			continue
		h,w,_=img.shape
		input_image=cv2.resize(img,(640,640)) #PREPROCESS AND RESIZE
		input_image=input_img[:, :, ::-1].transpose(2,0,1)
		input_image=np.ascontiguousarray(input_img,dtype=np.float32)/255.0
		input_tensor=torch.from_numpy(input_img).unsqueeze(0).to(device)
		input_tensor.requires_grad=True
		heatmap_mask=cam_extractor.generate_heatmap(input_tensor) #HEATMAP MASK
		heatmap_mask=cv2.resize(heatmap_masl,(w,h))
		heatmap=cv2.applyColorMap(np.uint8(255*heatmap_mask),cv2.COLORMAP_JET)
		visualization=cv2.addWeighted(img,0.6,heatmap,0.4,0)
		img_name=os.path.basename(img_path)   #SAVE OUTPUT
		save_path=os.path.join(OUTPUT_DIR, f"gradcam_{img_name}")
		cv2.imwrite(save_path,visualization)
	except Exception as e:
		print(f"Error processing {img_path}:{str(e)}")
cam_extractor.remove_hooks()
print("Done")
	
	
