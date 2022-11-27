from model.edsr import edsr
from model.srgan import sr_resnet
from model.wdsr import wdsr_a, wdsr_b, wdsr

import os
import tf2onnx
import onnx

# Number of residual blocks
depth = 32
# Super-resolution factor
scale = 4

# Location of model weights (needed for demo)
weights_dir = f'weights\weights\wdsr-b-32-x4'
weights_file = os.path.join(weights_dir, 'weights.h5')

model = wdsr_b(scale=scale, num_res_blocks=depth)
model.load_weights(weights_file)

onnx_model, _ = tf2onnx.convert.from_keras(model, opset=11)
onnx.save(onnx_model, "converted_model\wdsr_b.onnx")
