import cv2
import numpy as np
import time

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras import models 
from tensorflow.keras import losses
from tensorflow.keras import layers

## read video frames
vidcap = cv2.VideoCapture('cat.mp4')
success, image = vidcap.read()
count = 0
images = []
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     
  success, image = vidcap.read()
  images.append("frame" + str(count) + ".jpg")
  count += 1
print('Read in ', count, ' frames')


def load_img(path_to_img, max_dim=512):  
  img = cv2.imread(path_to_img) 
  (h, w) = img.shape[:2]
  dim = (int(w * max_dim / float(h)), max_dim)
  img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
  img = kp_image.img_to_array(img)
  return np.expand_dims(img, axis=0)


def load_and_process_img(path_to_img):
  img = load_img(path_to_img)
  return tf.keras.applications.vgg19.preprocess_input(img)

def deprocess_img(processed_img):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)  
  # perform the inverse of the preprocessing step
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  return np.clip(x[:, :, ::-1], 0, 255).astype('uint8')


def get_model():
  # Load our model. We load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  # Get output layers corresponding to style and content layers 
  style_outputs = [vgg.get_layer(name).output for name in style_layers]
  content_outputs = [vgg.get_layer(name).output for name in content_layers]
  model_outputs = style_outputs + content_outputs
  # Build model 
  return models.Model(vgg.input, model_outputs)


def style_representations(model, style_path):
  style_image = load_and_process_img(style_path)
  style_outputs = model(style_image)
  style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
  return style_features


def get_content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
  # We make the image channels first 
  channels = int(input_tensor.shape[-1])
  a = tf.reshape(input_tensor, [-1, channels])
  n = tf.shape(a)[0]
  gram = tf.matmul(a, a, transpose_a=True)
  return gram / tf.cast(n, tf.float32)


# global variables
style_path =  'images/Vassily_Kandinsky_1913_-_Composition_7.jpg'#'images/antichrist1.jpg' 
# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 
# Style layer we are interested in
style_layers = ['block1_conv1','block2_conv1','block3_conv1', 'block4_conv1', 'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
tf.enable_eager_execution()
model = get_model() 
style_features = style_representations(model, style_path)
gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]


def get_style_loss(base_style, gram_target):
  height, width, channels = base_style.get_shape().as_list()
  gram_style = gram_matrix(base_style)
  
  return tf.reduce_mean(tf.square(gram_style - gram_target))


def content_representations(model, content_path):
  content_image = load_and_process_img(content_path)
  content_outputs = model(content_image)
  content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
  return content_features


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
  style_weight, content_weight = loss_weights
  model_outputs = model(init_image)
  style_output_features = model_outputs[:num_style_layers]
  content_output_features = model_outputs[num_style_layers:]  
  style_score = 0
  content_score = 0
  weight_per_style_layer = 1.0 / float(num_style_layers)
  for target_style, comb_style in zip(gram_style_features, style_output_features):
    style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
    
  # Accumulate content losses from all layers 
  weight_per_content_layer = 1.0 / float(num_content_layers)
  for target_content, comb_content in zip(content_features, content_output_features):
    content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)
  
  style_score *= style_weight
  content_score *= content_weight
  loss = style_score + 2 * content_score 
  return loss, style_score, content_score        


def compute_grads(cfg):
  with tf.GradientTape() as tape: 
    all_loss = compute_loss(**cfg) # Compute gradients wrt input image
  total_loss = all_loss[0]
  return tape.gradient(total_loss, cfg['init_image']), all_loss


def style_transfer_iter(cfg, num_iterations, init_image, best_loss, best_img, norm_means):
  min_vals = -norm_means
  max_vals = 255 - norm_means 
  opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)
  for i in range(num_iterations):
    grads, all_loss = compute_grads(cfg)
    loss, style_score, content_score = all_loss
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    if loss < best_loss: # Update best loss and best image from total loss. 
      best_loss = loss
      best_img = deprocess_img(init_image.numpy())
  return best_img, best_loss


def run_style_transfer(content_path, num_iterations=50, content_weight=1e3, style_weight=1e-2): 
  content_features = content_representations(model, content_path)
  init_image = load_and_process_img(content_path)
  init_image = tfe.Variable(init_image, dtype=tf.float32)
  loss_weights = (style_weight, content_weight)
  best_loss, best_img = float('inf'), None
  cfg = {
      'model': model,
      'loss_weights': loss_weights,
      'init_image': init_image,
      'gram_style_features': gram_style_features,
      'content_features': content_features
  }
  norm_means = np.array([103.939, 116.779, 123.68])  
  return style_transfer_iter(cfg, num_iterations, init_image, best_loss, best_img, norm_means)
  

## Turn images into stylized images
for i in range(0, count):
  start_time = time.time()
  best, best_loss = run_style_transfer(images[i], num_iterations=50)
  cv2.imwrite(images[i], best.astype('uint8')) 
  print('Frame', i, ': {:.4f}s'.format(time.time() - start_time))


## Convert images to video
frame = cv2.imread(images[1])
height, width, channels = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
filename = 'cat_output1.mp4'
out = cv2.VideoWriter(filename, fourcc, 29.0, (width, height))

for image in images:
    out.write(cv2.imread(image)) # Write out frame to video

out.release()
cv2.destroyAllWindows()
print("The output video is {}".format(filename))