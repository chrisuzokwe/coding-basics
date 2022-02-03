import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_boundaries(image, xyxy_bounds, figsize=(10, 10)):

  fig, ax = plt.subplots(figsize=figsize)

  #image = Image.open(image_path)
  
  ax.imshow(image)

  for xyxy in xyxy_bounds: 

    rect = patches.Rectangle((xyxy[0], xyxy[1]), xyxy[3]-xyxy[1], xyxy[2]-xyxy[0], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

  plt.pause(0.001)

  #return image


#def make_grid():
	