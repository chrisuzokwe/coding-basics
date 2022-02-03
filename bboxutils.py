import numpy as np

def bbox_to_relative(ious, anchors, gt, img_shape):

  index_inside = np.where(
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] <= img_shape[1]) &
        (anchors[:, 3] <= img_shape[0])
    )[0]

  argmax_ious = ious.argmax(axis=1)[index_inside]
  max_iou_bbox = gt[argmax_ious]
  valid_anchors = anchors[index_inside]

  # conversion of our x1, x2, y1, y2 format to ctr, xy format
  height = valid_anchors[:, 2] - valid_anchors[:, 0]
  width = valid_anchors[:, 3] - valid_anchors[:, 1]
  ctr_y = valid_anchors[:, 0] + 0.5 * height
  ctr_x = valid_anchors[:, 1] + 0.5 * width

  base_height = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
  base_width = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
  base_ctr_y = max_iou_bbox[:, 0] + 0.5 * base_height
  base_ctr_x = max_iou_bbox[:, 1] + 0.5 * base_width

  #calculate location
  eps = np.finfo(height.dtype).eps # displays machine value limits for the type of variable height is
  height = np.maximum(height, eps) 
  width = np.maximum(width, eps)

  dy = (base_ctr_y - ctr_y) / height # calculate respective distance using formula from above
  dx = (base_ctr_x - ctr_x) / width
  dh = np.log(base_height / height)
  dw = np.log(base_width / width)

  anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()
  #print(anchor_locs)

  # final locations
  anchor_locations = np.empty((len(anchors),) + anchors.shape[1:], dtype=anchor_locs.dtype)
  anchor_locations.fill(0)
  anchor_locations[index_inside, :] = anchor_locs

  return anchor_locations


def relative_to_bbox(anchors, pred):

  # convert anchors from x1 y1 x2 y2 to ctrx, ctry, h, w
  anc_height = anchors[:, 2] - anchors[:, 0]
  anc_width = anchors[:, 3] - anchors[:, 1]
  anc_ctr_y = anchors[:, 0] + 0.5 * anc_height
  anc_ctr_x = anchors[:, 1] + 0.5 * anc_width

  # Reformat anchor targets to match our anchor target sizes
  pred_anchor_locs = pred.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)

  # convert predictions using the same formulas above
  pred_anchor_locs_numpy = pred_anchor_locs[0].data.numpy()

  dy = pred_anchor_locs_numpy[:, 0::4]
  dx = pred_anchor_locs_numpy[:, 1::4]
  dh = pred_anchor_locs_numpy[:, 2::4]
  dw = pred_anchor_locs_numpy[:, 3::4]

  #print("anchor targ:", dy.shape, anc_height[:, np.newaxis].shape, anc_ctr_y[:, np.newaxis].shape)
  ctr_y = dy * anc_height[:, np.newaxis] + anc_ctr_y[:, np.newaxis]
  ctr_x = dx * anc_width[:, np.newaxis] + anc_ctr_x[:, np.newaxis]
  h = np.exp(dh) * anc_height[:, np.newaxis]
  w = np.exp(dw) * anc_width[:, np.newaxis]  

  # convert center points to y1 x1 x2 y2 format
  roi = np.zeros(pred_anchor_locs_numpy.shape, dtype=anchors.dtype)
  roi[:, 0::4] = ctr_y - 0.5 * h
  roi[:, 1::4] = ctr_x - 0.5 * w
  roi[:, 2::4] = ctr_y + 0.5 * h
  roi[:, 3::4] = ctr_x + 0.5 * w

  return roi


def anchor_target_generator(sub_sample=16, fe_size=50, ratios=[0.5, 1, 2], scales=[8, 16, 32], mode="all"):

  # generate anchor targets by enumerating ratios and scales.

  # IN: (int): width and height of the base anchor
  #     (int): fe_size - size of image from feature extractor
  #     (list): ratios of width to height of anchors
  #     (list): scales

  # OUT: (numpy.ndarray): ((len(ratios)*len(scales)), 4)
  #       array of bounding box coordinates built from ratios and scales

  #print(ratios)
  #print(scales)
  n_anchors = len(ratios)*len(scales)
  # generate center points for all anchors
  ctr_x = np.arange(sub_sample, (fe_size+1)*sub_sample, sub_sample)
  ctr_y = np.arange(sub_sample, (fe_size+1)*sub_sample, sub_sample)
  
  
  if mode =="all":
      
      ctr = np.zeros((len(ctr_x) * len(ctr_x), 2))
    
      index = 0
      for x in range(len(ctr_x)):
          for y in range(len(ctr_y)):
              ctr[index, 0] = ctr_x[x] - 8
              ctr[index, 1] = ctr_y[y] - 8
              index +=1
  
  elif mode == "diagonal":
      
    ctr = np.zeros((len(ctr_x), 2))
    
    for i in range(len(ctr)):
      ctr[i][0] = ctr_x[i] - 8
      ctr[i][1] = ctr_x[i] - 8
      

  # calculate and fill anchor box array
  anchors = np.zeros((len(ctr) * n_anchors, 4), dtype=np.float32)

  index = 0
  for c in ctr:
    ctr_y, ctr_x = c
    for i in range(len(ratios)):
      for j in range(len(scales)):
        h = sub_sample * scales[j] * np.sqrt(ratios[i])
        w = sub_sample * scales[j] * np.sqrt(1./ ratios[i])
        anchors[index, 0] = ctr_y - h / 2.
        anchors[index, 1] = ctr_x - w / 2.
        anchors[index, 2] = ctr_y + h / 2.
        anchors[index, 3] = ctr_x + w / 2.
        index += 1

  return anchors
  # by looping through each scale and ratio we create the points that exist in one anchor box.


def bbox_ious(gt, pred):

  # calculate the intersection over union of each anchor over the ground truth boxes (2) by creating a box using:
  # the max of x1 and y1 in both boxes
  # the min of x2 and y2 in both boxes
  ious = np.empty((len(pred), len(gt)), dtype=np.float32) #empty iou totals array
  ious.fill(0)
  #print(gt)

  for num1, i in enumerate(pred): # enumerate goes through elements in the list while having a counter that can start from anywhere
      ya1, xa1, ya2, xa2 = i  
      anchor_area = (ya2 - ya1) * (xa2 - xa1)
      for num2, j in enumerate(gt): 
         # print(j)
          yb1, xb1, yb2, xb2 = j
          box_area = (yb2- yb1) * (xb2 - xb1)
          inter_x1 = max([xb1, xa1])
          inter_y1 = max([yb1, ya1])
          inter_x2 = min([xb2, xa2])
          inter_y2 = min([yb2, ya2])
          if (inter_x1 < inter_x2) and (inter_y1 < inter_y2): # check: if these boxes overlap then the area is the IOU if not the IOU is 0
              iter_area = (inter_y2 - inter_y1) * \
  (inter_x2 - inter_x1)
              iou = iter_area / \
  (anchor_area+ box_area - iter_area)            
          else:
              iou = 0.
              #print(num1, num2)
          ious[num1, num2] = iou

  return ious

def assign_labels(ious, anchors, img_shape, neg_thresh=0.3, pos_thresh=0.7, n_sample=256, pos_ratio=0.5):

  index_inside = np.where(
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] <= img_shape[1]) &
        (anchors[:, 3] <= img_shape[0])
    )[0]
    
  disable_diag_idx = np.where(
        (anchors[:, 0] != anchors[:, 1]) &
        (anchors[:, 2] != anchors[:, 3])
    )[0]

  # create an empty label array
  label = np.empty((len(index_inside), ), dtype=np.int32)
  label.fill(-1)

  ious = ious[index_inside]

  #print(ious)
  # get index and value of larget IoU per ground truth bbox (resulting in len(ground truth))
  gt_argmax_ious = ious.argmax(axis=0)
  gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]

  # get index and value of larget IoU per anchor box (resulting in len(anchors))
  argmax_ious = ious.argmax(axis=1)
  max_ious = ious[np.arange(len(index_inside)), argmax_ious] # for each anchor box return an array specifying which anchor box is larger

  # find anchor boxes that have the same value as the max IoUs
  gt_argmax_ious = np.where(ious == gt_max_ious)[0]

  label[max_ious < neg_thresh] = 0
  label[gt_argmax_ious] = 1
  label[max_ious >= pos_thresh] = 1

  n_pos = pos_ratio * n_sample

  pos_index = np.where(label == 1)[0] # return indices of positive labels
  #print(pos_index)
  #print(len(pos_index) - n_pos)

  if len(pos_index) > n_pos: # if there are more positive samples than 1:1 ratio, disable random samples
      disable_index = np.random.choice(pos_index, size=(int(len(pos_index) - n_pos)), replace=False)
      label[disable_index] = -1

  n_neg = n_sample - np.sum(label == 1)
  neg_index = np.where(label == 0)[0]

  if len(neg_index) > n_neg:
      disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace = False)
      label[disable_index] = -1

  anchor_labels = np.empty((len(anchors),), dtype=label.dtype)
  anchor_labels.fill(-1)
  anchor_labels[index_inside] = label
  #anchor_labels[disable_diag_idx] = 0

  return anchor_labels


def proposal_layer(roi, objectness_score_numpy, img_size=[800,800],nms_thresh=0.7, n_train_pre_nms = 12000, n_train_post_nms = 2000, n_test_pre_nms = 6000, n_test_post_nms = 300, min_size = 16, mode="train"):
  
  if mode == "train":
    pre_trunc = n_train_pre_nms
    post_trunc = n_train_post_nms

  elif mode == "test":
    pre_trunc = n_test_pre_nms
    post_trunc = n_test_pre_nms

  # clip boxes to the image i.e if its larger than image make its bound the border of the image
  img_size = (800, 800) #Image size
  roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
  roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

  # remove boxes below threshold
  hs = roi[:, 2] - roi[:, 0]
  ws = roi[:, 3] - roi[:, 1]
  keep = np.where((hs >= min_size) & (ws >= min_size))[0]
  roi = roi[keep, :]
  score = objectness_score_numpy[keep]

  # insert proposal score pairs from highest to lowest
  order = score.ravel().argsort()[::-1]
  order = order[:pre_trunc] # cut down predictions to pre_nms
  roi = roi[order, :]

  score = score[order]

  # Apply non-maximum supression while using the top proposals
  y1 = roi[:, 0]
  x1 = roi[:, 1]
  y2 = roi[:, 2]
  x2 = roi[:, 3]

  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  order = score.argsort()[::-1]

  keep = []

  while order.size > 0:
      i = order[0]
      keep.append(i)
      xx1 = np.maximum(x1[i], x1[order[1:]])
      yy1 = np.maximum(y1[i], y1[order[1:]])
      xx2 = np.minimum(x2[i], x2[order[1:]])
      yy2 = np.minimum(y2[i], y2[order[1:]])
      
      w = np.maximum(0.0, xx2 - xx1 + 1)
      h = np.maximum(0.0, yy2 - yy1 + 1)
      
      inter = w * h
      ovr = inter / (areas[i] + areas[order[1:]] - inter)
      
      inds = np.where(ovr <= nms_thresh)[0]
      order = order[inds + 1]
      
  keep = keep[:post_trunc] # while training/testing , use accordingly
  roi = roi[keep] # the final region proposals
  score = score[keep]

  return roi, score