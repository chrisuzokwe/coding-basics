class FasterRCNN(nn.Module):
  def __init__(self):

    super(FasterRCNN, self).__init__()

    self.rpn = RegionProposalNetwork()
    self.classifier = PoolingCNN()


  def forward(self, gt_bounds, image, labels, device="cpu"):

    rpn_loc, rpn_score, gt_rpn_loc, gt_rpn_score, out_map, roi = self.rpn.forward(image=image, bbox=gt_bounds, img_size=[800,800], device=device)
    #rpn_loc=rpn_loc.to(device)
    #print("RPN LOCS:", rpn_loc)
    #print("GT roi:", gt_bounds)

    
    gt_roi_labels, gt_roi_locs, roi_cls_loc, roi_cls_score = self.classifier.forward(roi, gt_bounds, labels, out_map)
   

    return gt_roi_labels, gt_roi_locs, rpn_loc, rpn_score, roi_cls_loc, roi_cls_score, gt_rpn_score, gt_rpn_loc

  def predict(self, image, device="cpu"):
    roi, score, out_map = self.rpn.predict(image, device=device)
    roi_cls_loc, roi_cls_score, bboxes, corr_bound = self.classifier.predict(out_map, torch.from_numpy(roi))

    return roi_cls_loc, roi_cls_score, bboxes, corr_bound


def sample_rois(gt_bounds, pred_bounds, labels,  n_sample=128, pos_ratio=0.5, pos_iou_thresh=0.5, neg_iou_thresh=0.5, neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0):

  ious = bbox_ious(gt_bounds, pred_bounds)

  gt_assignment = ious.argmax(axis=1)
  #print("GROUND TRUTH ASSIGNMENT:", gt_assignment)
  max_iou = ious.max(axis=1)
  #print(labels.shape)
  gt_roi_label = labels[gt_assignment]

  # randomly sample positive examples to achieve the predefined ratio as before
  pos_roi_per_image = int(n_sample * pos_ratio)
  pos_index = np.where(max_iou >= pos_iou_thresh)[0]
  pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
  if pos_index.size > 0:
      pos_index = np.random.choice(
          pos_index, size=pos_roi_per_this_image, replace=False)

  # same thing for our negative examples
  neg_index = np.where((max_iou < neg_iou_thresh_hi) &
                              (max_iou >= neg_iou_thresh_lo))[0]
  neg_roi_per_this_image = n_sample - pos_roi_per_this_image
  neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                  neg_index.size))
  if  neg_index.size > 0:
      neg_index = np.random.choice(
          neg_index, size=neg_roi_per_this_image, replace=False)

  # concatenate our examples...
  keep_index = np.append(pos_index, neg_index)
  gt_roi_labels = gt_roi_label[keep_index]
  gt_roi_labels[pos_roi_per_this_image:] = 0  # negative labels --> 0
  sample_roi = pred_bounds[keep_index]
  #print(sample_roi.shape)

  bbox_for_sampled_roi = gt_bounds[gt_assignment[keep_index]]
  #print(bbox_for_sampled_roi.shape)

  height = sample_roi[:, 2] - sample_roi[:, 0]
  width = sample_roi[:, 3] - sample_roi[:, 1]
  ctr_y = sample_roi[:, 0] + 0.5 * height
  ctr_x = sample_roi[:, 1] + 0.5 * width

  base_height = bbox_for_sampled_roi[:, 2] - bbox_for_sampled_roi[:, 0]
  base_width = bbox_for_sampled_roi[:, 3] - bbox_for_sampled_roi[:, 1]
  base_ctr_y = bbox_for_sampled_roi[:, 0] + 0.5 * base_height
  base_ctr_x = bbox_for_sampled_roi[:, 1] + 0.5 * base_width

  # use the final formulation to calculate the maximum bounds
  #print(height.dtype)
  eps = np.finfo(height.detach().numpy().dtype).eps
  height = np.maximum(height.detach().numpy(), eps)
  width = np.maximum(width.detach().numpy(), eps)

  dy = (base_ctr_y - ctr_y).detach().numpy() / height
  dx = (base_ctr_x - ctr_x).detach().numpy() / width
  dh = np.log(base_height / height).detach().numpy()
  dw = np.log(base_width / width).detach().numpy()

  gt_roi_locs = np.vstack((dy, dx, dh, dw)).transpose()
  #print(sample_roi)

  return gt_roi_labels, gt_roi_locs, sample_roi


def pooling_layer(sample_roi, out_map):

  # convert our sample_rois into numpy array
  rois = sample_roi.float()
  #print(rois[0:4])
  roi_indices = 0 * np.ones((len(rois),), dtype=np.int32)
  roi_indices = torch.from_numpy(roi_indices).float()

  # concat roi and their indexes
  indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
  xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
  indices_and_rois = xy_indices_and_rois.contiguous()

  # create input to classification layer
  size = 7
  adaptive_max_pool = nn.AdaptiveMaxPool2d(size)

  output = []
  rois = indices_and_rois.data.float()
  rois[:, 1:].mul_(1/16.0) # Subsampling ratio
  rois = rois.long()
  num_rois = rois.size(0)
  print("num rois:", num_rois)

  for i in range(num_rois):
      roi = rois[i]
      im_idx = roi[0]
      #print(roi[1:])
      im = out_map.narrow(0, im_idx, 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
      #print(im.shape)
      output.append(adaptive_max_pool(im))
      
  output = torch.cat(output, 0)

  # Reshape the tensor so that we can pass it through the feed forward layer.
  k = output.view(output.size(0), -1)
  print(k.shape)
  return k

def roi_to_bbox(anchors, pred_anchor_locs_numpy):
  print(pred_anchor_locs_numpy)

  # convert anchors from x1 y1 x2 y2 to ctrx, ctry, h, w
  anc_height = anchors[:, 2] - anchors[:, 0]
  anc_width = anchors[:, 3] - anchors[:, 1]
  anc_ctr_y = anchors[:, 0] + 0.5 * anc_height
  anc_ctr_x = anchors[:, 1] + 0.5 * anc_width

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
  roi = np.zeros(pred_anchor_locs_numpy.shape, dtype=pred_anchor_locs_numpy.dtype)
  roi[:, 0::4] = ctr_y - 0.5 * h
  roi[:, 1::4] = ctr_x - 0.5 * w
  roi[:, 2::4] = ctr_y + 0.5 * h
  roi[:, 3::4] = ctr_x + 0.5 * w  

  return roi
  
 
 
class PoolingCNN(nn.Module):

  def __init__(self, 
               sample_layer_params = {
                      "n_sample": 128,
                      "pos_ratio": 0.85,
                      "pos_iou_thresh": 0.9,
                      "neg_iou_thresh": 0.5,
                      "neg_iou_thresh_hi": 0.3,
                      "neg_iou_thresh_lo": 0.0,
                      "mode": "train"
                    },
               ):
    
    super(PoolingCNN, self).__init__()

    self.roi_head_classifier = nn.Sequential(*[nn.Linear(12544, 4096),
                                      nn.Linear(4096, 4096)])
    self.cls_loc = nn.Linear(4096, 2 * 4) # (VOC 20 classes + 1 background. Each will have 4 co-ordinates)
    self.cls_loc.weight.data.normal_(0, 0.01)
    self.cls_loc.bias.data.zero_()

    self.score = nn.Linear(4096, 2) # (VOC 20 classes + 1 background)

  def forward(self, pred_bounds, gt_bounds, labels, out_map):

    # print("pred_bounds", pred_bounds)
    gt_roi_labels, gt_roi_locs, sample_roi = sample_rois(gt_bounds, pred_bounds, labels)
    k = pooling_layer(sample_roi, out_map)
    #print(k.shape)
    #print(k.type)

    k = self.roi_head_classifier(k)
    #print(k.shape)

    roi_cls_loc = self.cls_loc(k)
    roi_cls_score = self.score(k)
    print(roi_cls_loc)

    return gt_roi_labels, gt_roi_locs, roi_cls_loc, roi_cls_score

  def predict(self, out_map, roi):

    #print(roi.shape)
    k = pooling_layer(roi, out_map)
    k = self.roi_head_classifier(k)
    roi_cls_loc = self.cls_loc(k)
    roi_cls_score = self.score(k)

    index = np.array(roi_cls_score.detach().argmax(axis=1))
    #print("index", index)
    #

    print(roi_cls_loc)
    corr_bound = np.asarray([roi_cls_loc[i, (index[i]*4):(index[i]*4+4)].detach().numpy() for i in range(len(roi_cls_loc))])

    #for i in range(len(roi_cls_loc)):
    #  print(roi_cls_loc[i, (index[i]*4):(index[i]*4+4)])

    #print(corr_bound.shape)
    
    #print(corr_bound)
    bboxes = roi_to_bbox(np.asarray(roi), corr_bound)
    #print(bboxes)

    return roi_cls_loc, roi_cls_score, bboxes, corr_bound


def rcnn_loss(gt_roi_locs, gt_roi_labels, roi_cls_score, roi_cls_loc, gt_rpn_score):

    print(roi_cls_score.shape)
    print(roi_cls_loc.shape)

    # convert ground truth labels and regions of interest to torch variables
    gt_roi_loc = gt_roi_locs
    gt_roi_label = np.float32(gt_roi_labels.long())


    # calculate our classification loss
    roi_cls_loss = F.cross_entropy(roi_cls_score, gt_roi_labels.long(), ignore_index=-1)

    # calculate regression loss using positive examples
    n_sample = roi_cls_loc.shape[0]
    roi_loc = roi_cls_loc.view(n_sample, -1, 4)

    roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_label]

    x_roi = torch.abs(torch.from_numpy(gt_roi_loc) - roi_loc)
    roi_loc_loss = ((x_roi < 1).float() * 0.5 * x_roi ** 2) + ((x_roi >= 1).float() * (x_roi - 0.5))

    # total roi loss
    roi_lambda = 10.
    N_reg_roi = (gt_rpn_score > 0).float().sum()
    roi_loc_loss = roi_loc_loss.sum() / N_reg_roi
    print("roi_cls:", roi_cls_loss, "roi_loc_loss:", roi_loc_loss)
    roi_loss = roi_cls_loss + (roi_lambda * roi_loc_loss)
    return roi_loss