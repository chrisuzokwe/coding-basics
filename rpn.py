class RegionProposalNetwork(nn.Module):

  def __init__(self, in_channels=256, mid_channels=256, ratios=[1], anchor_scales=[32, 48, 64], stride=16, mode="diagonal",
               
                anchor_gen_params = {
                      "ratios": [0.5, 1, 2],
                      "scales": [8, 16, 32],
                      "subsample": 8
                    },
                anchor_label_params = {
                      "neg_thresh": 0.3,
                      "pos_thresh": 0.7,
                      "n_sample": 256,
                      "pos_ratio": 0.5
                    },
                proposal_layer_params = {
                      "nms_thresh": 0.7,
                      "n_train_pre_nms": 12000,
                      "n_train_post_nms": 2000,
                      "n_test_pre_nms": 6000,
                      "n_test_post_nms": 300,
                      "min_size": 0,
                      "mode": "train"
                    }, 
               ):

      super(RegionProposalNetwork, self).__init__()

      self.anchor_gen_params = anchor_gen_params
      self.anchor_label_params = anchor_label_params
      self.proposal_layer_params = proposal_layer_params
      n_anchor = len(anchor_gen_params["ratios"])*len(anchor_gen_params["scales"])

      self.extractor = self.fe_init()

      self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
      self.reg_layer = nn.Conv2d(mid_channels, n_anchor *4, 1, stride=(1, 50), padding=0)
      self.cls_layer = nn.Conv2d(mid_channels, n_anchor *2, 1, stride=(1, 50), padding=0)
      # paper initialzes these layers with zero mean dn 0.01 standard deviation
      # conv sliding layer
      self.conv1.weight.data.normal_(0, 0.01)
      self.conv1.bias.data.zero_()

      # Regression layer
      self.reg_layer.weight.data.normal_(0, 0.01)
      self.reg_layer.bias.data.zero_()

      # classification layer
      self.cls_layer.weight.data.normal_(0, 0.01)
      self.cls_layer.bias.data.zero_()

      # Network Parameters
      #print(list(self.extractor.parameters()))
      #print(list(self.conv1.parameters()))
      #print(list(self.reg_layer.parameters()))
      #print(list(self.cls_layer.parameters()))

      #self.params = list(self.conv1.parameters()) + list(self.reg_layer.parameters()) + list(self.cls_layer.parameters()) #+ list(self.extractor.parameters())

  def forward(self, image, bbox, img_size, scale=16, device=torch.device("cpu")):

      # Anchors Generation
      anchors = anchor_target_generator(sub_sample=self.anchor_gen_params["subsample"], fe_size=50, ratios=self.anchor_gen_params["ratios"], scales=self.anchor_gen_params["scales"], mode="diagonal")
      ious = bbox_ious(bbox.cpu(), anchors)
      anchor_labels = assign_labels(ious, anchors, img_size, self.anchor_label_params["neg_thresh"], self.anchor_label_params["pos_thresh"], self.anchor_label_params["n_sample"], self.anchor_label_params["pos_ratio"])
      anchor_locations = bbox_to_relative(ious, anchors, bbox.cpu(), img_size)
   
      # Turn Generated Anchors and Labels into Tensor
      gt_rpn_loc = torch.from_numpy(anchor_locations)
      gt_rpn_score = torch.from_numpy(anchor_labels)

      # Kernel Transformation/Feature Extractor
      x = self.extractor(image)
      x = self.conv1(x)

      # prediction of object location with respect to the anchor and objectness scores
      pred_anchor_locs = self.reg_layer(x)
      pred_cls_scores = self.cls_layer(x)

      # Reformat anchor targets to match our anchor target sizes
      pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)

      pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
      pred_cls_scores  = pred_cls_scores.view(1, -1, 2)

      rpn_loc = pred_anchor_locs[0]
      rpn_score = pred_cls_scores[0]

      #rpn_score = rpn_score
      rpn_score = rpn_score.to(device)

      #rpn_loc = rpn_loc
      rpn_loc = rpn_loc.to(device)

      #gt_rpn_score = gt_rpn_score
      gt_rpn_score = gt_rpn_score.to(device)

      #gt_rpn_loc = gt_rpn_loc
      gt_rpn_loc = gt_rpn_loc.to(device)
      #rpn_loss = self.loss(rpn_loc, rpn_score, gt_rpn_loc, gt_rpn_score)

      return rpn_loc, rpn_score, gt_rpn_loc, gt_rpn_score

  def forwardcounts(self, image, bbox, img_size, scale=16, device=torch.device("cpu")):

    #print(len(bbox))
    # Anchors Generation
    anchors = anchor_target_generator(sub_sample=self.anchor_gen_params["subsample"], fe_size=50, ratios=self.anchor_gen_params["ratios"], scales=self.anchor_gen_params["scales"], mode="diagonal")
    ious = bbox_ious(bbox.cpu(), anchors)
    anchor_labels = assign_labels(ious, anchors, img_size, self.anchor_label_params["neg_thresh"], self.anchor_label_params["pos_thresh"], self.anchor_label_params["n_sample"], self.anchor_label_params["pos_ratio"])
    anchor_locations = bbox_to_relative(ious, anchors, bbox.cpu(), img_size)
  
    # Turn Generated Anchors and Labels into Tensor
    gt_rpn_loc = torch.from_numpy(anchor_locations)
    gt_rpn_score = torch.from_numpy(anchor_labels)

    # Kernel Transformation/Feature Extractor
    x = self.extractor(image)
    x = self.conv1(x)

    # prediction of object location with respect to the anchor and objectness scores
    pred_anchor_locs = self.reg_layer(x)
    pred_cls_scores = self.cls_layer(x)

    # Reformat anchor targets to match our anchor target sizes
    pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)

    pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
    pred_cls_scores  = pred_cls_scores.view(1, -1, 2)

    rpn_loc = pred_anchor_locs[0]
    rpn_score = pred_cls_scores[0]

    #rpn_score = rpn_score
    rpn_score = rpn_score.to(device)

    #rpn_loc = rpn_loc
    rpn_loc = rpn_loc.to(device)

    #gt_rpn_score = gt_rpn_score
    gt_rpn_score = gt_rpn_score.to(device)

    #gt_rpn_loc = gt_rpn_loc
    gt_rpn_loc = gt_rpn_loc.to(device)
    #rpn_loss = self.loss(rpn_loc, rpn_score, gt_rpn_loc, gt_rpn_score)

    return anchors, ious, anchor_labels, anchor_locations

  def predict(self, image, device=torch.device("cpu")):

      anchors = anchor_target_generator(sub_sample=self.anchor_gen_params["subsample"], fe_size=50, ratios=self.anchor_gen_params["ratios"], scales=self.anchor_gen_params["scales"], mode="diagonal")
      #tensor = tr.to_tensor(image)
      #tensor = tensor.reshape(1, 3, 800, 800)
      tensor = image.to(device)

      x = self.extractor(tensor)
      x = self.conv1(x)

      # print("pred: extacted")
      # prediction of object location with respect to the anchor and objectness scores
      pred_anchor_locs = self.reg_layer(x)
      pred_cls_scores = self.cls_layer(x)
      #print("pred: prediction")

      #locs = [x for _,x in sorted(zip(pred_cls_scores, pred_anchor_locs))]
      #scores = sorted(pred_cls_scores)

      # Reformat anchor targets to match our anchor target sizes
      pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
      #print(pred_anchor_locs.shape)

      # pred_cls_scores  = pred_cls_scores.view(1, -1, 2) <-- for softmax classification
      objectness_score = pred_cls_scores.view(1, 50, 1, len(self.anchor_gen_params["ratios"])*len(self.anchor_gen_params["scales"]), 2)[:, :, :, :, 1].contiguous().view(1, -1)
      
      # convert predictions using the same formulas above
      objectness_score_numpy = objectness_score[0].cpu().data.numpy()

      roi = relative_to_bbox(anchors, pred_anchor_locs.cpu())
      roi, score = proposal_layer(roi, objectness_score_numpy, nms_thresh=self.proposal_layer_params["nms_thresh"], n_test_post_nms=self.proposal_layer_params["n_test_post_nms"], n_test_pre_nms=self.proposal_layer_params["n_test_pre_nms"], n_train_post_nms=self.proposal_layer_params["n_train_post_nms"], n_train_pre_nms=self.proposal_layer_params["n_train_pre_nms"], min_size=self.proposal_layer_params["min_size"], mode=self.proposal_layer_params["mode"])

      return roi, score

  def predict_test(self, image, device=torch.device("cpu")):

      anchors = anchor_target_generator(sub_sample=self.anchor_gen_params["subsample"], fe_size=50, ratios=self.anchor_gen_params["ratios"], scales=self.anchor_gen_params["scales"], mode="diagonal")
      #tensor = tr.to_tensor(image)
      #tensor = tensor.reshape(1, 3, 800, 800)
      tensor = image.to(device)

      x = self.extractor(tensor)
      x = self.conv1(x)

      # print("pred: extacted")
      # prediction of object location with respect to the anchor and objectness scores
      pred_anchor_locs = self.reg_layer(x)
      pred_cls_scores = self.cls_layer(x)
      #print("pred: prediction")

      #locs = [x for _,x in sorted(zip(pred_cls_scores, pred_anchor_locs))]
      #scores = sorted(pred_cls_scores)

      # Reformat anchor targets to match our anchor target sizes
      pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
      #print(pred_anchor_locs.shape)

      # pred_cls_scores  = pred_cls_scores.view(1, -1, 2) <-- for softmax classification
      objectness_score = pred_cls_scores.view(1, 50, 1, len(self.anchor_gen_params["ratios"])*len(self.anchor_gen_params["scales"]), 2)[:, :, :, :, 1].contiguous().view(1, -1)
      
      # convert predictions using the same formulas above
      objectness_score_numpy = objectness_score[0].cpu().data.numpy()

      allroi = relative_to_bbox(anchors, pred_anchor_locs.cpu())
      roi, score = proposal_layer(allroi, objectness_score_numpy, nms_thresh=self.proposal_layer_params["nms_thresh"], n_test_post_nms=self.proposal_layer_params["n_test_post_nms"], n_test_pre_nms=self.proposal_layer_params["n_test_pre_nms"], n_train_post_nms=self.proposal_layer_params["n_train_post_nms"], n_train_pre_nms=self.proposal_layer_params["n_train_pre_nms"], min_size=self.proposal_layer_params["min_size"], mode=self.proposal_layer_params["mode"])

      return roi, score, allroi, objectness_score_numpy, anchors

  def fe_init(self):
      # initialize resnet

      from torchvision import models
      resnet18 = models.resnet18(pretrained=False)

      resnet_layers = []
      resnet_layers.append(resnet18.conv1)
      resnet_layers.append(resnet18.bn1)
      resnet_layers.append(resnet18.relu)
      resnet_layers.append(resnet18.maxpool)
      resnet_layers.append(resnet18.layer1)
      resnet_layers.append(resnet18.layer2) #--> torch.Size([1, 128, 100, 100])
      resnet_layers.append(resnet18.layer3) #--> torch.Size([1, 256, 50, 50])
      # resnet_layers.append(resnet18.layer4)
      extractor = nn.Sequential(*resnet_layers)

      #print(new_layers[0])
      #extractor = nn.Sequential(*new_layers)

      for param in extractor.parameters():
        param.requires_grad = True

      return extractor

# Loss
def rpn_loss(rpn_loc, rpn_score, gt_rpn_loc, gt_rpn_score, rpn_lambda=10):

  rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_score.long(), ignore_index = -1)
  #print(rpn_cls_loss)
  pos = gt_rpn_score > 0
  mask = pos.unsqueeze(1).expand_as(rpn_loc)

  # extract bounding boxes from positive labels
  mask_loc_preds = rpn_loc[mask].view(-1, 4)
  mask_loc_targets = gt_rpn_loc[mask].view(-1, 4)

  x = torch.abs(mask_loc_targets - mask_loc_preds)
  rpn_loc_loss = ((x < 1).float() * 0.5 * x**2) + ((x >= 1).float() * (x-0.5))

  # apply loss
  x = torch.abs(mask_loc_targets - mask_loc_preds)
  rpn_loc_loss = ((x < 1).float() * 0.5 * x**2) + ((x >= 1).float() * (x-0.5))

  # Combine and apply our class loss, using a regularization parameter
  N_reg = (gt_rpn_score >0).float().sum()
  rpn_loc_loss = rpn_loc_loss.sum() / N_reg
  rpn_loss = rpn_cls_loss + (rpn_lambda * rpn_loc_loss)

  #print("rpn_loc_loss:", rpn_loc_loss, "rpn_cls_loss:", rpn_cls_loss)

  return rpn_cls_loss, (rpn_lambda * rpn_loc_loss), rpn_loss
  
def load_rpn(subsample, scales, thresholds, samples, ratio, model_path):

  #load rpn
  anchor_gen_params = {
    "ratios": [1],
    "scales": scales,
    "subsample": subsample
  }

  anchor_label_params = {
    "neg_thresh": thresholds[1],
    "pos_thresh": thresholds[0],
    "n_sample": sample,
    "pos_ratio": ratio
  }

  proposal_layer_params = {
    "nms_thresh": 0.0,
    "n_train_pre_nms": 12000,
    "n_train_post_nms": 2000,
    "n_test_pre_nms": 6000,
    "n_test_post_nms": 300,
    "min_size": 0,
    "mode": "train"
  }

  rpn_lambda = 100

  # initialize device and load rpn with parameters or (optional) weights
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  rpn = RegionProposalNetwork(anchor_gen_params=anchor_gen_params, anchor_label_params=anchor_label_params, proposal_layer_params=proposal_layer_params)
  rpn.load_state_dict(torch.load(model_path, map_location=device))
  rpn = rpn.to(device)

  return rpn, device, anchor_label_params