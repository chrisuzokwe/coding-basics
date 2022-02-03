import time
import csv

def train_rpn(rpn, rpn_lambda, imgtrain, lbltrain, imgval, lblval, sclval, model_path, eval_path, training_path):

  img = imgtrain
  labels = lbltrain

  epochs = 2
  lr = .001
  optimizer = optim.Adam(rpn.parameters(), lr=lr)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=550, gamma=0.9)

  t0 = time.time()
  total_loss = 0
  total_cls_loss = 0
  total_loc_loss = 0

  # Set Up Logger (Evaluation)
  ############################
  with open(eval_path, 'w', newline='') as csvfile:
    fieldnames = ['p3trim', 'p5', 'p5trim', 'r3trim', 'r5', 'r5trim',  'f13trim', 'f15', 'f15trim']
    writerP = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writerP.writeheader()

    # Set Up Logger (Training)
    ##########################
    with open(training_path, 'w', newline='') as csvfile:
      fieldnames = ['epoch', 'iter', 'total_loss', 'cls_loss', 'loc_loss', 'lr', 'precision', 'recall', 'f1', 'average_iou']
      writerT = csv.DictWriter(csvfile, fieldnames=fieldnames)
      writerT.writeheader()

      for i in range(epochs):
        for j in range(len(labels)):

            ################## Forward
            optimizer.zero_grad()
            sampleimg = img[j].reshape(3, 800, 800).float().unsqueeze(0)
            rpn_loc, rpn_score, gt_rpn_loc, gt_rpn_score = rpn(image=sampleimg.to(device), bbox=torch.from_numpy(labels[j]).to(device), img_size=[800,800], device=device)
            cls_loss, loc_loss, loss = rpn_loss(rpn_loc, rpn_score, gt_rpn_loc, gt_rpn_score, rpn_lambda=rpn_lambda)

            ################# Log
            iter = i*len(labels) + j+1
            total_loss = total_loss + loss.item()
            total_cls_loss = total_cls_loss + cls_loss.item()
            total_loc_loss = total_cls_loss + loc_loss.item()
            print('iter', iter, 'cls_loss', cls_loss.item(), 'loc_loss:', loc_loss.item(), 'loss:', loss.item(), "lr:", optimizer.param_groups[0]['lr'])

            if not (iter%50):
              #print('iter', iter, 'cls_loss', cls_loss.item(), 'loc_loss:', loc_loss.item(), 'loss:', loss.item(), 'total_loss:', total_loss/50, "lr:", optimizer.param_groups[0]['lr'])
              prf3, prf3trim, prf5, prf5trim, ious = predict_set(imgval, lblval, sclval, rpn)
              writerP.writerow({'p3trim': prf3trim[0], 'p5': prf5[0], 'p5trim': prf5trim[0], 'r3trim':prf3trim[1], 'r5': prf5[1], 'r5trim':prf5trim[1],  'f13trim': prf3trim[2], 'f15': prf5[2], 'f15trim': prf5trim[2]})
              writerT.writerow({'epoch': i+1, 'iter': iter, 'total_loss': total_loss/50, 'cls_loss': total_cls_loss/50, 'loc_loss': total_loc_loss/50, 'lr': optimizer.param_groups[0]['lr'], 'precision': prf3[0] , 'recall': prf3[1], 'f1':prf3[2], 'average_iou': ious})
              total_loss = 0
              total_cls_loss = 0
              total_loc_loss = 0

            ################# Backward
            loss.backward()
            optimizer.step()
            scheduler.step()
          
        if not ((i+1)%1):
          torch.save(rpn.state_dict(), model_path + "state_dict_epoch_" +str(i+1))

      t1 = time.time()

      with open(model_path + 'time', 'w', newline='') as csvfile:
        fieldnames = ['training_time']
        writerC = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writerC.writeheader()
        writerC.writerow({'training_time': t1-t0})

      print("trained in", t1-t0, "seconds")

      for param in rpn.extractor.parameters():
        param.requires_grad = False

      epochs = 8
      lr = .001
      optimizer = optim.Adam(rpn.parameters(), lr=lr)
      scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=550, gamma=0.6)

      t0 = time.time()
      total_loss = 0

      for i in range(epochs):
        for j in range(len(labels)):

            # Forward
            optimizer.zero_grad()
            sampleimg = img[j].reshape(3, 800, 800).float().unsqueeze(0)
            rpn_loc, rpn_score, gt_rpn_loc, gt_rpn_score = rpn(image=sampleimg.to(device), bbox=torch.from_numpy(labels[j]).to(device), img_size=[800,800], device=device)
            cls_loss, loc_loss, loss = rpn_loss(rpn_loc, rpn_score, gt_rpn_loc, gt_rpn_score, rpn_lambda=rpn_lambda)

            # Log
            iter = i*len(labels) + j+1
            total_loss = total_loss + loss.item()
            cls_loss = cls_loss + cls_loss.item()
            loc_loss = loc_loss + loc_loss.item()

            #if loss.item() > 50:
            print('iter', iter, 'cls_loss', cls_loss.item(), 'loc_loss:', loc_loss.item(), 'loss:', loss.item(), "lr:", optimizer.param_groups[0]['lr'])

            if not (iter%50):
              #print('iter', iter, 'cls_loss', cls_loss.item(), 'loc_loss:', loc_loss.item(), 'loss:', loss.item(), 'total_loss:', total_loss/50, "lr:", optimizer.param_groups[0]['lr'])
              prf3, prf3trim, prf5, prf5trim, ious = predict_set(imgval, lblval, sclval, rpn)
              writerP.writerow({'p3trim': prf3trim[0], 'p5': prf5[0], 'p5trim': prf5trim[0], 'r3trim':prf3trim[1], 'r5': prf5[1], 'r5trim':prf5trim[1],  'f13trim': prf3trim[2], 'f15': prf5[2], 'f15trim': prf5trim[2]})
              writerT.writerow({'epoch': i+1, 'iter': iter, 'total_loss': total_loss/50, 'cls_loss': total_cls_loss/50, 'loc_loss': total_loc_loss/50, 'lr': optimizer.param_groups[0]['lr'], 'precision': prf3[0] , 'recall': prf3[1], 'f1':prf3[2], 'average_iou': ious})
              total_loss = 0
              total_cls_loss = 0
              total_loc_loss = 0

            # Backward
            loss.backward()
            optimizer.step()
            scheduler.step()
          
        if not ((i+1)%1):
          torch.save(rpn.state_dict(), model_path + "state_dict_epoch_rpn_" +str(i+1))

      t1 = time.time()
      with open(model_path + 'time', 'a', newline='') as csvfile:
        fieldnames = ['training_time']
        writerC = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writerC.writeheader()
        writerC.writerow({'training_time': t1-t0})
        
      print("trained in", t1-t0, "seconds")
      
      
def log_rpn(rpn, device, path, imagest, labelst, scaleset, targets, log_path, anchor_label_params):

  #prediction_sizes(rpn, device, imagest, labelst, scalest, path, scaleset)
  #plt.close('all')

  #score_dist(rpn, device, imagest, labelst, scalest, path, scaleset)
  #plt.close('all')

  #prediction_spread(imagest, scalest, labelst, len(scaleset), path)
  #plt.close('all')

  #for i in targets:
  #  time_viz(rpn, device, imagest, scalest, labelst, path, i)
  #  plt.close('all')

  #a, b, c, d = countbatches(anchor_label_params, len(scaleset), img, rpn)
  #save_batches(a, b, c, d, scaleset, path)
  #plt.close('all')

  a, b, c, d, e = predict_set(imagest, labelst, scalest, rpn)
  iou = average_1diou(imagest, labelst, scalest)
  ref_deviation, est_deviation = average_deviation(imagest, labelst, scalest)

  with open(log_path, 'a', newline='') as csvfile:
    fieldnames = ['path', 'p3', 'p3trim', 'p5', 'p5trim', 'r3', 'r3trim', 'r5', 'r5trim', 'f13', 'f13trim', 'f15', 'f15trim', 'average_iou', '1d_iou', 'md_from_ref', 'md_from_est']
    writerL = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #writerL.writeheader()
    writerL.writerow({'path': path.split('/')[-2],'p3':a[0], 'p3trim': b[0], 'p5': c[0], 'p5trim': d[0], 'r3':a[1], 'r3trim':b[1], 'r5': c[1], 'r5trim':d[1], 'f13':a[2] , 'f13trim': b[2], 'f15': c[2], 'f15trim': d[2], 'average_iou': e, '1d_iou': iou, 'md_from_ref': ref_deviation, 'md_from_est': est_deviation})
    
    
# RPN Inspector
Reload RPN for "Inspection"

#### Prediction Distribution for Each Anchor

def prediction_sizes(rpn, device, imagest, labelst, scalest, path, scales):

  n_anchor = len(scales)
  all_predictions = []

  for i in range(n_anchor):
    all_predictions.append(np.array([]))

  for i in range(len(imagest)):

      lbl = torch.from_numpy(labelst[i])
      scale = scalest[i]
      sampleimg = imagest[i].reshape(3, 800, 800).float().unsqueeze(0)
      rois, scores, allroi, allscores, anchors = rpn.predict_test(sampleimg.to(device), device=device)

      for i in range(n_anchor):
        all_predictions[i] = np.append(all_predictions[i], allroi[i::n_anchor])

  for i in range(n_anchor):
    all_predictions[i] = all_predictions[i].reshape((int(len(all_predictions[i])/4)), 4)

  all_areas = []

  for bounds in all_predictions:

    y1 = bounds[:, 0]
    x1 = bounds[:, 1]
    y2 = bounds[:, 2]
    x2 = bounds[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    all_areas.append(areas)

  square_lengths_pred = np.sqrt(all_areas)
  #feature_map_lengths_pred = square_lengths_pred/4

  plt.title(path.split('/')[-2])
  plt.xlabel('Anchor Scale')
  plt.ylabel('Size Predicted (Root of Area)')

  plt.boxplot(square_lengths_pred.T, labels=scales, sym="")
  plt.figsize = (100, 75)

  plt.savefig(path + 'size_distribution')

#### Score Distribution for Each Anchor

def score_dist(rpn, device, imagest, labelst, scalest, path, scales):
  n_anchor = len(scales)
  all_scores = []

  for i in range(n_anchor):
    all_scores.append(np.array([]))

  for i in range(len(imagest)):

      lbl = torch.from_numpy(labelst[i])
      scale = scalest[i]
      sampleimg = imagest[i].reshape(3, 800, 800).float().unsqueeze(0)
      rois, scores, allroi, allscores, anchors = rpn.predict_test(sampleimg.to(device), device=device)

      for i in range(n_anchor):
        all_scores[i] = np.append(all_scores[i], allscores[i::n_anchor])

  for i, scores in enumerate(all_scores):
    plt.hist(scores, label=scales[i])

  plt.legend()
  plt.title(path.split('/')[-2])
  plt.xlabel('Objectness Score')
  plt.ylabel('# of Prediction')
  plt.figsize = (70, 55)

  plt.savefig(path + 'score_distribution')

#### Visualize Predictions in Time

import matplotlib.pyplot as plt
def time_viz(rpn, device, imagest, scalest, labelst, path, i):
  # prediction
  lbl = torch.from_numpy(labelst[i])
  scale = scalest[i]

  sampleimg = imagest[i].reshape(3, 800, 800).float().unsqueeze(0)
  rois, scores = rpn.predict(sampleimg.to(device), device=device)

  fig, ax = plt.subplots(figsize=(10, 10))
  
  ax.imshow(imagest[i])
  ax.axis('off')

  for xyxy in rois: 

    rect = patches.Rectangle((xyxy[0], xyxy[1]), xyxy[3]-xyxy[1], xyxy[2]-xyxy[0], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

  fig.savefig(path + 'boundsample' + str(i))

  bounds = rois
  bounds1 = bounds[:, 0].copy()
  bounds2 = bounds[:, 3].copy()
  bounds_full = np.vstack((bounds1, bounds2))
  bounds_full.sort()
  bounds_full = bounds_full.T
  bounds_full = bounds_full / scale /((44100 / 8192) / 2)

  plt.figure(figsize=(10, 1))

  y = np.arange(-1, 1, 1)
  for pred in bounds_full:
    plt.fill_betweenx(y, pred[0], pred[1]) #ylabel=None)

  #plt.xlabel('Seconds')
  plt.yticks([])
  plt.title('Section Predictions of Trained Model: Sample ' + str(i))
  plt.savefig(path + 'timesample' + str(i))

#### Visualize Layers

def layer_viz(rpn, image):
  sampleimg = image.reshape(3, 800, 800).float().unsqueeze(0)
  #x400 = rpn.extractor[0:9](sampleimg.to(device))
  #x200 = rpn.extractor[0:16](sampleimg.to(device))
  #x100 = rpn.extractor[0:23](sampleimg.to(device))
  x = rpn.extractor(sampleimg.to(device))

  plt.figure()

  #subplot(r,c) provide the no. of rows and columns
  f, axarr = plt.subplots(int(len(x[0][0:127])/4), 4, figsize=(12,80)) 

  # use the created array to output your multiple images. In this case I have stacked 4 images vertically
  #axarr[0].imshow(v_slice[0])
  #axarr[1].imshow(v_slice[1])
  #axarr[2].imshow(v_slice[2])
  #axarr[3].imshow(v_slice[3])

  for i, ax in enumerate(axarr.ravel()):
    ax.imshow(x[0][0:127][i].detach().cpu())

  f.show()

#### Count Batch Information

def countbatches(anchor_label_params, n_anchor, img, rpn):

  rpn.anchor_label_params = anchor_label_params

  idxs = []
  ancsumpos = np.zeros(n_anchor)
  ancsumneg = np.zeros(n_anchor)  
  batch_total = 0
  batch_ratio = 0

  for i in range(n_anchor):
    ancidxs = np.arange(i, n_anchor*50, n_anchor)
    idxs.append(ancidxs)

  for i in range(len(img)):
    sampleimg = img[i].reshape(3, 800, 800).float().unsqueeze(0)
    anchors, ious, anchor_labels, anchor_locations = rpn.forwardcounts(image=sampleimg.to(device), bbox=torch.from_numpy(labels[i]).to(device), img_size=[800,800], device=device)

    pos_anchors = np.where(anchor_labels == 1)
    neg_anchors = np.where(anchor_labels == 0)

    batch_size = len(pos_anchors[0]) + len(neg_anchors[0])
    batch_total = batch_total + batch_size

    pos_ratio = len(pos_anchors[0])/batch_size
    batch_ratio = batch_ratio + pos_ratio

    for i in pos_anchors[0]:
      for idx, j in enumerate(idxs):
        if i in j:
          ancsumpos[idx] = ancsumpos[idx] + 1

    for i in neg_anchors[0]:
      for idx, j in enumerate(idxs):
        if i in j:
          ancsumneg[idx] = ancsumneg[idx] + 1

  print("Average positive ratio:", batch_ratio/len(img))
  print("Average batch size:", batch_total/len(img))
  print("Positive Anchors:", ancsumpos)
  print("Negative Anchors:", ancsumneg)

  return batch_ratio/len(img), batch_total/len(img), ancsumpos, ancsumneg

def save_batches(a, b, c, d, scales, path):

  import matplotlib.pyplot as plt
  import numpy as np

  scale_sizes = scales
  pos_anchors = c
  neg_anchors = d

  x = np.arange(len(scale_sizes))  # the label locations
  width = 0.35  # the width of the bars

  fig, ax = plt.subplots()
  rects1 = ax.bar(x - width/2, pos_anchors, width, label='Positve Anchors')
  rects2 = ax.bar(x + width/2, neg_anchors, width, label='Negative Anchors', color='red')

  # Add some text for scale_sizes, title and custom x-axis tick labels, etc.
  ax.set_ylabel('# Labels')
  ax.set_xlabel('Anchor Scale')# )\n\n') + "Average positive ratio: " + f"{a:.2f}" + "\nAverage batch size: " + f"{b:.2f}")
  ax.set_title('Positive and Negative Anchor Labels Over Training Dataset')
  ax.set_xticks(x)
  ax.set_xticklabels(scales)
  ax.legend()

  fig.tight_layout()

  plt.savefig(path + '_batchcounts')

#### Count Number of Predictions per Image

def prediction_spread(imagest, scalest, labelst, n_anchor, path):
  prediction_nums = []

  for i in range(len(imagest)):

      lbl = torch.from_numpy(labelst[i])
      scale = scalest[i]
      sampleimg = imagest[i].reshape(3, 800, 800).float().unsqueeze(0)
      rois, scores, allroi, allscores, anchors = rpn.predict_test(sampleimg.to(device), device=device)

      prediction_nums.append(len(rois))
      test_prediction_nums = [len(x) for x in labelst]

  plt.title('# of predictions per: ResNet 50')
  plt.xlabel('Data')
  plt.ylabel('# of predictions')

  plt.boxplot([prediction_nums, test_prediction_nums], labels = [path.split('/')[-2], 'Ground Truth'], sym="")
  plt.figsize = (40, 40)

  plt.savefig(path + 'predictions_per_image')

#### Calculate 1d IoU


def iou_1d(gt_bounds, bounds, scale):

  gt_bounds1 = gt_bounds[:, 0].numpy().copy()
  gt_bounds2 = gt_bounds[:, 3].numpy().copy()
  gt_bounds_full = np.vstack((gt_bounds1, gt_bounds2))
  gt_bounds_full = gt_bounds_full.T
  gt_bounds_full = gt_bounds_full / scale /((44100 / 8192) / 2)

  bounds1 = bounds[:, 0].copy()
  bounds2 = bounds[:, 3].copy()
  bounds_full = np.vstack((bounds1, bounds2))
  
  bounds_full.sort()
  bounds_full = bounds_full.T
  bounds_full = bounds_full / scale /((44100 / 8192) / 2)
  #print(bounds_full)
  #print(gt_bounds_full)

  bf = []
  for x in bounds_full:
    if x[1] - x[0] > 0:
      bf.append(x)
  bf = np.array(bf)

  gt = []
  for x in gt_bounds_full:
    if x[1] - x[0] > 10:
      gt.append(x)
  gt_bounds_full = np.array(gt)

  # find squared error of each two points
  errors = np.empty((len(gt_bounds_full), len(bf)), dtype=np.float32)

  for idx, gt_bound in enumerate(gt_bounds_full):
    for idx2, p_bound in enumerate(bf):
      errors[idx, idx2] = ((gt_bound[0] - p_bound[0])**2 + (gt_bound[1] - p_bound[1])**2)/ 2

  min_error_idx = errors.argmin(axis=1)
    
  ious  = np.empty((len(gt_bounds_full)), dtype=np.float32)

  for idx, boundary in enumerate(gt_bounds_full):
    if (boundary[1] >= bf[min_error_idx[idx]][0]) and (boundary[0] <= bf[min_error_idx[idx]][1]):

      union = max(boundary[1], bf[min_error_idx[idx]][1]) - min(boundary[0], bf[min_error_idx[idx]][0])
      inter = max(boundary[0], bf[min_error_idx[idx]][0]) - min(boundary[1], bf[min_error_idx[idx]][1])
      ious[idx] = np.abs(inter/union)

    else:
      ious[idx] = 0

  return ious

def average_1diou(images, labels, scales):

    prediction_total = 0
    running_iou = 0
    for i in range(len(images)):

      img = images[i].reshape(3, 800, 800).float().unsqueeze(0)
      lbl = torch.from_numpy(labels[i])
      scale = scales[i]

      bounds, scores = rpn.predict(img.to(device), device=device)

      
      ious = iou_1d(lbl, bounds, scale)
      prediction_total = prediction_total + len(ious)
      running_iou = running_iou + sum(ious)
                                      
    return running_iou/prediction_total

#### Calculate Median Deviation

def boundary_deviation(gt_bounds, bounds, scale, trim=False):

  gt_bounds1 = gt_bounds[:, 0].numpy().copy()
  gt_bounds2 = gt_bounds[:, 3].numpy().copy()
  gt_bounds_full = np.vstack((gt_bounds1, gt_bounds2))
  gt_bounds_full = gt_bounds_full.T
  gt_bounds_full = gt_bounds_full / scale /((44100 / 8192) / 2)

  bounds1 = bounds[:, 0].copy()
  bounds2 = bounds[:, 3].copy()
  bounds_full = np.vstack((bounds1, bounds2))
  
  bounds_full.sort()
  bounds_full = bounds_full.T
  bounds_full = bounds_full / scale /((44100 / 8192) / 2)
  #print(bounds_full)
  #print(gt_bounds_full)

  bf = []

  for x in bounds_full:
    if x[1] - x[0] > 0:
      bf.append(x)

  bf = np.array(bf)

  #if len(bf) == 0:
  #  continue
  #print(bf)

  #bounds_full = bounds_full[np.where(x[1] != x[0] for x in bounds_full)]
  #print(bounds_full)

  ref, est = mir_eval.segment.deviation(gt_bounds_full, bf, trim=trim)
  return ref, est, len(gt_bounds_full), len(bf)

def average_deviation(images, labels, scales):

    ref_total = 0
    ref_val = 0
    est_total = 0
    est_val = 0
    
    for i in range(len(images)):

      img = images[i].reshape(3, 800, 800).float().unsqueeze(0)
      lbl = torch.from_numpy(labels[i])
      scale = scales[i]

      bounds, scores = rpn.predict(img.to(device), device=device)

      
      ref_score, est_score, ref_len, est_len = boundary_deviation(lbl, bounds, scale, trim=False)
      ref_val = ref_val + ref_score
      ref_total = ref_total + ref_len
      
      est_val = est_val + est_score
      est_total = est_total + est_len
                                      
    return ref_val/ref_total, est_val/est_total