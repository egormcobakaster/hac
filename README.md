#pipeline

  extract face box with yolo
    
    outs = face_model(orgimg_t)[0]
  
  expand box:

  
    x1 = box[0] - w // 2
    y1 = box[1] - h // 2
    x2 = box[2] + w // 2
    y2 = box[3] + h // 2
  
  predict live prob
#test:
  
  change input path in test.py (line 14)
  out - ./out.txt:
    ...
    path_to_image live_prob
    path_to_image live prob
    ...
  
