#!/usr/bin/env python
import numpy as np
import rospy
import cv2
from cv_bridge   import CvBridge
from ros_occlusion_ssm.srv import Segmentation         as SegmentationSrv, \
                        SegmentationRequest  as SegmentationRequestMsg, \
                        SegmentationResponse as SegmentationResponseMsg
from ros_occlusion_ssm import SAM


if __name__ == '__main__':
    rospy.init_node('ros_occlusion_ssm')
    model = rospy.get_param('~model', 'vit_h')
    cuda  = rospy.get_param('~cuda', 'cuda')
    bridge = CvBridge()
    # really just loads the model, moves it to the gpu, and makes a predictor object that's called by sam.segment()
    print('Starting SAM...')
    sam = SAM(model, cuda)

    '''
        TODO: need to figure out what exactly the sekonix camera nodes send - check at CAVS
    '''

    def srv_segmentation(req : SegmentationRequestMsg):
        try:
            # currently uses a cv_bridge function that receives a ROS Image Message from the passed SegmentationRequestMsg
            img    = cv2.cvtColor(bridge.imgmsg_to_cv2(req.image), cv2.COLOR_BGR2RGB)
            # the SegmentationRequestMsg also has query_points member - check ROS documentation
            points = np.vstack([(p.x, p.y) for p in req.query_points])
            # seems to create the bounding boxes as an ndarray using req.boxes.data
            boxes  = np.asarray(req.boxes.data).reshape((len(req.boxes.data) // 4, 4))[0] if len(req.boxes.data) > 0 else None
            # seems to create and ndarray of labels for each box
            labels = np.asarray(req.query_labels)
            print("Segmenting at pixels:")
            # TODO: iterates over query_points - figure out if this should be all points in a region, a region border, or what
            for point in points:
                print(f"{point[0]}, {point[1]}")
            print(img.shape)
            # TODO: replace this with the inference function I need to add
            masks, scores, logits = sam.segment(img, points, labels, boxes, req.multimask)
            # seems like this forms the message that will be sent back to the calling location, holding everything returned by the inference pass
            res = SegmentationResponseMsg()
            # seems to need to convert the ndarrays returned above to ROS Image msg explicitly
            res.masks  = [bridge.cv2_to_imgmsg(m.astype(np.uint8)) for m in masks]
            res.scores = scores.tolist()
            if req.logits:
                res.logits = [bridge.cv2_to_imgmsg(l) for l in logits]
            return res
        except Exception as e:
            print(f'{e}')
            raise Exception('Failure during service call. Check output on SAM node.')

    srv = rospy.Service('~segment', SegmentationSrv, srv_segmentation)
    print('SAM is ready')
    while not rospy.is_shutdown():
        rospy.sleep(0.2)