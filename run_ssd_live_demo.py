from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite, create_mobilenetv3_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import os

if len(sys.argv) < 4:
    print(sys.argv)
    print('Usage: python run_ssd_live_example.py <net type>  <model path> <label path> [video file]')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]

if len(sys.argv) >= 5:
    cap = cv2.VideoCapture(sys.argv[4])  # capture from file
    print('capture from file')
else:
    cap = cv2.VideoCapture(0)   # capture from camera
    cap.set(3, 1920)
    cap.set(4, 1080)
    print('capture from camera')

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
print(f'Net classes num : {num_classes}')


if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
    print('Create_mobilenetv2_ssd_lite !')
elif net_type == 'mb3-ssd-lite':
    net = create_mobilenetv3_ssd_lite(len(class_names), is_test=True)
    print('Create_mobilenetv3_ssd_lite !')
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)


net.load(model_path)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
    print('Create_mobilenetv2_ssd_lite_predictor !')
elif net_type == 'mb3-ssd-lite':
    predictor = create_mobilenetv3_ssd_lite_predictor(net, candidate_size=200)
    print('Create_mobilenetv3_ssd_lite_predictor !')
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)


timer = Timer()
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(orig_image, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# timer = Timer()
# # cap = cv2.VideoCapture(sys.argv[4])  # capture from file
# if os.path.isdir(sys.argv[4]):
#     image_list = os.listdir(sys.argv[4])
#     for img in image_list:
#         image_path = os.path.join(sys.argv[4], img)
#         orig_image = cv2.imread(image_path)
#         image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
#
#         timer.start()
#         boxes, labels, probs = predictor.predict(image, 10, 0.4)
#         # print(f'Predict Res:boxes:{boxes.shape}, labels:{labels.shape}, probs:{probs.shape}')
#         interval = timer.end()
#
#         print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
#         for i in range(boxes.size(0)):
#             box = boxes[i, :]
#             label = f"{class_names[labels[i]]}{probs[i]:.2f}"
#             cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
#             cv2.putText(orig_image, label,
#                         (box[0] + 20, box[1] + 40),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1,  # font scale
#                         (255, 0, 255),
#                         2)  # line type
#         # cv2.imshow('annotated', orig_image)
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break
#         img_name = image_path[:-4]+'_tangle_text.jpg'
#         cv2.imwrite(img_name, orig_image)
#
#
# else:
#     cap = cv2.VideoCapture(sys.argv[4])  # capture from file
#     while True:
#         ret, orig_image = cap.read()
#         # print('ret, orig_image = cap.read()')
#         if orig_image is None:
#             continue
#         image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
#         timer.start()
#         boxes, labels, probs = predictor.predict(image, 10, 0.4)
#         interval = timer.end()
#         print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
#         for i in range(boxes.size(0)):
#             box = boxes[i, :]
#             label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
#             cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
#
#             cv2.putText(orig_image, label,
#                         (box[0]+20, box[1]+40),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1,  # font scale
#                         (255, 0, 255),
#                         2)  # line type
#         cv2.imshow('annotated', orig_image)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
cap.release()
cv2.destroyAllWindows()


