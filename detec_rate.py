from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import os

def detec_rate(net, class_names, file_path, device):
    predictor = create_mobilenetv3_ssd_lite_predictor(net, candidate_size=200, device=device)
    count_, correct_, wrong_, miss_ = 0,0,0,0
    image_class = class_names[1:]
    for idx, cla in enumerate(image_class):
        cla_dir = os.path.join(file_path, cla)
        image_list = os.listdir(cla_dir)
        count, correct, wrong, miss = 0,0,0,0
        for img in image_list:
            img_path = os.path.join(cla_dir, img)
            orig_image = cv2.imread(img_path)
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            # image = orig_image
            boxes, labels, probs = predictor.predict(image, 10, 0.4)
            count += 1
            if labels.size(0) == 1:
                if class_names[labels[0]] == cla:
                    correct += 1
                else:
                    wrong += 1
            elif labels.size(0) > 1:
                wrong += 1
                for lab in labels:
                    if class_names[lab] == cla:
                        correct += 1
                        break
            else:
                miss += 1

        print(f'Detection Situation: {cla}:{count}-Correct:{correct}({(correct/count)*100:.2f}%)'
              f'-Wrong:{wrong}({(wrong/count)*100:.2f}%)-Miss:{miss}({(miss/count)*100:.2f}%)')
        count_+=count
        correct_+=correct
        wrong_+=wrong
        miss_+=miss

    # cv2.destroyAllWindows()
    return count_, correct_, wrong_, miss_



