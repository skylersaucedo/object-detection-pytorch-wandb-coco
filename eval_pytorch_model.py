"""
USE THIS PATTERN TO LOAD MODEL FROM CHECKPOINTS
EVALUATE ON HOLDOUT DS
"""


import os
import wandb
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from coco_utils import get_coco, get_coco_kp
from engine import evaluate, train_one_epoch
from group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from torchvision.transforms import InterpolationMode
from transforms import SimpleCopyPaste
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import draw_bounding_boxes

print('is_available: ', torch.cuda.is_available())
print('device_count: ', torch.cuda.device_count())
print('current_device: ', torch.cuda.current_device())
print('current_device: ', torch.cuda.device(0))
print('get_device_name: ', torch.cuda.get_device_name(0))


def get_args_parser(add_help=True):

    import argparse

    output_dir_path = r'C:\Users\TSI\Desktop\object-detection-pytorch-wandb-coco-master\outputdir'
    data_dir_path = r"C:\Users\TSI\Desktop\object-detection-pytorch-wandb-coco-master"

    dataset_type = 'coco'
    model = "retinanet_resnet50_fpn"
    device_type = "cuda"
    batch_size = 4
    epochs = 250
    workers = 1
    optimizer = "sgd"
    norm_weight_decay = 0.9
    momentum = 0.9
    lr = 0.0005 #0.001
    weight_decay = 1e-4
    lr_step_size = 8

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default=data_dir_path, type=str, help="dataset path")
    parser.add_argument("--dataset", default=dataset_type, type=str, help="dataset name")
    parser.add_argument("--model", default=model, type=str, help="model name")
    parser.add_argument("--device", default=device_type, type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=batch_size, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=epochs, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=workers, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--opt", default=optimizer, type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=lr,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=momentum, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=weight_decay,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=norm_weight_decay,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=lr_step_size, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output_dir", default=output_dir_path, type=str, help="path to save outputs")
    parser.add_argument("--resume", default=output_dir_path, type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
    )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default='ResNet50_Weights.IMAGENET1K_V1', type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # Use CopyPaste augmentation training parameter
    parser.add_argument(
        "--use-copypaste",
        action="store_true",
        help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.",
    )

    return parser


def run_evaluation_script():
    model_path = r'C:\Users\TSI\Desktop\object-detection-pytorch-wandb-coco-master\outputdir\model_248.pth'

    yo = ['--weights-backbone', 'ResNet50_Weights.IMAGENET1K_V1']
    args = get_args_parser().parse_args(yo)


    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh

    model = torchvision.models.get_model(
        args.model, weights=args.weights, weights_backbone=args.weights_backbone, **kwargs
    )

    # load checkpoint weights

    #epoch = 249
    parameters = [p for p in model.parameters() if p.requires_grad]
    opt_name = args.opt.lower()
    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov="nesterov" in opt_name)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    scaler = torch.cuda.amp.GradScaler()


    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    #loss = checkpoint['loss']
    lr_scheduler = checkpoint['lr_scheduler']
    args = checkpoint['args']

    print('---------------------UPDATED MODEL ----------------')

    device = torch.device("cuda")

    holdout_ds_path = r'C:\Users\TSI\Desktop\object-detection-pytorch-wandb-coco-master\holdout_images_april_25'

    images = glob.glob(holdout_ds_path + '/*.jpg')

    CLASSES = ['scratch', 'dent', 'paint', 'pit']
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    model.eval()
    model.cuda() # send weights to gpu

    for img_path in images:
        img_r = cv2.imread(img_path)
        img_c = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

        img_t = img_c.transpose([2,0,1])

        img_t = np.expand_dims(img_t, axis=0)
        img_t = img_t/255.0
        img_t = torch.FloatTensor(img_t)

        print('shape of img_t ', np.shape(img_t))

        img_t = img_t.to(device)
        detections = model(img_t)[0]

        print('detections: , ' ,detections)

        #loop over the detections
        for i in range(0, len(detections["boxes"])):
            confidence = detections["scores"][i]

            print(confidence)

            con = 0.5

            if confidence > con:

                idx = int(detections["labels"][i])
                box = detections["boxes"][i].detach().cpu().numpy()
                (startX, startY, endX, endY) = box.astype("int")
                # display the prediction to our terminal
                label = "{}: {:.2f}%".format(CLASSES[idx-1], confidence * 100)
                print("[INFO] {}".format(label))
                # draw the bounding box and label on the image
                cv2.rectangle(img_c, (startX, startY), (endX, endY),
                    COLORS[idx-1], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(img_c, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx-1], 2)

        plt.imshow(img_c)
        plt.show()

if __name__ == "__main__":
    run_evaluation_script()