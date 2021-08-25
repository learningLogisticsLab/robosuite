import os
import pickle
from PIL import Image
import time

import numpy as np
import torch
import torch.nn as nn

from detectron2.engine import DefaultPredictor

from Trainer import Trainer

################################################################
'''
README

# VisualManager
    # constructor
    eyes = VisualManager(
        preprocessor_kwarg = dict(
            ...
        ),
        imagesaver_kwarg = dict(
            ...
        )
    )

    # use
    feature_vectors = eyes(obs['agentview_image'],env)

        ! obs can have any other camera name
        ! must be passed for annotation purposes


# Preprocessor
    # constructor
    preprocessor = Preprocessor(
        MODEL_ROOT = '{MODEL_ROOT}',  <= The Root directory of the model
        mask_size = (128,128),        <= The size of mask will be converted
        garyscale = True              <= if the image will be grayscaled
    )

    use

    preprocessed_feature_vector = preprocessor( img )  
        # input
        #   img should have shape => ( Height, Width, Channel )
        # output
        #   (N, embedded size)


'''
################################################################
"""
MODEL_ROOT
    L model_cfg.pickle    <==
    L model_final.pth
"""
"""
IMAGE_SAVE_DIR
    L {id}.pickle        <== a brunch of pickle and png pair
    L {id}.png
"""

class Preprocessor(nn.Module):
    def __init__(
        self,
        MODEL_ROOT      = None,
        mask_size       = (128,128),
        grayscale       = True,
        threshold       = 0.5,
        backbone        = None,
        getVec          = None, 
        norm            = None,
        acti            = None
        ):
        super(Preprocessor, self).__init__()
        assert MODEL_ROOT is not None
        self.MODEL_ROOT = MODEL_ROOT
        # Load the config and weight of model and construct the predictor 
        self.threshold = threshold
        self.load_model()

        self.mask_size = mask_size
        self.format = 'L' if grayscale else "1"

        ##############################################################################
        #  Learnable Network
        ##############################################################################
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 16, 3, 1, 1),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.MaxPool2d(2, stride = 2),  # size shrink half
            nn.ReLU(),

            nn.Conv2d(16, 16, 3, 1, 1),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.MaxPool2d(2, stride = 2),  # size shrink half
            nn.ReLU(),

            nn.Conv2d(16, 16, 3, 1, 1),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.MaxPool2d(2, stride = 2),  # size shrink half
            nn.ReLU()
        ) if backbone is None else backbone

        self.getVec = nn.Sequential(
            nn.Linear( int(mask_size[0]/8 * mask_size[1]/8) + 6, 128),
            nn.Linear( 128, 128),
            nn.Linear( 128, 64),
        ) if getVec is None else getVec

        self.norm = nn.BatchNorm2d(3) if norm is None else norm
        self.acti = nn.Tanh() if acti is None else acti
        ##############################################################################
        self.testdrive()

        
    def forward(self, img):
        # make sure the input shape is ( Height, Width, Channel )
        assert len(img.shape) == 3, "ERROR: The input is not in a shape of ( Height, Width, Channel ), input does not have 3 dimension"
        assert img.shape[2]   == 3, "ERROR: The input is not in a shape of ( Height, Width, Channel ), input does not have 3 channel"

        img = np.array(img)


        instances = self.predictor(img)["instances"]
        N = len(instances)

        if N == 0: return torch.tensor([[0]])

        '''
        instances.pred_boxes
            Boxes object storing N object
            instances.pred_boxes.tensor return => (N, 4) matrix
        instances.pred_classes shape: (N)
        instnaces.pred_mask    shape: (N, H, W)
        instances.score        shape: (N)
        
        img                    shape: (H, W, C)
        '''
        info = torch.cat( 
        (
            instances.pred_boxes.tensor,
            instances.pred_classes.unsqueeze(1),
            instances.scores.unsqueeze(1)
        ), dim = 1)

        masks = [ 
            np.asarray(
                Image.fromarray(
                    m.detach().numpy()
                ).convert( self.format ).resize( self.mask_size )
            )
            for m in instances.pred_masks
        ]
        masks = torch.tensor( np.asarray(masks) , dtype = torch.float).unsqueeze(1)
        masks = self.acti(masks)

        image = torch.tensor(
            np.asarray(Image.fromarray(img).resize( self.mask_size )),
            dtype=torch.float32
        ).permute((2,0,1))
        image = self.norm(image.repeat(N,1,1,1))
        '''
        N       : number of instances idenify in the image
        HS, WS  : pre-defined number of the resized mask, default (128, 128)
        info    : tensor shape: (N, 6) <- the six dim are : (x1, y1, x2, y2, classes_id, score)
        masks   : tensor shape: (N, 1, HS, WS)
        image   : tensor shape: (N, 3, HS, WS)
        imgnseg : tensor shape: (N, 4, HS, WS)
        '''
        
        imgnseg = torch.cat((masks,image),dim = 1)

        assert imgnseg.shape == (N, 4, *self.mask_size)
        assert info.shape    == (N, 6)

        feature_maps = self.backbone(imgnseg).reshape((N,-1))
        '''
        feature_map
            tensor shape: (N, HS/8 * WS/8)
        '''

        vector = torch.cat( (feature_maps,info) ,dim=1)

        return self.getVec(vector)

    def testdrive(self):
        N   = 12
        H,W = self.mask_size
        test_masks = torch.rand(N, 4, H, W)
        test_info  = torch.rand(N, 6)
        with torch.no_grad():
            test_map = self.backbone( test_masks ).reshape((N,-1))
            vector   = torch.cat( (test_map,test_info) ,dim=1)
            vec      = self.getVec( vector )
            assert len(vec.shape) == 2

    def load_model(self):
        with open(os.path.join(self.MODEL_ROOT, 'model_cfg.pickle'), 'rb') as f:
            cfg = pickle.load(f)

        cfg.MODEL.WEIGHTS = os.path.join(self.MODEL_ROOT, "model_final.pth")  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold # set a custom testing threshold

        self.predictor = DefaultPredictor(cfg)

class ImageSaver():
    def __init__(
        self,
        save_mode          = False,
        save_freq          = None,
        IMAGE_DIR          = None,
        ):
        self.save_mode = save_mode
        if self.save_mode is True:
            assert save_freq is not None, "save mode is on but saves frequency is not provide, try save_mode = False, or save_freq = 1_000"
            assert IMAGE_DIR is not None, "save mode is on but Image directory is not provide, try save_mode = False, or provide directory to save the data"
            self.IMAGE_DIR = IMAGE_DIR
            self.save_freq = save_freq
            self.counter   = 0
        

    def __call__(self, img, seg, env):
        if self.save_mode is False:
            return None
        if self.counter == 0:
            self.save(img.astype(np.uint8), seg, env)
        self.counter = (self.counter + 1) % self.save_freq
        return self.counter == 1

    def pre_save(self):
        if not os.path.isdir(self.IMAGE_DIR):
            os.mkdir(self.IMAGE_DIR)

    def save(self, img, seg, env):
        self.pre_save()

        id = int(time.time())
        while id in os.listdir():
            id = int(time.time())
        
        with open(os.path.join(self.IMAGE_DIR,f'{id}.pickle') , 'wb') as f:
            pickle.dump( self.seg2anno(seg, env) ,f)
        Image.fromarray(img).save(os.path.join(self.IMAGE_DIR,f'{id}.png'))
        
        raise Exception("This function is not finish yet")
        
    def seg2anno(self, seg, env):
        # TODO 
        #   figure out how to programmatically annotation mask
        #   maybe save in coco format

        # objects: a dictionary [id] => name
        # ids: list of M id 
        objects = {}
        ids = []
        for i in seg.reshape(-1,3):
            name = env.sim.model.geom_id2name(i[1])
            objects[i[1]] = name
            if i[1] not in ids: ids.append(i[1])
        ids = np.array(sorted(ids))
        #for k, v in sorted(objects.items()): print(k, v.split("_") if v is not None else v)


        # mask (256,256,1) with M ID
        _,mask,_ = np.split(seg,3,axis=2)

        # mask[np.newaxis]                           ==> ( 1, 256, 256, 1)
        # ids[:, np.newaxis, np.newaxis, np.newaxis] ==> ( M,   1,   1, 1)
        #                                                 L Broadcastable

        # masks  ==> (M, 256, 256, 1) 
        masks = ( mask[np.newaxis] == ids[:, np.newaxis, np.newaxis, np.newaxis]).squeeze()#.astype(np.uint8)
        #masks = np.array(list( filter( lambda m: m.sum() > 100, masks ) )) outdated
        #masks = masks * 255
        return masks

class VisualManager():
    def __init__(
        self,
        _preprocessor      = Preprocessor,
        preprocessor_kwarg = None,
        _imagesaver        = ImageSaver,
        imagesaver_kwarg   = None,
        _trainer           = Trainer,
        trainer_kwarg      = None,
        train_schedule     = (10_000,),
        ):
        # not sure if it sure inherit from nn.module
        #super(VisualManager, self).__init__()
        self.preprocessor   = _preprocessor(**preprocessor_kwarg) 
        self.imagesaver     = _imagesaver(**imagesaver_kwarg)
        self.trainer        = _trainer(**trainer_kwarg)
        self.image_saved    = 0
        self.train_schedule = train_schedule


        print("Finished Init VisualManage")


    def __call__(self,vis,env):
        print("Using Visual Manager")
        img, seg = np.array(vis).astype(np.uint8)

        img = np.rot90(img,k=2)
        seg = np.rot90(seg,k=2)

        if self.imagesaver(img, seg, env): 

            self.image_saved += 1

            if self.image_saved in self.train_schedule: 
                self.preprocessor.MODEL_ROOT = self.trainer.train()
                self.preprocessor.load_model()


        # return embedded vectors
        return self.preprocessor(img)

