################################################################
'''
README

# VisualManager
    The Visual Manager is a class which handle all visual related task,
    which composited by:
    - Preprocessor
        extract object's feature vector form a image input
    - ImageSaver
        annotate and save data for futher training
    - Trainer
        tune a new model form the old model on the data that we generated
    

    

    # EXAMPLE
    # constructor

    ----------------------------------------------------------------------------------------------------------------------------
    eyes = VisualManager(
        MODEL_ROOT = path,               # The directory to the model
            
        DATA_ROOT  = path,               # THe directoey to save image and data
            
        verbose    = True,               # verbose

        train_schedule     = (10_000,),  # The trainer will tune the model when saved image hit the number listed

        preprocessor_kwarg = dict(
            mask_size = (128,128),       # size that image will be wrap to
            grayscale  = True,           # allow gray for more information
            threshold  = 0.5,            # thresold of confident score
            backbone   = None,           # backbone for image and masks
            getVec     = None,           # get vector from feature map
            norm       = None,           # norm layer for image and masks
            acti       = None,           # activation layer for image and masks
        ),   
        imagesaver_kwarg = dict(   
            save_mode = True,            # True to turn on image saving mode
            save_freq = 100              # how often will save image and annotations
        ),   
        trainer_kwarg = dict(   
            NUM_CLASSES    = 20,         # Number for classes for classify
            train_mode     = True,       # True to turn on training mode
            NEW_MODEL_ROOT = path,       # The directory that all newly tuned model will be saved 
        )
    )
    ----------------------------------------------------------------------------------------------------------------------------

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


import os
import pickle
from PIL import Image, ImageFilter
import time

import numpy as np
import torch
import torch.nn as nn

from pycocotools.mask import encode as Mask2RLE

from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode

from robosuite.utils.visual.Trainer import Trainer

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
        # Load the config and weight of model and construct the predictor 
        self.threshold = threshold
        self.load_model(MODEL_ROOT)

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

        self.norm = nn.LayerNorm(self.mask_size) if norm is None else norm
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

        image = torch.tensor(
            np.asarray(Image.fromarray(img).resize( self.mask_size )),
            dtype=torch.float32
        ).permute((2,0,1))
        image = self.acti(self.norm(image.repeat(N,1,1,1)))
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

        with torch.no_grad():

            test_info  = torch.rand(N, 6)
            test_masks = torch.rand(N, 1, H, W)
            test_image = torch.rand(H, W, 3).permute((2,0,1))
            
            try:
                test_masks = self.acti(test_masks)
            except:
                raise Exception("The specified acti layer is not compatible")

            try:
                test_image = self.norm(test_image.repeat(N,1,1,1))
            except:
                raise Exception("THe specifed norm layer is not compatible")


            test_imgnseg = torch.cat((test_masks,test_image),dim=1)

            try:
                test_map = self.backbone( test_imgnseg ).reshape((N,-1))
            except:
                raise Exception("The specifed backbone layer is not compatible")
            vector   = torch.cat( (test_map,test_info) ,dim=1)

            try:
                vec      = self.getVec( vector )
            except:
                raise Exception("The specified getVec is not compatible")
            assert len(vec.shape) == 2

    def load_model(self, MODEL_ROOT):
        self.MODEL_ROOT = MODEL_ROOT
        with open(os.path.join(self.MODEL_ROOT, 'model_cfg.pickle'), 'rb') as f:
            cfg = pickle.load(f)

        cfg.MODEL.WEIGHTS = os.path.join(self.MODEL_ROOT, "model_final.pth")  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold # set a custom testing threshold

        self.predictor = DefaultPredictor(cfg)

class ImageSaver():
    def __init__(
        self,
        save_mode = False,
        save_freq = None,
        DATA_ROOT = None,
        ):
        self.save_mode = save_mode
        if self.save_mode is True:
            assert save_freq is not None, "save mode is on but saves frequency is not provide, try save_mode = False, or save_freq = 1_000"
            assert DATA_ROOT is not None, "save mode is on but Image directory is not provide, try save_mode = False, or provide directory to save the data"
            self.DATA_ROOT = DATA_ROOT
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
        if not os.path.isdir(self.DATA_ROOT):
            os.mkdir(self.DATA_ROOT)

    def save(self, img, seg, env):
        self.pre_save()

        id = int(time.time())
        while id in os.listdir():
            id = int(time.time())
        
        with open(os.path.join(self.DATA_ROOT,f'{id}.pickle') , 'wb') as f:
            pickle.dump( self.seg2anno(seg, env, id) ,f)
        Image.fromarray(img).save(os.path.join(self.DATA_ROOT,f'{id}.png'))
        
        #raise Exception("This function is not finish yet")
        
    def seg2anno(self, seg, env, img_id):
        # objects: a dictionary [id] => name
        # ids: list of M id 
        returnDict = {}
        returnDict['file_name']   = os.path.join(self.DATA_ROOT, f'{img_id}.png')
        returnDict['height']      = len(seg)
        returnDict['width']       = len(seg[0])
        returnDict['image_id']    = img_id
        returnDict['annotations'] = []


        # mask (256,256,1) with M ID
        _,mask,_ = np.split(seg,3,axis=2)

        objects = {}
        ids = np.unique(mask)
        for id in ids:
            name = env.sim.model.geom_id2name(id)
            objects[id] = name

        for k, v in sorted(objects.items()): print(k, v.split("_") if v is not None else v)

        # mask.squeeze()[np.newaxis]                 ==> ( 1, 256, 256)
        # ids[:, np.newaxis, np.newaxis, np.newaxis] ==> ( M,   1,   1)
        #                                                 L Broadcastable to
                                                        #( M, 256, 256)
        # masks  ==> (M, 256, 256, 1) 
        masks = (mask.squeeze()[np.newaxis] == ids[:, np.newaxis, np.newaxis]).astype(np.uint8)
        #masks = np.asarray(masks)  

        # id     : list of id
        # object : map<id,name>
        # masks  : masks
        have_name = False
        no_name_counter = 0
        name2idMask = {}
        for _id, _mask in zip(ids,masks):
            _name = objects[_id]
            if _name is not None:
                _name = _name.split('_')[0]
            if _name is None and have_name:
                have_name = False
                no_name_counter += 1
            if _name is not None and not have_name:
                have_name = True

            if _name is None:
                _name = f'bin{no_name_counter}'
            if _name in name2idMask:
                old_mask, old_id = name2idMask[_name]
                name2idMask[_name] = (old_mask + _mask, old_id + [_id])
            else:
                name2idMask[_name] = (_mask, [_id])

        # name2idMask : dict< one word name : ( mask<256,256> , list<id> ) >



        for k,v in sorted(name2idMask.items()):
            _mask, _ids = v
            #if not this_Instance_Should_be_Saved(_ids): continue
            annoDict = {}

            annoDict['bbox']        = self.mask2BBox(_mask)
            annoDict['bbox_mode']   = BoxMode.XYXY_ABS

            annoDict['category_id'] = 1
            # annoDict['category_id'] = mapGeomIDtoCategoryID(_ids)


            _vis  = np.asarray(_mask * 255 / _mask.max()).astype(np.uint8)
            _vis  = Image.fromarray(_vis, mode='L').filter(ImageFilter.MinFilter(3)).convert('1')
            _mask = np.asarray(_vis)

            _RLE = Mask2RLE( np.asarray( _mask,dtype=np.uint8, order= 'F') )
            annoDict['segmentation'] = _RLE
            
            
            returnDict['annotations'].append(annoDict)
            _vis.save(os.path.join('.','imgseg',f'{img_id}_{k}.png'))


        return returnDict

    def mask2BBox(self, mask):
        rows       = np.any(mask,axis=0)
        cols       = np.any(mask,axis=1)
        rmin, rmax = np.where(rows)[0][[0,-1]]
        cmin, cmax = np.where(cols)[0][[0,-1]]
        return [rmin,cmin, rmax, cmax]

class VisualManager():
    def __init__(
        self,
        MODEL_ROOT         = None,
        DATA_ROOT          = None,
        verbose            = True,
        train_schedule     = (10_000,),
        _preprocessor      = Preprocessor,
        preprocessor_kwarg = dict(),
        _imagesaver        = ImageSaver,
        imagesaver_kwarg   = dict(),
        _trainer           = Trainer,
        trainer_kwarg      = dict(),
        ):
        preprocessor_kwarg["MODEL_ROOT"] = MODEL_ROOT
        self.preprocessor = _preprocessor(**preprocessor_kwarg)

        imagesaver_kwarg["DATA_ROOT"] = DATA_ROOT
        self.imagesaver = _imagesaver(**imagesaver_kwarg)
        
        self.trainer = _trainer(
            MODEL_ROOT = MODEL_ROOT,
            DATA_ROOT = DATA_ROOT,
            **trainer_kwarg
            )
        
        self.verbose        = verbose
        self.image_saved    = 0
        self.train_schedule = train_schedule

        if not self.imagesaver.save_mode and self.trainer.train_mode:
            print("[VisualManager]Warning: the train_mode is on but save_mode is not on, it will not train when VisualManager is call, ")
            _sanity = input('[VisualManager]but you can call VisualManager.train() to force train, are you sure?[y/n(default, will raise error)]')
            assert _sanity == 'Y' or _sanity == 'y', '[VisaulManager]train_mode is on, while save_mode is not'
        if self.verbose: print("[VisualManager]Finished Init")


    def __call__(self,vis,env):
        if self.verbose and self.imagesaver.save_mode :
            print(f"[VisualManager]Datasaver  : {self.imagesaver.counter}/{self.imagesaver.save_freq}")
            print(f"[VisualManager]image saved: {self.image_saved}")

        img, seg = np.array(vis).astype(np.uint8)

        img = np.rot90(img,k=2)
        seg = np.rot90(seg,k=2)

        if self.imagesaver(img, seg, env): 

            self.image_saved += 1
            if self.verbose:    
                print("[VisualManager]One Image saved")
                print(f"[VisualManager]train schedule:", self.train_schedule)

            if self.image_saved in self.train_schedule and self.trainer.train_mode: 
                if self.verbose: print("[VisualManager]Trainer start training")
                self.trainer.train(sche = f"tune_model_{self.image_saved}")
                self.preprocessor.load_model(self.trainer.get_current_root())

        # return embedded vectors
        if self.verbose: print("[VisualManager]returning feature vectors...")
        
        return self.preprocessor(img)


    def train(self,train_name = "force-train"):
        print("[VisualManager]unschedule train")
        
        if self.verbose: print("[VisualManager]Trainer start training")

        self.trainer.train(sche = f"tune_model_{train_name}")
        self.preprocessor.load_model(self.trainer.get_current_root())
