from typing import Dict
import torch

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, pickle

#import some common detectron2 utilities
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog

class Trainer():
    def __init__(
        self,
        NUM_CLASSES    = 20,
        train_mode     = False,
        DATA_ROOT      = None,
        MODEL_ROOT     = None,
        NEW_MODEL_ROOT = None   
        ):
        self.train_mode     = train_mode
        self.DATA_ROOT      = DATA_ROOT
        self.MODEL_ROOT     = [MODEL_ROOT]
        self.NEW_MODEL_ROOT = NEW_MODEL_ROOT
        self.NUM_CLASSES    = NUM_CLASSES
        if self.train_mode:
            assert MODEL_ROOT     is not None, "Need to provide MODEL_ROOT"
            assert DATA_ROOT      is not None, "Need to provide DATA_ROOT"
            assert NEW_MODEL_ROOT is not None, "Need to provide NEW_DATA_ROOT"

        
    def train(self, sche, hyperparam_kwarg = None):
        self.current_dir = sche
        self.set_hyperparam(**hyperparam_kwarg)

        trainer = DefaultTrainer(self.cfg) 
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        with open(os.path.join(self.NEW_MODEL_ROOT, sche , 'model_cfg.pickle'), 'wb') as f:
            pickle.dump(self.cfg,f)
            
        self.MODEL_ROOT.append( os.path.join(self.NEW_MODEL_ROOT, self.current_dir) )
        
        
    def get_current_root(self):
        return self.MODEL_ROOT[-1]

    def set_hyperparam(self):

        def VisualManager_Trainer_dataset_function():
            returnList = []
            for file in sorted(os.listdir(self.DATA_ROOT)):
                if not file.endswith('.pickle'): continue
                
                f_path = os.path.join(self.DATA_ROOT,file)

                with open(f_path,'rb') as f:
                    returnList.append(pickle.load(f))
            return returnList
        
        DatasetCatalog.register('VisualManager_Trainer_Dataset', VisualManager_Trainer_dataset_function)
        

        with open(os.path.join(self.MODEL_ROOT[-1], 'model_cfg.pickle'), 'rb') as f:
            self.cfg = pickle.load(f)

        self.cfg.DATASETS.TRAIN = ('VisualManager_Trainer_Dataset',)

        self.cfg.MODEL.WEIGHTS = os.path.join(self.MODEL_ROOT[-1], "model_final.pth")
        
        # Detectron default 4
        self.cfg.DATALOADER.NUM_WORKERS = 4
        # Detectron default 40000
        self.cfg.SOLVER.MAX_ITER = 120_000
        '''
        Detectron default 
        Base Learning rate 0.001
        GAMMA              0.1 
        STEP               (30000,)
            GAMMA : Learning rate decay factor
            STEPS: num of iter for learning rate decay by gamma
        
        MASK RCNN PAPER : https://arxiv.org/pdf/1703.06870.pdf
            Base LR 0.02
            decay by 10 @ 120k/160k
            
            Cityscapes finetuning 
                Base LR 0.001
                decay by 10 @ 18k/24k
            
            update baseline
                Base LR 0.001
                decay by 10 @ 120k,160k/180k
            
            Benefit form deeper model
        '''   
        self.cfg.SOLVER.BASE_LR      = 0.001  
        self.cfg.SOLVER.GAMMA        = 0.1 
        self.cfg.SOLVER.STEPS        = (90_000,)
        self.cfg.SOLVER.WEIGHT_DECAY = 0.000_1


        #   ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
        #   E.g., a common configuration is: 512 * 16 = 8192
        # Detectron default 16
        self.cfg.SOLVER.IMS_PER_BATCH = 32
        # Detectron default 512
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 2048

        # Number of classes 
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.NUM_CLASSES 

        # Confident Level
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

        self.cfg.OUTPUT_DIR = os.path.join(self.NEW_MODEL_ROOT, self.current_dir)

