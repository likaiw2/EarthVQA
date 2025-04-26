import ever as er
from ever.core.builder import make_model, make_dataloader
import torch
import numpy as np
import os, json
from tqdm import tqdm
import logging
from ever.core.checkpoint import load_model_state_dict_from_ckpt
from ever.core.config import import_config
import argparse
from PIL import Image
from torchvision import transforms
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
import h5py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.earthvqa import EarthVQADataset

QUESTIONS = ['Are there any villages in this scene?', 'Is there a commercial area near the residential area?', 'Are there any playgrounds in this scene?', 'Is there any commercial land in this scene?', 'Is there any forest in this scene?', 'Is there any agriculture in this scene?', 'What are the types of residential buildings?', 'Are there any urban villages in this scene?', 'What are the needs for the renovation of villages?', 'Is there any barren in this scene?', 'Whether greening need to be supplemented in residential areas?', 'Is there any woodland in this scene?', 'What are the land use types in this scene?', 'What are the needs for the renovation of residents?', 'Are there any buildings in this scene?', 'Is there any agricultural land in this scene?', 'What is the area of roads?', 'What is the area of playgrounds?', 'Is it a rural or urban scene?', 'What is the area of barren?', 'Are there any bridges in this scene?', 'Are there any eutrophic waters in this scene?', 'Are there any viaducts in this scene?', 'What is the area of water?', 'Are there any roads in this scene?', 'Is there any residential land in this scene?', 'How many eutrophic waters are in this scene?', 'Is there any industrial land in this scene?', 'Is there any park land in this scene?', 'Is there any uncultivated agricultural land in this scene?', 'Is there a school near the residential area?', 'Are there any large driveways (more than four lanes)?', 'What are the comprehensive traffic situations in this scene?', 'What is the area of buildings?', 'Is there any construction land in this scene?', 'What are the water types in this scene?', 'Are there any viaducts near the residential area?', 'Is there a construction area near the residential area?', 'Is there a park near the residential area?', 'What are the road materials around the village?', 'Are there any intersections in this scene?', 'What are the road types around the residential area?', 'What are the water situations around the agricultural land?', 'What is the situation of barren land?', 'What is the area of the forest?', 'Are there any intersections near the school?', 'Is there any water in this scene?', 'Is there any educational land in this scene?', 'How many intersections are in this scene?', 'Are there any greenhouses in this scene?', 'What is the area of agriculture?']
QUESTION_VOC = [' ', 'materials', 'are', 'driveways', 'How', 'construction', 'industrial', 'be', 'area', 'barren', 'agricultural', 'road', 'educational', 'intersections', 'village', 'greenhouses', 'any', 'many', 'bridges', 'areas', 'scene', 'there', 'buildings', 'uncultivated', 'traffic', 'Is', 'residential', 'forest', 'woodland', 'supplemented', 'near', 'than', 'Are', 'eutrophic', 'this', 'playgrounds', 'situations', 'commercial', 'urban', 'land', 'school', 'residents', '(more', 'it', 'rural', 'viaducts', 'is', 'types', 'to', 'roads', 'the', 'for', 'greening', 'of', 'four', 'a', 'park', 'comprehensive', 'agriculture', 'in', 'What', 'villages', 'needs', 'around', 'water', 'situation', 'use', 'waters', 'or', 'large', 'need', 'lanes)', 'renovation', 'Whether']
ANSWER_VOC = [0, 1, 2, 3, 4, 5, 6, '0%-10%', '10%-20%', '20%-30%','30%-40%','40%-50%','50%-60%','60%-70%','70%-80%','80%-90%', '90%-100%',  'The roads need to be improved, and waters need to be cleaned up',  'This is an important traffic area with 3 intersections', 'There are residential, educational, park, and agricultural areas', 'Developing', 'There are railways',  'This is a very important traffic area with 1 intersection, several viaducts, and several bridges', 'There are cement roads', 'There are educational, construction, and agricultural areas', 'Underdeveloped', 'There are unsurfaced roads, and cement roads', 'There are residential, commercial, park, and agricultural areas', 'There are commercial areas', 'This is a very important traffic area with 2 intersections, and several viaducts', 'There are commercial, construction, and park areas', 'There are residential, commercial, park, industrial, and agricultural areas', 'There are commercial, and construction areas', 'This is not an important traffic area', 'This is a very important traffic area with 2 intersections, and several bridges', 'There are unsurfaced roads, and railways', 'There are woodland, industrial, and agricultural areas', 'There are park areas', 'There are construction, park, and agricultural areas', 'There are residential, and industrial areas', 'There are residential, and construction areas', 'There is no water', 'There are residential, construction, and park areas', 'There are commercial buildings', 'There are agricultural areas', 'There are educational areas', 'There are residential, and commercial areas', 'There are commercial, educational, park, and industrial areas', 'There are clean waters near the agriculture land', 'There are ponds', 'There are residential, commercial, park, and industrial areas', 'There are educational, park, industrial, and agricultural areas', 'There are unsurfaced roads, cement roads, railways, and asphalt roads', 'There are one-way lanes, and railways', 'There are residential, commercial, educational, park, and industrial areas', 'There are no water area', 'There are railways, and asphalt roads', 'There are construction areas', 'The urban villages need attention', 'There are unsurfaced roads, railways, and asphalt roads', 'There are residential, and agricultural areas', 'There are residential, commercial, and agricultural areas', 'No', 'This is a very important traffic area with 1 intersection, and several viaducts', 'The greening needs to be supplemented', 'There are residential, commercial, educational, and construction areas', 'This is an important traffic area with several bridges', 'There are residential, commercial, educational, and industrial areas', 'There are woodland areas', 'There are residential, commercial, and construction areas', 'Rural', 'There are residential, construction, park, industrial, and agricultural areas', 'There are residential, woodland, industrial, and agricultural areas', 'This is an important traffic area with 4 intersections', 'There are private buildings', 'There are woodland, and agricultural areas', 'There are residential, commercial, construction, and park areas',  'There are rivers and ponds', 'There are residential, construction, and agricultural areas', 'There are residential, and educational areas', 'There are commercial, and educational areas', 'There are polluted waters near the agriculture land', 'There are one-way lanes, wide lanes, and railways', 'There are one-way lanes, and wide lanes', 'Urban', 'There are residential, commercial, and educational areas', 'There are commercial, and park areas', 'There are unsurfaced roads, cement roads, and asphalt roads', 'There are commercial buildings, and private buildings',  'This is an important traffic area with 1 intersection', 'There are commercial, industrial, and agricultural areas', 'There are residential, commercial, construction, park, and industrial areas', 'There are asphalt roads', 'There are residential, commercial, and park areas', 'There are no agricultural land',  'There are commercial, construction, park, and agricultural areas', 'There are residential, educational, and construction areas', 'There are commercial, construction, and industrial areas', 'There are residential, commercial, construction, and industrial areas', 'There are park, and industrial areas', 'There are commercial, and agricultural areas', 'There are residential, educational, construction, and park areas', 'No obvious land use types', 'There are construction, park, and industrial areas', 'There are residential, educational, park, and industrial areas', 'There are commercial, park, and industrial areas', 'This is an important traffic area with several viaducts', 'This is a very important traffic area with 1 intersection, and several bridges', 'There are residential, park, and agricultural areas', 'There are residential, commercial, construction, and agricultural areas', 'There are residential, commercial, educational, construction, park, and agricultural areas', 'There are wide lanes, and railways', 'There are residential, park, and industrial areas', 'There are residential, industrial, and agricultural areas', 'There are construction, and park areas', 'There are residential, commercial, construction, park, industrial, and agricultural areas', 'There are residential, park, industrial, and agricultural areas', 'There are residential areas', 'There are residential, commercial, educational, park, and agricultural areas', 'There are residential, commercial, industrial, and agricultural areas', 'There are residential, commercial, educational, and park areas', 'There are construction, and agricultural areas', 'There are no water nor agricultural land', 'The waters need to be cleaned up', 'There are park, and agricultural areas', 'There are rivers', 'This is a very important traffic area with 3 intersections, and several viaducts', 'This is an important traffic area with 2 intersections', 'There are industrial areas', 'There are unsurfaced roads, and asphalt roads', 'This is a very important traffic area with 2 intersections, several viaducts, and several bridges', 'There are commercial, park, and agricultural areas', 'There are one-way lanes', 'There are residential, educational, construction, and agricultural areas', 'There are no roads', 'There are residential, construction, park, and agricultural areas', 'There are residential, and park areas', 'There are commercial, construction, and agricultural areas', 'There are cement roads, and asphalt roads',  'There are residential, educational, and agricultural areas', 'There are commercial, and industrial areas', 'There are park, industrial, and agricultural areas', 'This is a very important traffic area with several viaducts, and several bridges', 'There are educational, construction, and park areas', 'There are residential, woodland, and agricultural areas', 'There are residential, and woodland areas', 'There are unsurfaced roads, cement roads, and railways', 'There are educational, park, and agricultural areas', 'There are residential, educational, and park areas', 'There are commercial, educational, and park areas', 'There are wide lanes', 'There are cement roads, and railways', 'There are no residential buildings', 'There are commercial, park, industrial, and agricultural areas', 'There are residential, commercial, and industrial areas', 'The greening needs to be supplemented and urban villages need attention', 'There is no barren land', 'There are educational, and agricultural areas', 'The roads need to be improved', 'Yes', 'There are unsurfaced roads', 'There are residential, commercial, construction, park, and agricultural areas', 'There are residential, construction, and industrial areas', 'There are cement roads, railways, and asphalt roads', 'There are educational, and park areas', 'There are no needs']

QUESTION_TYPES = ['Basic Judging', 
                  'Reasoning-based Judging',  
                  'Basic Counting', 
                  'Reasoning-based Counting', 
                  'Object Situation Analysis', 
                  'Comprehensive Analysis']


er.registry.register_all()

logger = logging.getLogger(__name__)

er.registry.register_all()

def convert2str(indexes, map_dict=QUESTION_VOC):
    if isinstance(indexes, np.int64):
        converted_str = map_dict[indexes]
    else:
        converted_str = ' '.join([map_dict[idx] for idx in indexes if map_dict[idx] != ' ']) + '?'
    return converted_str

def process_single_image(feat_path, question_str):
    feat = h5py.File(feat_path, 'r')
    
    vqa_image = np.array(feat['feature']).transpose([1, 2, 0])
    vqa_image = torch.from_numpy(vqa_image).float()
    vqa_image = vqa_image.permute([2, 0, 1]).unsqueeze(0)
    
    mask = np.array(feat['pred_mask']).astype(np.int64) - 1    # (H, W)
    mask = torch.tensor(mask, dtype=torch.int64).unsqueeze(0)
    # Build vocab mapping: word -> index
    ques = question_str.strip('?').split(' ')
    encoded_ques = np.zeros(len(QUESTION_VOC)).astype(np.int64)
    for i, word in enumerate(ques):
        encoded_ques[i] = QUESTION_VOC.index(word)
    ques = torch.from_numpy(encoded_ques).unsqueeze(0)
    
    questype = 'Basic Judging'

    ret = {
        'imagen': os.path.basename(feat_path),
        'vqaimagen': os.path.basename(feat_path),
        'seg_mask': mask,
        'question': ques,
        'questype': questype,
        'question_len': len(ques),
        'answer': None  # placeholder
    }

    return vqa_image, ret
    # image_tensor.shape = ([batch_size,2048,32,32])
    # ret.keys =['imagen', 'vqaimagen', 'seg_mask', 'question', 'questype', 'question_len', 'answer']

def predict_soba(feat_path=None, question=None):
    cfg = import_config('/home/liw324/code/Segment/EarthVQA/configs/soba.py')
    ckpt_path = '/home/liw324/code/Segment/EarthVQA/weights/soba.pth'

    model_state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model = make_model(cfg['model'])
    model.load_state_dict(model_state_dict)
    model.cuda()
    model.eval()

    img, ret = process_single_image(feat_path, question)

    ques, questypes, imagen = ret['question'], ret['questype'], ret['imagen']
    
    preds = model(img, ret)

    # if isinstance(ques, str):
    #     ques_str = ques + '?'
    # else:
    #     ques_str = convert2str(ques, QUESTION_VOC)

    ans_idx = preds.argmax(dim=1).cpu().numpy()[0]
    
    ans_str = convert2str(ans_idx, ANSWER_VOC)

    print(f"Question: {question}")
    print(f"Answer: {ans_str}")
    
    return ans_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval methods')
    feature_path = '/home/liw324/code/Segment/EarthVQA/streamlit_script/out/tmpcmohs32g.hdf5'
    question = 'What is the area of buildings?'
    args = parser.parse_args()
    predict_soba(feature_path, question)
