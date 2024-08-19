import numpy as np
import pandas as pd
import os
import sys
import lxml.etree as ET
import cv2
import tiffslide as openslide
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation,binary_erosion
from skimage.morphology import diamond
from skimage import measure
import tifffile as ti
import ome_types as ot
from glob import glob
from tqdm import tqdm
import time
# import bioformats
from PIL import Image
from xml_to_mask_ome import xml_to_mask
print(cv2.__version__)


def run_convert(filepath):
    file_ome_tif = filepath

    # file_output = filepath[:-1]
    file_output = filepath.split('.ome')[0] + '.segmentations.ome.tif'

    file_ome_xml = filepath.split('.tiff')[0]+ '.xml'

    cmdstring = 'tiffcomment -set '+ file_ome_xml + ' ' + file_ome_tif
    cmdstring2 = 'BF_MAX_MEM=32000M bfconvert -tilex 512 -tiley 512 -pyramid-resolutions 6 -pyramid-scale 2 -compression LZW '+filepath + ' ' + file_output

    cmdstring3 = 'rm ' + filepath

    os.system(cmdstring)
    os.system(cmdstring2)
    os.system(cmdstring3)

    return file_output

#
#SCRIPT STARTS HERE
wsi_ext = ['.svs','.scn']

channel_names = ['cortical-interstitium','medullary-interstitium','glomeruli','tubules','arteries-arterioles']
ontology_names = ['0005270','0005211','0000074','0009773','0001637']

slide_dir = '/orange/pinaki.sarder/nlucarelli/ReferenceSCNsFixed/'
save_dir = '/orange/pinaki.sarder/nlucarelli/TEST2/'

csv_file = 'segmentation-masks.csv'
csv_columns = ['Channel number','Mask name','Source file','Ontology abbreviation','Ontology ID','Protocol','Is entire image masked','Num objects']
protocol_doi = 'dx.doi.org/10.17504/protocols.io.dm6gp35p8vzp/v1'

things = [3,4,5,6]
stuff = [1,2]
mpp=.25

slides = []
for ext in wsi_ext:
    slides.extend(glob(slide_dir + '*' + ext))

si=-1
csv_lines = []

for slide in slides:
    si+=1

    slide_name = save_dir + slide.split('/')[-1].split('.')[0] + '.ome.tiff'
    seg_name = slide_name.split('.ome')[0] + '.segmentations.ome.tiff'

    if os.path.exists(seg_name):
        print(f'Already completed, skipping: {seg_name}')
        continue

    sl = openslide.OpenSlide(slide)

    if slide[-4:] == '.svs':
    #THIS BLOCK JUST FOR THE SVS
        x,y = sl.dimensions
        x=int(x)
        y=int(y)
    #THIS BLOCK JUST FOR THE SCNS
    elif slide[-4:] == '.scn':
        coords = pd.read_csv('/orange/pinaki.sarder/nlucarelli/ReferenceSCNsFixed/coords.csv').to_numpy()[:,1:]
        x = coords[si,0]
        y = coords[si,1]
    else:
        print(f'Extension {slide[-4:]} not recognized, skipping {slide}')
        continue

    nm = slide.rsplit(".", 1)[0]+'.xml'

    if not os.path.exists(nm):
        print(f'No annotation file {nm} found... skipping {slide}')
        continue


    full = np.zeros((len(channel_names),y,x))
    full = full.astype(np.uint16)

    mask = xml_to_mask(nm,(0,0),(x,y),[1,2,3,4,5,6])

    obj_num = []
    channel_num = []
    ont_nm = []
    channel_nm = []

    ii=-1

    for cl in tqdm(range(len(channel_names)+1)):
        # print(cl)

        if cl+1 in stuff:
            mask_layer = (mask==cl+1).astype(np.uint8)
            obj_num.append(1)
            channel_num.append(cl)
            channel_nm.append(channel_names[cl])
            ont_nm.append(ontology_names[cl])
            ii+=1
        elif cl+1 in things:
            mask_layer = (mask==cl+1).astype(np.uint8)
            mask_layer = measure.label(mask_layer)
            if cl == 2:
                max_gloms = np.max(mask_layer)
                # obj_num.append(np.max(mask_layer))
                channel_num.append(cl)
                channel_nm.append(channel_names[cl])
                ont_nm.append(ontology_names[cl])
                ii+=1
            elif cl > 3:
                obj_num.append(np.max(mask_layer))
                channel_num.append(cl-1)
                channel_nm.append(channel_names[cl-1])
                ont_nm.append(ontology_names[cl-1])
                ii+=1
            else:
                mask_layer2 = mask_layer + max_gloms
                mask_layer2[mask_layer==0] = 0
                mask_layer = mask_layer2
                obj_num.append(np.max(mask_layer))
                del mask_layer2
            # mask_layer = mask_layer.astype(np.uint16)
        else:
            print(f'Channel {cl} not listed...')
            break

        full[ii,:,:] = full[ii,:,:] + mask_layer

    del mask, mask_layer

    save_dir + slide.split('/')[-1].split('.')[0] + '.ome.tiff'

    for channels in range(len(channel_num)):
        current_line = []
        current_line.append(channel_num[channels])
        current_line.append(channel_nm[channels])
        current_line.append(slide.split('/')[-1].split('.')[0] + '.segmentations.ome.tiff')
        current_line.append('Uberon')
        current_line.append(ont_nm[channels])
        current_line.append(protocol_doi)
        current_line.append('Yes')
        current_line.append(obj_num[channels])
        csv_lines.append(current_line)


    #WRITING

    tiff_writer = ti.TiffWriter(slide_name,ome=True,bigtiff=True)
    tiff_writer.write(full,metadata={'axes':'CYX'})
    tiff_writer.close()

    tiff_file = ot.from_tiff(slide_name)

    full_list = channel_names
    for i,c in enumerate(tiff_file.images[0].pixels.channels):
        c.name = full_list[i]
        print(full_list[i])


    xml_name = save_dir + slide.split('/')[-1].split('.')[0] + '.ome.xml'
    xml_data = ot.to_xml(tiff_file)
    xml_data = xml_data.replace('<Pixels','<Pixels PhysicalSizeXUnit="\u03BCm" PhysicalSizeYUnit="\u03BCm"')
    with open(xml_name,'wt+') as fh:
        fh.write(xml_data)


    output_name = run_convert(slide_name)
    print(output_name)
    #

    csv_df = pd.DataFrame(csv_lines,columns=csv_columns)
    csv_df.to_csv(save_dir + csv_file,index=False)
