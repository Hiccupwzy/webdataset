import webdataset as wds
import os
import tarfile
from PIL import Image
import yaml
import sys
import pickle
import json
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from diffusers.models import AutoencoderKL
import os
os.environ['FORCE_TQDM'] = '1'

image_model_id = "../../../stabilityai/sd-vae-ft-ema"
vae = AutoencoderKL.from_pretrained(image_model_id).cuda()
vae.eval()

with open('../train/filenames.pickle', 'rb') as handle:
    train_data_list = pickle.load(handle)

with open('../test/filenames.pickle', 'rb') as handle:
    test_data_list = pickle.load(handle)

image_folder_path = "../image/images"
captions_folder_path = "../text/celeba-caption"

# Check if the path exists
if not os.path.exists(image_folder_path) or not os.path.exists(captions_folder_path):
    print(f"Error: The path {image_folder_path} or {captions_folder_path} does not exist.")
    sys.exit(1)

output_dir = "../../storage/facehq_sharded"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

train_images = [os.path.join(image_folder_path,index+".jpg") for index in train_data_list]
test_images = [os.path.join(image_folder_path,index+".jpg") for index in test_data_list]
train_captions = [os.path.join(captions_folder_path,index+".txt") for index in train_data_list]
test_captions = [os.path.join(captions_folder_path,index+".txt") for index in test_data_list]


print(f"There are totally {len(train_images)} training images and {len(test_images)} testing images")
print(f"There are totally {len(train_captions)} training captions and {len(test_captions)} testing captions")
transf = transforms.Compose([transforms.Resize(1024),transforms.ToTensor()])


def shard_data(data, shard_count=66, output_dir='.', prefix='train-', rename_dict=None):
    shard_size = len(data) // shard_count + (len(data) % shard_count > 0)
    shard_generator = (data[i:i+shard_size] for i in range(0, len(data), shard_size))
    

    for shard_idx, shard in tqdm(enumerate(shard_generator),total=shard_count):
        
        shard_num_str = f"{shard_idx:06d}"
        tar_filename = os.path.join(output_dir, f"{prefix}{shard_num_str}.tar")
        sink = wds.TarWriter(tar_filename, encoder=False)

        for idx, (image_path, caption_path) in tqdm(enumerate(shard),total=shard_size):
            # split the shards so that they can be calculated using two GPUs
            if shard_idx>=26:
                print(f"processing shard {shard_idx}")
                break
            # print(f"loading image {image_path}")
            with open(image_path, "rb") as stream:
                image = stream.read()

            image_content = Image.open(image_path)
            img_tensor = transf(image_content)
            image_content = vae.encode(img_tensor.unsqueeze(0).cuda()).latent_dist.sample()[0]
            assert len(image_content.shape)==3
            latent = image_content.detach().cpu().numpy()
            np.save("temp.npy",latent)
            with open("temp.npy", "rb") as stream:
                latent = stream.read()
            sample = {
                "__key__": f"{idx:06d}.", # idx,   
                "image.png": image,
                "latent.npy": latent
            }
            sink.write(sample)
        sink.close()

    print(f"Sharding completed. {shard_count} shards created.")

train_images.sort()
train_captions.sort()
train_data= list(zip(train_images,train_captions))

test_images.sort()
test_captions.sort()
test_data= list(zip(test_images,test_captions))

shard_data(train_data, shard_count=66, output_dir=output_dir, prefix='train-')
shard_data(test_data, shard_count=12, output_dir=output_dir, prefix='test-')