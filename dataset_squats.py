import numpy as np
import os
import pickle
import torch
import cv2 as cv
import re


from torch.utils.data import Dataset, DataLoader

CATEGORY_INDEX = {
    "boxing": 0,
    "handclapping": 1,
    "handwaving": 2,
    "jogging": 3,
    "running": 4,
    "squatting": 5
#     "squatting": 5,
#     "walking": 6
}

class RawDataset(Dataset):
    def __init__(self, directory, dataset="train"):
        self.instances, self.labels = self.read_dataset(directory, dataset)

        self.instances = torch.from_numpy(self.instances)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, idx):
        sample = { 
            "instance": self.instances[idx],
            "label": self.labels[idx] 
        }

        return sample

    def zero_center(self, mean):
        self.instances -= float(mean)

    def read_dataset(self, directory, dataset="train"):
        if dataset == "train":
            filepath = os.path.join(directory, "train.p")
        elif dataset == "dev":
            filepath = os.path.join(directory, "dev.p")
        else:
            filepath = os.path.join(directory, "test.p")

        videos = pickle.load(open(filepath, "rb"))

        instances = []
        labels = []
        for video in videos:
            for frame in video["frames"]:
                instances.append(frame.reshape((1, 60, 80)))
                labels.append(CATEGORY_INDEX[video["category"]])

        instances = np.array(instances, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint8)

        self.mean = np.mean(instances)

        return instances, labels

class BlockFrameDataset(Dataset):
    def __init__(self, directory, dataset="train"):
        self.instances, self.labels = self.read_dataset(directory, dataset)

        self.instances = torch.from_numpy(self.instances)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, idx):
        sample = { 
            "instance": self.instances[idx], 
            "label": self.labels[idx] 
        }

        return sample

    def zero_center(self, mean):
        self.instances -= float(mean)

    def read_dataset(self, directory, dataset="train", mean=None):
        if dataset == "train":
            filepath = os.path.join(directory, "train.p")
        elif dataset == "dev":
            filepath = os.path.join(directory, "dev.p")
        else:
            filepath = os.path.join(directory, "test.p")

        videos = pickle.load(open(filepath, "rb"))

        instances = []
        labels = []
        current_block = []
        for video in videos:
            for i, frame in enumerate(video["frames"]):
                current_block.append(frame)
                if len(current_block) % 15 == 0:
                    current_block = np.array(current_block)
                    instances.append(current_block.reshape((1, 15, 60, 80)))
                    labels.append(CATEGORY_INDEX[video["category"]])
                    current_block = []

        instances = np.array(instances, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint8)

        self.mean = np.mean(instances)

        return instances, labels

class BlockFrameDataset(Dataset):
    def __init__(self, directory, dataset="train"):
        self.instances, self.labels = self.read_dataset(directory, dataset)

        self.instances = torch.from_numpy(self.instances)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, idx):
        sample = { 
            "instance": self.instances[idx], 
            "label": self.labels[idx] 
        }

        return sample

    def zero_center(self, mean):
        self.instances -= float(mean)

    def read_dataset(self, directory, dataset="train", mean=None):
        if dataset == "train":
            filepath = os.path.join(directory, "train.p")
        elif dataset == "dev":
            filepath = os.path.join(directory, "dev.p")
        else:
            filepath = os.path.join(directory, "test.p")

        videos = pickle.load(open(filepath, "rb"))

        instances = []
        labels = []
        current_block = []
        for video in videos:
            for i, frame in enumerate(video["frames"]):
                current_block.append(frame)
                if len(current_block) % 15 == 0:
                    current_block = np.array(current_block)
                    instances.append(current_block.reshape((1, 15, 60, 80)))
                    labels.append(CATEGORY_INDEX[video["category"]])
                    current_block = []

        instances = np.array(instances, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint8)

        self.mean = np.mean(instances)

        return instances, labels
    
    
    
    
    
    
class SquatDataset(Dataset):
    def __init__(self, directory, dataset="train"):
        self.instances, self.labels = self.read_dataset(directory, dataset)

        self.instances = torch.from_numpy(self.instances)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, idx):
        sample = { 
            "instance": self.instances[idx], 
            "label": self.labels[idx] 
        }

        return sample

    def zero_center(self, mean):
        self.instances -= float(mean)
        
# def load_data_for_persons(root_folder, start_index, finish_index, frames_per_clip)
    def read_dataset(self, directory, dataset="train", mean=None):
        
        _nsre = re.compile('([0-9]+)')
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]
        
        if dataset == "train":
            root_folder = directory
            start_index = 1
            finish_index = 14
            frames_per_clip = 25
#             filepath = os.path.join(directory, "train.p")
        elif dataset == "dev":
            root_folder = directory
            start_index = 15
            finish_index = 18
            frames_per_clip = 25        
#             filepath = os.path.join(directory, "dev.p")
        else:
            filepath = os.path.join(directory, "test.p")

#         videos = pickle.load(open(filepath, "rb"))

        instances = []
        labels = []
        current_block = []
        class_labels = ["boxing", "handclapping", "handwaving", "jogging", "running", "squatting"] # 6 labels
        frame_path = "/frames2/"
        frame_set_prefix = "person" # 2 digit person ID [01..25] follows
        rec_prefix = "d" # seq ID [1..4] follows
        rec_count = 4
        seg_prefix = "seg" # seq ID [1..4] follows
        seg_count = 4

        data_array = []
#     data_array_np = np.zeros((frames_per_clip, rec_count*seg_count*25*len(class_labels), 120, 160))
        data_array_np = np.zeros((rec_count*seg_count*(finish_index-start_index+1)*len(class_labels), frames_per_clip, 120, 160, 3))
    
        classes_array = []
        z = 0

    # let's make a couple of loops to generate all of them
        for i in range(0, len(class_labels)):
        # class
            class_folder = root_folder + class_labels[i] + frame_path

            for j in range(start_index, finish_index+1):
            # person
                if j<10:
                    person_folder = class_folder + frame_set_prefix + "0" + str(j) + "_" + class_labels[i] + "_"
                else:
                    person_folder = class_folder + frame_set_prefix + str(j) + "_" + class_labels[i] + "_"

                for k in range(1,rec_count+1):
                # recording
                    rec_folder = person_folder + rec_prefix + str(k) + "/"
                    for m in range(1,seg_count+1):
                    # segment
                        seg_folder = rec_folder + seg_prefix + str(m) + "/"

                    # get the list of files
                        file_list = [f for f in os.listdir(seg_folder)]
                        example_size = len(file_list)

                    # for larger segments, we can change the starting point to augment the data
                        clip_start_index = 0
                        if example_size > frames_per_clip:
                        # set a random starting point but fix length - augments data, but slows training
                        #clip_start_index = randint(0, (example_size - frames_per_clip))
                        # sample the frames from the center
                            clip_start_index = int(example_size/2 - frames_per_clip/2)
                            example_size = frames_per_clip

                    # need natural sort before loading data
                        file_list.sort(key=natural_sort_key)

                    #create a list for each segment
                        current_seg_temp = []
                        for n in range(clip_start_index,example_size+clip_start_index):
                            file_path = seg_folder + file_list[n]
                            print(file_path)
                            frame = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
                            frame = cv.resize(frame, (80, 60), interpolation = cv.INTER_AREA)
#                             frame = np.array(frame)
                            current_block.append(frame)
                            if len(current_block) % 15 == 0:
                                current_block = np.array(current_block)
                                instances.append(current_block.reshape((1, 15, 60, 80)))
                                labels.append(i)
                                current_block = []
                            
                            
#                             data = np.asarray( Image.open(file_path), dtype='uint8' )
#                             try:
#                             # remove unnecessary channels
#                                 data_gray = np.delete(data,[1,2],2)
#                                 data_gray = data_gray.astype('float32')/255.0
#                                 current_seg_temp.append(data_gray)
#                                 data_array_np[z,n-clip_start_index,:,:,:] = data_gray
#                             except:
#                                 data_array_np[z,n-clip_start_index,:,:,:] = data.reshape(120,160,1)
#                 frame = Image.fromarray(np.array(frame))
#                 frame = frame.convert("L")
#                 frame = np.array(frame.getdata(),
#                                  dtype=np.uint8).reshape((120, 160))
#                 im = Image.fromarray(frame)
#                 size = tuple((np.array(im.size) * 0.5).astype(int))
#                 frame = np.array(im.resize(size, PIL.Image.BICUBIC))



#         backSub = cv.createBackgroundSubtractorMOG2()
#         for video in videos:
#             for i, frame in enumerate(video["frames"]):
#                 fgMask = backSub.apply(frame)
#                 current_block.append(fgMask)
#                 if len(current_block) % 15 == 0:
#                     current_block = np.array(current_block)
#                     instances.append(current_block.reshape((1, 15, 60, 80)))
#                     labels.append(CATEGORY_INDEX[video["category"]])
#                     current_block = []



        instances = np.array(instances, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint8)

        self.mean = np.mean(instances)

        return instances, labels

class CuboidDataset(Dataset):
    def __init__(self, directory, dataset="train"):
        self.instances, self.labels = self.read_dataset(directory, dataset)

        self.instances = torch.from_numpy(self.instances)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, idx):
        sample = { 
            "instance": self.instances[idx], 
            "label": self.labels[idx] 
        }

        return sample

    def zero_center(self, mean):
        self.instances -= float(mean)

    def read_dataset(self, directory, dataset="train", mean=None):
        if dataset == "train":
            filepath = os.path.join(directory, "train.p")
        elif dataset == "dev":
            filepath = os.path.join(directory, "dev.p")
        else:
            filepath = os.path.join(directory, "test.p")

        videos = pickle.load(open(filepath, "rb"))

        instances = []
        labels = []
        current_block = []
        for video in videos:
            firstimage = None
            first = True
            for i, frame in enumerate(video["frames"]):
                secondimage = frame
                if first:
                    first = False
                    firstimage = secondimage
                    continue
                result = None
                try:
                    diff = cv.absdiff(firstimage, secondimage)
                    threshold_value = 40
                    set_to_value = 255
                    ret, result = cv.threshold(diff, threshold_value, set_to_value, cv.THRESH_BINARY)
                except:
                    pass
                firstimage = secondimage
                if result is not None:
                    current_block.append(result)
                if len(current_block) % 15 == 0:
                    current_block = np.array(current_block)
                    instances.append(current_block.reshape((1, 15, 60, 80)))
                    labels.append(CATEGORY_INDEX[video["category"]])
                    current_block = []

        instances = np.array(instances, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint8)

        self.mean = np.mean(instances)

        return instances, labels
                    

class BlockFrameFlowDataset(Dataset):
    def __init__(self, directory, dataset="train", mean=None):
        self.instances, self.labels = self.read_dataset(directory, dataset)

        for i in range(len(self.instances)):
            self.instances[i]["frames"] = torch.from_numpy(
                self.instances[i]["frames"])
            self.instances[i]["flow_x"] = torch.from_numpy(
                self.instances[i]["flow_x"])
            self.instances[i]["flow_y"] = torch.from_numpy(
                self.instances[i]["flow_y"])

        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        sample = { 
            "instance": self.instances[idx], 
            "label": self.labels[idx] 
        }

        return sample

    def zero_center(self, mean):
        for i in range(len(self.instances)):
            self.instances[i]["frames"] -= float(mean["frames"])
            self.instances[i]["flow_x"] -= float(mean["flow_x"])
            self.instances[i]["flow_y"] -= float(mean["flow_y"])

    def read_dataset(self, directory, dataset="train", mean=None):
        if dataset == "train":
            frame_path = os.path.join(directory, "train.p")
            flow_path = os.path.join(directory, "train_flow.p")
        elif dataset == "dev":
            frame_path = os.path.join(directory, "dev.p")
            flow_path = os.path.join(directory, "dev_flow.p")
        else:
            frame_path = os.path.join(directory, "test.p")
            flow_path = os.path.join(directory, "test_flow.p")

        video_frames = pickle.load(open(frame_path, "rb"))
        video_flows = pickle.load(open(flow_path, "rb"))

        instances = []
        labels = []

        mean_frames = 0
        mean_flow_x = 0
        mean_flow_y = 0

        for i_video in range(len(video_frames)):
            current_block_frame = []
            current_block_flow_x = []
            current_block_flow_y = []

            frames = video_frames[i_video]["frames"]
            flow_x = [0] + video_flows[i_video]["flow_x"]
            flow_y = [0] + video_flows[i_video]["flow_y"]

            for i_frame in range(len(frames)):
                current_block_frame.append(frames[i_frame])

                if i_frame % 15 > 0:
                    current_block_flow_x.append(flow_x[i_frame])
                    current_block_flow_y.append(flow_y[i_frame])

                if (i_frame + 1) % 15 == 0:
                    current_block_frame = np.array(
                        current_block_frame,
                        dtype=np.float32).reshape((1, 15, 60, 80))
                    current_block_flow_x = np.array(
                        current_block_flow_x,
                        dtype=np.float32).reshape((1, 14, 30, 40))
                    current_block_flow_y = np.array(
                        current_block_flow_y,
                        dtype=np.float32).reshape((1, 14, 30, 40))

                    mean_frames += np.mean(current_block_frame)
                    mean_flow_x += np.mean(current_block_flow_x)
                    mean_flow_y += np.mean(current_block_flow_y)

                    instances.append({
                        "frames": current_block_frame,
                        "flow_x": current_block_flow_x,
                        "flow_y": current_block_flow_y
                    })

                    labels.append(
                        CATEGORY_INDEX[video_frames[i_video]["category"]])

                    current_block_frame = []
                    current_block_flow_x = []
                    current_block_flow_y = []

        mean_frames /= len(instances)
        mean_flow_x /= len(instances)
        mean_flow_y /= len(instances)

        self.mean = {
            "frames": mean_frames,
            "flow_x": mean_flow_x,
            "flow_y": mean_flow_y
        }

        labels = np.array(labels, dtype=np.uint8)

        return instances, labels
    