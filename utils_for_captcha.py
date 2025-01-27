import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from concurrent.futures import ThreadPoolExecutor
import sys
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
global_centers = None
backgrounds = []
for i in range(1, 4):
    background_path = fr"background\{i}.jpg"
    background = cv2.imread(background_path)
    if background is None:
        print(f"Failed to read background images: {background_path}")
    else:
        backgrounds.append(background)

def calculate_difference(image, background):
    background_resized = cv2.resize(background, (image.shape[1], image.shape[0]))
    diff_b = cv2.absdiff(image[:, :, 0], background_resized[:, :, 0])
    diff_g = cv2.absdiff(image[:, :, 1], background_resized[:, :, 1])
    diff_r = cv2.absdiff(image[:, :, 2], background_resized[:, :, 2])
    diff = cv2.merge([diff_b, diff_g, diff_r])
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(diff_gray, 50, 255, cv2.THRESH_BINARY)
    return thresh

def find_best_background(image, backgrounds):
    min_diff = float('inf')
    best_background = None
    for background in backgrounds:
        thresh = calculate_difference(image, background)
        diff_count = np.sum(thresh == 255)
        if diff_count < min_diff:
            min_diff = diff_count
            best_background = background
    best_background = cv2.resize(best_background, (image.shape[1], image.shape[0]))
    return best_background


def process_image_with_clustering(image, backgrounds):
    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, np.ndarray):
        pass
    elif isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("Unsupported image format")
    if image is None:
        print(f"Failed to read the image")
        sys.exit("The process has ended,exit code 1")
    best_background = find_best_background(image, backgrounds)
    best_background = cv2.cvtColor(best_background, cv2.COLOR_BGR2RGB)
    diff = cv2.absdiff(image, best_background)
    threshold = 25
    result = np.where(diff > threshold, diff, 255)
    final_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    foreground_coords = np.column_stack(np.where(final_gray < 255))
    if len(foreground_coords) == 0:
        print("Invalid image")
        sys.exit("The process has ended,exit code 1")
    dbscan = DBSCAN(eps=20, min_samples=50)
    labels = dbscan.fit_predict(foreground_coords)
    unique_labels = np.unique(labels)
    centers = []
    for label in unique_labels:
        if label != -1:
            cluster_points = foreground_coords[labels == label]
            center = cluster_points.mean(axis=0).astype(int)
            centers.append(center)
    centers = np.array(centers)
    global global_centers
    global_centers=centers
    return centers

def crop_and_save(image):
    Segmentl = []
    image = Image.fromarray(image)
    resized_image = image.resize((800, 600), Image.Resampling.LANCZOS)
    centers = process_image_with_clustering(resized_image, backgrounds)
    sorted_centers = sorted(centers, key=lambda x: x[1])
    for center in sorted_centers:
        left = max(0, center[1] - 80)
        upper = max(0, center[0] - 80)
        right = min(800, center[1] + 80)
        lower = min(600, center[0] + 80)
        segment = resized_image.crop((left, upper, right, lower))
        segment = segment.resize((60, 60), Image.Resampling.LANCZOS)
        Segmentl.append(segment)
    return Segmentl

label_to_index = {'一': 1, '万': 2, '上': 3, '不': 4, '临': 5, '久': 6, '之': 7, '乐': 8, '书': 9, '争': 10, '事': 11, '云': 12, '亡': 13, '以': 14, '体': 15, '俭': 16, '倦': 17, '具': 18, '军': 19, '农': 20, '凌': 21, '出': 22, '别': 23, '到': 24, '刺': 25, '前': 26, '力': 27, '劝': 28, '功': 29, '务': 30, '勃': 31, '勇': 32, '勤': 33, '匠': 34, '十': 35, '千': 36, '卧': 37, '厌': 38, '厚': 39, '发': 40, '只': 41, '合': 42, '同': 43, '和': 44, '咬': 45, '图': 46, '地': 47, '坐': 48, '坚': 49, '壮': 50, '夕': 51, '大': 52, '天': 53, '头': 54, '奋': 55, '奔': 56, '孙': 57, '孜': 58, '学': 59, '定': 60, '寒': 61, '寝': 62, '小': 63, '尝': 64, '尺': 65, '屈': 66, '帜': 67, '平': 68, '年': 69, '幼': 70, '废': 71, '康': 72, '延': 73, '开': 74, '异': 75, '强': 76, '往': 77, '待': 78, '德': 79, '心': 80, '志': 81, '忘': 82, '怡': 83, '急': 84, '恒': 85, '息': 86, '悬': 87, '愤': 88, '慕': 89, '懈': 90, '成': 91, '扬': 92, '技': 93, '折': 94, '持': 95, '挠': 96, '换': 97, '推': 98, '改': 99, '故': 100, '斗': 101, '新': 102, '日': 103, '旦': 104, '旷': 105, '昂': 106, '映': 107, '智': 108, '曼': 109, '朋': 110, '朝': 111, '来': 112, '杵': 113, '标': 114, '树': 115, '根': 116, '格': 117, '梁': 118, '步': 119, '民': 120, '气': 121, '水': 122, '池': 123, '游': 124, '点': 125, '牙': 126, '牢': 127, '独': 128, '画': 129, '百': 130, '直': 131, '睛': 132, '神': 133, '移': 134, '窗': 135, '立': 136, '竿': 137, '紧': 138, '练': 139, '继': 140, '羊': 141, '而': 142, '股': 143, '胆': 144, '腾': 145, '自': 146, '舍': 147, '舞': 148, '节': 149, '苦': 150, '蓬': 151, '薪': 152, '虫': 153, '行': 154, '补': 155, '读': 156, '起': 157, '跛': 158, '身': 159, '躬': 160, '车': 161, '辍': 162, '追': 163, '里': 164, '针': 165, '铁': 166, '锲': 167, '闻': 168, '陈': 169, '雕': 170, '雪': 171, '革': 172, '顾': 173, '食': 174, '马': 175, '鱼': 176, '鸡': 177, '鼎': 178, '龙': 179}
index_to_label = {v-1: k for k, v in label_to_index.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 179
model = models.mobilenet_v2(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, num_labels)
model.load_state_dict(torch.load(r"single character recognition.pth", map_location=device,weights_only=True))
model = model.to(device)
model.eval()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_top_k_predictions(output, k=5):
    probabilities = F.softmax(output, dim=1)
    top_probs, top_indices = probabilities.topk(k)
    top_labels = [index_to_label[idx.item()] for idx in top_indices[0]]
    top_probs = top_probs[0].tolist()
    return list(zip(top_labels, top_probs))

def predict_segment(image):
    result = {}
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        top_predictions = get_top_k_predictions(output)
    print("Top 5 predictions with probabilities:")
    for label, prob in top_predictions:
        result[label]=(f"{prob:.4f}")
    return result

label_to_index = {'同德一心': 1, '车水马龙': 2, '百尺竿头': 3, '大智大勇': 4, '坚持不懈': 5, '持之以恒': 6, '奋不顾身': 7, '朝气蓬勃': 8, '咬紧牙根': 9, '志坚行苦': 10, '奋身独步': 11, '废寝忘食': 12, '鱼龙曼延': 13, '不屈不挠': 14, '乐事劝功': 15, '悬梁刺股': 16, '锲而不舍': 17, '坐以待旦': 18, '亡羊补牢': 19, '临池学书': 20, '孜孜不辍': 21, '勇往直前': 22, '幼学壮行': 23, '自强不息': 24, '画龙点睛': 25, '坐薪悬胆': 26, '节俭力行': 27, '革故鼎新': 28, '十年寒窗': 29, '推陈出新': 30, '只争朝夕': 31, '雕虫小技': 32, '标新立异': 33, '奋发图强': 34, '继往开来': 35, '壮志凌云': 36, '力争上游': 37, '别具匠心': 38, '铁杵成针': 39, '急起直追': 40, '改天换地': 41, '卧薪尝胆': 42, '孙康映雪': 43, '马到成功': 44, '坚定不移': 45, '务农息民': 46, '百折不挠': 47, '勤学苦练': 48, '久坐地厚': 49, '心慕力追': 50, '独树一帜': 51, '朋心合力': 52, '朝夕不倦': 53, '躬体力行': 54, '别具一格': 55, '跛行千里': 56, '一日千里': 57, '发愤忘食': 58, '心旷神怡': 59, '千军万马': 60, '万马奔腾': 61, '心平气和': 62, '闻鸡起舞': 63, '折节读书': 64, '发奋图强': 65, '学而不厌': 66, '斗志昂扬': 67}
keys=list(label_to_index.keys())

def process_segment(segment):
    segment = transform(segment).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(segment)
    return dict(get_top_k_predictions(output, k=179))
def match_word(image):
    segments = crop_and_save(image)
    out = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(process_segment, segments)
        out = list(results)
    best_word = None
    best_score = 0
    best_segments = []

    for word in keys:
        word_score = 1
        word_segments = [-1] * 4
        char_segment_probs = []

        for i, char in enumerate(word):
            for segment_idx in range(4):
                prob = out[segment_idx].get(char, 0)
                char_segment_probs.append((char, segment_idx + 1, prob))
        char_segment_probs.sort(key=lambda x: x[2], reverse=True)
        assigned_segments = set()
        assigned_char = set()
        for char, segment, prob in char_segment_probs:
            if segment not in assigned_segments and char not in assigned_char:
                idx = word.index(char)
                word_segments[idx] = segment
                word_score += -np.log(1.01 - prob)
                assigned_segments.add(segment)
                assigned_char.add(char)
                if len(assigned_segments) == 4:
                    break

        if len(assigned_segments) == 3 and word_score > best_score:
            remaining_segments = {1, 2, 3, 4} - assigned_segments
            remaining_segment = remaining_segments.pop()
            for i in range(4):
                if word_segments[i] == -1:
                    word_segments[i] = remaining_segment
            best_score = word_score
            best_word = word
            best_segments = word_segments
        if len(assigned_segments) == 4 and word_score > best_score:
            best_score = word_score
            best_word = word
            best_segments = word_segments
    return best_word, best_score, best_segments