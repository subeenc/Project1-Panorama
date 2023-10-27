import cv2
import numpy as np
import random
import itertools
from tqdm import tqdm

# step 1) image load
def load_image(img_path):
    # 이미지 불러오기
    return cv2.imread(img_path)

# step 2) ORB를 수행하여 각 이미지에 대한 keypoints, descriptors 추출
def ORB(image):    
    # Convert the images to grayscale (for keypoint and descriptor matching)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    
    return keypoints, descriptors

# step 3) Hamming distance를 사용하여 Bruteforce matching 수행
def bruteforce_matching(keypoints1, keypoints2, descriptors1, descriptors2):
    # Hamming distance로 BruteForce matching 수행
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # 매칭 수행
    matches = bf.match(descriptors1, descriptors2)
    
    # 매칭 결과를 거리에 따라 정렬
    matches = sorted(matches, key=lambda x: x.distance)
    # 상위 N개 매칭 결과 선택 (N을 원하는 개수로 변경)
    N = 30
    top_matches = matches[:N]
    
    # 좋은 매칭으로 필터링
    good_matches = []
    for match in top_matches:
        if match.distance < 50:  # 거리를 조절하여 좋은 매칭을 선택
            good_matches.append(match)


    # 매칭된 키포인트 쌍을 추출
    matched_keypoints1 = np.float32([keypoints1[match.queryIdx].pt for match in good_matches])
    matched_keypoints2 = np.float32([keypoints2[match.trainIdx].pt for match in good_matches])
    
    return matched_keypoints1, matched_keypoints2


def find_homography(matched_keypoints1, matched_keypoints2):
    # RANSAC을 사용하여 호모그래피 행렬 추정
    best_inliers = 0
    best_H = None

    # 모든 가능한 4개의 키포인트 쌍 선택
    # 모든 가능한 조합의 개수 계산
    num_combinations = len(list(itertools.combinations(range(len(matched_keypoints1)), 4)))
    all_combinations = itertools.combinations(range(len(matched_keypoints1)), 4)
    for indices in tqdm(all_combinations, desc="RANSAC for Homography matrix..", total=num_combinations):
        selected_keypoints1 = matched_keypoints1[list(indices)]
        selected_keypoints2 = matched_keypoints2[list(indices)]
        
        # 호모그래피 행렬을 계산하기 위한 선형 방정식 설정
        A = np.zeros((8, 9))
        for i in range(4):
            x1, y1 = selected_keypoints1[i]
            x2, y2 = selected_keypoints2[i]
            A[i * 2] = [-x2, -y2, -1, 0, 0, 0, x2 * x1, y2 * x1, x1]
            A[i * 2 + 1] = [0, 0, 0, -x2, -y2, -1, x2 * y1, y2 * y1, y1]

        # A에 대한 특이값 분해를 수행하여 호모그래피 행렬 추출
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)

        # 정규화 (가장 마지막 원소를 1로 만들기)
        H /= H[2, 2]

        # RANSAC을 사용하여 inliers 찾기
        num_inliers = 0
        threshold = 5  # 임계값 조절 가능
        for i in range(len(matched_keypoints1)):
            x1, y1 = matched_keypoints1[i]
            x2, y2 = matched_keypoints2[i]

            # 이미지 2의 좌표를 호모그래피 행렬을 통해 이미지 1의 좌표로 변환
            transformed_point = np.dot(H, np.array([x2, y2, 1]))
            transformed_point /= transformed_point[2]

            # 이미지 1의 좌표와 변환된 좌표를 비교하여 inliers 판별
            dist = np.sqrt((x1 - transformed_point[0]) ** 2 + (y1 - transformed_point[1]) ** 2)
            if dist < threshold:
                num_inliers += 1

        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_H = H
    
    print('----------[best_H]----------')
    print(best_H)
    print('----------[best_inliers]----------')
    print(best_inliers)
    
    return best_H


def apply_homography(H, x, y):
    # Apply homography transformation to coordinates (x, y)
    homogenous_coords = np.array([x, y, 1])
    new_coords = np.dot(H, homogenous_coords)
    new_coords /= new_coords[2]
    return new_coords[:2]


def create_panorama(img1, img2, H):
    # Get dimensions of input images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Determine the output image dimensions
    min_x, min_y = apply_homography(H, 0, 0)
    max_x, max_y = apply_homography(H, w2, h2)
    panorama_width = int(max(max_x, w1))
    panorama_height = int(max(max_y, h1))

    # Initialize the panorama image
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)

    # Copy img1 to the panorama image
    panorama[:h1, :w1, :] = img1

    # Iterate through each pixel in img2 and apply the homography to map it to the panorama
    print('Create Panorama..')
    for y in tqdm(range(h2)):
        for x in range(w2):
            new_x, new_y = apply_homography(H, x, y)
            new_x = int(new_x)
            new_y = int(new_y)

            # Check if the new coordinates are within the bounds of the panorama
            if 0 <= new_x < panorama_width and 0 <= new_y < panorama_height:
                panorama[new_y, new_x, :] = img2[y, x, :]

    return panorama


if __name__ == "__main__":
    image1 = cv2.imread('image/sogang1.jpg')
    image2 = cv2.imread('image/sogang2.jpg')
    
    image1_gray = cv2.cvtColor(image1, cv2.IMREAD_GRAYSCALE)
    image2_gray = cv2.cvtColor(image2, cv2.IMREAD_GRAYSCALE)
    
    keypoints1, descriptors1 = ORB(image1_gray)
    keypoints2, descriptors2 = ORB(image2_gray)
    
    matched_keypoints1, matched_keypoints2 = bruteforce_matching(keypoints1,keypoints2, descriptors1, descriptors2)
    
    best_H = find_homography(matched_keypoints1, matched_keypoints2)

    # Create the panorama
    panorama = create_panorama(image1, image2, best_H)

    # Save the resulting panorama image
    cv2.imwrite('./image/sogang_panorama.jpg', panorama)
    print('Done.')