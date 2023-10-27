# [Project 1] 2D homography computation with ORB and RANSAC

## &#128204; 과제 내용
1. Choose two images

2. compute ORB keypoint and descriptors (opencv)

3. apply Bruteforce matching with Hamming distance (opencv)

4. implement RANSAC algorithm to compute the homography matrix. (DIY)

5. prepare a panorama image of larger size (DIY)

6. warp two images to the panorama image using the homography matrix (DIY)

## &#128193; File Setting
📦Project1-Panorama<br>
 ┣ 📂image<br>
 ┃ ┣ 📜sogang1.jpg<br>
 ┃ ┣ 📜sogang2.jpg<br>
 ┃ ┗ 📜sogang_panorama.jpg<br>
 ┣ 📜Project1_homography_최수빈.py<br>
 ┃  ┣ &#128196;ORB(func) &nbsp;&nbsp;&nbsp; # extract keypoints, descriptors<br>
 ┃  ┣ &#128196;bruteforce_matching(func) &nbsp; # extract matched keypoints<br>
 ┃  ┣ &#128196;find_homography(func) &nbsp;&nbsp; # extract best homography matrix using RANSAC <br>
 ┃  ┗ &#128196;create_panorama(func) &nbsp;&nbsp;&nbsp; # create panorama using homography matrix <br> 
 ┗ 📜README.md

## &#128204; Usage
```bash
python Project1_homography_최수빈.py
```

## &#9989; Result
**sogang1.jpg (left)** <br>
<img src='./image/sogang1.jpg' width=400 height=300>


**sogang2.jpg (right)** <br>
<img src='./image/sogang2.jpg' width=400 height=300>


**sogang_panorama.jpg (panorama)** <br>
<img src='./image/sogang_panorama.jpg'>

