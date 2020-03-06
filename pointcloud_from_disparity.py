import numpy as np
import cv2
import open3d as o3d
######-----Reading poses-----#######
pose=[]
g_truth=open("poses.txt","r")
L=np.array([0,0,0,1])
for i in range(0,21):
    truc=np.array(list(map(float,g_truth.readline().split()))).reshape(3,4)
    truc=np.vstack((truc,L))
    # print(truc)
    pose.append(truc)

count=0
for pair in range(460,461):
    left=cv2.imread('img2/'+'0000000' + str(pair)+'.png')
    right=cv2.imread('img3/'+'0000000' + str(pair)+'.png')
    ######--------DISPARITY CALCULATION------######
    window_size = 5
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,             # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=3,
        P1=4  * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32  * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63
    )
    minDisparity=-39.0
    numDisparities=144.0
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    #WLS Filter
    lmbda = 60000
    sigma = 2
    visual_multiplier = 1
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(left, right).astype('float32')  
    dispr = right_matcher.compute(right, left)  
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, left, None, dispr)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)/16
    cv2.imshow('Disparity Map', filteredImg)
    cv2.waitKey(3000)
    print("Computed Disparity")
    #######------------#########
    # window_size = 5
    # stereo = cv2.StereoSGBM_create(minDisparity=-39,
    #     numDisparities=144,             
    #     blockSize=5,
    #     P1=8 * 3 * window_size ** 2,    
    #     P2=64 * 3 * window_size ** 2,
    #     disp12MaxDiff=1,
    #     uniquenessRatio=10,
    #     speckleWindowSize=100,
    #     speckleRange=32,
    #     preFilterCap=63)
    # numDisparities=144
    # minDisparity=-39
    # disparity = stereo.compute(right,left).astype('float32')/16
    # disparity = (disparity - minDisparity) / numDisparities
    # # disparity = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    # print(np.min(disparity),np.max(disparity))
    # cv2.imshow('Disparity Map', disparity)
    # cv2.waitKey(3000)
    #######------------#########
  
    
    depth=filteredImg
    augmented_disparity=[]
    print(depth.shape)
    for v in range(left.shape[0]):
        for u in range(right.shape[1]):
            augmented_disparity.append([u,v,depth[v,u],1])
    # print(augmented_disparity[57000])
    augmented_disparity=np.array(augmented_disparity)
    # print("Unique aug:",len(np.unique(augmented_disparity[:,2])))
    ######--------COLOUR EXTRACTION------------######
    left=cv2.cvtColor(left,cv2.COLOR_BGR2RGB)
    color=left.reshape((left.shape[0]*left.shape[1],3))
    # print("color shape:",color[300031:300057,:])
    ######--------Baseline Matrix Q--------------######
    focalLength = 7.070912e+02
    centerX = 6.018873e+02
    centerY = 1.831104e+02
    scalingFactor = 380.347
    baseline=0.53790448812 
    width=left.shape[0]
    height=right.shape[1]
    Q = np.array([[1,0,0,-width/2],[0,1,0,-height/2],[0,0,0,focalLength],[0,0,-1/baseline,0]])
    ######--------POINT CLOUD CALCULATION------######
    p_cloud=[]
    pc_color=[]
    mask=((augmented_disparity[:,2] > 30) & (augmented_disparity[:,2] < 10000))
    augmented_disparity=augmented_disparity[mask,:]
    color=(color[mask,:]/255.0).astype('float64')
    pc_color.append(color)
    threeDpts=Q.dot(augmented_disparity.T)
    threeDpts=threeDpts/threeDpts[3]
    threeDpts=pose[count].dot(threeDpts)
    p_cloud.append(threeDpts.T)    
    # p_cloud.append(np.hstack((augmented_disparity[:,:2],(scalingFactor/augmented_disparity[:,2]).reshape((augmented_disparity.shape[0],1)))))
    # print(p_cloud[0][300031:300057,:])
    ######--------PAIR COUNT-------------------######
    count=count+1
    print("Pair:",count)
cum_p_clouds=p_cloud[0]
cum_pc_colors=pc_color[0]
for i in range(1,len(p_cloud)):
    cum_p_clouds = np.concatenate((cum_p_clouds,p_cloud[i]),axis=0)
    cum_pc_colors= np.concatenate((cum_pc_colors,pc_color[i]),axis=0)
    print(i)
pcd = o3d.geometry.PointCloud()
print("Shape:", cum_p_clouds.shape)
pcd.points = o3d.utility.Vector3dVector(cum_p_clouds[:,:3])
pcd.colors = o3d.utility.Vector3dVector(cum_pc_colors)
o3d.io.write_point_cloud("all_pcs.ply", pcd)
o3d.visualization.draw_geometries([pcd])