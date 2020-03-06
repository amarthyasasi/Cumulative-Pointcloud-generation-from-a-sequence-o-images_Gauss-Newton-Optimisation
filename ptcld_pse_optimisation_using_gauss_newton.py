import numpy as np
from sklearn.preprocessing import normalize
import cv2
import open3d as o3d
left = cv2.imread('left.png')
right = cv2.imread('right.png')

window_size = 3;
left_matcher = cv2.StereoSGBM_create(
    minDisparity=-39,
    numDisparities=144,             
    blockSize=5,
    P1=8 * 3 * window_size ** 2,    
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
#WLS Filter
lmbda = 80000
sigma = 1.7
visual_multiplier = 1.0
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
displ = left_matcher.compute(left, right)  
dispr = right_matcher.compute(right, left)  
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, left, None, dispr)
# cv2.imshow('Filtered Map', filteredImg)
filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
# print(filteredImg[:10,:])
filteredImg = np.uint8(filteredImg)
# cv2.imshow('Disparity Map', filteredImg)
# cv2.waitKey(30)
print "Computed Disparity"
# print("Shape:",filteredImg.shape)
x=filteredImg.reshape((filteredImg.shape[0]*filteredImg.shape[1],1))
# print("Unique:",len(np.unique(x)))
# create point cloud with different files for the images
focalLength = 7.070912e+02
centerX = 6.018873e+02
centerY = 1.831104e+02
scalingFactor = 380.347
baseline=0.53790448812 
rgb = left
depth =filteredImg
# print rgb.shape
# print depth.shape

points = []    

depth=depth+0.
threeDpts=[]
i=0
for v in range(rgb.shape[1]):
    for u in range(rgb.shape[0]):
        color = rgb[u,v]
        # if depth[u,v]>10:
        Z = depth[u,v]/scalingFactor
        if Z==0: continue
        X = (u - centerX) * Z**0 / focalLength
        Y = (v - centerY) * Z**0 / focalLength
        points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
        # threeDpts.append([X,Y,Z])

file = open('out.ply',"w")
file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points)))
file.close()

######-----GAUSS NEWTON------#######
pcd = o3d.io.read_point_cloud("./all_pcs.ply")
K = np.array([[7.070912e+02, 0.000000e+00, 6.018873e+02], [0.000000e+00, 7.070912e+02, 1.831104e+02], [0.000000e+00, 0.000000e+00, 1.000000e+00]])
# K=np.eye(3,3)
pose=np.array([[-9.098548e-01,5.445376e-02,-4.113381e-01,-1.872835e+02], [4.117828e-02, 9.983072e-01, 4.107410e-02, 1.870218e+00], [4.128785e-01,2.043327e-02,-9.105569e-01,5.417085e+01]])
# pose=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
# print "pose.shape:",pose.shape
threeDpts=pcd.points
threeDpts=np.array(threeDpts)
# print("points.shape:",threeDpts[365:375,:])
# print("points.shape:",threeDpts.shape)
# print("max z:",np.min(threeDpts[:,2]))
Proj=K.dot(pose)
# Proj=pose
# print Proj
threeDpts  = np.hstack((threeDpts,np.ones((threeDpts.shape[0],1)))).T
twoDpts=Proj.dot(threeDpts)
twoDpts=twoDpts/twoDpts[2,:]
#####################
dlt_ind = np.random.randint(0,twoDpts.shape[1],55)
dlt_ind = dlt_ind.reshape(dlt_ind.shape[0],1)
# print dlt_ind
zero=np.zeros((1,4))
A=np.array([])
count=0
for i in dlt_ind:    
    r1=np.hstack((-threeDpts[:,i].T,zero,twoDpts[0,i]*threeDpts[:,i].T))
    r2=np.hstack((zero,-threeDpts[:,i].T,twoDpts[1,i]*threeDpts[:,i].T))
    if count==0:
        A=np.vstack((r1,r2))
        count=count+1
    else:
        A=np.vstack((A,np.vstack((r1,r2))))
# print(A.shape)
U,D,Vt=np.linalg.svd(A)
P_est=Vt[Vt.shape[0]-1,:]
P_est=P_est.reshape((3,4))
# print("DLT OUT:",P_est)
noise=np.ones((3,4))
# P_est=P_est+0.00001*noise
# print "Vt",Vt
# print "P",P_est
# P_est = np.array([[1,2,1,0.5],[0,1,0,0.7],[0.5,0.5,0.5,1]])
#############################
def computeJ(K,P_est,threeDpts,twoDpts):
    count=0
    zero=np.zeros((1,4))
    temp_2D=P_est.dot(threeDpts)
    # print("Temp",temp_2D.shape)
    for i in range(300370,301450):
        # print("shape",((threeDpts[:,i].T)/temp_2D[2,i]).shape)
        r1=-np.hstack(( ((threeDpts[:,i].T)/temp_2D[2,i]).reshape((1,4)), zero , -(temp_2D[0,i]/(temp_2D[2,i]**2))*((threeDpts[:,i].T).reshape((1,4))) ))
        r2=-np.hstack(( zero, ((threeDpts[:,i].T)/temp_2D[2,i]).reshape((1,4)) , -(temp_2D[1,i]/(temp_2D[2,i]**2))*((threeDpts[:,i].T).reshape((1,4))) ))
        if count==0:
            J=np.vstack((r1,r2))
            count=count+1
            # print("Jacobian:",J)
        else:
            J=np.vstack((J,np.vstack((r1,r2))))
    # print("shape:",J.shape)
    return J
p=0
prev_grad=999999999
prev_error=99999999999999
while True:
# for i in range(55):
    estd_2D=P_est.dot(threeDpts)
    estd_2D=estd_2D/estd_2D[2,:]
    E=twoDpts[:2,300370:301450]-estd_2D[:2,300370:301450]
    E=E.reshape((E.shape[0]*E.shape[1],1))
    Mean_Error=(E.T).dot(E)
    J=computeJ(K,P_est,threeDpts,twoDpts)
    print("Iteration no. :", p)
    # print("Mean Error=",Mean_Error)
    grad_new=np.linalg.norm((J.T).dot(E))
    print("Gradient=",grad_new)
    # if (prev_error-Mean_Error)<10e-33:
        # break
    if np.linalg.norm(prev_grad-grad_new)<0.5:
        break

    update=np.linalg.pinv((J.T).dot(J)).dot(J.T)
    delP=update.dot(E)
    P_est=P_est.reshape((12,1))
    P_est=P_est-0.5*delP
    P_est=P_est.reshape((3,4))
    prev_grad=grad_new
    prev_error=Mean_Error
    p=p+1
Rt=np.linalg.inv(K).dot(P_est)
print("Mean Error=",Mean_Error)
print "Expected pose:\n",pose
print "Estimated pose:\n",Rt*(pose[2,3]/Rt[2,3])