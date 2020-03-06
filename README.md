# Cumulative-Pointcloud-generation-from-a-sequence-o-images_Gauss-Newton-Optimisation
We take a sequence of frames from kitti dataset, compute disparity maps and then use them and generate point clouds for the frames individually and then combine them to get a cummulative point cloud.

Then using these 3D pts from the point cloud, we optimise the previously estimated poses by using gauss newton method.

