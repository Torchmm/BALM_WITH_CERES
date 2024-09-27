#include "tools.hpp"
#include <Eigen/Eigenvalues>
#include <random>
#include <ctime>
#include "bavoxel.hpp"
#include <stdio.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <malloc.h>
#include <stdlib.h>
#include "optim.h"
using namespace std;

int read_pose(vector<double> &tims, PLM(3) &rots, PLV(3) &poss, string prename)
{
  string readname = prename + "alidarPose.csv";

  cout << readname << endl;
  ifstream inFile(readname);

  if(!inFile.is_open())
  {
    printf("open fail\n"); return 0;
  }

  int pose_size = 0;
  string lineStr, str;
  Eigen::Matrix4d aff;
  vector<double> nums;

  int ord = 0;
  while(getline(inFile, lineStr))
  {
    ord++;
    stringstream ss(lineStr);
    while(getline(ss, str, ','))
      nums.push_back(stod(str));

    if(ord == 4)
    {
      for(int j=0; j<16; j++)
        aff(j) = nums[j];

      Eigen::Matrix4d affT = aff.transpose();

      rots.push_back(affT.block<3, 3>(0, 0));
      poss.push_back(affT.block<3, 1>(0, 3));
      tims.push_back(affT(3, 3));
      nums.clear();
      ord = 0;
      pose_size++;
      if (pose_size >= 333)
      {
          return pose_size;
      }
    }
  }

  return pose_size;
}

void read_file(vector<IMUST> &x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls, string &prename)
{
    if (prename == "")
    {
        prename = "datas/benchmark_realworld/";
    }
    else
    {
        prename = prename + "/datas/benchmark_realworld/";
    }
  

  PLV(3) poss; PLM(3) rots;
  vector<double> tims;
  int pose_size = read_pose(tims, rots, poss, prename);
  
  for(int m=0; m<pose_size; m++)
  {
    string filename = prename + "full" + to_string(m) + ".pcd";

    pcl::PointCloud<PointType>::Ptr pl_ptr(new pcl::PointCloud<PointType>());
    pcl::PointCloud<pcl::PointXYZI> pl_tem;
    pcl::io::loadPCDFile(filename, pl_tem);
    for(pcl::PointXYZI &pp: pl_tem.points)
    {
      PointType ap;
      ap.x = pp.x; ap.y = pp.y; ap.z = pp.z;
      ap.intensity = pp.intensity;
      pl_ptr->push_back(ap);
    }

    pl_fulls.push_back(pl_ptr);

    IMUST curr;
    curr.R = rots[m]; curr.p = poss[m]; curr.t = tims[m];
    x_buf.push_back(curr);
  }
  

}

void data_show(vector<IMUST> x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls,std::string file)
{
  IMUST es0 = x_buf[0];
  for(size_t i=0; i<x_buf.size(); i++)
  {
    x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
    x_buf[i].R = es0.R.transpose() * x_buf[i].R;
  }

  pcl::PointCloud<PointType> pl_send, pl_path;

  std::ofstream ofile(file);
  int winsize = x_buf.size();
  for(int i=0; i<winsize; i++)
  {
    pcl::PointCloud<PointType> pl_tem = *pl_fulls[i];
    down_sampling_voxel(pl_tem, 0.05);
    pl_transform(pl_tem, x_buf[i]);
    //pl_send += pl_tem;

    for (size_t i = 0; i < pl_tem.size(); i++)
    {
        ofile << pl_tem[i].x << " " << pl_tem[i].y << " " << pl_tem[i].z << std::endl;
    }

    //PointType ap;
    //ap.x = x_buf[i].p.x();
    //ap.y = x_buf[i].p.y();
    //ap.z = x_buf[i].p.z();
    //ap.curvature = i;
    //pl_path.push_back(ap);
  }

  ofile.close();
}

int main()
{
  //ros::init(argc, argv, "benchmark2");
  //ros::NodeHandle n;
  //pub_test = n.advertise<sensor_msgs::PointCloud2>("/map_test", 100);
  //pub_path = n.advertise<sensor_msgs::PointCloud2>("/map_path", 100);
  //pub_show = n.advertise<sensor_msgs::PointCloud2>("/map_show", 100);
  //pub_cute = n.advertise<sensor_msgs::PointCloud2>("/map_cute", 100);

  string prename, ofname;
  vector<IMUST> x_buf;
  vector<pcl::PointCloud<PointType>::Ptr> pl_fulls;

  //n.param<double>("voxel_size", voxel_size, 1);
  string file_path="";
  /*n.param<string>("file_path", file_path, "");*/

  read_file(x_buf, pl_fulls, file_path);

  IMUST es0 = x_buf[0];
  for(size_t i=0; i<x_buf.size(); i++)
  {
    x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
    x_buf[i].R = es0.R.transpose() * x_buf[i].R;
  }

  win_size = x_buf.size();
  printf("The size of poses: %d\n", win_size);

  data_show(x_buf, pl_fulls,"bef.txt");
  printf("Check the point cloud with the initial poses.\n");
  printf("If no problem, input '1' to continue or '0' to exit...\n");
 /* int a; cin >> a; if(a==0) exit(0);*/

  pcl::PointCloud<PointType> pl_full, pl_surf, pl_path, pl_send;
  for(int iterCount=0; iterCount<1; iterCount++)
  { 
    unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

    eigen_value_array[0] = 1.0 / 16;
    eigen_value_array[1] = 1.0 / 16;
    eigen_value_array[2] = 1.0 / 9;

    for(int i=0; i<win_size; i++)
      cut_voxel(surf_map, *pl_fulls[i], x_buf[i], i);

    pcl::PointCloud<PointType> pl_send;


    pcl::PointCloud<PointType> pl_cent; pl_send.clear();
    VOX_HESS voxhess;
    for(auto iter=surf_map.begin(); iter!=surf_map.end(); iter++)
    {
      iter->second->recut(win_size);
      iter->second->tras_opt(voxhess, win_size);
      iter->second->tras_display(pl_send, win_size);
    }

    printf("\nThe planes (point association) cut by adaptive voxelization.\n");
    printf("If the planes are too few, the optimization will be degenerated and fail.\n");
    printf("If no problem, input '1' to continue or '0' to exit...\n");
    /*int a; cin >> a; if(a==0) exit(0);*/
    pl_send.clear();

    if(voxhess.plvec_voxels.size() < 3 * x_buf.size())
    {
      printf("Initial error too large.\n");
      printf("Please loose plane determination criteria for more planes.\n");
      printf("The optimization is terminated.\n");
      exit(0);
    }

    //BALM2 opt_lsv;
    //opt_lsv.damping_iter(x_buf, voxhess);

    OptimCeres optim_ceres;
    optim_ceres.Optimilize(x_buf, voxhess);

    for(auto iter=surf_map.begin(); iter!=surf_map.end();)
    {
      delete iter->second;
      surf_map.erase(iter++);
    }
    surf_map.clear();
  }

  printf("\nRefined point cloud is publishing...\n");

  data_show(x_buf, pl_fulls,"aft.txt");
  printf("\nRefined point cloud is published.\n");

  return 0;

}


