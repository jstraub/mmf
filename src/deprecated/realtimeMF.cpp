#include <realtimeMF.hpp>


RealtimeMF::~RealtimeMF()
{
  if(!cuda_ready) return;

  checkCudaErrors(cudaFree(d_depth));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_z));
  checkCudaErrors(cudaFree(d_n));
  checkCudaErrors(cudaFree(d_xyz));
  checkCudaErrors(cudaFree(d_xu));
  checkCudaErrors(cudaFree(d_yu));
  checkCudaErrors(cudaFree(d_zu));
  checkCudaErrors(cudaFree(d_xv));
  checkCudaErrors(cudaFree(d_yv));
  checkCudaErrors(cudaFree(d_zv));
  checkCudaErrors(cudaFree(a));
  checkCudaErrors(cudaFree(b));
  checkCudaErrors(cudaFree(c));
#ifdef WEIGHTED
  checkCudaErrors(cudaFree(d_weights));
#endif

  free(h_n);
  delete optSO3_;
}

Matrix3f RealtimeMF::depth_cb(const uint16_t *data, int w, int h) 
{
//  static double dtAssign = 0.0f;
//  static double dtCost = 0.0f;
//  static double dtJacob = 0.0f;
  static double dtNormals = 0.0f;
  static double dtPrep = 0.0f;
  static double dtCG = 0.0f;
  static double dtTotal = 0.0f;
  static double Ncb = 0.0;

  Timer t0;
  prepareCUDA(w,h);
  float invF = 1.0f/570.f; //float(d->getFocalLength());

  checkCudaErrors(cudaMemcpy(d_depth, data, w * h * sizeof(uint16_t),
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaDeviceSynchronize());

#ifdef SMOOTH_DEPTH
  depth2smoothXYZ(invF,w,h); 
#else
  depth2xyzGPU(d_depth,d_x,d_y,d_z,invF,w,h,d_xyz); 
#endif
#ifdef WEIGHTED    
//  weightsFromCovGPU(d_z, d_weights, 30.0f, invF, w,h);
  weightsFromAreaGPU(d_z, d_weights, w,h);
#endif
  // obtain derivatives using sobel 
  computeDerivatives(w,h);
#ifndef SMOOTH_DEPTH
  // now smooth the derivatives
  smoothDerivatives(2,w,w);
#endif
  // obtain the normals using mainly cross product on the derivatives
  derivatives2normalsGPU(
      d_x,d_y,d_z,
      d_xu,d_yu,d_zu,
      d_xv,d_yv,d_zv,
      d_n,w,h);

  dtNormals_ = t0.toctic("normal extraction");
  dtNormals += dtNormals_;

#ifndef NDEBUG
  cout<<"OptSO3: sigma="<<optSO3_->sigma_sq_<<endl;
#endif
  Matrix3f kRwBefore = kRw_;
//  cout<<"kRw_ before = \n"<<kRw_<<endl;
  double residual = optSO3_->conjugateGradientCUDA(kRw_,nCGIter_);
  cout<<"delta rotation kRw_ = \n"<<kRwBefore*kRw_.transpose()<<endl;
  dtPrep_ = optSO3_->dtPrep();
  dtCG_ = optSO3_->dtCG();
  dtPrep += dtPrep_;
  dtCG += dtCG_;
//  t0.toctic(" full conjugate gradient");

  //    int N=0;
  //    cout<<optSO3_->computeAssignment(kRw_,N)<<endl;
  //    dtAssign += t0.toctic("assignment");
  //    cout<<optSO3_->evalCostFunction(kRw_)<<endl;
  //    dtCost += t0.toctic("costFct evaluation");
  //    Matrix3f J = Matrix3f::Identity();
  //    optSO3_->computeJacobian(J,kRw_);
  //    dtJacob += t0.toctic("Jacobian");
  //    cout<<"J:"<<endl<<J<<endl;

  double D_KL = optSO3_->D_KL_axisUnif();

  //cout<<"d_cb_ getting lock"<<endl;
  boost::mutex::scoped_lock updateLock(updateModelMutex);
  D_KL_= D_KL;
  residual_ = residual;

  checkCudaErrors(cudaDeviceSynchronize());
  //memcpy(h_dbg,data,w*h*sizeof(float));
#ifdef SHOW_WEIGHTS
  checkCudaErrors(cudaMemcpy(h_dbg, d_weights, w*h *sizeof(float), 
        cudaMemcpyDeviceToHost));
#endif
#ifdef SHOW_LOW_ERR
  checkCudaErrors(cudaMemcpy(h_dbg, optSO3_->d_errs_, w*h *sizeof(float), 
        cudaMemcpyDeviceToHost));
#endif
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(h_n, d_n, w*h* X_STEP *sizeof(float), 
        cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(h_xyz, d_xyz, w*h*4 *sizeof(float), 
        cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaDeviceSynchronize());
  // update viewer
  update = true;
  updateLock.unlock();

  dtTotal_ = t0.dtFromInit();
  dtTotal += dtTotal_;
  cout<<"total dt: "<<dtTotal_<<"ms"<<endl;
  Ncb ++;
  cout<<"---------------------------------------------------------------------------"<<endl
    <<" dtNormals="<<dtNormals/Ncb
    <<"\t dtPrep="<<dtPrep/Ncb
//    <<"\t dtCost="<<dtCost/Ncb
//    <<"\t dtJacob="<<dtJacob/Ncb
    <<"\t dtCG="<<dtCG/Ncb
    <<"\t dtTotal="<<dtTotal/Ncb <<endl;
  cout<<" residual="<<residual_<<"\t D_KL="<<D_KL_<<endl;
  cout<<"---------------------------------------------------------------------------"<<endl;

  return kRw_;
}


void RealtimeMF::prepareCUDA(uint32_t w,uint32_t h)
{
  if (cuda_ready) return;
  // CUDA preparations
  printf("Allocating and initializing CUDA arrays...\n");
  checkCudaErrors(cudaMalloc((void **)&d_depth, w * h * sizeof(uint16_t)));
  checkCudaErrors(cudaMalloc((void **)&d_x, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_y, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_z, w * h * sizeof(float)));

  checkCudaErrors(cudaMalloc((void **)&d_n, w * h * X_STEP* sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_xyz, w * h * 4* sizeof(float)));

  checkCudaErrors(cudaMalloc((void **)&d_xu, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_yu, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_zu, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_xv, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_yv, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_zv, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&a, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&b, w * h * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&c, w * h * sizeof(float)));
#ifdef WEIGHTED
  checkCudaErrors(cudaMalloc((void **)&d_weights, w * h * sizeof(float)));
#else
  d_weights = NULL;
#endif
  cout<<"cuda allocations done "<<d_n<<endl;

  h_sobel_dif[0] = 1;
  h_sobel_dif[1] = 0;
  h_sobel_dif[2] = -1;

  h_sobel_sum[0] = 1;
  h_sobel_sum[1] = 2;
  h_sobel_sum[2] = 1;

  // sig =1.0
  // x=np.arange(7) -3.0
  // 1.0/(np.sqrt(2*np.pi)*sig)*np.exp(-0.5*(x*x/sig**2))
  // 0.00443185,  0.05399097,  0.24197072,  0.39894228,  0.24197072,
  // 0.05399097,  0.00443185
  // sig = 2.0
  // 0.0647588 ,  0.12098536,  0.17603266,  0.19947114,  0.17603266,
  // 0.12098536,  0.0647588 
  /*
     h_kernel_avg[0] = 0.00443185;
     h_kernel_avg[1] = 0.05399097;
     h_kernel_avg[2] = 0.24197072;
     h_kernel_avg[3] = 0.39894228;
     h_kernel_avg[4] = 0.24197072;
     h_kernel_avg[5] = 0.05399097;
     h_kernel_avg[6] = 0.00443185;
     */

  h_kernel_avg[0] = 0.0647588;
  h_kernel_avg[1] = 0.12098536;
  h_kernel_avg[2] = 0.17603266;
  h_kernel_avg[3] = 0.19947114;
  h_kernel_avg[4] = 0.17603266;
  h_kernel_avg[5] = 0.12098536;
  h_kernel_avg[6] = 0.0647588;

  n_ = pcl::PointCloud<pcl::PointXYZRGB>(w,h);
  n_cp_ = pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr(&n_);
  Map<MatrixXf, Aligned, OuterStride<> > nMat = 
    n_.getMatrixXfMap(X_STEP,X_STEP,0);
  h_n = nMat.data();//(float *)malloc(w *h *3* sizeof(float));

  pc_ = pcl::PointCloud<pcl::PointXYZ>(w,h);
  pc_cp_ = pcl::PointCloud<pcl::PointXYZ>::ConstPtr(&pc_);
  Map<MatrixXf, Aligned, OuterStride<> > pcMat = 
    pc_.getMatrixXfMap(X_STEP,X_STEP,0);
  h_xyz = pcMat.data();//(float *)malloc(w *h *3* sizeof(float));

  h_dbg = (float *)malloc(w *h * sizeof(float));


  cout<<"inititalizing optSO3"<<endl;
  // TODO: use weights
  // TODO: make sure N is float all over the place
  //optSO3_ = new OptSO3(25.0f*M_PI/180.0f,d_n,w,h,NULL);//,d_weights);
  if(mode_.compare("direct") == 0)
  {
#ifndef WEIGHTED
    optSO3_ = new OptSO3(25.0f*M_PI/180.0f,d_n,w,h);//,d_weights);
#else
    optSO3_ = new OptSO3(25.0f*M_PI/180.0f,d_n,w,h,d_weights);
#endif
    nCGIter_ = 10; // cannot do that many iterations
  }else if (mode_.compare("approx") == 0){
#ifndef WEIGHTED
    optSO3_ = new OptSO3Approx(25.0f*M_PI/180.0f,d_n,w,h);//,d_weights);
#else
    optSO3_ = new OptSO3Approx(25.0f*M_PI/180.0f,d_n,w,h,d_weights);
#endif
    nCGIter_ = 25;
  }

  cuda_ready = true;
}

void RealtimeMF::visualizePc()
{
  // Block signals in this thread
  sigset_t signal_set;
  sigaddset(&signal_set, SIGINT);
  sigaddset(&signal_set, SIGTERM);
  sigaddset(&signal_set, SIGHUP);
  sigaddset(&signal_set, SIGPIPE);
  pthread_sigmask(SIG_BLOCK, &signal_set, NULL);

  bool showNormals =true;
  float scale = 2.0f;
  // prepare visualizer named "viewer"
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (
      new pcl::visualization::PCLVisualizer ("3D Viewer"));

  //      viewer->setPointCloudRenderingProperties (
  //          pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->initCameraParameters ();

  cv::namedWindow("normals");
  //cv::namedWindow("dbg");
  cv::namedWindow("dbgNan");
  cv::namedWindow("rgb");

  int v1(0);
  viewer->createViewPort (0.0, 0.0, 0.5, 1.0, v1);
  viewer->setBackgroundColor (0, 0, 0, v1);
  viewer->addText ("normals", 10, 10, "v1 text", v1);
  viewer->addCoordinateSystem (1.0,v1);

  int v2(0);
  viewer->createViewPort (0.5, 0.0, 1.0, 1.0, v2);
  viewer->setBackgroundColor (0.1, 0.1, 0.1, v2);
  viewer->addText ("pointcloud", 10, 10, "v2 text", v2);
  viewer->addCoordinateSystem (1.0,v2);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr n;
  pcl::PointCloud<pcl::PointXYZ>::Ptr pc;

  Timer t;
  while (!viewer->wasStopped ())
  {
    cout<<"viewer"<<endl;
    viewer->spinOnce (10);
    cv::waitKey(10);
    // break, if the last update was less than 2s ago
//    if (t.dtFromInit() > 20000.0)
//    {
//      cout<<" ending visualization - waited too long"<<endl;
//      break;
//    }
//    cout<<" after break"<<endl;

    // Get lock on the boolean update and check if cloud was updated
    boost::mutex::scoped_lock updateLock(updateModelMutex);
    if (updateRGB_)
    {
      cout<<"show rgb"<<endl;
      imshow("rgb",rgb_);
      updateRGB_ = false;
      t=Timer();
    } 
    if (update)
    {
      cout<<"show pc"<<endl;
      stringstream ss;
      ss <<"residual="<<residual_;
      if(!viewer->updateText(ss.str(),10,20,"residual"))
        viewer->addText(ss.str(),10,20, "residual", v1);
      ss.str(""); ss << "D_KL="<<D_KL_;
      if(!viewer->updateText(ss.str(),10,30,"D_KL"))
        viewer->addText(ss.str(),10,30, "D_KL", v1);

      Matrix3f wRk = kRw_.transpose();
      Matrix4f wTk;
      wTk<< wRk, MatrixXf::Zero(3,1),MatrixXf::Zero(1,3),1.0;

      cv::Mat nI(n_.height,n_.width,CV_32FC3); 
      for(uint32_t i=0; i<n_.width; ++i)
        for(uint32_t j=0; j<n_.height; ++j)
        {
          // nI is BGR but I want R=x G=y and B=z
          nI.at<cv::Vec3f>(j,i)[0] = (1.0f+n_.points[i+j*n_.width].z)*0.5f; // to match pc
          nI.at<cv::Vec3f>(j,i)[1] = (1.0f+n_.points[i+j*n_.width].y)*0.5f; 
          nI.at<cv::Vec3f>(j,i)[2] = (1.0f+n_.points[i+j*n_.width].x)*0.5f; 
        }
      cv::imshow("normals",nI); 

//      cv::Mat Ierr(n_.height,n_.width,CV_32FC1); 
//      for(uint32_t i=0; i<n_.width; ++i)
//        for(uint32_t j=0; j<n_.height; ++j)
//        {
//          // nI is BGR but I want R=x G=y and B=z
//          nI.at<float>(j,i) = h_n[(i+j*n_.width)*X_STEP+6];
//        }
//      normalizeImg(Ierr);

#ifdef SHOW_WEIGHTS
              cv::Mat Idbg(n_.height,n_.width,CV_32FC1,h_dbg);
              normalizeImg(Idbg);
              cv::imshow("dbg",Idbg);
#endif
      //        cv::Mat IdbgNan = Idbg.clone();
      //        showNans(IdbgNan);
      //        //showZeros(IdbgNan);
      //        cv::imshow("dbgNan",IdbgNan);
      
#ifdef SHOW_LOW_ERR
      cv::Mat Idbg(n_.height,n_.width,CV_32FC1,h_dbg);
      cv::Mat IgoodRGB = cv::Mat::zeros(n_.height,n_.width,CV_8UC3); 
      rgb_.copyTo(IgoodRGB,Idbg < 20.0*M_PI/180.0);
      normalizeImg(Idbg);
      cv::imshow("dbgNan",IgoodRGB); // Idbg
#endif

      if(showNormals){
        n = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(
            new pcl::PointCloud<pcl::PointXYZRGB>);
        for(uint32_t i=0; i<n_.points.size(); i+= 5)
          if (*((&n_.points[i].rgb)+1) < 6)
          {
            n->push_back(n_.points[i]);
            n->back().x *= scale;
            n->back().y *= scale;
            n->back().z *= scale;
          }
        //pcl::transformPointCloud(*n, *n, wTk);
      }

      pc = pcl::PointCloud<pcl::PointXYZ>::Ptr(
          new pcl::PointCloud<pcl::PointXYZ>);
      for(uint32_t i=0; i<pc_.width; i+= 5)
        for(uint32_t j=0; j<pc_.height; j+=5)
          pc->points.push_back(pc_cp_->points[i+j*pc_.width]);
      pcl::transformPointCloud(*pc, *pc , wTk);

      if(!updateCosy(viewer, kRw_,"mf",2.0f))
        addCosy(viewer,kRw_,"mf",2.0f, v1);

      Matrix3d vRw = vicon_q_.toRotationMatrix();
      Matrix3d kRw_fromVicon = kRv*vRw;
      if(!updateCosy(viewer, kRw_fromVicon.cast<float>(),
            "vicon",2.5f))
        addCosy(viewer,kRw_fromVicon.cast<float>(),
            "vicon",2.5f, v1);

      if(showNormals)
        if(!viewer->updatePointCloud(n, "normals"))
          viewer->addPointCloud(n, "normals",v1);

      if(!viewer->updatePointCloud(pc, "pc"))
        viewer->addPointCloud(pc, "pc",v2);

//      viewer->saveScreenshot("./test.png");
      update = false;
      t=Timer();
    }
    updateLock.unlock();
  }
}

void RealtimeMF::getAxisAssignments()
{
  
}

void RealtimeMF::computeDerivatives(uint32_t w,uint32_t h)
{
  setConvolutionKernel_small(h_sobel_dif);
  convolutionRowsGPU_small(a,d_x,w,h);
  convolutionRowsGPU_small(b,d_y,w,h);
  convolutionRowsGPU_small(c,d_z,w,h);
  setConvolutionKernel_small(h_sobel_sum);
  convolutionColumnsGPU_small(d_xu,a,w,h);
  convolutionColumnsGPU_small(d_yu,b,w,h);
  convolutionColumnsGPU_small(d_zu,c,w,h);
  convolutionRowsGPU_small(a,d_x,w,h);
  convolutionRowsGPU_small(b,d_y,w,h);
  convolutionRowsGPU_small(c,d_z,w,h);
  setConvolutionKernel_small(h_sobel_dif);
  convolutionColumnsGPU_small(d_xv,a,w,h);
  convolutionColumnsGPU_small(d_yv,b,w,h);
  convolutionColumnsGPU_small(d_zv,c,w,h);
}

void RealtimeMF::smoothDerivatives(uint32_t iterations, uint32_t w,uint32_t h)
{
  setConvolutionKernel(h_kernel_avg);
  for(uint32_t i=0; i<iterations; ++i)
  {
    convolutionRowsGPU(a,d_xu,w,h);
    convolutionRowsGPU(b,d_yu,w,h);
    convolutionRowsGPU(c,d_zu,w,h);
    convolutionColumnsGPU(d_xu,a,w,h);
    convolutionColumnsGPU(d_yu,b,w,h);
    convolutionColumnsGPU(d_zu,c,w,h);
    convolutionRowsGPU(a,d_xv,w,h);
    convolutionRowsGPU(b,d_yv,w,h);
    convolutionRowsGPU(c,d_zv,w,h);
    convolutionColumnsGPU(d_xv,a,w,h);
    convolutionColumnsGPU(d_yv,b,w,h);
    convolutionColumnsGPU(d_zv,c,w,h);
  }
}


void RealtimeMF::depth2smoothXYZ(float invF, uint32_t w,uint32_t h)
{
  depth2floatGPU(d_depth,a,w,h);

//  for(uint32_t i=0; i<3; ++i)
//  {
//    depthFilterGPU(a,w,h);
//  }
  //TODO compare:
  // now smooth the derivatives
  setConvolutionKernel(h_kernel_avg);
  for(uint32_t i=0; i<3; ++i)
  {
    convolutionRowsGPU(b,a,w,h);
    convolutionColumnsGPU(a,b,w,h);
  }
  // convert depth into x,y,z coordinates
  depth2xyzFloatGPU(a,d_x,d_y,d_z,invF,w,h,d_xyz); 
}


void RealtimeMF::run ()
{
  boost::thread visualizationThread(&RealtimeMF::visualizePc,this); 

  this->run_impl();
  while (42) boost::this_thread::sleep (boost::posix_time::seconds (1));
  this->run_cleanup_impl();
  visualizationThread.join();
}

