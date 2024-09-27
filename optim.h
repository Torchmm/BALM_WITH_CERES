#pragma once
#include <ceres/ceres.h>
#include "bavoxel.hpp"

Eigen::Matrix<double, 3, 3> skew(Eigen::Matrix<double, 3, 1>& mat_in) 
{
    Eigen::Matrix<double, 3, 3> skew_mat;
    skew_mat.setZero();
    skew_mat(0, 1) = -mat_in(2);
    skew_mat(0, 2) = mat_in(1);
    skew_mat(1, 2) = -mat_in(0);
    skew_mat(1, 0) = mat_in(2);
    skew_mat(2, 0) = -mat_in(1);
    skew_mat(2, 1) = mat_in(0);
    return skew_mat;
}

void getTransformFromSe3(const Eigen::Matrix<double, 6, 1>& se3, Eigen::Quaterniond& q, Eigen::Vector3d& t)
{
    Eigen::Vector3d omega(se3.data());
    Eigen::Vector3d upsilon(se3.data() + 3);
    Eigen::Matrix3d Omega = skew(omega);

    double theta = omega.norm();
    double half_theta = 0.5 * theta;

    double imag_factor;
    double real_factor = cos(half_theta);
    if (theta < 1e-10)
    {
        double theta_sq = theta * theta;
        double theta_po4 = theta_sq * theta_sq;
        imag_factor = 0.5 - 0.0208333 * theta_sq + 0.000260417 * theta_po4;
    }
    else
    {
        double sin_half_theta = sin(half_theta);
        imag_factor = sin_half_theta / theta;
    }

    q = Eigen::Quaterniond(real_factor, imag_factor * omega.x(), imag_factor * omega.y(), imag_factor * omega.z());


    Eigen::Matrix3d J;
    if (theta < 1e-10)
    {
        J = q.matrix();
    }
    else
    {
        Eigen::Matrix3d Omega2 = Omega * Omega;
        J = (Eigen::Matrix3d::Identity() + (1 - cos(theta)) / (theta * theta) * Omega + (theta - sin(theta)) / (pow(theta, 3)) * Omega2);
    }

    t = J * upsilon;
}


// 自定义SE3流形
class EigenSE3 : public ceres::Manifold
{
public:
    int AmbientSize() const override { return 7; }
    int TangentSize() const override { return 6; }

    // 对四元数x左乘增量李代数delta，返回四元数x_plus_delta
    bool Plus(const double* x,
        const double* delta,
        double* x_plus_delta) const override
    {
        Eigen::Map<const Eigen::Vector3d> trans(x + 4);

        Eigen::Quaterniond delta_q;
        Eigen::Vector3d delta_t;
        getTransformFromSe3(Eigen::Map<const Eigen::Matrix<double, 6, 1>>(delta), delta_q, delta_t);
        Eigen::Map<const Eigen::Quaterniond> quater(x);
        Eigen::Map<Eigen::Quaterniond> quater_plus(x_plus_delta);
        Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta + 4);

        quater_plus = delta_q * quater;
        trans_plus = delta_q * trans + delta_t;

        return true;
    };

    // 四元素左扰动对李代数的雅可比矩阵
    bool PlusJacobian(const double* x, double* jacobian_ptr) const override
    {
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian_ptr);
        (j.topRows(6)).setIdentity();
        (j.bottomRows(1)).setZero();
        return true;
    };

    // 四元数y - 四元数x , 返回李代数y_minus_x，这个函数实现是错误的，好像没有调用，
    bool Minus(const double* y,
        const double* x,
        double* y_minus_x) const override
    {
        double ambient_y_minus_x[4];
        ambient_y_minus_x[3] =
            y[3] * x[3] + y[0] * x[0] +
            y[1] * x[1] + y[2] * x[2];
        ambient_y_minus_x[0] =
            -y[3] * x[0] + y[0] * x[3] -
            y[1] * x[2] + y[2] * x[1];
        ambient_y_minus_x[1] =
            -y[3] * x[1] + y[0] * x[2] +
            y[1] * x[3] - y[2] * x[0];
        ambient_y_minus_x[2] =
            -y[3] * x[2] - y[0] * x[1] +
            y[1] * x[0] + y[2] * x[3];

        const double u_norm =
            std::sqrt(ambient_y_minus_x[0] * ambient_y_minus_x[0] +
                ambient_y_minus_x[1] * ambient_y_minus_x[1] +
                ambient_y_minus_x[2] * ambient_y_minus_x[2]);
        if (u_norm > 0.0)
        {
            const double theta = std::atan2(u_norm, ambient_y_minus_x[3]);
            y_minus_x[0] = theta * ambient_y_minus_x[0] / u_norm;
            y_minus_x[1] = theta * ambient_y_minus_x[1] / u_norm;
            y_minus_x[2] = theta * ambient_y_minus_x[2] / u_norm;
        }
        else {
            y_minus_x[0] = 0.0;
            y_minus_x[1] = 0.0;
            y_minus_x[2] = 0.0;
        }
        std::cout << "minus =====" << std::endl;
        return true;
    };

    // 实现是错误的 ，这个函数好像没有使用
    bool MinusJacobian(const double* x, double* jacobian_ptr) const override
    {
        Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian(
            jacobian_ptr);

        (jacobian.leftCols(3)).setIdentity();
        (jacobian.rightCols(1)).setZero();

        std::cout << "plus jaco=====" << std::endl;
        return true;
    };
};

// SO3的流形,修改了雅可比矩阵为单位阵，和CERES的欧式流形组成SE3流形不可用，不知道为什么
class EigenQuternionManifoldMySelf : public ceres::Manifold
{
public:
    int AmbientSize() const override { return 4; }
    int TangentSize() const override { return 3; }

    // 对四元数x左乘增量李代数delta，返回四元数x_plus_delta
    bool Plus(const double* x,
        const double* delta,
        double* x_plus_delta) const override 
    {
        Eigen::Map<Eigen::Quaterniond> x_plus_delta_result(x_plus_delta);
        Eigen::Map<const Eigen::Quaterniond> x_src(x);

        const double norm_delta =
            sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
        if (norm_delta > 0.0) 
        {
            const double sin_delta_by_delta = sin(norm_delta) / norm_delta;

            // Note, in the constructor w is first.
            Eigen::Quaterniond delta_q(cos(norm_delta),
                sin_delta_by_delta * delta[0],
                sin_delta_by_delta * delta[1],
                sin_delta_by_delta * delta[2]);
            x_plus_delta_result = delta_q * x_src;
        }
        else {
            x_plus_delta_result = x_src;
        }

        return true;
    };

    // 四元素左扰动对李代数的雅可比矩阵
    bool PlusJacobian(const double* x, double* jacobian_ptr) const override
    {
        Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> jacobian(
            jacobian_ptr);

        (jacobian.topRows(3)).setIdentity();
        (jacobian.bottomRows(1)).setZero();
        return true;
    };

    // 四元数y - 四元数x , 返回李代数y_minus_x
    bool Minus(const double* y,
        const double* x,
        double* y_minus_x) const override
    {
        double ambient_y_minus_x[4];
        ambient_y_minus_x[3] =
            y[3] * x[3] + y[0] * x[0] +
            y[1] * x[1] + y[2] * x[2];
        ambient_y_minus_x[0] =
            -y[3] * x[0] + y[0] * x[3] -
            y[1] * x[2] + y[2] * x[1];
        ambient_y_minus_x[1] =
            -y[3] * x[1] + y[0] * x[2] +
            y[1] * x[3] - y[2] * x[0];
        ambient_y_minus_x[2] =
            -y[3] * x[2] - y[0] * x[1] +
            y[1] * x[0] + y[2] * x[3];

        const double u_norm =
            std::sqrt(ambient_y_minus_x[0] * ambient_y_minus_x[0] +
                ambient_y_minus_x[1] * ambient_y_minus_x[1] +
                ambient_y_minus_x[2] * ambient_y_minus_x[2]);
        if (u_norm > 0.0)
        {
            const double theta = std::atan2(u_norm, ambient_y_minus_x[3]);
            y_minus_x[0] = theta * ambient_y_minus_x[0] / u_norm;
            y_minus_x[1] = theta * ambient_y_minus_x[1] / u_norm;
            y_minus_x[2] = theta * ambient_y_minus_x[2] / u_norm;
        }
        else {
            y_minus_x[0] = 0.0;
            y_minus_x[1] = 0.0;
            y_minus_x[2] = 0.0;
        }

        return true;
    };

    bool MinusJacobian(const double* x, double* jacobian_ptr) const override
    {
        Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> jacobian(
            jacobian_ptr);

        (jacobian.leftCols(3)).setIdentity();
        (jacobian.rightCols(1)).setZero();
        return true;
    };
};

// 自动求导中的计算实对阵矩阵特征值
template <typename T>
void Compute(const Eigen::Matrix<T, 3, 3> matrix, T values[3])
{
    T a11 = matrix(0, 0);
    T a22 = matrix(1, 1);
    T a33 = matrix(2, 2);

    T a12 = matrix(0, 1);
    T a13 = matrix(0, 2);
    T a23 = matrix(1, 2);

    T a = -(a11 + a22 + a33);
    T b = a11 * a22 + a22 * a33 + a33 * a11 - a12 * a12 - a13 * a13 - a23 * a23;
    T c = -matrix.determinant();

    T Q = (a * a - T(3) * b) / T(9);
    T R = (T(2) * a * a * a - T(9) * a * b + T(27) * c) / T(54);

    if (R * R < Q * Q * Q)
    {
        T theta = ceres::acos(R / sqrt(Q * Q * Q));
        values[0] = -T(2) * sqrt(Q) * cos(theta / T(3)) - a / T(3);
        values[1] = -T(2) * sqrt(Q) * cos((theta + T(2) * M_PI) / T(3)) - a / T(3);
        values[2] = -T(2) * sqrt(Q) * cos((theta - T(2) * M_PI) / T(3)) - a / T(3);
    }
    else
    {
        T A = -(R / ceres::abs(R)) * ceres::cbrt(ceres::abs(R) + ceres::sqrt(R * R - Q * Q * Q));
        T B = A == T(0) ? T(0) : Q / A;
        values[0] = (A + B) - a / T(3);
        values[1] = -T(0.5) * (A + B) - a / T(3);
        values[2] = -T(0.5) * (A + B) - a / T(3);
    }
}

class LdiarBACostFunctionNoAuto:public ceres::DynamicCostFunction
{
public:
    LdiarBACostFunctionNoAuto(const PointCluster& sig_vec, const std::vector<PointCluster>& plvec_voxels, const double& coeff, int index) :
        sig_vec_(sig_vec), plvec_voxels_(plvec_voxels), coeff_(coeff), index_(index) 
    {
        for (int i = 0; i < plvec_voxels.size(); i++)
        {
            AddParameterBlock(7);
        }

        SetNumResiduals(1);
    }

    // 右扰动
    //bool Evaluate(const double* const* parameters, double* residuals, double** jacobians) const override
    //{
    //    int win_size = plvec_voxels_.size();
    //    int kk = 0;
    //     //所有lidar帧的位姿信息集合
    //    PLM(4) T(win_size);// 4x4矩阵
    //    for (int i = 0; i < win_size; i++)
    //    {
    //        Eigen::Quaterniond  qi = Eigen::Map<const Eigen::Quaternion<double>>(&parameters[i][0]);
    //        qi.normalize();
    //        Eigen::Vector3d ti = Eigen::Map<const Eigen::Vector3d>(&parameters[i][4]);
    //        T[i] << qi.toRotationMatrix(),ti, 0, 0, 0, 1;
    //    }

    //    PointCluster sig = sig_vec_;
    //    for (int i = 0; i < win_size; i++)
    //    {
    //        if (plvec_voxels_[i].N != 0)
    //        {
    //            PointCluster clusterTemp;
    //            clusterTemp.transform(plvec_voxels_[i], T[i].block<3, 3>(0, 0), T[i].block<3, 1>(0, 3));
    //            sig += clusterTemp;
    //        }
    //    }

    //    const Eigen::Vector3d& vBar = sig.v / sig.N;
    //    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P / sig.N - vBar * vBar.transpose());
    //    const Eigen::Vector3d& lmbd = saes.eigenvalues();
    //    const Eigen::Matrix3d& U = saes.eigenvectors();
    //    int NN = sig.N;

    //    residuals[0] = NN * lmbd[kk];

    //    Eigen::Vector3d u[3] = { U.col(0), U.col(1), U.col(2) };

    //    const Eigen::Vector3d& uk = u[kk];
    //    Eigen::Matrix3d ukukT = uk * uk.transpose();
    //   
    //    PLV(3) viRiTuk(win_size);
    //    PLM(3) viRiTukukT(win_size);
    //    std::vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> Auk(win_size);
    //    for (int i = 0; i < win_size; i++)
    //    {
    //        if (jacobians)
    //        {
    //            if (jacobians[i])
    //            {
    //                Eigen::Map<Eigen::Matrix<double, 1, 7/*, Eigen::RowMajor*/>> jaco(jacobians[i]);
    //                jaco.setZero();

    //                if (plvec_voxels_[i].N != 0)
    //                {
    //                    Eigen::Matrix3d Pi = plvec_voxels_[i].P;
    //                    Eigen::Vector3d vi = plvec_voxels_[i].v;
    //                    Eigen::Matrix3d Ri = T[i].block<3, 3>(0, 0);
    //                    double ni = plvec_voxels_[i].N;

    //                    Eigen::Matrix3d vihat; vihat << SKEW_SYM_MATRX(vi);
    //                    Eigen::Vector3d RiTuk = Ri.transpose() * uk;
    //                    Eigen::Matrix3d RiTukhat; RiTukhat << SKEW_SYM_MATRX(RiTuk);

    //                    Eigen::Vector3d PiRiTuk = Pi * RiTuk;
    //                    viRiTuk[i] = vihat * RiTuk;
    //                    viRiTukukT[i] = viRiTuk[i] * uk.transpose();

    //                    Eigen::Vector3d ti_v = T[i].block<3, 1>(0, 3) - vBar;
    //                    double ukTti_v = uk.dot(ti_v);

    //                    Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
    //                    Eigen::Vector3d combo2 = Ri * vi + ni * ti_v;
    //                    Auk[i].block<3, 3>(0, 0) = (Ri * Pi + ti_v * vi.transpose()) * RiTukhat - Ri * combo1;
    //                    Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * I33;
    //                    Auk[i] /= NN;

    //                    const Eigen::Matrix<double, 6, 1>& jjt = Auk[i].transpose() * uk;

    //                    jaco.block<1, 6>(0, 0) += NN * jjt.transpose();

    //                }

    //            }
    //        }
    //        
    //    }

    //    return true;
    //}

    // 左扰动，ceres的EigenQuaternionManifold是左更新
    bool Evaluate(const double* const* parameters, double* residuals, double** jacobians) const override 
    {
        PointCluster cluster = sig_vec_;

        int l = 0;
        int win_size = plvec_voxels_.size();

        // 所有lidar帧的位姿信息集合
        PLM(4) T(win_size);// 4x4矩阵
        for (int i = 0; i < win_size; i++)
        {
            Eigen::Quaterniond  qi = Eigen::Map<const Eigen::Quaternion<double>>(&parameters[i][0]);
            Eigen::Vector3d ti = Eigen::Map<const Eigen::Vector3d>(&parameters[i][4]);
            T[i] << qi.toRotationMatrix(),ti, 0, 0, 0, 1;
            
        }

        
        // 包含所有点簇信息的矩阵集合

        PLM(4) Co;
        Co.resize(win_size);

        int Num = 0;
        for (int i = 0; i < win_size; i++)
        {
            Co[i] << plvec_voxels_[i].P, plvec_voxels_[i].v, plvec_voxels_[i].v.transpose(), plvec_voxels_[i].N;
            Num += plvec_voxels_[i].N;
        }

         double coe = Num;
         Eigen::Matrix4d C;
         C.setZero();

         vector<int> Ns(win_size);

         PLM(4) TC(win_size), TCT(win_size);
         for (int j = 0; j < win_size; j++)
         {
             if ((int)Co[j](3, 3) > 0)
             {
                 TC[j] = T[j] * Co[j];
                 TCT[j] = TC[j] * T[j].transpose();
                 C += TCT[j];

                 Ns[j] = Co[j](3, 3);
             }
         }


         double NN = C(3, 3);
         C = C / NN;
         Eigen::Vector3d v_bar = C.block<3, 1>(0, 3);
         Eigen::Matrix3d cov = C.block<3, 3>(0, 0) - v_bar * v_bar.transpose();
         Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov);
         Eigen::Vector3d lmbd = saes.eigenvalues();
         Eigen::Matrix3d Uev = saes.eigenvectors();

         // 最小特征值作为残差，coe应该是平面点数
         residuals[0] = coe * lmbd[l];

         // std::cout << "index: " << index_ << " coe: " << coe << " res: " << lmbd[l];
         
         Eigen::Vector3d u[3] = { Uev.col(0), Uev.col(1), Uev.col(2) };
         Eigen::Matrix<double, 6, 4> U[3];
         PLV(6) g_kl[3];
         for (int k = 0; k < 3; k++)
         {
             g_kl[k].resize(win_size);
             U[k].setZero();
             U[k].block<3, 3>(0, 0) = hat(-u[k]);
             U[k].block<3, 1>(3, 3) = u[k];
         }

         PLV(6) UlTCF(win_size, Eigen::Matrix<double, 6, 1>::Zero());

         if (jacobians)
         {
             for (size_t i = 0; i < win_size; i++)
             {
                 if (jacobians[i])
                 {
                     Eigen::Map<Eigen::Matrix<double, 1, 7/*, Eigen::RowMajor*/>> jaco(jacobians[i]);
                     jaco.setZero();
                     if (Ns[i] != 0)
                     {
                         // 公式32中 SP * (Tq - 1/N * CF*
                         Eigen::Matrix<double, 3, 4> temp = T[i].block<3, 4>(0, 0);
                         temp.block<3, 1>(0, 3) -= v_bar;

                         Eigen::Matrix<double, 4, 3> TC_TCFSp = TC[i] * temp.transpose();
                         for (int k = 0; k < 3; k++)
                         {
                             Eigen::Matrix<double, 6, 1> g1, g2;
                             g1 = U[k] * TC_TCFSp * u[l];
                             g2 = U[l] * TC_TCFSp * u[k];

                             g_kl[k][i] = (g1 + g2) / NN;
                         }

                         UlTCF[i] = (U[l] * TC[i]).block<6, 1>(0, 3);
                         jaco.block<1, 6>(0, 0) += coe * g_kl[l][i].transpose();
                     }
                 }
             }

         }

         return true;
    }

private:
    const PointCluster sig_vec_;
    const std::vector<PointCluster> plvec_voxels_;
    const double coeff_;
    int index_;
};


class LdiarBACostFunction
{
public:
    LdiarBACostFunction(const PointCluster& sig_vec, const std::vector<PointCluster>& plvec_voxels, const double& coeff, int index) :
        sig_vec_(sig_vec), plvec_voxels_(plvec_voxels), coeff_(coeff),index_(index){}

    static ceres::CostFunction* Create(const PointCluster& sig_vec, const std::vector<PointCluster>& plvec_voxels, const double& coeff,int index)
    {
        auto cost_function = new ceres::DynamicAutoDiffCostFunction<LdiarBACostFunction, 4>(
            		new LdiarBACostFunction(sig_vec, plvec_voxels,coeff,index));

        for (int i = 0; i < plvec_voxels.size(); i++)
        {
        	cost_function->AddParameterBlock(7);
        }
        
        cost_function->SetNumResiduals(1);

        return cost_function;
    }
    // q1 t1 q2 t2 .... qn tn
    template <typename T>
    bool operator()(T const* const *  parameters, T* residuals) const
    {
        Eigen::Matrix<T, 3, 3> pAll = sig_vec_.P.cast<T>();
        Eigen::Matrix<T, 3, 1> vAll = sig_vec_.v.cast<T>();
        T NAll = T(sig_vec_.N);

        for (size_t i = 0; i < plvec_voxels_.size(); i++)
        {
            Eigen::Matrix<T, 3, 3> sig_p = plvec_voxels_[i].P.cast<T>();
            Eigen::Matrix<T, 3, 1> sig_v = plvec_voxels_[i].v.cast<T>();
            T N = T(plvec_voxels_[i].N);

            Eigen::Quaternion<T> qi = Eigen::Map<const Eigen::Quaternion<T>>(&parameters[i][0]);
            Eigen::Matrix<T, 3, 1> ti = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(&parameters[i][4]);
            
            Eigen::Matrix<T, 3, 3> rp = qi * sig_v * ti.transpose();
            Eigen::Matrix<T, 3, 3> p = qi.toRotationMatrix() * sig_p* qi.toRotationMatrix().transpose() + rp + rp.transpose() + N * ti * ti.transpose();
            
            NAll += N;
            vAll += qi * sig_v + N * ti;
            pAll += p;
        }

        Eigen::Matrix<T, 3, 1> center = vAll / NAll;
        Eigen::Matrix<T, 3, 3> cov = pAll / NAll - center * center.transpose();

        T eigenvalues[3];
        Compute<T>(cov, eigenvalues);

        // 找到最大特征值
        T min_eigenvalue = T(0);
        if (eigenvalues[0] <= eigenvalues[1] && eigenvalues[0] <= eigenvalues[2])
        {
            min_eigenvalue = eigenvalues[0];
        }
        else if (eigenvalues[1] <= eigenvalues[0] && eigenvalues[1] <= eigenvalues[2])
        {
            min_eigenvalue = eigenvalues[1];
        }
        else
            min_eigenvalue = eigenvalues[2];
 
        // 设置残差
        
        residuals[0] = (min_eigenvalue) * NAll;
        return true;
    }

private:
    const PointCluster sig_vec_;
    const std::vector<PointCluster> plvec_voxels_;
    const double coeff_;
    int index_;
};

class OptimCeres
{
public:

    struct PoseStruct
    {
        double pose[7];
    };

	void Optimilize(vector<IMUST>& x_stats, VOX_HESS& voxhess)
	{

        std::cout << "beg optim!" << std::endl;
        vector<int> planes(x_stats.size(), 0);
        for (int i = 0; i < voxhess.plvec_voxels.size(); i++)
        {
            for (int j = 0; j < voxhess.plvec_voxels[i]->size(); j++)
                if (voxhess.plvec_voxels[i]->at(j).N != 0)
                    planes[j]++;
        }
        sort(planes.begin(), planes.end());

        // 检查平面数量，数量小说明初始误差大，需要更多平面
        if (planes[0] < 20)
        {
            printf("Initial error too large.\n");
            printf("Please loose plane determination criteria for more planes.\n");
            printf("The optimization is terminated.\n");
            return;
        }

        int win_size = x_stats.size();
        PLM(4) T(win_size);

        ceres::Problem problem;
        ceres::LossFunction* loss = new ceres::CauchyLoss(0.1);

        
        std::cout << "beg optim22!" << std::endl;
        std::vector<Eigen::Quaterniond> quaternions(win_size);
        std::vector<Eigen::Vector3d> translates(win_size);
        std::vector<PoseStruct> poses(win_size);
        std::vector<double*> parameter_blocks;
        parameter_blocks.reserve(win_size * 4 + win_size * 3);
        for (int i = 0; i < win_size; i++)
        {
            Eigen::Quaterniond qq = Eigen::Quaterniond(x_stats[i].R);
            
            memcpy(poses[i].pose, qq.coeffs().data(), sizeof(double) * 4);
            memcpy(poses[i].pose + 4, x_stats[i].p.data(), sizeof(double) * 3);

            parameter_blocks.push_back(poses[i].pose);
        } 

        std::vector<const ceres::Manifold*> manifolds;
        for (size_t i = 0; i < win_size; i++)
        {
            // 创建复合流形
            //ceres::Manifold* quaternion_manifold = new EigenQuternionManifoldMySelf;
            //ceres::Manifold* euclidean_manifold = new ceres::EuclideanManifold<3>;
            //ceres::Manifold* product_manifold = new ceres::ProductManifold(quaternion_manifold, euclidean_manifold);
            ceres::Manifold* product_manifold = new EigenSE3;
            manifolds.push_back(product_manifold);
        }

        double resALL = 0;
        for (size_t i = 0; i < voxhess.plvec_voxels.size(); i++)
        {
            // 自动求导的损失函数，非常耗时，自动求导耗时占了90%以上
            //ceres::CostFunction* cost = LdiarBACostFunction::Create(*voxhess.sig_vecs[i], *voxhess.plvec_voxels[i], 1.0,i);
            
            // 解析导数损失函数，收敛没有自动求导收敛快
            ceres::DynamicCostFunction* cost = new LdiarBACostFunctionNoAuto(*voxhess.sig_vecs[i], *voxhess.plvec_voxels[i], 1.0, i);
            // 创建并配置 GradientChecker，检查雅可比矩阵
            //ceres::NumericDiffOptions opt;
            //ceres::GradientChecker gradient_checker(cost, &manifolds, opt);

            //ceres::GradientChecker::ProbeResults results;
            //if (!gradient_checker.Probe(parameter_blocks.data(), 1e-9, &results)) 
            //{
            //    LOG(ERROR) << "An error has occurred:\n" << results.error_log;
            //}

            problem.AddResidualBlock(cost, loss, parameter_blocks);
            std::cout << "add res ! ===  " << i << std::endl;
        }




        // TODO流形需要修改
        for (size_t i = 0; i < win_size; i++)
        {
            // 创建复合流形
            //ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold;
            //ceres::Manifold* euclidean_manifold = new ceres::EuclideanManifold<3>;
            //ceres::Manifold* product_manifold = new ceres::ProductManifold(quaternion_manifold, euclidean_manifold);
            ceres::Manifold* product_manifold = new EigenSE3;
            // 设置复合流形
            problem.SetManifold(poses[i].pose, product_manifold);
        }



        problem.SetParameterBlockConstant(poses[0].pose);
        
        auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::cout << " Optimize! " << std::endl;
        ceres::Solver::Options options;
        options.num_threads = 8;
        options.max_num_iterations = 50;
        options.preconditioner_type = ceres::SCHUR_JACOBI;
        options.line_search_direction_type = ceres::BFGS;
        options.minimizer_progress_to_stdout = true;   // 输出优化进度
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;  // 设置LM算法
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;  // 也可以选择其他二阶求解器
        ceres::Solver::Summary sum;
        ceres::Solve(options, &problem, &sum);
        std::cout << sum.FullReport() << std::endl;

        auto end = std::chrono::high_resolution_clock::now().time_since_epoch().count();

        std::cout << "spend: " << double(end - start) / 1e9 << " second!" << std::endl;
        for (size_t i = 0; i < x_stats.size(); i++)
        {
            Eigen::Quaterniond q;
            Eigen::Vector3d p;
            memcpy(q.coeffs().data(), poses[i].pose, sizeof(double) * 4);
            memcpy(p.data(), poses[i].pose + 4, sizeof(double) * 3);
            q.normalize();
            x_stats[i].R = q.toRotationMatrix();
            x_stats[i].p = p;
        }
        
	}
private:

};