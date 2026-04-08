#include <iostream>
#include <traj_utils/polynomial_traj.h>

/**
 * @brief 生成经过多个航点的最小 Snap (四阶导数) 多项式轨迹 (闭式求解)
 *
 * 求解流程:
 *   1. 构建映射矩阵 A: 多项式系数 → 端点导数值  (A * c = d)
 *   2. 构建选择矩阵 C: 端点导数 → 不重复的决策变量 [d_F; d_P]  (d = Cᵀ * [d_F; d_P])
 *   3. 构建 Snap 代价矩阵 Q: Q(i,j) = ∫ (d³pᵢ/dt³)(d³pⱼ/dt³) dt
 *   4. 转换到导数空间: R = C * A⁻ᵀ * Q * A⁻¹ * Cᵀ
 *   5. 令 ∂J/∂d_P = 0, 解出: d_P* = -R_PP⁻¹ * R_FPᵀ * d_F
 *   6. 反算多项式系数: c = A⁻¹ * Cᵀ * [d_F; d_P*]
 *
 * @param Pos       3×(seg_num+1) 矩阵, 每列是一个航点 [x,y,z]ᵀ
 *                  col(0)=起点, col(1)~col(seg_num-1)=中间航点, col(seg_num)=终点
 * @param start_vel 起点速度 [vx, vy, vz]ᵀ
 * @param end_vel   终点速度 [vx, vy, vz]ᵀ
 * @param start_acc 起点加速度 [ax, ay, az]ᵀ
 * @param end_acc   终点加速度 [ax, ay, az]ᵀ
 * @param Time      seg_num 维向量, Time(k) = 第 k 段飞行时间
 * @return          多段多项式轨迹对象
 */
PolynomialTraj PolynomialTraj::minSnapTraj(const Eigen::MatrixXd &Pos, const Eigen::Vector3d &start_vel,
                                           const Eigen::Vector3d &end_vel, const Eigen::Vector3d &start_acc,
                                           const Eigen::Vector3d &end_acc, const Eigen::VectorXd &Time)
{
  int seg_num = Time.size();
  Eigen::MatrixXd poly_coeff(seg_num, 3 * 6); // seg_num × 18, 每行 = [cx0~5 | cy0~5 | cz0~5]
  Eigen::VectorXd Px(6 * seg_num), Py(6 * seg_num), Pz(6 * seg_num);

  int num_f, num_p; // num_f: 固定导数变量个数, num_p: 自由导数变量个数 (待优化)
  int num_d;        // 所有段端点导数总个数 = 6 * seg_num

  const static auto Factorial = [](int x) {
    int fac = 1;
    for (int i = x; i > 0; i--)
      fac = fac * i;
    return fac;
  };

  /* ---------- 端点导数向量 D ----------
   * 每段 6 个端点导数, 排列: D[k*6+0]=p(0), D[k*6+1]=p(T),
   *   D[k*6+2]=v(0), D[k*6+3]=v(T), D[k*6+4]=a(0), D[k*6+5]=a(T)
   * 位置来自航点(全部已知), 速度/加速度只在首段起点和末段终点已知
   */
  Eigen::VectorXd Dx = Eigen::VectorXd::Zero(seg_num * 6);
  Eigen::VectorXd Dy = Eigen::VectorXd::Zero(seg_num * 6);
  Eigen::VectorXd Dz = Eigen::VectorXd::Zero(seg_num * 6);

  for (int k = 0; k < seg_num; k++)
  {
    // 每段起点和终点的位置 (来自航点坐标)
    Dx(k * 6) = Pos(0, k);
    Dx(k * 6 + 1) = Pos(0, k + 1);
    Dy(k * 6) = Pos(1, k);
    Dy(k * 6 + 1) = Pos(1, k + 1);
    Dz(k * 6) = Pos(2, k);
    Dz(k * 6 + 1) = Pos(2, k + 1);

    if (k == 0)
    {
      // 第一段起点: 速度和加速度由用户指定
      Dx(k * 6 + 2) = start_vel(0);
      Dy(k * 6 + 2) = start_vel(1);
      Dz(k * 6 + 2) = start_vel(2);

      Dx(k * 6 + 4) = start_acc(0);
      Dy(k * 6 + 4) = start_acc(1);
      Dz(k * 6 + 4) = start_acc(2);
    }
    else if (k == seg_num - 1)
    {
      // 最后一段终点: 速度和加速度由用户指定
      Dx(k * 6 + 3) = end_vel(0);
      Dy(k * 6 + 3) = end_vel(1);
      Dz(k * 6 + 3) = end_vel(2);

      Dx(k * 6 + 5) = end_acc(0);
      Dy(k * 6 + 5) = end_acc(1);
      Dz(k * 6 + 5) = end_acc(2);
    }
  }

  /* ---------- 映射矩阵 A ----------
   * A 将多项式系数 c 映射到端点导数 d:  A * c = d
   * 对 5 次多项式 p(t) = c0 + c1*t + c2*t² + c3*t³ + c4*t⁴ + c5*t⁵:
   *   行 2i  : 第 i 阶导在 t=0 的值  (仅第 i 列非零, 值为 i!)
   *   行 2i+1: 第 i 阶导在 t=T 的值  (展开式)
   * 整体 A 是分块对角矩阵, 每个 6×6 块对应一段
   */
  Eigen::MatrixXd Ab;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(seg_num * 6, seg_num * 6);

  for (int k = 0; k < seg_num; k++)
  {
    Ab = Eigen::MatrixXd::Zero(6, 6);
    for (int i = 0; i < 3; i++) // i=0:位置, i=1:速度, i=2:加速度
    {
      Ab(2 * i, i) = Factorial(i);            // t=0 时: 第 i 阶导 = i! * c_i
      for (int j = i; j < 6; j++)
        Ab(2 * i + 1, j) = Factorial(j) / Factorial(j - i) * pow(Time(k), j - i); // t=T 时
    }
    A.block(k * 6, k * 6, 6, 6) = Ab;
  }

  /* ---------- 选择矩阵 Cᵀ ----------
   * 将端点导数 d (长度 6m, 含重复项) 重排为不重复的决策变量:
   *   d = Cᵀ * [d_F; d_P]
   *
   * d_F (固定变量, 长度 num_f = 2m+4):
   *   所有航点位置(m+1个) + 起点v/a + 终点v/a
   * d_P (自由变量, 长度 num_p = 2m-2):
   *   中间航点的速度和加速度 (每个中间点 2 个)
   *
   * 相邻段共享连接点的导数被映射到同一个变量, 自动保证 v/a 连续性
   */
  Eigen::MatrixXd Ct, C;

  num_f = 2 * seg_num + 4; // 固定变量: (m+1)位置 + 起点v,a + 终点v,a = 2m+4
  num_p = 2 * seg_num - 2; // 自由变量: (m-1)个中间点 × 2(v,a) = 2m-2
  num_d = 6 * seg_num;
  Ct = Eigen::MatrixXd::Zero(num_d, num_f + num_p);

  // --- 第一段起点: 位置/速度/加速度 → d_F 的前 3 个 ---
  Ct(0, 0) = 1;  // 起点位置 → d_F[0]
  Ct(2, 1) = 1;  // 起点速度 → d_F[1]
  Ct(4, 2) = 1;  // 起点加速度 → d_F[2]

  // --- 第一段终点: 位置已知, 速度/加速度为自由变量 ---
  Ct(1, 3) = 1;                    // 终点位置 → d_F[3]
  Ct(3, 2 * seg_num + 4) = 1;      // 终点速度 → d_P[0]
  Ct(5, 2 * seg_num + 5) = 1;      // 终点加速度 → d_P[1]

  // --- 最后一段: 起/终点位置, 终点速度/加速度 ---
  Ct(6 * (seg_num - 1) + 0, 2 * seg_num + 0) = 1;  // 末段起点位置 → d_F
  Ct(6 * (seg_num - 1) + 1, 2 * seg_num + 1) = 1;  // 末段终点位置 → d_F
  Ct(6 * (seg_num - 1) + 2, 4 * seg_num + 0) = 1;  // 末段起点速度 → d_P (=倒数第二航点的v)
  Ct(6 * (seg_num - 1) + 3, 2 * seg_num + 2) = 1;  // 末段终点速度 → d_F (用户指定)
  Ct(6 * (seg_num - 1) + 4, 4 * seg_num + 1) = 1;  // 末段起点加速度 → d_P
  Ct(6 * (seg_num - 1) + 5, 2 * seg_num + 3) = 1;  // 末段终点加速度 → d_F (用户指定)

  // --- 中间段 (第 2 段 ~ 第 seg_num-1 段): 相邻段共享连接点 ---
  for (int j = 2; j < seg_num; j++)
  {
    Ct(6 * (j - 1) + 0, 2 + 2 * (j - 1) + 0) = 1;                   // 本段起点位置 → d_F
    Ct(6 * (j - 1) + 1, 2 + 2 * (j - 1) + 1) = 1;                   // 本段终点位置 → d_F
    Ct(6 * (j - 1) + 2, 2 * seg_num + 4 + 2 * (j - 2) + 0) = 1;     // 本段起点速度 → d_P (=上段终点v)
    Ct(6 * (j - 1) + 3, 2 * seg_num + 4 + 2 * (j - 1) + 0) = 1;     // 本段终点速度 → d_P
    Ct(6 * (j - 1) + 4, 2 * seg_num + 4 + 2 * (j - 2) + 1) = 1;     // 本段起点加速度 → d_P (=上段终点a)
    Ct(6 * (j - 1) + 5, 2 * seg_num + 4 + 2 * (j - 1) + 1) = 1;     // 本段终点加速度 → d_P
  }

  C = Ct.transpose();

  // 通过 C 将原始 D 映射到 [d_F; d_P] 空间
  Eigen::VectorXd Dx1 = C * Dx;
  Eigen::VectorXd Dy1 = C * Dy;
  Eigen::VectorXd Dz1 = C * Dz;

  /* ---------- Snap 代价矩阵 Q ----------
   * 对 5 次多项式, 3 阶导 (jerk) 的积分:
   *   Q(i,j) = ∫₀ᵀ [d³(t^i)/dt³] * [d³(t^j)/dt³] dt
   *          = i(i-1)(i-2) * j(j-1)(j-2) / (i+j-5) * T^{i+j-5}
   * 只有 i,j ≥ 3 时才非零 (低次项的 3 阶导为 0)
   * Q 也是分块对角的, 每段独立
   */
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(seg_num * 6, seg_num * 6);

  for (int k = 0; k < seg_num; k++)
  {
    for (int i = 3; i < 6; i++)
    {
      for (int j = 3; j < 6; j++)
      {
        Q(k * 6 + i, k * 6 + j) =
            i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2) / (i + j - 5) * pow(Time(k), (i + j - 5));
      }
    }
  }

  /* ---------- R 矩阵: 将代价从系数空间转到导数空间 ----------
   * 原始代价: J = cᵀ Q c
   * 因 A*c = d, c = A⁻¹*d; 又 d = Cᵀ * [d_F; d_P]
   * 代入得 J = [d_F; d_P]ᵀ * (C * A⁻ᵀ * Q * A⁻¹ * Cᵀ) * [d_F; d_P]
   *          = [d_F; d_P]ᵀ * R * [d_F; d_P]
   */
  Eigen::MatrixXd R = C * A.transpose().inverse() * Q * A.inverse() * Ct;

  // 提取固定变量 d_F 部分
  Eigen::VectorXd Dxf(2 * seg_num + 4), Dyf(2 * seg_num + 4), Dzf(2 * seg_num + 4);

  Dxf = Dx1.segment(0, 2 * seg_num + 4);
  Dyf = Dy1.segment(0, 2 * seg_num + 4);
  Dzf = Dz1.segment(0, 2 * seg_num + 4);

  /* 将 R 分为 4 个子块:
   *   R = [ R_FF  R_FP ]    代价展开:
   *       [ R_PF  R_PP ]    J = d_Fᵀ R_FF d_F + 2 d_Fᵀ R_FP d_P + d_Pᵀ R_PP d_P
   */
  Eigen::MatrixXd Rff(2 * seg_num + 4, 2 * seg_num + 4);
  Eigen::MatrixXd Rfp(2 * seg_num + 4, 2 * seg_num - 2);
  Eigen::MatrixXd Rpf(2 * seg_num - 2, 2 * seg_num + 4);
  Eigen::MatrixXd Rpp(2 * seg_num - 2, 2 * seg_num - 2);

  Rff = R.block(0, 0, 2 * seg_num + 4, 2 * seg_num + 4);
  Rfp = R.block(0, 2 * seg_num + 4, 2 * seg_num + 4, 2 * seg_num - 2);
  Rpf = R.block(2 * seg_num + 4, 0, 2 * seg_num - 2, 2 * seg_num + 4);
  Rpp = R.block(2 * seg_num + 4, 2 * seg_num + 4, 2 * seg_num - 2, 2 * seg_num - 2);

  /* ---------- 闭式最优解 ----------
   * 令 ∂J/∂d_P = 0:
   *   2 * R_PF * d_F + 2 * R_PP * d_P = 0
   *   d_P* = -R_PP⁻¹ * R_PF * d_F = -R_PP⁻¹ * R_FPᵀ * d_F  (R 对称)
   */
  Eigen::VectorXd Dxp(2 * seg_num - 2), Dyp(2 * seg_num - 2), Dzp(2 * seg_num - 2);
  Dxp = -(Rpp.inverse() * Rfp.transpose()) * Dxf;
  Dyp = -(Rpp.inverse() * Rfp.transpose()) * Dyf;
  Dzp = -(Rpp.inverse() * Rfp.transpose()) * Dzf;

  // 将最优自由变量填回完整导数向量
  Dx1.segment(2 * seg_num + 4, 2 * seg_num - 2) = Dxp;
  Dy1.segment(2 * seg_num + 4, 2 * seg_num - 2) = Dyp;
  Dz1.segment(2 * seg_num + 4, 2 * seg_num - 2) = Dzp;

  // 反算多项式系数: c = A⁻¹ * d = A⁻¹ * Cᵀ * [d_F; d_P*]
  Px = (A.inverse() * Ct) * Dx1;
  Py = (A.inverse() * Ct) * Dy1;
  Pz = (A.inverse() * Ct) * Dz1;

  // 整理系数到 poly_coeff 矩阵
  for (int i = 0; i < seg_num; i++)
  {
    poly_coeff.block(i, 0, 1, 6) = Px.segment(i * 6, 6).transpose();
    poly_coeff.block(i, 6, 1, 6) = Py.segment(i * 6, 6).transpose();
    poly_coeff.block(i, 12, 1, 6) = Pz.segment(i * 6, 6).transpose();
  }

  /* ---------- 构建轨迹对象 ---------- */
  PolynomialTraj poly_traj;
  for (int i = 0; i < poly_coeff.rows(); ++i)
  {
    vector<double> cx(6), cy(6), cz(6);
    for (int j = 0; j < 6; ++j)
    {
      cx[j] = poly_coeff(i, j), cy[j] = poly_coeff(i, j + 6), cz[j] = poly_coeff(i, j + 12);
    }
    // reverse: 存储顺序从 [c0,...,c5] 转为 [c5,...,c0] (高次在前)
    reverse(cx.begin(), cx.end());
    reverse(cy.begin(), cy.end());
    reverse(cz.begin(), cz.end());
    double ts = Time(i);
    poly_traj.addSegment(cx, cy, cz, ts);
  }

  return poly_traj;
}

/**
 * @brief 生成单段 5 次多项式轨迹
 *
 * 给定起点和终点的位置/速度/加速度 (共 6 个约束),
 * 求解 5 次多项式 p(t) = c5*t⁵ + c4*t⁴ + c3*t³ + c2*t² + c1*t + c0
 * 6 个约束唯一确定 6 个系数, 直接解线性方程组 C * coeff = B
 *
 * @param start_pt  起点位置 [x, y, z]ᵀ
 * @param start_vel 起点速度 [vx, vy, vz]ᵀ
 * @param start_acc 起点加速度 [ax, ay, az]ᵀ
 * @param end_pt    终点位置 [x, y, z]ᵀ
 * @param end_vel   终点速度 [vx, vy, vz]ᵀ
 * @param end_acc   终点加速度 [ax, ay, az]ᵀ
 * @param t         总飞行时间 T
 * @return          单段多项式轨迹对象
 */
PolynomialTraj PolynomialTraj::one_segment_traj_gen(const Eigen::Vector3d &start_pt, const Eigen::Vector3d &start_vel, const Eigen::Vector3d &start_acc,
                                                    const Eigen::Vector3d &end_pt, const Eigen::Vector3d &end_vel, const Eigen::Vector3d &end_acc,
                                                    double t)
{
  /* 约束矩阵 C, 使得 C * [c5,c4,c3,c2,c1,c0]ᵀ = B
   * 多项式: p(t) = c5*t⁵ + c4*t⁴ + c3*t³ + c2*t² + c1*t + c0
   * (系数从高次到低次排列)
   *
   *   行 0: p(0)   = c0                    → [0, 0, 0, 0, 0, 1]
   *   行 1: p'(0)  = c1                    → [0, 0, 0, 0, 1, 0]
   *   行 2: p''(0) = 2*c2                  → [0, 0, 0, 2, 0, 0]
   *   行 3: p(T)   = c5T⁵+c4T⁴+...+c0      → [T⁵, T⁴, T³, T², T, 1]
   *   行 4: p'(T)  = 5c5T⁴+4c4T³+...       → [5T⁴, 4T³, 3T², 2T, 1, 0]
   *   行 5: p''(T) = 20c5T³+12c4T²+...     → [20T³, 12T², 6T, 2, 0, 0]
   */
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(6, 6), Crow(1, 6);
  Eigen::VectorXd Bx(6), By(6), Bz(6);

  C(0, 5) = 1;  // p(0) = c0
  C(1, 4) = 1;  // p'(0) = c1
  C(2, 3) = 2;  // p''(0) = 2*c2
  Crow << pow(t, 5), pow(t, 4), pow(t, 3), pow(t, 2), t, 1;
  C.row(3) = Crow;  // p(T)
  Crow << 5 * pow(t, 4), 4 * pow(t, 3), 3 * pow(t, 2), 2 * t, 1, 0;
  C.row(4) = Crow;  // p'(T)
  Crow << 20 * pow(t, 3), 12 * pow(t, 2), 6 * t, 2, 0, 0;
  C.row(5) = Crow;  // p''(T)

  // 右端向量 B: 每个轴独立
  Bx << start_pt(0), start_vel(0), start_acc(0), end_pt(0), end_vel(0), end_acc(0);
  By << start_pt(1), start_vel(1), start_acc(1), end_pt(1), end_vel(1), end_acc(1);
  Bz << start_pt(2), start_vel(2), start_acc(2), end_pt(2), end_vel(2), end_acc(2);

  // QR 分解求解 C * coeff = B → coeff = C⁻¹ * B
  Eigen::VectorXd Cofx = C.colPivHouseholderQr().solve(Bx);
  Eigen::VectorXd Cofy = C.colPivHouseholderQr().solve(By);
  Eigen::VectorXd Cofz = C.colPivHouseholderQr().solve(Bz);

  // 系数顺序: [c5, c4, c3, c2, c1, c0] (高次在前)
  vector<double> cx(6), cy(6), cz(6);
  for (int i = 0; i < 6; i++)
  {
    cx[i] = Cofx(i);
    cy[i] = Cofy(i);
    cz[i] = Cofz(i);
  }

  PolynomialTraj poly_traj;
  poly_traj.addSegment(cx, cy, cz, t);

  return poly_traj;
}
