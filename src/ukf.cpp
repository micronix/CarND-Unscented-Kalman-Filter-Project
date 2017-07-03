#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

#define EPS 0.001
#define PI 3.1415926

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.57;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  x_ << 0, 0, 0, 0, 0;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  n_x_ = x_.size();
  n_aug_ = n_x_ + 2;
  lambda_ = 3 - n_aug_;
  n_sig_ = 2 * n_aug_ + 1;

  Xsig_pred_ = MatrixXd(n_x_, n_sig_);
  weights_ = VectorXd(n_sig_);

  // noise matrices
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;

  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;
}

UKF::~UKF() {}

void UKF::normalizeAngle(double *angle){
  while(*angle > M_PI){
    *angle -= 2. * M_PI;
  }
  while(*angle < -M_PI){
    *angle += 2. * M_PI;
  }
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  // initialize state if not initialized
  if (!is_initialized_){
    if (meas_package.sensor_type_ == MeasurementPackage::LASER){
      float x = meas_package.raw_measurements_[0];
      float y = meas_package.raw_measurements_[1];

      x_ << x, y, 0, 0, 0;
    } else {
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float rho_dot = meas_package.raw_measurements_[2];

      // for radar measurement convert from polar coordinates
      float px = rho * cos(phi);
      float py = rho * sin(phi);
      float vx = rho_dot * cos(phi);
      float vy = rho_dot * cos(phi);
      float v = sqrt(vx * vx + vy * vy);
      x_ << px, py, v, 0, 0;
    }

    // initialize weights
    weights_(0) = lambda_ / (n_aug_ + lambda_);
    for(int i = 1; i < weights_.size(); i++){
      weights_(i) = 0.5 / (n_aug_ + lambda_);
    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  // update based on measurement
  double dt = meas_package.timestamp_ - time_us_;
  dt = dt / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_){
    UpdateRadar(meas_package);
  }
  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // augment mean and covarance
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // sigma points
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);

  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;

  MatrixXd L = P_aug.llt().matrixL();

  // create sigman points
  Xsig_aug.col(0) = x_aug;
  double coeff = sqrt(lambda_ + n_aug_);
  VectorXd Lp;
  for(int i = 0; i < n_aug_; i++){
    Lp = coeff * L.col(i);
    Xsig_aug.col(i + 1) = x_aug + Lp;
    Xsig_aug.col(i + 1 + n_aug_) =  x_aug - Lp;
  }

  // sigman points prediction
  for(int i = 0; i < n_sig_; i++){
    double px = Xsig_aug(0, i);
    double py = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double theta = Xsig_aug(3, i);
    double thetad = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_thetadd = Xsig_aug(6, i);

    double tsin = sin(theta);
    double tcos = cos(theta);
    double total_theta = theta + thetad * delta_t;

    // turning vs no turning
    double px2;
    double py2;
    if (fabs(thetad) > EPS) {
      double coeff = v / thetad;
      px2 = px + coeff * (sin(total_theta) - tsin);
      py2 = py + coeff * (tcos - cos(total_theta));
    } else {
      px2 = px + v * delta_t * tcos;
      py2 = py + v * delta_t * tsin;
    }

    double v2 = v;
    double theta2 = total_theta;
    double thetad2 = thetad;

    // add noise
    px2 += 0.5 * nu_a * delta_t * delta_t * tcos;
    py2 += 0.5 * nu_a * delta_t * delta_t * tsin;
    v2 += nu_a * delta_t;
    theta2 += 0.5 * nu_thetadd * delta_t * delta_t;
    thetad2 += nu_thetadd * delta_t;

    Xsig_pred_(0, i) = px2;
    Xsig_pred_(1, i) = py2;
    Xsig_pred_(2, i) = v2;
    Xsig_pred_(3, i) = theta2;
    Xsig_pred_(4, i) = thetad2;
  }

  x_ = Xsig_pred_ * weights_;

  P_.fill(0.0);
  for(int i = 0; i < n_sig_; i++){
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    normalizeAngle(&x_diff(3));
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  int n_z = 2;
  MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z, n_sig_);
  UpdateUKF(meas_package, Zsig, n_z);

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);

  // convert state to radar format
  for(int i = 0; i < n_sig_; i++){
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double theta = Xsig_pred_(3, i);

    Zsig(0, i) = sqrt(px * px + py * py);
    Zsig(1, i) = atan2(py, px);
    Zsig(2, i) = (px * cos(theta)  + py * sin(theta)) * v / Zsig(0, i);
  }
  UpdateUKF(meas_package, Zsig, n_z);
}

void UKF::UpdateUKF(MeasurementPackage meas_package, MatrixXd Zsig, int n_z){
  VectorXd z_pred = VectorXd(n_z);
  z_pred = Zsig * weights_;

  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for(int i = 0; i < n_sig_; i++){
    VectorXd z_diff = Zsig.col(i) - z_pred;
    normalizeAngle(&z_diff(1));
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  MatrixXd R = MatrixXd(n_z, n_z);
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
    R = R_radar_;
  } else {
    R = R_lidar_;
  }
  S = S + R;

  // calculate cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for(int i = 0; i < n_sig_; i++){
    // calculate z_diff
    VectorXd z_diff = Zsig.col(i) - z_pred;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
      normalizeAngle(&z_diff(1));
    }

    // calculate x_diff
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    normalizeAngle(&x_diff(3));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // measurements
  VectorXd z = meas_package.raw_measurements_;

  // kalman gain
  MatrixXd K = Tc * S.inverse();

  // update the state and covariance normalize angle if radar
  VectorXd z_diff = z - z_pred;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
    normalizeAngle(&z_diff(1));
  }

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
    //NIS_radar_ = z.transpose() * S.inverse() * z;
  }
  if (meas_package.sensor_type_ == MeasurementPackage::LASER){
    //NIS_laser_ = z.transpose() * S.inverse() * z;
  }
}
