#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
	VectorXd delta(4);
	rmse << 0,0,0,0;

  if (estimations.size() != ground_truth.size()){
    return rmse;
  }

	//accumulate squared residuals
	for(int i=0; i < estimations.size(); ++i){
    delta = estimations.at(i) - ground_truth.at(i);
		delta = delta.array() * delta.array();
		rmse = delta + rmse;
	}

	//calculate the mean
	rmse = rmse / estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}
