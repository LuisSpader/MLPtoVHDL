//Numpy array shape [64, 32]
//Min -1.000000000000
//Max 0.937500000000
//Number of zeros 251

#ifndef W2_H_
#define W2_H_

#ifndef __SYNTHESIS__
weight2_t w2[2048];
#else
weight2_t w2[2048] = {0.0000, -0.1875, 0.5625, -0.2500, 0.1250, 0.3750, -0.1875, 0.2500, 0.0000, -0.1875, -0.6875, -0.0625, 0.1875, -0.1875, 0.1875, 0.1250, -0.1875, 0.1875, 0.0625, -0.4375, 0.1875, -0.2500, 0.8750, -0.3125, -0.1875, 0.0000, 0.1250, 0.1250, 0.2500, 0.1250, -0.8125, -0.1875, 0.2500, 0.1250, 0.1250, -0.5000, -0.5625, -0.0625, -0.8125, 0.4375, -0.1250, 0.0625, -0.5000, 0.0000, 0.1875, -0.6250, 0.0000, -0.0625, -0.0625, 0.2500, -0.3750, -0.5625, -0.0625, -0.4375, 0.2500, -0.2500, -0.3750, 0.3750, 0.0000, 0.2500, 0.1875, 0.1250, -0.3125, 0.0000, 0.2500, -0.3125, -0.0625, -0.7500, -0.5000, -0.6875, -0.5000, 0.3750, -0.0625, 0.0000, -0.1875, -0.3750, 0.1250, -0.3125, 0.0000, -0.5625, -0.8750, 0.2500, -0.4375, -0.3125, -0.1875, -0.2500, 0.3750, 0.0625, -0.5000, 0.6250, 0.1250, 0.1250, 0.1250, 0.1250, -0.0625, 0.1250, 0.4375, -0.0625, -0.1250, -0.2500, -0.5625, -0.8125, -0.3125, 0.4375, -0.0625, 0.1875, 0.0000, 0.0625, -0.0625, -0.0625, 0.0625, -0.5625, -0.6250, 0.1250, 0.0000, -0.4375, -0.0625, 0.0000, -0.1250, 0.0000, 0.0000, 0.5000, 0.1875, 0.0625, 0.1875, 0.1875, 0.0000, -0.1875, 0.1250, -0.0625, -0.1250, -0.4375, -0.6250, 0.0000, 0.0000, 0.6875, 0.1250, 0.1250, 0.0625, -0.0625, -0.0625, 0.0000, 0.2500, -1.0000, -0.3125, 0.0000, 0.1250, -0.1875, -0.1875, 0.0625, 0.0000, 0.1875, 0.0000, 0.1875, 0.1250, 0.1250, -0.3750, 0.0000, 0.3125, -0.1250, 0.1250, 0.1875, 0.0000, -0.2500, -0.5625, -0.1250, -0.1875, 0.6875, -0.0625, 0.0000, 0.1875, 0.1250, 0.0625, -0.3750, 0.1875, -0.9375, 0.0000, 0.0000, 0.1875, -0.3125, 0.0625, 0.0000, -0.0625, 0.1250, -0.0625, 0.1875, 0.0625, 0.0625, -0.1875, 0.1875, 0.2500, 0.1250, -0.1250, -0.0625, -0.1250, -0.3750, -0.2500, 0.0000, -0.3125, 0.3125, -0.5625, -0.2500, 0.2500, 0.1875, 0.1250, -0.6250, -0.1875, 0.1250, 0.0625, -0.0625, 0.2500, 0.1875, 0.0000, -0.1250, 0.0625, 0.2500, 0.1250, 0.1250, 0.1250, 0.2500, 0.0625, -0.1875, 0.2500, -0.0625, -0.2500, 0.3125, -0.5000, -0.8750, -0.1875, 0.1875, -0.4375, 0.2500, -0.1250, -0.1250, 0.3125, -0.2500, 0.1875, -0.3750, -0.7500, -0.2500, 0.0625, -0.1250, 0.3750, 0.6250, 0.4375, -0.1250, -0.1250, 0.2500, 0.3125, -0.5000, 0.1250, 0.0000, 0.6250, 0.0000, 0.6875, -0.0625, 0.0000, 0.0000, 0.0000, 0.1875, 0.1250, 0.6250, 0.1250, -0.0625, 0.0625, 0.1875, -1.0000, -0.6875, -0.1250, 0.1250, 0.2500, -0.0625, -0.1250, 0.0625, -0.1875, -0.3750, 0.1250, 0.2500, 0.3750, -0.3125, 0.0625, -0.0625, 0.1250, 0.1875, -0.0625, 0.0625, -0.4375, -0.6250, 0.0625, -0.1875, -0.0625, 0.1875, 0.1250, 0.0625, 0.1875, 0.1875, -0.1250, -0.3125, -0.3750, -0.8750, -0.3125, 0.3125, 0.0000, 0.1875, -0.0625, 0.1875, -0.1875, -0.1875, 0.1250, 0.0625, 0.3125, 0.0000, -0.3750, 0.1250, 0.3125, 0.1250, 0.0000, 0.4375, -0.0625, -0.3125, 0.1875, 0.0625, -0.0625, -0.0625, 0.0000, -0.8750, -0.1250, 0.0000, 0.4375, -0.1875, 0.0000, -0.5000, -0.2500, 0.1250, 0.1250, 0.1250, -0.9375, 0.2500, -0.1875, -0.0625, 0.0000, -0.0625, 0.0000, 0.0000, -0.3750, -0.0625, 0.1250, 0.1250, 0.1250, 0.4375, 0.0000, -0.0625, 0.0625, 0.0000, -0.0625, -0.0625, 0.0000, -0.5000, -0.5625, 0.0625, 0.2500, -0.1875, 0.0625, -0.1875, -0.1875, 0.1250, 0.0000, 0.0000, -0.3750, 0.1875, -0.1250, -0.3125, 0.2500, 0.0625, -0.4375, -0.0625, 0.0625, 0.1875, 0.3125, 0.1875, 0.1250, 0.2500, 0.1250, 0.0000, 0.0000, -0.0625, -0.2500, -0.1875, -0.0625, 0.0000, -0.3750, 0.1250, 0.1875, 0.2500, -0.0625, 0.0625, -0.0625, -0.0625, 0.1875, 0.0000, 0.0000, 0.1250, 0.0625, 0.0625, -0.0625, 0.1250, 0.1250, 0.0625, -0.0625, 0.2500, 0.1875, 0.1250, -1.0000, 0.0625, 0.2500, 0.0000, -0.0625, 0.0625, -0.5000, -0.0625, 0.0000, -0.0625, -0.1875, 0.0625, -0.0625, 0.2500, 0.2500, -0.0625, -0.1875, 0.0000, 0.1250, 0.3750, 0.3125, 0.0625, 0.0000, 0.0625, 0.0000, 0.2500, 0.3125, 0.0625, -0.2500, 0.2500, 0.1250, 0.0625, 0.1875, 0.0625, 0.2500, -0.2500, 0.0000, 0.2500, -0.8125, 0.0000, -0.1250, 0.0000, 0.0625, -0.1250, 0.0625, 0.1875, 0.3125, -0.0625, 0.0000, 0.1250, 0.0625, 0.1875, 0.1250, 0.0625, -0.0625, 0.0625, 0.0000, 0.1250, -0.0625, 0.0625, 0.0625, 0.1250, -0.0625, 0.0625, 0.2500, 0.0625, 0.3750, -0.2500, 0.2500, 0.8125, -0.9375, -0.0625, -0.1250, -0.0625, 0.6250, -0.1875, 0.9375, -0.3125, 0.4375, -0.3750, 0.0000, 0.1875, -0.4375, 0.0625, -0.0625, -0.0625, 0.0625, 0.6250, -0.0625, 0.0000, -0.1250, 0.1250, 0.1875, 0.1875, -0.1250, -0.1250, 0.3125, 0.0625, 0.1875, -0.1875, 0.3125, 0.0000, -0.3750, 0.2500, -0.1250, 0.0000, -0.0625, -0.0625, 0.1875, 0.3750, -0.6875, -0.8750, -0.4375, 0.3125, 0.4375, 0.0625, -0.1875, -0.0625, -0.4375, -0.1875, -0.1250, 0.0625, -0.1875, -0.1875, -0.1875, 0.0000, 0.1875, 0.0625, 0.1250, 0.0625, -0.1250, -0.5625, 0.0000, 0.1875, 0.0000, 0.0000, 0.0000, -0.1875, -0.3125, 0.3125, 0.0000, 0.0625, 0.0625, -0.5000, -0.5625, 0.3750, 0.2500, 0.1250, -0.8125, 0.0000, -0.5000, 0.1250, 0.0000, 0.0000, -0.4375, -0.0625, -0.1875, -0.0625, 0.1875, 0.0000, 0.0000, 0.3750, 0.0625, -0.2500, -0.1875, 0.0625, 0.0000, 0.0625, 0.0625, -0.6250, -0.3125, 0.0625, 0.2500, 0.0625, 0.1875, -0.3125, -0.2500, 0.2500, -0.1250, 0.0625, -0.0625, 0.0625, -0.1250, 0.2500, 0.1250, 0.0000, -0.6875, 0.0000, -0.0625, -0.3750, 0.1875, -0.1250, 0.0625, 0.3125, 0.1875, 0.1250, -0.1250, 0.2500, -0.2500, 0.3750, 0.1250, -0.5000, -0.3125, -0.1250, -0.1250, -0.3125, -0.0625, 0.2500, -0.1250, 0.3750, -0.1250, -0.0625, 0.3750, -0.1875, 0.0625, -0.0625, 0.1250, 0.0625, -0.4375, -0.0625, 0.1875, -0.1875, -0.0625, 0.0000, 0.0000, 0.0000, 0.0625, 0.2500, -0.1250, -0.1250, -0.2500, 0.0000, 0.1875, 0.3125, -0.0625, 0.0000, -0.1875, -0.4375, -0.6250, 0.1250, -0.2500, 0.5625, 0.1250, 0.1250, 0.3750, -0.0625, 0.1875, -0.0625, 0.0625, 0.0000, 0.3750, 0.0625, -0.0625, -0.0625, 0.0625, 0.2500, -1.0000, 0.0625, -0.6250, -0.1875, 0.0625, 0.1875, -0.2500, -0.0625, 0.0625, 0.2500, 0.0000, -0.2500, 0.0625, 0.5625, -0.8125, -0.4375, -0.2500, 0.2500, -0.0625, 0.3750, 0.0625, 0.1250, 0.0000, 0.1875, 0.2500, 0.3125, 0.0625, 0.0625, -0.1875, 0.1250, 0.1875, -0.1250, 0.3125, 0.1875, -0.7500, 0.0625, 0.1250, 0.0625, -0.0625, 0.0625, -0.2500, -0.1875, 0.1875, -0.2500, 0.3750, 0.3750, -0.8750, -0.3125, -0.1250, 0.1250, 0.0000, 0.1875, 0.1250, -0.0625, 0.0000, 0.0625, 0.0000, 0.1875, -0.3125, 0.0000, 0.0625, 0.0625, 0.1875, -0.1875, 0.0625, 0.4375, -0.8750, -0.1250, 0.3125, 0.2500, 0.0625, 0.1250, -0.4375, -0.0625, 0.1875, -0.3750, 0.8750, -0.3750, -0.5000, -0.6875, -0.7500, 0.3125, -0.1875, -0.0625, 0.1875, 0.0625, 0.1875, 0.1875, -0.1250, 0.1250, -0.2500, -0.1250, 0.2500, -0.3750, 0.1875, -0.4375, -0.0625, 0.3750, -0.6875, 0.1250, -0.0625, -0.3125, -0.0625, 0.2500, -0.0625, 0.3125, -0.1875, 0.0625, 0.4375, 0.1250, 0.1875, -0.8750, -0.3125, -0.2500, -0.5000, 0.1875, -0.1875, -1.0000, -0.5625, 0.1250, -0.0625, -0.0625, -0.4375, -0.1250, -0.1875, -0.3750, -0.1250, 0.2500, -0.1250, 0.2500, -0.1250, -0.4375, -0.2500, -0.1875, -0.1875, 0.1250, 0.2500, -0.1250, 0.1250, 0.0000, 0.3750, 0.0625, 0.1250, 0.0000, -0.3750, 0.0000, -0.6250, -0.1250, 0.2500, -1.0000, -0.3125, 0.4375, 0.0000, 0.0625, -0.1250, -0.0625, 0.3125, -0.4375, -0.1875, -0.1250, -0.0625, 0.0625, 0.0625, 0.3750, 0.0625, -0.0625, -0.3125, 0.2500, 0.1250, -0.4375, 0.5000, -0.3125, 0.5625, 0.0625, 0.0625, 0.3125, 0.0000, -0.3125, -0.4375, -0.1875, 0.0625, -1.0000, 0.2500, 0.6250, -0.0625, 0.0625, -0.1875, -0.1250, 0.0625, -0.0625, 0.0625, 0.0000, 0.0625, -0.1875, 0.0000, 0.1875, 0.2500, 0.1875, 0.0000, -0.1250, -0.1250, 0.0000, 0.4375, 0.0625, 0.1250, -0.3750, 0.2500, 0.3125, 0.0000, -0.5000, 0.0000, -0.3125, 0.0000, -0.8750, 0.5000, 0.2500, -0.2500, -0.1875, 0.1250, -0.2500, -0.3125, 0.0625, 0.1875, 0.1250, -0.0625, -0.3125, -0.0625, 0.0000, 0.2500, -0.0625, 0.1250, -0.3125, 0.2500, 0.3750, 0.0625, 0.2500, 0.1875, -0.0625, 0.0625, -0.0625, -0.1875, -0.3750, 0.4375, 0.2500, 0.0000, -0.3125, 0.0625, 0.1250, -0.1250, -0.8125, 0.1250, -0.0625, 0.1875, -0.1875, 0.1250, 0.3125, -0.1250, -0.0625, -0.2500, -0.0625, -0.0625, -0.0625, 0.0625, 0.2500, 0.4375, 0.0625, -0.1875, 0.2500, 0.3750, 0.5000, -0.0625, 0.0625, -0.1875, -0.3750, 0.0625, 0.3750, -0.1875, -0.1875, 0.1250, 0.1250, -0.0625, -0.1250, -0.2500, 0.0000, 0.1250, 0.0625, 0.2500, 0.3125, 0.0625, 0.0000, -0.2500, 0.1250, -0.5625, -0.2500, 0.0625, 0.2500, 0.1250, -0.3750, -0.1250, 0.6250, 0.3750, 0.6875, -0.1250, -0.1250, 0.0000, -0.5625, -0.2500, 0.3750, -0.0625, 0.0000, -0.4375, 0.3125, 0.1250, 0.0625, -0.4375, 0.3125, 0.4375, -0.0625, 0.0000, 0.0000, 0.0625, 0.3125, -0.0625, 0.1875, -0.8125, -0.1250, 0.1875, 0.1250, -0.1250, -0.1250, -0.4375, 0.3125, -0.1250, -0.5625, -1.0000, -0.3750, -0.0625, -0.7500, -0.1250, -0.3750, 0.1875, -0.1250, -0.1875, -0.1875, -0.0625, 0.3125, -0.5000, 0.3125, 0.5625, -0.6875, -0.4375, -0.3125, 0.1875, 0.3750, -0.0625, 0.2500, -0.0625, -0.1250, -0.5625, -0.1875, 0.7500, 0.0000, 0.3750, 0.0000, 0.2500, 0.1875, -0.1875, -0.4375, 0.3125, -0.6875, -0.2500, -0.0625, 0.0625, -0.0625, -0.1250, 0.0625, 0.1875, 0.1250, 0.1875, 0.6250, -0.5625, 0.0625, -0.0625, -0.0625, 0.0000, -0.1250, 0.0000, -0.0625, -0.1875, -0.3125, 0.0625, 0.0000, 0.2500, 0.0000, 0.3750, -0.4375, -0.2500, 0.0625, -0.0625, -0.0625, 0.1875, -0.5625, -0.3125, -0.0625, -0.1875, -0.0625, -0.5625, 0.2500, 0.1250, 0.1250, 0.4375, 0.3125, -0.2500, -0.0625, -0.3125, -0.6250, 0.0000, -0.1250, 0.0000, 0.3750, -0.0625, 0.1250, 0.1250, -0.1250, 0.0625, -0.1250, 0.3125, -0.1250, -0.0625, -0.1875, 0.0000, 0.1875, 0.3125, -0.3750, -0.1250, 0.0000, 0.0000, 0.1875, 0.0625, 0.3750, 0.1875, 0.2500, 0.0000, -0.1250, -0.1250, -0.1250, -0.3750, -0.3125, -0.0625, -0.1875, -0.1250, 0.0000, 0.0000, 0.2500, 0.0625, -1.0000, -0.1250, 0.0000, 0.1250, 0.1875, 0.0000, 0.0625, 0.1875, 0.0000, 0.1875, -0.3125, 0.3125, -0.1875, 0.1250, 0.1875, 0.0625, 0.1250, -0.0625, -0.4375, 0.0625, -0.3125, 0.2500, -0.0625, -0.1250, -0.1250, 0.0000, -0.1875, -0.1250, -0.0625, -0.1250, -0.0625, 0.2500, -0.1250, 0.1250, 0.0000, 0.1250, 0.0625, 0.0625, 0.2500, 0.1250, 0.0000, -0.0625, 0.0000, 0.3750, 0.1250, -0.1875, 0.0625, 0.0000, 0.1875, -0.0625, -1.0000, 0.0000, -0.1250, 0.2500, -0.1875, -0.0625, 0.0625, 0.1250, -0.1250, -0.3750, -0.2500, -0.2500, -0.1875, 0.1250, -0.0625, 0.4375, -0.3125, 0.2500, -0.1875, 0.2500, 0.1875, -0.1250, 0.0625, -0.0625, -0.0625, -0.1875, 0.1875, -0.1875, -0.0625, 0.0000, 0.1250, 0.0000, -0.0625, 0.0000, -0.0625, -0.1250, 0.0625, 0.0625, 0.0625, 0.0625, 0.1875, -0.1875, 0.1250, -0.1250, -0.5625, 0.1250, -0.1250, 0.3750, -0.7500, 0.1250, 0.8125, 0.0625, -0.1875, -0.2500, 0.1875, -0.0625, 0.0625, 0.0000, -0.0625, -0.4375, -0.1250, -0.0625, 0.0625, 0.0000, 0.0000, 0.3125, 0.0000, -0.4375, 0.0000, -0.0625, 0.1250, 0.0625, 0.1875, 0.1875, 0.1875, -0.1875, -0.2500, 0.0625, -0.1875, 0.0000, -0.5000, 0.1875, 0.5000, 0.0000, -0.6250, -0.2500, -0.5000, 0.1250, 0.0000, 0.0625, -0.2500, -0.2500, -0.1875, -0.3750, -0.1250, -0.3125, 0.1250, 0.1250, -0.2500, -0.8750, -0.4375, -0.1875, 0.1250, 0.0000, -0.0625, 0.3125, 0.0000, 0.1875, 0.2500, -0.1250, 0.0625, 0.2500, 0.3125, 0.2500, 0.6250, -0.2500, 0.0625, -0.0625, -0.2500, 0.0625, -0.3750, 0.2500, -0.5625, 0.1875, 0.2500, 0.0000, -0.6250, -0.4375, 0.1875, 0.5000, 0.0000, -0.4375, 0.1875, 0.1250, 0.2500, -0.0625, 0.0625, -0.0625, -0.1875, -0.2500, 0.2500, 0.1875, -0.1875, 0.4375, -0.0625, 0.3750, 0.3125, 0.0000, -0.1875, -0.0625, -0.0625, 0.0625, -0.1875, 0.0000, -0.5000, 0.3750, 0.1875, 0.0000, 0.0625, -0.0625, 0.0625, 0.0000, 0.9375, -0.0625, -0.1875, -0.2500, -0.2500, -0.0625, 0.0000, -0.0625, -0.1250, -0.4375, 0.4375, 0.0000, -0.8125, 0.0000, 0.0000, 0.0000, 0.3750, 0.0000, -0.1875, 0.0625, 0.0000, 0.5000, -0.1250, 0.1250, -0.1875, 0.1250, 0.0625, -0.0625, 0.3125, 0.0625, 0.2500, -0.3125, 0.9375, 0.0000, -0.3750, -0.1250, -0.8750, -0.1875, -0.0625, -0.1875, 0.0625, -0.3125, 0.1250, 0.1250, -0.3125, -0.0625, 0.3125, -0.0625, 0.1875, -0.1250, 0.1250, 0.0625, -0.0625, 0.3125, 0.1875, 0.3125, -0.1250, -0.3125, 0.2500, -0.0625, 0.2500, 0.0625, -0.6250, -0.2500, 0.8125, 0.0625, -0.3750, -0.0625, -1.0000, 0.1875, -0.2500, -0.1250, 0.0625, 0.0625, 0.1250, 0.1875, 0.0000, 0.3125, -0.1250, 0.2500, 0.0000, -0.3750, 0.3750, 0.0625, 0.1250, 0.1875, 0.1875, 0.0000, 0.1250, 0.1875, 0.0000, 0.0000, 0.1250, -0.1250, -0.1875, 0.1875, 0.3750, 0.0625, -0.2500, -0.0625, 0.0000, 0.0625, 0.0000, -0.3125, -0.2500, 0.1250, -0.1250, 0.1875, 0.0000, 0.1875, -0.6250, 0.0625, -0.0625, -0.0625, -0.0625, 0.0000, -0.1250, 0.3750, -0.1875, -0.0625, 0.0000, 0.0625, -0.1875, 0.0000, -0.0625, 0.0625, 0.0625, 0.0625, 0.1250, -0.2500, 0.2500, 0.0625, 0.0625, 0.1250, 0.3125, 0.1250, 0.1875, 0.2500, -0.1875, 0.1250, 0.0000, 0.0000, -0.2500, -0.0625, 0.2500, -0.0625, -0.2500, 0.0000, -0.0625, 0.3750, -0.1250, 0.0625, -0.2500, -0.5625, -0.1875, -0.0625, -0.2500, -0.0625, 0.0625, 0.0000, 0.0000, -0.1250, 0.2500, 0.0000, 0.0625, 0.0000, 0.1250, 0.1250, -0.0625, 0.6250, 0.1250, 0.0625, -0.5000, -0.0625, 0.0000, -0.1250, 0.6875, 0.0000, 0.0000, 0.0625, -0.5000, 0.3750, 0.1250, 0.3125, -0.7500, -0.6250, 0.0000, -0.5000, -0.2500, -0.1875, 0.0625, -0.0625, 0.0625, -1.0000, 0.1875, 0.0625, 0.0625, 0.0000, 0.1250, 0.0625, -0.0625, 0.0000, 0.1875, -0.2500, -0.3750, -0.6875, 0.0625, -0.4375, 0.5000, -0.0625, 0.0625, 0.1250, -0.1875, 0.0625, -0.1250, 0.1875, -0.4375, -0.3125, 0.0625, 0.0000, -0.4375, -0.5000, 0.0000, 0.5625, -0.2500, -0.3750, 0.5625, 0.0625, 0.2500, -0.1875, 0.1250, 0.1875, -0.1250, 0.3125, 0.4375, -0.0625, -1.0000, -0.9375, -0.0625, -0.5625, 0.1875, -0.0625, 0.1875, 0.0625, -0.3750, 0.0000, 0.1250, 0.0625, -0.3750, -0.0625, 0.0625, 0.0000, -0.1875, -0.8750, 0.0000, 0.1875, -0.3125, -0.1875, 0.3125, 0.0000, 0.2500, -0.2500, 0.0625, 0.0625, -0.3125, 0.0625, 0.3750, 0.0625, 0.0000, -0.5625, -0.0625, -0.6250, 0.1250, -0.0625, -0.0625, 0.1875, -0.3750, 0.1875, 0.2500, -0.0625, -0.4375, 0.1250, 0.0625, 0.0625, -0.1875, -1.0000, 0.1250, 0.1250, -0.2500, -0.0625, -0.1875, 0.0000, 0.1250, -0.3125, 0.0625, 0.1250, -0.0625, 0.0000, -0.0625, 0.1875, 0.0625, -0.4375, -0.1250, -0.2500, 0.0625, 0.0625, -0.2500, -0.0625, 0.1250, 0.1875, 0.3125, -0.2500, -0.1875, -0.1250, 0.1250, 0.0000, -0.1250, -1.0000, -0.0625, -0.0625, -0.2500, 0.0625, 0.0625, -0.0625, -0.2500, 0.1250, 0.1250, 0.1250, 0.1875, 0.0000, -0.2500, 0.0000, 0.0625, -0.5000, -0.3750, -0.1875, 0.1250, -0.0625, -0.0625, 0.0000, 0.2500, -0.0625, 0.1250, -0.1250, -0.1875, 0.0625, -0.0625, 0.0625, -0.3125, 0.0625, 0.0000, 0.3125, 0.1875, -0.0625, 0.3125, 0.0625, 0.2500, 0.1250, 0.0625, 0.1875, 0.1875, 0.0625, -0.1875, -0.1250, 0.0625, -0.6875, -0.7500, -0.1250, 0.1250, -0.0625, -0.1250, 0.1250, -0.0625, 0.0625, -0.1875, -0.0625, -0.5625, 0.0625, -0.0625, 0.0625, -0.3750, 0.1250, -0.1250, 0.3125, 0.3125, -0.1250, 0.1250, 0.0000, 0.1250, 0.0000, 0.1250, 0.1250, 0.2500, 0.3750, 0.0625, -0.1875, 0.0625, -0.6875, -0.5000, -0.1250, 0.5000, -0.0625, -0.1875, 0.3750, -0.1250, 0.3125, -0.1875, 0.0625, -0.5000, -1.0000, 0.0625, -0.3750, -0.3750, 0.1875, -0.0625, -0.0625, 0.0000, -0.6875, 0.2500, -0.1250, -0.0625, 0.0000, 0.0000, -0.0625, -0.1250, 0.6250, 0.4375, 0.3125, 0.3750, -0.8125, 0.1875, -0.3125, 0.6250, 0.0000, 0.0000, 0.6250, -0.3750, 0.1875, 0.0625, 0.0000, -0.7500, -0.6250, 0.2500, -0.4375, -0.1875, 0.2500, 0.0000, -0.3125, 0.0625, -0.7500, 0.5625, -0.3125, -0.2500, 0.0625, 0.0625, 0.1250, -0.1875, 0.3750, 0.0000, -0.4375, 0.0000, -0.6875, 0.1250, -0.1250, 0.0000, -0.0625, -0.1875, 0.2500, -0.3125, 0.0625, -0.0625, 0.1250, 0.1250, -0.3750, 0.0625, -0.3125, -0.6875, 0.3750, -0.1250, 0.3125, -0.3750, -0.6250, 0.1250, 0.0000, 0.1250, 0.0000, 0.0625, 0.0000, 0.0000, 0.0000, 0.1250, -0.0625, -0.0625, 0.0625, 0.0625, -0.1250, 0.1250, 0.1250, 0.1250, 0.0000, -0.9375, -0.1250, 0.0000, 0.0625, 0.1875, -0.0625, 0.1250, -0.1250, -0.2500, 0.3125, 0.0625, 0.2500, -0.1250, -0.0625, 0.3750, 0.0000, -0.1875, -0.1250, 0.0000, -0.1250, -0.1250, 0.1250, 0.1875, -0.2500, 0.0625, -0.1250, 0.0000, 0.0625, -0.0625, 0.1250, 0.1250, 0.0000, -0.0625, 0.0000, -0.0625, 0.0000, -0.2500, 0.0000, -0.0625, 0.0000, -0.1875, 0.0625, 0.0625, 0.1875, -0.2500, 0.1875, 0.4375, -0.2500, -0.1250, -0.3750, 0.3125, 0.0625, -0.1875, 0.0625, 0.3125, -0.6875, 0.0625, -0.4375, 0.0000, -0.0625, 0.3750, 0.0625, -0.0625, -0.1250, 0.0000, -0.0625, 0.0625, 0.1250, -0.1875, 0.0000, -0.0625, 0.2500, 0.1250, 0.1875, 0.0625, -0.0625, -0.1250, 0.1875, 0.3125, 0.0625, -0.1875, -0.1250, 0.4375, 0.1875, -0.0625, 0.0625, 0.2500, -0.6250, 0.1250, -0.4375, 0.0625, 0.0000, 0.3125, -0.0625, -0.1250, -0.0625, 0.0000, -0.1250, -0.0625, 0.1250, -0.1875, -0.1250, -0.0625, 0.1875, 0.1250, 0.1250, 0.0000, -0.0625, 0.1250, 0.0625, 0.1875, 0.1250, 0.1875, 0.0000, 0.2500, 0.1875, -0.0625, 0.0000, 0.0625, -1.0000, 0.0625, -0.3125, -0.1875, -0.0625, -0.0625, -0.1250, -0.0625, 0.0000, 0.2500, -0.3125, -0.1875, 0.1250, -0.7500, -0.3125, 0.0000, -0.1875, -0.0625, 0.0000, -0.1250, 0.0000, 0.2500, 0.0000, 0.0000, 0.0625, 0.2500, -0.0625, 0.0625, 0.1250, -0.1875, 0.0000, 0.0625, -0.7500, -0.2500, 0.0000, -0.5000, -0.0625, 0.3750, -0.3125, 0.1875, 0.2500, -0.1875, -0.0625, -0.2500, -0.1250, -0.3750, -0.5625, 0.0625, -0.3750, 0.2500, 0.0000, -0.4375, 0.0000, 0.3125, -1.0000, 0.0625, -0.0625, -0.1875, 0.0000, -0.2500, 0.0625, -0.1250, 0.2500, 0.0000, -0.3125, -0.1875, 0.3125, 0.0000, -0.1250, 0.0625, 0.3125, 0.8750, 0.3125, -0.3125, -0.5000, -0.4375, -0.4375, -0.0625, -0.1875, 0.1875, -0.1875, 0.0000, 0.1250, -0.4375, -0.1250, 0.0625, -0.4375, 0.2500, 0.1250, -0.8750, 0.0000, 0.1250, 0.1250, 0.2500};
#endif

#endif