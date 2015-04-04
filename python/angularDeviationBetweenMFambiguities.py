# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
# under the MIT license. See the license file LICENSE.

import numpy as np
from js.geometry.rotations import *

axes = np.c_[np.eye(3), -np.eye(3)].T
print axes

cosies = []
for axisA in axes:
  for axisB in axes:
    if axisA.dot(axisB) == 0:
      cosies.append(np.c_[axisA, axisB, np.cross(axisA,axisB)])
      print axisA, axisB, cosies[-1]
print len(cosies)
angs = np.zeros((len(cosies), len(cosies)))
for i in range(len(cosies)):
  for j in range(len(cosies)):
    angs[i,j] = Rot3(cosies[i]).dot(Rot3(cosies[j])).toQuat().toAxisAngle()[0]

print angs*180./np.pi
print np.unique(angs*180./np.pi)
