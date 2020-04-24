/*
 * ray.cuh
 *
 *  Created on: 27 Mar 2020
 *      Author: asaouli
 */

#ifndef RAY_CUH_
#define RAY_CUH_

#include <stdio.h>
#include <cuda.h>
#include "glm/glm.hpp"

class ray
{
public:
	glm::vec4 origin;
	glm::vec4 direction;

public:
	__device__ ray()
	{
		// Empty
	};

	__device__ ray(const glm::vec4& _origin, const glm::vec4& _dir)
	{
		origin = _origin;
		direction = glm::normalize(glm::vec4(_dir.x,_dir.y,_dir.z,0.0));
	};

	__device__ glm::vec4 point_at_depth(float t)const
	{
		 return origin + t * direction;
	}

};


#endif /* RAY_CUH_ */
