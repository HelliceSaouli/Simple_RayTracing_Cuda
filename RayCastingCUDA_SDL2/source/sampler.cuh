/*
 * sampler.cuh
 *
 *  Created on: 30 Mar 2020
 *      Author: asaouli
 */

#ifndef SAMPLER_CUH_
#define SAMPLER_CUH_

#include <curand_kernel.h>
#include "glm/glm.hpp"

__device__ glm::vec4 random_in_unit_sphere(curandState *state)
{

	float theta = 2.0f *  M_PI * curand_uniform(state);
	float gama  = acos(1.0f - 2.0f * curand_uniform(state));

	return glm::vec4 (sin(gama) * cos(theta),
				 sin(theta) * sin(gama),
				 cos(gama),1.0);


/*
	glm::vec4 v;
	do
	{
		v = glm::vec4(curand_uniform(state),curand_uniform(state),curand_uniform(state),1.0)   - glm::vec4(1.0);
 	}while(glm::length(v) * glm::length(v) >= 1.0);

	return v;
*/
}


#endif /* SAMPLER_CUH_ */
