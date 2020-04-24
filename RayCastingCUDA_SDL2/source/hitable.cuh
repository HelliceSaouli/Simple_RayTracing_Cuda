/*
 * hitable.cuh
 *
 *  Created on: 29 Mar 2020
 *      Author: asaouli
 */

#ifndef HITABLE_CUH_
#define HITABLE_CUH_

#include "ray.cuh"

class material;

struct hits
{
	float depth;
	glm::vec4 position;
	glm::vec4 normal;
	material *mat_ptr;
};

class hitable
{
public:
	// pure virtual function for array object interaction
	__device__ virtual bool hit(const ray& aray, float tmin, float tmax, hits& records)const=0;

};


#endif /* HITABLE_CUH_ */
