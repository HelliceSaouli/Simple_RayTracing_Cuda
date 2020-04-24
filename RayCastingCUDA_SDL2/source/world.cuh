/*
 * world.cuh
 *
 *  Created on: 29 Mar 2020
 *      Author: asaouli
 */

#ifndef WORLD_CUH_
#define WORLD_CUH_

#include "hitable.cuh"


class world: public hitable
{

public:
		unsigned int size;
		hitable** objects;

public:
	__device__ world()
	{
		size = 0;
		objects = nullptr;
	}

	__device__ world(hitable** object_list, unsigned int size_list)
	{
		size = size_list;
		objects = object_list;
	}


	__device__ virtual bool hit(const ray& aray, float tmin, float tmax, hits& records)const;
};


__device__ inline  bool world::hit(const ray& aray, float tmin, float tmax, hits& records)const
{
	bool anyhit = false;
	hits recod_temp;

	float closeHit = tmax;

	for(unsigned int i = 0; i < size; i++)
	{
		if(objects[i]->hit(aray,tmin,closeHit,recod_temp))
		{
			anyhit = true;
			closeHit = recod_temp.depth;
			records = recod_temp;
		}
	}
	return anyhit;
}
#endif /* WORLD_CUH_ */
