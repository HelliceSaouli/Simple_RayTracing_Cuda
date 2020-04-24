/*
 * sphere.cuh
 *
 *  Created on: 29 Mar 2020
 *      Author: asaouli
 */

#ifndef SPHERE_CUH_
#define SPHERE_CUH_

#include "world.cuh"

class sphere: public hitable
{
public :
	 glm::vec4 origin;
	 float raduis;
	 material *mat_ptr;
public:
	 __device__ sphere()
	 {
		 raduis = 0.0;
		 mat_ptr = nullptr;
	 };
	 __device__ sphere(const glm::vec4& org, float r,material *mat):origin(org),raduis(r),mat_ptr(mat)
	 {
	 };
	 __device__ virtual bool hit(const ray& aray, float tmin, float tmax, hits& records)const;
};

__device__ inline  bool sphere::hit(const ray& aray, float tmin, float tmax, hits& records)const
{
    glm::vec4 oc = aray.origin - origin;
    float a = glm::dot(aray.direction, aray.direction);
    float b = 2.0 * glm::dot(oc, aray.direction);
    float c = glm::dot(oc, oc) - raduis*raduis;
    float delta = b*b - 4.0f*a*c;

    if(delta > 0)
    {
    	float t = (-b - sqrt(delta)) / (2 * a);

    	if( t < tmax && t > tmin)
    	{
    		records.depth = t;
    		records.position = aray.point_at_depth(t);
    		records.normal =  (records.position - origin) / raduis;
    		records.mat_ptr = mat_ptr;
    		return true;
    	}
    	t = (-b + sqrt(delta)) / (2 * a);
    	if( t < tmax && t > tmin)
    	{
    		records.depth = t;
    		records.position = aray.point_at_depth(t);
    		records.normal =  (records.position - origin) / raduis ;
    		records.mat_ptr = mat_ptr;
    		return true;
    	}

    }
	return false;
}




#endif /* SPHERE_CUH_ */
