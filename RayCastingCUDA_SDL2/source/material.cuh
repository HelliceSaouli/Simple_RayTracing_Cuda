/*
 * material.h
 *
 *  Created on: 26 Mar 2020
 *      Author: asaouli
 */

#ifndef MATERIAL_H_
#define MATERIAL_H_

#include "hitable.cuh"
#include "sampler.cuh"


class material
{
public :
	__device__ virtual bool scatter(const ray& in_ray, const hits& record, glm::vec4& attunation, ray& scattered, curandState *state)const = 0;
};


class lambertian: public material
{
public:
	glm::vec4 albedo;
public:
	__device__ lambertian(const glm::vec4& _albedo):albedo(_albedo)
	{
		//Empty
	};
	__device__ virtual bool scatter(const ray& in_ray, const hits& record, glm::vec4& attunation, ray& scattered,curandState *state)const
	{
		glm::vec4 target = record.position + record.normal  + random_in_unit_sphere(state);
	    scattered = ray(record.position, target - record.position);
	    attunation = albedo;
		return true;
	}
};


__device__ glm::vec4 reflect(const glm::vec4& in_ray,const glm::vec4& normal)
{
	return in_ray -  2.0f * glm::dot(in_ray,normal) * normal;
}

class metal: public material
{
public:
	glm::vec4 albedo;
	float fuzzy;

public:
	__device__ metal(const glm::vec4& _albedo, const float& f):albedo(_albedo)
	{
		if(f < 1.0)
			fuzzy = f;
		else
			fuzzy = 1.0;
	};
	__device__ virtual bool scatter(const ray& in_ray, const hits& record, glm::vec4& attunation, ray& scattered, curandState *state)const
	{
		glm::vec4 normalized = glm::normalize(in_ray.direction);
		glm::vec4 reflected = reflect(normalized, record.normal) + (fuzzy * random_in_unit_sphere(state));
		scattered = ray(record.position, reflected);
		attunation = albedo;

		return (glm::dot(scattered.direction,record.normal) > 0.0f);
	};
};

__device__ bool refract(const glm::vec4& in_ray, const glm::vec4& normal, float ratio, glm::vec4& refracted)
{
	glm::vec4 v = glm::normalize(in_ray);
	float costheta = glm::dot(v,normal);

	float k = 1.0 - ratio * ratio *  (1.0- costheta * costheta);

	if (k > 0.0)
	{
		//refracted = (v * ratio) + (normal * (ratio * costheta - std::sqrt(k)));
		refracted =  (v - normal * costheta) * ratio -( normal * sqrt(k));
		return true;
	}
	return false;
}


__device__ float shilick(float cosin, float ref_index)
{
	float r = (1.0 - ref_index) / (1.0 + ref_index);
	r = r * r;
	return  r + (1.0 - r) * std::pow((1.0 - cosin), 5.0);
}

class dielectric : public material
{

public :
	float ref_index;

public:
	__device__ dielectric(float ri):ref_index(ri)
	{
		// EMPTY
	};

	__device__ virtual bool scatter(const ray& in_ray, const hits& record, glm::vec4& attunation, ray& scattered, curandState *state)const
	{
		glm::vec4 outward_normal;
		float ratio = 0.0;
		attunation = glm::vec4(1.0,1.0,1.0,1.0);
		glm::vec4 reflected = reflect(in_ray.direction, record.normal);
		float reflect_prob = 0.0;
		float cosine = 0.0;
		glm::vec4 refracted;

		if(glm::dot(in_ray.direction,record.normal) > 0.0f)
		{
			outward_normal = - record.normal;
			ratio = ref_index;
			cosine = ref_index * glm::dot(in_ray.direction,record.normal) / glm::length(in_ray.direction);
		}
		else
		{
			outward_normal = record.normal;
			ratio = 1.0/ ref_index;
			cosine = - glm::dot(in_ray.direction,record.normal) / glm::length(in_ray.direction);
		}

		if(refract(in_ray.direction,outward_normal,ratio,refracted))
		{
			reflect_prob = shilick(cosine,ref_index);

		}
		else
		{
			scattered = ray(record.position, reflected);
			reflect_prob = 1.0;

		}

		if(curand_uniform(state) < reflect_prob)
		{
			scattered = ray(record.position, reflected);

		}
		else
		{
			scattered = ray(record.position, refracted);
		}

		return true;
	}
};

#endif /* MATERIAL_H_ */
