#include <stdio.h>
#include <cuda.h>

#include "ray.cuh"
#include "sphere.cuh"
#include "world.cuh"
#include "material.cuh"
#include "sampler.cuh"

__constant__ glm::mat4 d_invProjectMatrix;
__constant__ glm::mat4 d_viewMatrix;
__constant__ glm::vec4 d_camerapos;




__device__ glm::vec4 shade(const ray& aray, hitable** world, curandState *state)
{
	ray current_ray = aray;
	glm::vec4 current_attenuation = glm::vec4(1.0);

	for(unsigned int depth = 0; depth < 10; depth++)	// number of scattering and stuff
	{
		hits courrent;
		if ((*world)->hit(current_ray,0.0001f,FLT_MAX,courrent))
		{
            ray scattered;
            glm::vec4 attenuation;

            if(courrent.mat_ptr->scatter(current_ray,courrent,attenuation,scattered,state))
            {
            	current_attenuation *= attenuation;
            	current_ray = scattered;
            }
            else
            	return glm::vec4(0.0);
		}
		else
		{
			//glm::normalize(current_ray.direction);
			float t = 0.5f * (current_ray.direction.y + 1.0);
			glm::vec4 color = (1.0f - t) * glm::vec4(1.0,1.0,1.0,1.0)  +   t * glm::vec4(0.5,0.7,1.0,1.0);
			return color * current_attenuation;
		}
	}

	return glm::vec4(0.0);
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__global__ void setup_kernel ( curandState * state,int imageW, int imageH )
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((x >= imageW) || (y >= imageH))
    	return;

    int indx = y*imageW + x;
	curand_init(x , indx , 0, &state[indx]) ;
}


__global__ void Kernel1(uint *d_output,hitable** world,curandState * state, int imageW, int imageH)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    int indx =  y * imageW + x;
    float4 rgba ;
    if ((x >= imageW) || (y >= imageH)) return;

    curandState local_rand_state = state[indx];
    glm::vec4 color(0.0);

    for(unsigned int s = 0; s < 5 ; s++)
    {
		float u =  1.0 - 2.0  * (float(x) + curand_uniform(&local_rand_state))  / float(imageW);
		float v =  1.0 - 2.0  * (float(y) + curand_uniform(&local_rand_state))  / float(imageH);

		glm::vec4 origin =  d_camerapos;
		glm::vec4 dir	 =   d_invProjectMatrix * glm::vec4(u,v,1.0,1.0);
		dir =  d_viewMatrix * dir;

		ray aray(origin,dir);
		color += shade(aray,world,&local_rand_state);
    }

    //state[indx] = local_rand_state;
    color /= 5.0f;
    rgba = make_float4(color.r,color.g,color.b,1.0);
    d_output[indx]  = rgbaFloatToInt(rgba);
}

__global__ void setupWorld(hitable** d_objects, hitable** d_world, curandState * state)
{
	// only the first thread will do this
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		curandState local_rand_state = *state;
		d_objects[0] = new sphere(glm::vec4(0.0,-100.5,0.0,1.0),100.5,
								new lambertian(glm::vec4(0.5,0.5,0.5,1.0)));

        int i = 1;
        for(int a = -2; a < 2; a++)
        {
            for(int b = -2; b < 2; b++)
            {
            	float choose_mat = curand_uniform(&local_rand_state);
            	glm::vec4 origin((a + curand_uniform(&local_rand_state)) * 15.0,
            					 0.65,
            					 (b + curand_uniform(&local_rand_state)) * 15.0,
            					 1.0);

                if(choose_mat < 0.8f) {
                	d_objects[i++] = new sphere(origin, 1.5,
                                             new lambertian(glm::vec4 (curand_uniform(&local_rand_state)
                                            		 , curand_uniform(&local_rand_state)
                                            		 , curand_uniform(&local_rand_state),
                                            		 1.0)));
                }
                else if(choose_mat < 0.95f) {
                	d_objects[i++] = new sphere(origin, 1.5,
                                             new  metal(glm::vec4(curand_uniform(&local_rand_state),
                                            		 	 0.6, curand_uniform(&local_rand_state)
                                            		 	 ,1.0),
                                            		 	 curand_uniform(&local_rand_state)));
                }
                else {
                	d_objects[i++] = new sphere(origin, 1.5, new dielectric(1.3));
                }
            }
        }

		d_objects[i++] = new sphere(glm::vec4(-4.0,1.0,22.0,1.0), 2.0,  new dielectric(0.5));
		d_objects[i++] = new sphere(glm::vec4(4.0, 1.0, 10.0,1.0),  2.0, new metal(glm::vec4(0.7, 0.6, 0.5,1.0), 0.0));


		*state = local_rand_state;
		*d_world = new world(d_objects,4*4 + 1 + 2);
	}
}

__global__ void rand_init(curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1080, 0, 0, rand_state);
    }
}


__global__ void cleanWorld(hitable** d_objects, hitable** d_world)
{
    for(int i=0; i < 4*4+1+2; i++) {
        delete ((sphere *)d_objects[i])->mat_ptr;
        delete d_objects[i];
    }
	delete *d_world;
}


extern "C"
void CallKernel(uint *d_output, hitable** d_world, curandState * state,int imageW, int imageH,dim3 Blocks,dim3 Threads )
{
	cudaError_t err ;
	Kernel1<<<Blocks,Threads>>>(d_output,d_world, state,imageW, imageH);
	err = cudaGetLastError();
	if ( cudaSuccess != err)
		fprintf(stderr, "Failed (error code %s)!\n", cudaGetErrorString(err));
}

extern"C"
void SetupWorld(hitable** d_objects, hitable** d_world, curandState * state)
{
	cudaError_t err ;
	setupWorld<<<1,1>>>(d_objects,d_world, state);
	err = cudaGetLastError();
	if ( cudaSuccess != err)
		fprintf(stderr, "Failed (error code %s)!\n", cudaGetErrorString(err));
}

extern"C"
void CleanWorld(hitable** d_objects, hitable** d_world)
{
	cudaError_t err ;
	cleanWorld<<<1,1>>>(d_objects,d_world);
	err = cudaGetLastError();
	if ( cudaSuccess != err)
		fprintf(stderr, "Failed (error code %s)!\n", cudaGetErrorString(err));
}

extern "C"
void Callsetupkernel(curandState * state,int imageW, int imageH,dim3 Blocks,dim3 Threads )
{
	cudaError_t err ;
	setup_kernel<<<Blocks,Threads>>>(state,imageW, imageH);
	err = cudaGetLastError();
	if ( cudaSuccess != err)
		fprintf(stderr, "Failed setup(error code %s)!\n", cudaGetErrorString(err));
}

extern"C"
void randInitCall(curandState *rand_state)
{
	cudaError_t err ;
	rand_init<<<1,1>>>(rand_state);
	err = cudaGetLastError();
	if ( cudaSuccess != err)
		fprintf(stderr, "Failed (error code %s)!\n", cudaGetErrorString(err));
}

extern "C"
void copydata(float* viewmatrix, float* invProjectMatrix, size_t sizeofMatrix, float* camerapos, size_t sizeofcamerapos)
{
	cudaMemcpyToSymbol(d_viewMatrix, viewmatrix, sizeofMatrix);
	cudaMemcpyToSymbol(d_invProjectMatrix, invProjectMatrix, sizeofMatrix);
	cudaMemcpyToSymbol(d_camerapos, camerapos, sizeofcamerapos);
}
