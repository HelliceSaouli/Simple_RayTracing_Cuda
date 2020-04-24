/*
 * PixelBuffer.cpp
 *
 *  Created on: Aug 5, 2015
 *      Author: hakosoft
 */

#include "PixelBuffer.h"

extern "C"
void CallKernel(uint *d_output, hitable** d_world, curandState * state,int imageW, int imageH,dim3 Blocks,dim3 Threads );

extern "C"
void Callsetupkernel(curandState * state,int imageW, int imageH,dim3 Blocks,dim3 Threads );

PixelBuffer::PixelBuffer(int w,int h)
{
	this->w      = w;
	this->h      = h;
	this->_image = 0;
	this->_pbo   = 0;

}

PixelBuffer::~PixelBuffer()
{
	// TODO Auto-generated destructor stub
}

void PixelBuffer::clear()
{
	cudaGraphicsUnregisterResource(this->cuda_pbo_rsc);
	glDeleteBuffers(1,&this->_pbo);
	glDeleteTextures(1,&this->_image);
	cudaDeviceReset();
}

void PixelBuffer::creatPBO()
{
	if(this->_pbo != 0)
	{
		this->clear();
	}

	glGenBuffers(1,&this->_pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER,this->_pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, this->w * this->h* sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW); // RGBA
}

void PixelBuffer::registerPBOToCUDA()
{
	cudaGraphicsGLRegisterBuffer(&this->cuda_pbo_rsc, this->_pbo, cudaGraphicsMapFlagsWriteDiscard);
}

void PixelBuffer::creatTexture()
{
	glGenTextures(1,&this->_image);
	glBindTexture(GL_TEXTURE_2D,this->_image);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,this->w,this->h,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
    glBindTexture(GL_TEXTURE_2D,0);
}

void PixelBuffer::fromPBOToTexture()
{
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, this->_pbo);
    glBindTexture(GL_TEXTURE_2D, this->_image);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->w, this->h, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void PixelBuffer::bindTargetTex()
{
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D,this->_image);
}

void PixelBuffer::setrandomdevice()
{
	size_t size = this->w * this->h * sizeof ( curandState );
	cudaMalloc(( void **) & devStates , size);
    dim3 blockSize(16,16);
    dim3 gridSize(iDivUp(this->w, blockSize.x), iDivUp(this->h, blockSize.y));
	Callsetupkernel(this->devStates,this->w,this->h,gridSize,blockSize);

}
void PixelBuffer::render(hitable** d_world)
{

	// do some matrix calculation here next time
    uint *d_output;
    cudaGraphicsMapResources(1, &this->cuda_pbo_rsc, 0);
    size_t bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&d_output, &bytes,this->cuda_pbo_rsc);
    cudaMemset(d_output, 0,this->w* this->h*4);
    // call CUDA kernel, writing results to PBO
    dim3 blockSize(8,8);
    dim3 gridSize(iDivUp(this->w, blockSize.x), iDivUp(this->h, blockSize.y));
    CallKernel(d_output,d_world,this->devStates,this->w,this->h,gridSize,blockSize);
    cudaGraphicsUnmapResources(1, &this->cuda_pbo_rsc, 0);
}

