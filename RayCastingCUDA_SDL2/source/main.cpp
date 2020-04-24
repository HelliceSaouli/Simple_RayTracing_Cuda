#include "HeaderCpp.h"

#include <cuda_runtime.h>

#include "Display.h"
#include "Shader.h"
#include "Mesh.h"	// i don't think i need this now
#include "Transform.h"
#include "Camera.h"
#include "TextureGL.h"
#include "Fps.h"
#include "PixelBuffer.h"
#include "input.h"
#include "sphere.cuh"
#include "world.cuh"

#include "glm/glm.hpp"


using namespace std;

extern "C"
void copydata(float* viewmatrix, float* invProjectMatrix, size_t sizeofMatrix, float* camerapos, size_t sizeofcamerapos);

extern "C"
void SetupWorld(hitable** d_objects, hitable** d_world, curandState * state);

extern "C"
void CleanWorld(hitable** d_objects, hitable** d_world);

extern "C"
void randInitCall(curandState *rand_state);


/****************************************************************************************/
int main()
{

	Vertex quard[] = { Vertex(glm::vec3(-1.0,-1.0,0.0),glm::vec3(0.0,0.0,0.0),glm::vec2(0.0,0.0)),
					   Vertex(glm::vec3( 1.0,-1.0,0.0),glm::vec3(0.0,0.0,0.0),glm::vec2(1.0,0.0)),
			           Vertex(glm::vec3(-1.0,1.0,0.0),glm::vec3(0.0,0.0,0.0), glm::vec2(0.0,1.0)),
			           Vertex(glm::vec3(-1.0,1.0,0.0), glm::vec3(0.0,0.0,0.0),glm::vec2(0.0,1.0)),
			           Vertex(glm::vec3( 1.0,-1.0,0.0),glm::vec3(0.0,0.0,0.0),glm::vec2(1.0,0.0)),
			           Vertex(glm::vec3( 1.0,1.0, 0.0),glm::vec3(0.0,0.0,0.0),glm::vec2(1.0,1.0)),
	};


	Fps _FramePerSc;
	input in;
	Display _display(600,400,"Simple Ray Tracing GPU - FPS :");
	Shader _RenderToTex("shaders/RenderToTex");

	GLint attribut[3];
	attribut[0] = _RenderToTex.GetAttributeLocation("position");
	attribut[1] = _RenderToTex.GetAttributeLocation("color");
	attribut[2] = _RenderToTex.GetAttributeLocation("texCoord");

	Mesh _drawingquard(quard,sizeof(quard) / sizeof(quard[0]),attribut); // this is where am going to put texture on




	PixelBuffer _PBO(600,400);

	/********************** set up  camera ***************************/
	 Camera   cam(glm::vec3(0.0,10.0,50.0),glm::vec3(0.0,0.0,0.0),120.0f,(float)600/(float)400
			 	  ,0.0001f,10000.0f);

	 /**************************************** set up movement ***********************/
		float mouseSpeed = 5.5;
		float walkspeed  = 1.8;

	/*************************** set up render ****************************************/
	_PBO.creatPBO();
	_PBO.registerPBOToCUDA();
	_PBO.creatTexture();
	_FramePerSc.Start(); // start counting fps
	_PBO.setrandomdevice();

	/******************* main render loop *****************************************/

	size_t sizeofMatrix = sizeof(glm::mat4);
	size_t sizeofcampos = sizeof(glm::vec4);
	copydata(cam.getViewMatrix(),cam.getInverseProjMatrix(),sizeofMatrix,
			 cam.getPosition(),sizeofcampos);


	/************************ set up the world *****************/
	/* 		 this are going to be initialized inside the GPU    *
	 * 			no need to CUDA copy from CPU to GPU		   */
	/***********************************************************/


	hitable** d_objects;
	int num_hitables = 4*4 +1 + 2;
	cudaMalloc((void**)&d_objects, num_hitables * sizeof(hitable*));

	hitable** d_world;
    curandState *d_rand_state_simple;
    cudaMalloc((void **)&d_rand_state_simple, 1*sizeof(curandState));
	cudaMalloc((void**)&d_world,  sizeof(hitable*));
	randInitCall(d_rand_state_simple);
	SetupWorld(d_objects,d_world, d_rand_state_simple);

	cudaDeviceSynchronize();

	/********************** render ************************/
	while(!in.QUIT())
	{

		/************************************** consume input first ****************/

		_display.Clear(0.0f,0.0,0.0f,0.0f);
		in.updateEvents();


		if(in.getKey(SDL_SCANCODE_UP) == KEY_DOWN)
		{

			std::cout<<" moving up"<<std::endl;
			cam.move(walkspeed * _FramePerSc.deltatime(),MOVE_FORWARD);
			cam.ViewMatrixUpdate();
		}

		if(in.getKey(SDL_SCANCODE_DOWN) == KEY_DOWN)
		{
			std::cout<<" moving down"<<std::endl;
			cam.move(walkspeed * _FramePerSc.deltatime(),MOVE_BACKWARD);
			cam.ViewMatrixUpdate();

		}

		if(in.getKey(SDL_SCANCODE_LEFT) == KEY_DOWN)
		{
			std::cout<<" moving lef"<<std::endl;
			cam.move(walkspeed * _FramePerSc.deltatime(),MOVE_LEFT);
			cam.ViewMatrixUpdate();
		}

		if(in.getKey(SDL_SCANCODE_RIGHT) == KEY_DOWN)
		{
			std::cout<<" moving right"<<std::endl;
			cam.move(walkspeed * _FramePerSc.deltatime(),MOVE_RIGHT);
			cam.ViewMatrixUpdate();
		}

/*********************************************************************************************************/
		if(in.getKey(SDL_SCANCODE_W) == KEY_DOWN)
		{

			cam.rotateX(mouseSpeed * _FramePerSc.deltatime());
			cam.ViewMatrixUpdate();
		}

		if(in.getKey(SDL_SCANCODE_S) == KEY_DOWN)
		{

			cam.rotateX(-mouseSpeed * _FramePerSc.deltatime());
			cam.ViewMatrixUpdate();

		}

		if(in.getKey(SDL_SCANCODE_A) == KEY_DOWN)
		{

			cam.rotateY(-mouseSpeed * _FramePerSc.deltatime());
			cam.ViewMatrixUpdate();
		}

		if(in.getKey(SDL_SCANCODE_D) == KEY_DOWN)
		{

			cam.rotateY(mouseSpeed * _FramePerSc.deltatime());
			cam.ViewMatrixUpdate();
		}
/********************************************************************************************************/
		if(in.getKey(SDL_SCANCODE_C) == KEY_DOWN)
		{
			 in.exit();
		}

/***************** send the inverse updated camera matrix view to GPU *************/

		// too much my be i should optimize it  ?????

		copydata(cam.getViewMatrix(),cam.getInverseProjMatrix(),sizeofMatrix,
				 cam.getPosition(),sizeofcampos);

			_PBO.render(d_world);
			_PBO.fromPBOToTexture();
				_RenderToTex.bind();
					 _PBO.bindTargetTex();
					 _display.Clear(0.0f,0.0,0.0f,0.0f);
					_drawingquard.DrawQuad();


			_display.Update();
			//frame counter
			_FramePerSc.Count();// count
			char* temp = new char[128];
			sprintf(temp,"Simple Ray Tracing GPU - FPS : %d",(int)_FramePerSc.getFPS());
			SDL_SetWindowTitle(_display.getWindow(),temp);
	}


	/******************** clean GPU ******************/
	_RenderToTex.deleteShader();
	CleanWorld(d_objects,d_world);
	cudaFree(d_objects);
	cudaFree(d_world);

	_PBO.clear();
	return 0;
}
