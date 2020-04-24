/*********************************************************************
 * Camera.h                                                         *
 *                                                                   *
 *  Created on: Aug 9, 2015                                          *
 *      Author: hakosoft saouli                                      *
 *      this a virtual camera 2.0 it will use  GLM data structure    *
 *********************************************************************/

#ifndef Camera_H_
#define Camera_H_

#include <cmath>
#include "glm/glm.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/transform.hpp"
#include <glm/gtc/type_ptr.hpp>

enum direction
{
		MOVE_FORWARD
       ,MOVE_BACKWARD
       ,MOVE_LEFT
       ,MOVE_RIGHT
};

class Camera
{
	private:
		glm::mat4 _v;	  // the view matrix
		glm::mat4 _p;    //  the projection matrix
		glm::mat4 _m;	//   the model matrix
		glm::mat4 _mvp;//    the model view projection matrix
		float *inverseViewMatrix; /*** remove this later it is useless ***/

		glm::vec3 _camPosition;		 /* the position of the camera in world space or the eye*/
		glm::vec3 _target;           /* the focus point where the camera is looking Camera - target = the vector direction*/
		glm::vec3 _up;				 /* the up vector for camera*/
		float fav;

		/***** extra stuff for gpu  fix this later  ****/
		glm::mat4 inv;
		glm::vec4 pos;
	public:
		float width;
		float hight;
		float near;		             /* the near plane of Frustrum*/
		float far;                   /* the Far plane of  Frustrum*/

		void buildMVP(); /* this will only multiplay the 3 matrices togther*/

	public:
		Camera(const glm::vec3&,const glm::vec3&, float ,float ,float,float); /* one constractor that do every thing*/
		virtual ~Camera();

		/****************************************************************
		 *                                                              *
		 *  this is bunch of methodes setter and getter and other stuff *
		 *                                                              *
		 ****************************************************************/
		void setPosition(const glm::vec3&);
		void setTarget(const glm::vec3&);
		void setUpvector(const glm::vec3&);

		float getFav()const;

		void setModelMatrix(const glm::mat4&); /* this to feed the model matrix to the system manual */

		/****************************************************************/
		float* getPosition();
		float* getViewMatrix();
		glm::mat4 getProjectionMatrix()const;
		glm::mat4 getModelViewProjectionMatrix()const;
		glm::mat4 getModelMatrix()const;
		glm::vec3 getTarget()const;
		float* getInverseProjMatrix();
		glm::vec3 getFront()const;
		glm::vec3 getUp()const;


		void move(const float speed,direction d);
		void rotateX(float angle);
		void rotateY(float angle);

		/****************************************************************/
		void ViewMatrixUpdate();
	//	void ProjectionMatrixUpdate();
		void update();              /*  this update function will recalculate everything if the camera changed position or where it looks at*/

	private:
		glm::vec3 get_left()const;
		glm::vec3 get_right()const;
};

/* ------------------------ Implementation ------------------------ */

inline
void Camera::setPosition(const glm::vec3& eye)
{
	this->_camPosition = eye;
}

inline
void Camera::setTarget(const glm::vec3& target)
{
	this->_target = target;
}

inline
void Camera::setUpvector(const glm::vec3& UP)
{
	this->_up = UP;
}

inline
float Camera::getFav()const
{
	return this->fav;
}

inline
void Camera::setModelMatrix(const glm::mat4& model)
{
	this->_m = model;
}

/**************************************************************************************/
inline
float* Camera::getPosition()
{
	pos = glm::vec4(_camPosition,0.0);
	return glm::value_ptr(pos);
}

inline
glm::vec3 Camera::getTarget()const
{
 return this->_target;
}

inline
float* Camera::getViewMatrix()
{
	return glm::value_ptr(this->_v);
}

inline
glm::mat4 Camera::getModelMatrix()const
{
	return this->_m;
}

inline
float* Camera::getInverseProjMatrix()
{
	inv =  glm::inverse(this->_p);

	return glm::value_ptr(inv);
}

inline
glm::mat4 Camera::getProjectionMatrix()const
{
	return this->_p;
}

inline
glm::mat4 Camera::getModelViewProjectionMatrix()const
{
	return this->_mvp;
}

/**************************************************************************************/
inline
void Camera::buildMVP()
{
	this->_mvp = this->_p * this->_v * this->_m;
}

/*************************************************************************************/
inline
void Camera::update()
{
	this->buildMVP();
}
/*************************************************************************************/
inline
void Camera::ViewMatrixUpdate()
{
	this->_v           = glm::lookAt(this->_camPosition,this->_target, this->_up);
}

/**************************************************************************************/
inline
glm::vec3 Camera::getFront()const
{
	return glm::normalize(this->_target - this->_camPosition);
}
/**************************************************************************************/
inline
glm::vec3 Camera::getUp()const
{
	return this->_up;
}
/**************************************************************************************/
inline
glm::vec3
Camera::get_left()const
{
	glm::vec3 left(0.0);
	left = glm::cross(this->_up,getFront());
	return glm::normalize(left);
}
/**************************************************************************************/
inline
glm::vec3
Camera::get_right()const
{
	glm::vec3 right(0.0);
	right = glm::cross(getFront(),this->_up);
	return glm::normalize(right);
}

inline
float DegreeToRad(float Angle)
{
	return (Angle  * ( M_PI / 180.00));
}

#endif /* Camera_H_ */
