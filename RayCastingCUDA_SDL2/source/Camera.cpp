/*
 * Camera.cpp
 *
 *  Created on: Aug 9, 2015
 *      Author: hakosoft
 */

#include "Camera.h"
#include <stdexcept>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/rotate_vector.hpp>



Camera::Camera(const glm::vec3& eye,const glm::vec3& target, float fav,float aspect,float zNear,float zFar)
	:_camPosition(eye)
	,_target(target)
	,fav(fav)
	,far(zFar)
	,near(zNear)
{

	this->_up          = glm::vec3(0.0,-1.0,0.0);	 					                    /* this is the up vector */
	this->_p           = glm::perspective(this->fav,aspect,this->near,this->far);           /* this is projection matrix */
	this->_v           = glm::lookAt(this->_camPosition,this->_target, this->_up);
	this->_m  = glm::mat4(1.0);
}

Camera::~Camera()
{
	// TODO Auto-generated destructor stub
}



void Camera::move(const float speed,direction d)
{
	  if(d == MOVE_FORWARD)
	  	this->_camPosition = this->_camPosition + speed  * this->getFront();
	  else if(d == MOVE_BACKWARD)
		this->_camPosition = this->_camPosition - speed  * this->getFront();
	  else if(d == MOVE_LEFT)
		this->_camPosition = this->_camPosition + speed  * this->get_left();
	  else if(d == MOVE_RIGHT)
		this->_camPosition = this->_camPosition + speed  * this->get_right();
	  else
		throw std::invalid_argument("DIRECTION invalid");

	  this->_target = this->_camPosition + this->getFront();	// wieried why i do have to recompute target , any why this removed the flikering
}

void Camera::rotateX(float angle)
{
	angle = DegreeToRad(angle);
	glm::vec3 f = this->getFront();
	glm::vec3 forward(f.x,f.y,f.z);
	glm::vec3 yAxis(0.0,0.0,-1.0);
	glm::vec3 hAxis = glm::cross(yAxis,forward);

	glm::normalize(hAxis);
	forward = glm::rotate(forward,angle,hAxis);
	glm::normalize(forward);

	glm::vec3 up = glm::cross(forward,hAxis);
	glm::normalize(up);

	this->_up = up;
	this->_target = this->_camPosition + glm::vec3(forward[0],forward[1],forward[2]);

}

void Camera::rotateY(float angle)
{

	angle = DegreeToRad(angle);
	glm::vec3 f = this->getFront();
	glm::vec3 forward(f.x,f.y,f.z);
	glm::vec3 yAxis(0.0,0.0,-1.0);
	glm::vec3 hAxis = glm::cross(yAxis,forward);

	glm::normalize(yAxis);
	forward = glm::rotate(forward,angle,yAxis);
	glm::normalize(forward);

	glm::vec3 up = glm::cross(forward,hAxis);
	glm::normalize(up);

	this->_up = up;
	this->_target = this->_camPosition + glm::vec3(forward[0],forward[1],forward[2]);
}
