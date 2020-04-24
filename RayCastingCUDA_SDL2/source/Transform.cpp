/*
 * Transform.cpp
 *
 *  Created on: Jun 13, 2015
 *      Author: hakosoft
 */

#include "Transform.h"

Transform::Transform(const glm::vec3& pos , const glm::vec3& rot , const glm::vec3& scale):
_pos(pos),
_rot(rot),
_scale(scale)
{
}
Transform::~Transform()
{

}

void Transform::setPos(const glm::vec3& pos)
{
	this->_pos = pos;
}

void Transform::setRot(const glm::vec3& rot)
{
	this->_rot = rot;
}

void Transform::setScale(const glm::vec3& scale)
{
	this->_scale = scale;
}

glm::vec3 Transform::getPos()
{
	return this->_pos;
}

glm::vec3 Transform::getRot()
{
	return this->_rot;
}

glm::vec3 Transform::getScale()
{
	return this->_scale;
}

glm::mat4 Transform::getModel()const
{
	glm::mat4 posMatrix   = glm::translate(this->_pos);

	glm::mat4 rotXmatrix  = glm::rotate(this->_rot.x,glm::vec3(1.0,0.0,0.0));
	glm::mat4 rotYmatrix  = glm::rotate(this->_rot.y,glm::vec3(0.0,1.0,0.0));
	glm::mat4 rotZmatrix  = glm::rotate(this->_rot.z,glm::vec3(0.0,0.0,1.0));

	glm::mat4 rotMatrix	  = rotZmatrix * rotYmatrix * rotXmatrix;
	glm::mat4 scaleMartix = glm::scale(this->_scale);


	return posMatrix * rotMatrix * scaleMartix;
}


