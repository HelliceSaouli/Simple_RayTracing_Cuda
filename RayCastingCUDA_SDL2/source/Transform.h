/*
 * Transform.h
 *
 *  Created on: Jun 13, 2015
 *      Author: hakosoft
 */

#ifndef TRANSFORM_H_
#define TRANSFORM_H_

#include "glm/glm.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/transform.hpp"

class Transform
{
    private:
		glm::vec3 _pos;
		glm::vec3 _rot;
		glm::vec3 _scale;
	public:
		Transform(const glm::vec3& pos = glm::vec3(), const glm::vec3& rot = glm::vec3(), const glm::vec3& scale = glm::vec3(1.0,1.0,1.0));
		virtual ~Transform();

		// some setter ang getter
		void setPos(const   glm::vec3&);
		void setRot(const   glm::vec3&);
		void setScale(const glm::vec3&);

		glm::vec3 getPos();
		glm::vec3 getRot();
		glm::vec3 getScale();

		glm::mat4 getModel()const;
};

#endif /* TRANSFORM_H_ */
