/*
 * Mesh.cpp
 *
 *  Created on: Jun 13, 2015
 *      Author: hakosoft
 */

#include "Mesh.h"

//for mesh that we extracted from
Mesh::Mesh(std::vector< glm::vec3 > vertices,std::vector< glm::vec3 > colors, unsigned int numVertices, GLint *Attribut)
{
	this->_drawCount = numVertices;
	glGenVertexArrays(1,&this->_vao);
	glBindVertexArray(this->_vao);

	//Vertices
	glGenBuffers(NUM_BUFFER-1,this->_vbo);
	glBindBuffer(GL_ARRAY_BUFFER,this->_vbo[POSITION_VB]);

	glBufferData(GL_ARRAY_BUFFER,vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);//it's look like cudaMemCopy xD
	glEnableVertexAttribArray(Attribut[0]); // there is only one attribute for now
	glVertexAttribPointer(Attribut[0], 3, GL_FLOAT, GL_FALSE,0,0); // how to read our data

	//colors
	glBindBuffer(GL_ARRAY_BUFFER,this->_vbo[COLOR_VB]);
	glBufferData(GL_ARRAY_BUFFER,colors.size() * sizeof(glm::vec3), &colors[0], GL_STATIC_DRAW);//it's look like cudaMemCopy xD
	glEnableVertexAttribArray(Attribut[1]); // there is only one attribute for now
	glVertexAttribPointer(Attribut[1], 3, GL_FLOAT, GL_FALSE,0,0); // how to read our data

	glBindVertexArray(0); // stop the binding after this instruction

}

//for the quard that we are gonna draw on it i know 2 constructor  is ugly but hey what i can do XD
Mesh::Mesh(Vertex* vertices, unsigned int numVertices, GLint *Attribut)
{
	this->_drawCount = numVertices;
	glGenVertexArrays(1,&this->_vao);
	glBindVertexArray(this->_vao);

	std::vector<glm::vec3> position;
	std::vector<glm::vec2> TexCoord;
	std::vector<glm::vec3> colors;

	position.reserve(numVertices);
	TexCoord.reserve(numVertices);
	colors.reserve(numVertices);

	for(int i = 0; i < numVertices; i++)
	{
		position.push_back(vertices[i].getVertexPos());
		TexCoord.push_back(vertices[i].getTexCoords());
		colors.push_back(vertices[i].getColors());
	}

	// position
	glGenBuffers(NUM_BUFFER,this->_vbo);
	glBindBuffer(GL_ARRAY_BUFFER,this->_vbo[POSITION_VB]);

	glBufferData(GL_ARRAY_BUFFER,position.size() * sizeof(glm::vec3), &position[0], GL_STATIC_DRAW);//it's look like cudaMemCopy xD
	glEnableVertexAttribArray(Attribut[0]); // there is only one attribute for now
	glVertexAttribPointer(Attribut[0], 3, GL_FLOAT, GL_FALSE,0,0); // how to read our data

	//colors
	glBindBuffer(GL_ARRAY_BUFFER,this->_vbo[COLOR_VB]);

	glBufferData(GL_ARRAY_BUFFER,colors.size() * sizeof(glm::vec3), &colors[0], GL_STATIC_DRAW);//it's look like cudaMemCopy xD
	glEnableVertexAttribArray(Attribut[1]); // there is only one attribute for now
	glVertexAttribPointer(Attribut[1], 3, GL_FLOAT, GL_FALSE,0,0); // how to read our data

	//UV
	glBindBuffer(GL_ARRAY_BUFFER,this->_vbo[TEXCOORD_VB]);

	glBufferData(GL_ARRAY_BUFFER,TexCoord.size() * sizeof(glm::vec2), &TexCoord[0], GL_STATIC_DRAW);//it's look like cudaMemCopy xD
	glEnableVertexAttribArray(Attribut[2]); // there is only one attribute for now
	glVertexAttribPointer(Attribut[2], 2, GL_FLOAT, GL_FALSE,0,0); // how to read our data


	glBindVertexArray(0); // stop the binding after this instruction
}

Mesh::~Mesh()
{
	glDeleteVertexArrays(1,&this->_vao);
}

void Mesh::Draw()
{
	glBindVertexArray(this->_vao);
	// the drawing
	glDrawArrays(GL_POINTS, 0,this->_drawCount);
	glBindVertexArray(0);
}

void Mesh::DrawQuad()
{
	glBindVertexArray(this->_vao);
	// the drawing
	glDrawArrays(GL_TRIANGLES, 0,this->_drawCount);
	glBindVertexArray(0);
}

