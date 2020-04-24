/*
 * input.cpp
 *
 *  Created on: Aug 10, 2015
 *      Author: hakosoft
 */

#include "input.h"

input::input()
{
	this->mouse_x = 0;
	this->mouse_y = 0;

	this->xRel = 0;
	this->yRel = 0;

	this->quit    = false;

	//Keyboread botton initialize
	for(int i = 0 ; i < SDL_NUM_SCANCODES; i++)
		this->keybored[i] = false;				/* all botton are not pressed*/

	// Mouse botton initialize
	for(int i = 0 ; i < 8; i++)
		this->MouseBotton[i] = false;

	 SDL_SetRelativeMouseMode(SDL_TRUE);
}

input::~input()
{
	// TODO Auto-generated destructor stub
}
/********************************************************************************************/
void input::updateEvents()
{
	this->xRel = 0;
	this->yRel = 0;
	while(SDL_PollEvent(&this->_events))
	{
		switch(this->_events.type)
		{
			case SDL_KEYDOWN :
					this->keybored[this->_events.key.keysym.scancode] = true;

					break;
			case SDL_KEYUP:
					this->keybored[this->_events.key.keysym.scancode] = false;
					break;

			case SDL_MOUSEBUTTONDOWN:
				this->MouseBotton[this->_events.button.button] = true;
				break;

			case SDL_MOUSEBUTTONUP:
				this->MouseBotton[this->_events.button.button] = false;
				break;

			case SDL_MOUSEMOTION:
				this->mouse_x = this->_events.motion.x;
				this->mouse_y = this->_events.motion.y;

				this->xRel    = this->_events.motion.xrel;
				this->yRel    = this->_events.motion.yrel;

				break;

			case SDL_QUIT:
				this->quit = true;
				break;

			default :
				break;
		}
	}
}
