/******************************************************
 * input.h                                            *
 *                                                    *
 *  Created on: Aug 10, 2015                          *
 *      Author: hakosoft  saouli                      *
 *                                                    *
 *      this class will handl SDL 2 events,mouse and  *
 *      Keyboread.                                    *
 *                                                    *
 ******************************************************/

#ifndef INPUT_H_
#define INPUT_H_

#define KEY_UP 0
#define KEY_DOWN 1

#include <SDL2/SDL.h>

class input
{
	private:
	    SDL_Event _events;							/* this is the event holder used in SDL 2 with we can see what's going on*/

	    bool keybored[SDL_NUM_SCANCODES];		    /* this array will hold the state of all keyboread key true if pressed false if not*/
	    bool MouseBotton[8];						/* this array will hold the state of all mouse bottons*/

	    bool quit;									/* this will indicate if the quite event is activated if true than stop  everything*/

	    int mouse_x;								/* this is the x position of the mouse */
	    int mouse_y;								/* this is the y position of the mouse */

	    int xRel;
	    int yRel;

	public:
		input();									/* simple constractor it will initialize the key arrays to false since no botton is pressed*/
		virtual ~input();

		void updateEvents();						  /* will update the status of the key all time*/
		int  getKey(const SDL_Scancode)const;	      /* this will be used to check if the  giving key is pressed (1) or not (0) */
		int  getMouseButton(const Uint8)const; /* thi will be used to check on mouse button */
		bool mouseMotion()const;							  /* to check if the mouse moved or not*/

		int getMouseX()const;
		int getMouseY()const;

		void HideMouse(bool);
		void exit();
		bool QUIT()const;							  /* this to check if we are  closing or not */
};

/*********************** implementation ****************/
inline
bool input::QUIT()const
{
	return this->quit;
}

inline
void input::exit()
{
	this->quit = true;
}

inline
void input::HideMouse(bool v)
{
    if(v == false)
        SDL_ShowCursor(SDL_ENABLE);
    else
        SDL_ShowCursor(SDL_DISABLE);
}

inline
int input::getKey(const SDL_Scancode key)const
{
	return (this->keybored[key]== true)? 1:0;
}

inline
int input::getMouseButton(const Uint8 button)const
{
	return (this->MouseBotton[button]==true)?1:0;
}

inline
bool input::mouseMotion()const
{
	if(this->xRel == 0 && this->yRel == 0)
		return false;
	else
		return true;
}

inline
int input::getMouseX()const
{
	return this->xRel;//this->mouse_x;
}

inline
int input::getMouseY()const
{
	return this->yRel;//this->mouse_y;
}
#endif /* INPUT_H_ */
