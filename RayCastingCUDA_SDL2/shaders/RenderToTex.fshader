#version 450

uniform sampler2D renderedTexture;

in vec2 texCoord0;

void main()
{
	gl_FragColor = texture2D(renderedTexture,texCoord0);
}