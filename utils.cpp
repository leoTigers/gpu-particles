#include "utils.h"

sf::Vector2f Diff(sf::Vector2f const& v1, sf::Vector2f const& v2, float const& dt)
{
	sf::Vector2f diff = (v1 - v2) / dt;
	return diff;
}

float Diff(float const& a1, float const& a2, float const& dt)
{
	float diff = (a1 - a2) / dt;
	return diff;
}

float Distance(sf::Vector2f const& v1, sf::Vector2f const& v2)
{
	float dist = sqrt(pow(v1.x - v2.x, 2) + pow(v1.y - v2.y, 2));
	return dist;
}

float Norm(sf::Vector2f const& v)
{
	float norm = sqrt(v.x * v.x + v.y * v.y);
	return norm;
}

float map(float value, float start1, float stop1, float start2, float stop2)
{
	return start2 + (stop2 - stop1) * ((value - start1) / (stop1 - start1));
}

sf::Color mapc(float value, float start1, float stop1)
{
	float start2 = 0;
	float stop2 = 1536;

	value = value > stop1 ? stop1 : value;
	float ratio = start2 + (stop2 - stop1) * ((value - start1) / (stop1 - start1));
	sf::Color c;

	if (ratio < 256)
	{
		c.r = 255;
		c.g = ratio;
	}
	else if (ratio < 512)
	{
		c.r = 512 - ratio;
		c.g = 255;
	}
	else if (ratio < 768)
	{
		c.g = 255;
		c.b = ratio - 512;
	}
	else if (ratio < 1024)
	{
		c.g = 1024 - ratio;
		c.b = 255;
	}
	else if (ratio < 1280)
	{
		c.b = 255;
		c.r = ratio - 1024;
	}
	else
	{
		c.b = 1536 - ratio;
		c.r = 255;
	}
	return c;
}
