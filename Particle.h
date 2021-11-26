#pragma once

#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Color.hpp>

#include "utils.h"


typedef struct {
	float m_mass;
	sf::Vector2f m_speed;
	sf::Vector2f m_force;
} Particle;
/*
class Particle
{
public:
	float m_mass;
	bool m_movable;	
	
	sf::Vector2f m_pos;
	sf::Vector2f m_speed;
	sf::Vector2f m_accel;
	
	sf::Vector2f m_force;

	sf::Color m_color;


public:
	Particle();
	Particle(float mass, sf::Vector2f position, sf::Vector2f speed);

	void setMovable(bool b);
	bool isMovable() const;

	void setMass(float const& mass);
	float getMass() const;

	void setPosition(sf::Vector2f const& position);
	sf::Vector2f getPosition() const;

	void setSpeed(sf::Vector2f const& speed);
	sf::Vector2f getSpeed() const;

	void setAcceleration(sf::Vector2f const& acceleration);
	sf::Vector2f getAcceleration() const;

	void addForce(sf::Vector2f const& force);
	sf::Vector2f getForce() const;
	void clearForce();

	void updatePosition(float const& dt);
	void updateColor();

	sf::Color getColor() const;
};
*/
