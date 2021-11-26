#include "Particle.h"
/*
Particle::Particle()
{
	m_mass = 50;
	m_pos = sf::Vector2f(0, 0);
	m_speed = sf::Vector2f(0, 0);
	m_accel = sf::Vector2f(0, 0);
	m_force = sf::Vector2f(0, 0);
	m_movable = true;
	m_color = sf::Color::Cyan;
}

Particle::Particle(float mass, sf::Vector2f position, sf::Vector2f speed)
{
	m_mass = mass;
	m_pos = position;
	m_speed = speed;
	m_accel = sf::Vector2f(0, 0);
	m_force = sf::Vector2f(0, 0);
	m_movable = true;
	m_color = sf::Color::Cyan;
}

void Particle::setMovable(bool b)
{
	m_movable = b;
}

bool Particle::isMovable() const
{
	return m_movable;
}

void Particle::setMass(float const& mass)
{
	m_mass = mass > 0 ? mass : 0;
}

float Particle::getMass() const
{
	return m_mass;
}

void Particle::setPosition(sf::Vector2f const& position)
{
	m_pos = position;
}

sf::Vector2f Particle::getPosition() const
{
	return m_pos;
}

void Particle::setSpeed(sf::Vector2f const& speed)
{
	m_speed = speed;
}

sf::Vector2f Particle::getSpeed() const
{
	return m_speed;
}

void Particle::setAcceleration(sf::Vector2f const& acceleration)
{
	m_accel = acceleration;
}

sf::Vector2f Particle::getAcceleration() const
{
	return m_accel;
}

void Particle::addForce(sf::Vector2f const& force)
{
	m_force += force;
}

sf::Vector2f Particle::getForce() const
{
	return m_force;
}

void Particle::clearForce()
{
	m_force = { 0, 0 };
}

void Particle::updatePosition(float const& dt)
{
	sf::Vector2f prev_pos = m_pos;
	m_pos = m_pos + m_speed * dt + .5f * m_force / m_mass * dt * dt;
	m_speed = (m_pos - prev_pos) / dt;
}

void Particle::updateColor()
{
	m_color.b = map(Norm(m_speed), 0, 1000, 0, 255); 
}

sf::Color Particle::getColor() const
{
	return m_color;
}
*/