#pragma once

#include <SFML/Graphics.hpp>
#include <cmath>


sf::Vector2f Diff(sf::Vector2f const& v1, sf::Vector2f const& v2, float const& dt);
float Diff(float const& a1, float const& a2, float const& dt);

float Distance(sf::Vector2f const& v1, sf::Vector2f const& v2);
float Norm(sf::Vector2f const& v);

float map(float value, float start1, float stop1, float start2, float stop2);

sf::Color mapc(float value, float start1, float stop1);