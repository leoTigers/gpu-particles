#include "kernel.cuh"
#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include <SFML/Window.hpp>
#include "Particle.h"

#define DIST 0.5



int main()
{
    int width = 1000; // window definition
    int height = 1000;
    sf::Window window(sf::VideoMode(width, height, 32), "OpenGL particles");
	window.setFramerateLimit(60) ;

    glViewport(0, 0, width, height); // viewport definition
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width, height, 0, -1, 1);
    glEnable(GL_POINT_SMOOTH); // allow to have rounded dots
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPointSize(1);

    Particle* particles;
    particles = new Particle[PARTICLE_TOTAL];
	float* vertexCoords = new float[2 * PARTICLE_TOTAL];
	unsigned char* colors = new unsigned char[3 * PARTICLE_TOTAL];


    for (size_t i = 0; i < PARTICLE_BLOCK_WIDTH; i++)
    {
        for (size_t j = 0; j < PARTICLE_BLOCK_HEIGHT; j++)
        {
			Particle particle{ 20, 
				sf::Vector2f(0, 0),
				sf::Vector2f(0, 0)
			};
            particles[i * PARTICLE_BLOCK_WIDTH + j] = particle;
			vertexCoords[(i * PARTICLE_BLOCK_WIDTH + j) * 2] = rand() / (float)RAND_MAX * 50 * cos(rand() / (float)RAND_MAX * 6.28);
			vertexCoords[(i * PARTICLE_BLOCK_WIDTH + j) * 2 + 1] = rand() / (float)RAND_MAX * 50 * sin(rand() / (float)RAND_MAX * 6.28);
			colors[3 * (i * PARTICLE_BLOCK_WIDTH + j)] = 0;
			colors[3 * (i * PARTICLE_BLOCK_WIDTH + j) + 1] = 0;
			colors[3 * (i * PARTICLE_BLOCK_WIDTH + j) + 2] = 0;
        }
    }

    sf::Clock deltaTime;
    float dt = 1e-7;

    bool LMB = false; // is left mouse button hit ?
    float zoom = 1; // zoom factor controled by Z and S keys
    sf::Vector2f camPos(0, 0); // camera position controled with arrow keys
    sf::Vector2f mousePos(0, 0);
	sf::Event event;

	Particle* d_particles;
	float* d_vertexCoords;
	unsigned char* d_colors;

	param p;
	param* d_p;


	cudaError_t error = cudaMalloc((void**)&d_particles, PARTICLE_TOTAL * sizeof(Particle));
	printf("GPUassert: %s \n", cudaGetErrorString(error));
	
	error = cudaMalloc((void**)&d_p, sizeof(param));
	printf("GPUassert: %s \n", cudaGetErrorString(error));

	error = cudaMemcpy(d_particles, particles, PARTICLE_TOTAL * sizeof(Particle), cudaMemcpyHostToDevice);
	printf("GPUassert: %s \n", cudaGetErrorString(error));

	error = cudaMalloc((void**)&d_vertexCoords, 2 * PARTICLE_TOTAL * sizeof(float));
	printf("GPUassert: %s \n", cudaGetErrorString(error));
	error = cudaMalloc((void**)&d_colors, 3 * PARTICLE_TOTAL * sizeof(unsigned char));
	printf("GPUassert: %s \n", cudaGetErrorString(error));

	error = cudaMemcpy(d_vertexCoords, vertexCoords, 2 * PARTICLE_TOTAL * sizeof(float), cudaMemcpyHostToDevice);
	printf("GPUassert: %s \n", cudaGetErrorString(error));
	error = cudaMemcpy(d_colors, colors, 3 * PARTICLE_TOTAL * sizeof(unsigned char), cudaMemcpyHostToDevice);
	printf("GPUassert: %s \n", cudaGetErrorString(error));


	while (window.isOpen()) // main loop, each time this loop is finished, we produce a new frame (so this while loop must run at least 20 times per seconds)
	{
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}

		glClearColor(0, 0, 0, 0); // we clear the screen with black (else, frames would overlay...)
		glClear(GL_COLOR_BUFFER_BIT); // clear the buffer

		// CONTROLS (click, zoom, scroll) ////////////////////

		if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
			LMB = true;
		else
			LMB = false;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Z))
			zoom += 1 * dt * zoom;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
			zoom -= 1 * dt * zoom;
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
			camPos.x += 500 * dt / zoom;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
			camPos.x -= 500 * dt / zoom;
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))
			camPos.y += 500 * dt / zoom;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))
			camPos.y -= 500 * dt / zoom;

		mousePos = (sf::Vector2f(sf::Mouse::getPosition(window).x, sf::Mouse::getPosition(window).y) / zoom - sf::Vector2f(width / 2, height / 2) / zoom - camPos); // we store the current mouse position in this variable

		//CUDA COMPUTE
		p.LMB = LMB;
		p.mousePos = mousePos;
		p.dt = dt;
		cudaMemcpy(d_p, &p, sizeof(param), cudaMemcpyHostToDevice);
		computeParticles(particles, d_particles, d_p, vertexCoords, d_vertexCoords, colors, d_colors);

		/*
		for (int i = 0; i < PARTICLE_TOTAL; i++) // we convert Vector2f positions to the OpenGL's way of storing positions : static arrays of floats
		{
			/*
			sf::Vector2f prev_pos = particles[i].m_pos;
			particles[i].m_pos = particles[i].m_pos + particles[i].m_speed * dt + .5f * particles[i].m_force / particles[i].m_mass * dt * dt;
			particles[i].m_speed = (particles[i].m_pos - prev_pos) / dt;
			//particles[i].updatePosition(dt);
			//particles[i].updateColor();

			vertexCoords[2 * i] = particles[i].m_pos.x;
			vertexCoords[2 * i + 1] = particles[i].m_pos.y;
			//sf::Color c = mapc(Norm(particles[i].m_speed), 0, 200);
			colors[3 * i] = particles[i].m_color.r;//particles[i].getColor().r;
			colors[3 * i + 1] = particles[i].m_color.g;//particles[i].getColor().g;
			colors[3 * i + 2] = particles[i].m_color.b;// map(Norm(particles[i].m_speed), 0, 1000, 0, 255);;// particles[i].getColor().b;
		}*/

		glPushMatrix(); // time to draw the frame

		glTranslatef(width / 2.f, height / 2.f, 0); // apply zoom
		glScaled(zoom, zoom, zoom);

		glTranslated(camPos.x, camPos.y, 0); // apply scroll

		glEnableClientState(GL_VERTEX_ARRAY); // we are using VBAs : here's how to draw them
		glEnableClientState(GL_COLOR_ARRAY);

		glVertexPointer(2, GL_FLOAT, 0, vertexCoords);
		glColorPointer(3, GL_UNSIGNED_BYTE, 0, colors);
		glDrawArrays(GL_POINTS, 0, PARTICLE_TOTAL);
		
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);

		glPopMatrix();

		glFlush();
		window.display(); // show the window with its frame drawn in it

		dt = deltaTime.restart().asSeconds(); // measure the time it took to complete all the calculations for this particular frame (dt=1/framerate) 
		printf("\r                              \rFPS: %f", 1.f / dt);
	}

	cudaFree(d_particles);
	//cudaFree(d_vertexCoords);
	//cudaFree(d_colors);
	return 0;
}